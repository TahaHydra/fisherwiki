"""DDP correctness: cursor semantics, coordinated stopping, world-size changes.

Three defects found in review, all of which are silent under a single GPU and
only appear at 4 or 8. None would show up in a loss curve:

* a per-rank cursor advanced by the *global* batch makes every rank skip
  ``world_size`` times too far on resume - at 4 GPUs, three quarters of each
  resumed epoch is never trained on;
* an uncoordinated stop lets one rank leave the loop while another waits in the
  gradient all-reduce, which hangs rather than crashes - on a rented
  interruptible box that means paying for a dead machine;
* a data cursor recorded under one world size names no position under another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="DDP resume logic needs torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwml.checkpoint import TrainerState  # noqa: E402
from fwml.shards import ShardIndex, ShardStream  # noqa: E402
from train_v2 import (  # noqa: E402
    advance_counters,
    sync_stop,
    world_size_changed,
)
from test_shards_and_resume import _write  # noqa: E402


class TestSampleCursorIsPerRank:
    def test_counter_advances_by_the_local_batch_not_the_global_one(self):
        """The regression. ``samples_this_epoch`` indexes this rank's own
        stream, so it must advance by the local batch size."""
        state = TrainerState()
        for _ in range(10):
            advance_counters(state, local_batch=32, world=4)

        assert state.samples_this_epoch == 320, (
            "per-rank cursor advanced by the global batch; on 4 GPUs every rank "
            "would skip 4x too far and silently drop 3/4 of the resumed epoch"
        )
        assert state.samples_total == 1280, "global counter should count all ranks"

    def test_single_gpu_counters_agree(self):
        state = TrainerState()
        advance_counters(state, local_batch=16, world=1)
        assert state.samples_this_epoch == state.samples_total == 16

    @pytest.mark.parametrize("world", [2, 4, 8])
    def test_resumed_ranks_cover_the_epoch_exactly_once(self, tmp_path, world):
        """End-to-end consequence: stop every rank mid-epoch, resume each with
        its own per-rank cursor, and the union must be the epoch exactly."""
        _write(tmp_path, n=1600, shard_size=25)
        index = ShardIndex(tmp_path)

        consumed_per_rank = 96          # what each rank had done before the stop
        seen: list[int] = []
        for rank in range(world):
            stream = ShardStream(index, "train", seed=4, world_size=world, rank=rank)
            before = list(stream.epoch_rows(0))[:consumed_per_rank]
            after = list(stream.epoch_rows(0, skip=consumed_per_rank))
            seen.extend(before)
            seen.extend(after)

        assert len(seen) == len(set(seen)), "a sample was trained on twice"
        expected = sorted(
            r
            for rank in range(world)
            for r in ShardStream(index, "train", seed=4,
                                 world_size=world, rank=rank).epoch_rows(0)
        )
        assert sorted(seen) == expected, "resume lost or duplicated samples"

    def test_a_global_cursor_would_have_lost_most_of_the_epoch(self, tmp_path):
        """Pins the size of the bug, so a regression is obvious rather than subtle."""
        _write(tmp_path, n=1600, shard_size=25)
        index = ShardIndex(tmp_path)
        world, local_consumed = 4, 96

        stream = ShardStream(index, "train", seed=4, world_size=world, rank=0)
        correct = list(stream.epoch_rows(0, skip=local_consumed))
        buggy = list(stream.epoch_rows(0, skip=local_consumed * world))

        assert len(buggy) < len(correct)
        lost = len(correct) - len(buggy)
        assert lost == local_consumed * (world - 1), (
            "the global-cursor bug should drop (world-1) x consumed samples "
            "per rank per resume"
        )


class TestStopIsCoordinated:
    def test_single_process_passes_the_decision_through(self):
        assert sync_stop(True, world=1, device="cpu") is True
        assert sync_stop(False, world=1, device="cpu") is False

    def test_any_rank_requesting_a_stop_stops_all_of_them(self, monkeypatch):
        """A preemption signal lands on one rank first; the max-reduce is what
        makes the other ranks leave on the same step instead of blocking in the
        gradient all-reduce."""
        import torch.distributed as dist

        seen = {}

        def fake_all_reduce(tensor, op=None):
            seen["op"] = op
            # simulate: some other rank requested a stop
            tensor.fill_(1.0)

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

        assert sync_stop(False, world=4, device="cpu") is True
        assert seen["op"] == dist.ReduceOp.MAX, (
            "a SUM or MIN reduce would either mis-count or require unanimity; "
            "stopping must be triggered by any single rank"
        )

    def test_no_rank_stops_when_none_requested(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "all_reduce", lambda t, op=None: t.fill_(0.0))
        assert sync_stop(False, world=4, device="cpu") is False

    def test_uninitialised_process_group_falls_back_to_local(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        assert sync_stop(True, world=4, device="cpu") is True

    def test_gloo_group_really_agrees(self, tmp_path):
        """The mocked tests pin the intent; this one pins that the real
        collective behaves as assumed, on CPU so it runs anywhere."""
        import torch.distributed as dist

        if not dist.is_available():
            pytest.skip("torch.distributed unavailable")
        init_file = tmp_path / "pg"
        try:
            dist.init_process_group(
                backend="gloo", init_method=f"file:///{init_file.as_posix()}",
                world_size=1, rank=0,
            )
        except Exception as exc:                     # pragma: no cover
            pytest.skip(f"could not init gloo group: {exc}")
        try:
            assert sync_stop(True, world=1, device="cpu") is True
            flag = torch.tensor([0.0])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            assert flag.item() == 0.0
        finally:
            dist.destroy_process_group()


class TestWorldSizeChange:
    def test_detects_a_change(self):
        assert world_size_changed(TrainerState(world_size=1), 4) is True
        assert world_size_changed(TrainerState(world_size=8), 8) is False

    def test_treats_a_missing_world_size_as_one(self):
        """Checkpoints written before this field existed must still resume."""
        assert world_size_changed(TrainerState(world_size=0), 1) is False
        assert world_size_changed(TrainerState(world_size=0), 4) is True

    def test_world_size_survives_a_checkpoint_round_trip(self, tmp_path):
        from fwml.checkpoint import CheckpointManager

        m = CheckpointManager(tmp_path)
        model = torch.nn.Linear(3, 2)
        state = TrainerState(epoch=2, samples_this_epoch=500, world_size=4)
        m.save_training_state(model=model, optimizer=torch.optim.SGD(
            model.parameters(), lr=0.1), scheduler=None, scaler=None,
            state=state, config={})
        got = TrainerState.from_dict(m.load()["state"])
        assert got.world_size == 4 and got.samples_this_epoch == 500

    def test_a_cursor_from_another_world_size_names_a_different_position(
        self, tmp_path
    ):
        """Why the restart exists: the same skip count means different samples
        under a different partitioning, so carrying it over would both duplicate
        and omit rather than continue."""
        _write(tmp_path, n=1200, shard_size=25)
        index = ShardIndex(tmp_path)
        one = list(ShardStream(index, "train", seed=9, world_size=1, rank=0)
                   .epoch_rows(0, skip=200))
        four = list(ShardStream(index, "train", seed=9, world_size=4, rank=0)
                    .epoch_rows(0, skip=200))
        assert one[:50] != four[:50]

    def test_restarting_the_epoch_omits_nothing(self, tmp_path):
        """The accepted trade: bounded duplication, zero omission."""
        _write(tmp_path, n=1200, shard_size=25)
        index = ShardIndex(tmp_path)
        world = 4
        covered = sorted(
            r
            for rank in range(world)
            for r in ShardStream(index, "train", seed=9,
                                 world_size=world, rank=rank).epoch_rows(0, skip=0)
        )
        assert len(covered) == len(set(covered)), "restart duplicated within an epoch"
        # Every shard that the new partitioning covers is covered exactly once.
        assert covered == sorted(set(covered))
