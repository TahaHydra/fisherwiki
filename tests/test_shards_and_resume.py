"""Shard integrity and resume correctness.

These are the two properties a multi-night run depends on and that no metric
would reveal if broken. A stream that is not deterministic makes a resumed epoch
silently different from an uninterrupted one; a checkpoint that is not atomic
turns a power cut into a lost week; a skip-ahead that is off by one quietly
re-trains or skips samples every single night.
"""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch", reason="shards/resume need torch")

from fwml.checkpoint import (  # noqa: E402
    CheckpointManager,
    TrainerState,
    capture_rng,
    restore_rng,
)
from fwml.shards import (  # noqa: E402
    Sample,
    ShardIndex,
    ShardReader,
    ShardStream,
    ShardWriter,
    verify_shards,
)


def _write(tmp_path, n=500, shard_size=64, split_of=lambda i: "train"):
    with ShardWriter(tmp_path, shard_size=shard_size) as w:
        for i in range(n):
            data = f"image-payload-{i}".encode() * 3
            w.write(Sample(
                key=f"k{i:05d}", data=data, class_id=i % 7, taxon_id=1000 + i % 7,
                sha256=hashlib.sha256(data).hexdigest(),
                split=split_of(i), group_id=f"g{i // 3}",
            ))
    return tmp_path


class TestShardStorage:
    def test_roundtrip_bytes_are_exact(self, tmp_path):
        _write(tmp_path, n=200, shard_size=32)
        index = ShardIndex(tmp_path)
        assert len(index) == 200
        with ShardReader(index) as r:
            for row in range(200):
                expected = f"image-payload-{row}".encode() * 3
                assert r.read(row) == expected

    def test_verify_detects_corruption(self, tmp_path):
        _write(tmp_path, n=120, shard_size=40)
        assert verify_shards(tmp_path, log=lambda m: None)["mismatched"] == 0

        # Flip a byte inside the first shard's payload region.
        shard = sorted(tmp_path.glob("*.tar"))[0]
        raw = bytearray(shard.read_bytes())
        index = ShardIndex(tmp_path)
        offset = index.locate(0)[1]
        raw[offset] ^= 0xFF
        shard.write_bytes(bytes(raw))

        report = verify_shards(tmp_path, log=lambda m: None)
        assert report["mismatched"] >= 1, (
            "a corrupted shard passed verification - the sha256 in the index is "
            "the only thing standing between a bad transfer and silent accuracy loss"
        )

    def test_interrupted_shard_is_not_left_looking_complete(self, tmp_path):
        w = ShardWriter(tmp_path, shard_size=1000)
        for i in range(10):
            data = f"x{i}".encode()
            w.write(Sample(f"k{i}", data, 0, 1, hashlib.sha256(data).hexdigest(),
                           "train", "g0"))
        # Simulate a crash: never call close().
        assert list(tmp_path.glob("*.tar")) == []
        assert list(tmp_path.glob("*.tar.tmp")), "no in-progress shard present"


class TestStreamDeterminism:
    def test_same_seed_and_epoch_give_the_same_order(self, tmp_path):
        _write(tmp_path, n=400, shard_size=50)
        index = ShardIndex(tmp_path)
        a = list(ShardStream(index, "train", seed=7).epoch_rows(3))
        b = list(ShardStream(index, "train", seed=7).epoch_rows(3))
        assert a == b

    def test_different_epochs_give_different_orders(self, tmp_path):
        _write(tmp_path, n=400, shard_size=50)
        index = ShardIndex(tmp_path)
        a = list(ShardStream(index, "train", seed=7).epoch_rows(0))
        b = list(ShardStream(index, "train", seed=7).epoch_rows(1))
        assert a != b
        assert sorted(a) == sorted(b), "an epoch must be a permutation, not a resample"

    def test_every_sample_appears_exactly_once(self, tmp_path):
        _write(tmp_path, n=400, shard_size=50)
        index = ShardIndex(tmp_path)
        rows = list(ShardStream(index, "train", seed=1).epoch_rows(0))
        assert sorted(rows) == list(range(400))

    def test_split_filtering_is_respected(self, tmp_path):
        _write(tmp_path, n=300, shard_size=40,
               split_of=lambda i: "train" if i % 3 else "validation")
        index = ShardIndex(tmp_path)
        rows = list(ShardStream(index, "validation", seed=1).epoch_rows(0))
        splits = index.table.column("split").to_pylist()
        assert rows and all(splits[r] == "validation" for r in rows)


class TestSkipAheadResume:
    """The core of nightly training: stopping mid-epoch must lose nothing and
    repeat nothing."""

    @pytest.mark.parametrize("skip", [0, 1, 137, 399])
    def test_skip_resumes_exactly_where_it_stopped(self, tmp_path, skip):
        _write(tmp_path, n=400, shard_size=50)
        index = ShardIndex(tmp_path)
        full = list(ShardStream(index, "train", seed=11).epoch_rows(2))
        tail = list(ShardStream(index, "train", seed=11).epoch_rows(2, skip=skip))
        assert tail == full[skip:], (
            "a resumed epoch diverged from the uninterrupted one - samples "
            "would be silently retrained or skipped every night"
        )

    def test_many_stops_cover_the_epoch_exactly_once(self, tmp_path):
        """Simulate a week of nights: stop and resume repeatedly, and assert the
        union is exactly the epoch with no duplicates."""
        _write(tmp_path, n=500, shard_size=40)
        index = ShardIndex(tmp_path)
        seen: list[int] = []
        consumed = 0
        for chunk in (60, 25, 180, 5, 90, 400):
            part = list(ShardStream(index, "train", seed=5).epoch_rows(
                0, skip=consumed))[:chunk]
            seen.extend(part)
            consumed += len(part)
            if consumed >= 500:
                break
        assert sorted(seen) == list(range(500))
        assert len(seen) == len(set(seen)), "a sample was trained on twice"

    def test_rank_shards_are_disjoint_and_balanced(self, tmp_path):
        _write(tmp_path, n=800, shard_size=50)
        index = ShardIndex(tmp_path)
        per_rank = [
            list(ShardStream(index, "train", seed=3, world_size=4, rank=r)
                 .epoch_rows(0))
            for r in range(4)
        ]
        flat = [r for rows in per_rank for r in rows]
        assert len(flat) == len(set(flat)), "two ranks trained on the same sample"
        counts = {len(rows) for rows in per_rank}
        assert len(counts) == 1, (
            f"ranks got different sample counts {counts} - DDP all-reduce "
            f"deadlocks when one rank runs out of batches first"
        )


class TestCheckpointCrashSafety:
    def test_a_failed_write_leaves_the_previous_checkpoint_intact(self, tmp_path):
        m = CheckpointManager(tmp_path)
        m.save({"format": 2, "value": "good"})
        assert m.latest.exists()

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("disk full")

        with pytest.raises(Exception):
            m.save({"format": 2, "value": Unpicklable()})

        restored = m.load()
        assert restored["value"] == "good", (
            "a failed checkpoint destroyed the last good one"
        )
        assert not list(tmp_path.glob("*.tmp")), "left a temp file behind"

    def test_a_corrupt_latest_falls_back_to_the_previous(self, tmp_path):
        m = CheckpointManager(tmp_path)
        m.save({"format": 2, "value": "first"})
        m.save({"format": 2, "value": "second"})
        m.latest.write_bytes(b"not a torch file at all")

        restored = m.load()
        assert restored is not None and restored["value"] == "first"

    def test_state_survives_a_round_trip(self, tmp_path):
        m = CheckpointManager(tmp_path)
        state = TrainerState(epoch=3, global_step=12345, samples_this_epoch=987,
                             best_metric=0.61, resolution=384)
        model = torch.nn.Linear(4, 3)
        opt = torch.optim.AdamW(model.parameters())
        m.save_training_state(model=model, optimizer=opt, scheduler=None,
                              scaler=None, state=state, config={"backbone": "x"})
        got = TrainerState.from_dict(m.load()["state"])
        assert (got.epoch, got.global_step, got.samples_this_epoch) == (3, 12345, 987)
        assert got.resolution == 384

    def test_ddp_and_single_gpu_checkpoints_interchange(self, tmp_path):
        """A run started on one GPU must resume on eight, and the reverse."""
        from fwml.checkpoint import load_model_state

        model = torch.nn.Linear(4, 3)
        wrapped_state = {f"module.{k}": v for k, v in model.state_dict().items()}
        fresh = torch.nn.Linear(4, 3)
        load_model_state(fresh, wrapped_state)
        for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
            assert torch.allclose(a, b)


class TestRngCapture:
    def test_python_numpy_and_torch_streams_all_restore(self):
        import random

        import numpy as np

        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        snapshot = capture_rng()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        random.seed(999); np.random.seed(999); torch.manual_seed(999)
        restore_rng(snapshot)
        got = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        assert got == expected, (
            "an RNG stream was not restored; a resumed run would augment and "
            "sample differently from an uninterrupted one"
        )
