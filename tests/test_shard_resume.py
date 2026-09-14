"""Shard preparation must survive interruption without redoing finished work.

The failure this guards against is expensive and quiet: the first implementation
kept every index row in RAM and wrote one combined ``index.parquet`` only when
the writer closed. Stopping a long preparation run near the end therefore left
a directory full of perfectly valid tar files that nothing could describe, and
a rerun started from image one and overwrote them.
"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("pyarrow", reason="shard indexes are parquet")

from fwml.shards import (  # noqa: E402
    Sample,
    ShardIndex,
    ShardReader,
    ShardWriter,
    adopt_shard,
    rebuild_index,
    scan_finalised,
    shard_sidecar,
)


def _sample(i: int, split: str = "train") -> Sample:
    data = f"payload-{i:05d}".encode() * 4
    return Sample(key=f"k{i:05d}", data=data, class_id=i % 5, taxon_id=1000 + i % 5,
                  sha256=hashlib.sha256(data).hexdigest(), split=split,
                  group_id=f"g{i // 3}")


def _write(out, first, last, *, shard_size=8, start_index=0):
    w = ShardWriter(out, prefix="train", shard_size=shard_size,
                    start_index=start_index)
    for i in range(first, last):
        w.write(_sample(i))
    return w


class TestFinalisedShardsAreDurable:
    def test_each_finalised_shard_gets_its_own_index(self, tmp_path):
        w = _write(tmp_path, 0, 24)
        w.close()
        shards = sorted(tmp_path.glob("*.tar"))
        assert len(shards) == 3
        for s in shards:
            assert shard_sidecar(s).exists(), f"{s.name} has no durable index"

    def test_an_interrupted_writer_still_leaves_finalised_shards_described(
        self, tmp_path
    ):
        """Simulates the real failure: writer never closes."""
        w = _write(tmp_path, 0, 20)          # 2 full shards + 4 in flight
        del w                                 # no close()

        state = scan_finalised(tmp_path)
        assert len(state["shards"]) == 2
        assert len(state["keys"]) == 16, "finalised shards are not self-describing"
        assert state["stale_tmp"], "the in-flight shard should still be a .tmp"
        assert state["next_index"] == 2


class TestResume:
    def test_resume_skips_finished_and_continues_numbering(self, tmp_path):
        w = _write(tmp_path, 0, 20)
        del w
        state = scan_finalised(tmp_path)
        for stale in state["stale_tmp"]:
            stale.unlink()

        done = state["keys"]
        remaining = [i for i in range(24) if f"k{i:05d}" not in done]
        assert len(remaining) == 8

        w2 = ShardWriter(tmp_path, prefix="train", shard_size=8,
                         start_index=state["next_index"])
        for i in remaining:
            w2.write(_sample(i))
        w2.close()

        names = sorted(p.name for p in tmp_path.glob("*.tar"))
        assert names == ["train-00000.tar", "train-00001.tar", "train-00002.tar"]

        index = ShardIndex(tmp_path)
        keys = index.table.column("key").to_pylist()
        assert len(keys) == len(set(keys)) == 24, "duplicate or missing samples"
        assert sorted(keys) == [f"k{i:05d}" for i in range(24)]

    def test_two_interruptions_still_produce_every_sample_once(self, tmp_path):
        written: list[int] = []
        for stop_after in (10, 9):
            state = scan_finalised(tmp_path)
            for stale in state["stale_tmp"]:
                stale.unlink()
            done = state["keys"]
            todo = [i for i in range(32) if f"k{i:05d}" not in done]
            w = ShardWriter(tmp_path, prefix="train", shard_size=8,
                            start_index=state["next_index"])
            for i in todo[:stop_after]:
                w.write(_sample(i))
                written.append(i)
            del w                             # interrupted again

        state = scan_finalised(tmp_path)
        for stale in state["stale_tmp"]:
            stale.unlink()
        todo = [i for i in range(32) if f"k{i:05d}" not in state["keys"]]
        w = ShardWriter(tmp_path, prefix="train", shard_size=8,
                        start_index=state["next_index"])
        for i in todo:
            w.write(_sample(i))
        w.close()

        keys = ShardIndex(tmp_path).table.column("key").to_pylist()
        assert len(keys) == len(set(keys)), "a sample was written twice"
        assert sorted(keys) == [f"k{i:05d}" for i in range(32)], "a sample was lost"

    def test_resumed_shards_still_read_back_correctly(self, tmp_path):
        w = _write(tmp_path, 0, 16)
        del w
        for stale in scan_finalised(tmp_path)["stale_tmp"]:
            stale.unlink()
        state = scan_finalised(tmp_path)
        w2 = ShardWriter(tmp_path, prefix="train", shard_size=8,
                         start_index=state["next_index"])
        for i in range(16, 24):
            w2.write(_sample(i))
        w2.close()

        index = ShardIndex(tmp_path)
        with ShardReader(index) as r:
            for row in range(len(index)):
                key = index.table.column("key")[row].as_py()
                i = int(key[1:])
                assert r.read(row) == f"payload-{i:05d}".encode() * 4


class TestAdoptingLegacyShards:
    def test_a_shard_with_no_sidecar_is_reconstructed_from_the_tar(self, tmp_path):
        """The real directory left by the old implementation: valid tars, no
        index at all. The tar itself carries every member name and offset, so
        the index is recoverable rather than the work being lost."""
        w = _write(tmp_path, 0, 16)
        del w
        for s in tmp_path.glob("*.tar"):
            shard_sidecar(s).unlink(missing_ok=True)      # legacy state
        for stale in tmp_path.glob("*.tar.tmp"):
            stale.unlink()

        assert scan_finalised(tmp_path)["keys"] == set(), "precondition"

        meta = {f"k{i:05d}": {"class_id": i % 5, "taxon_id": 1000 + i % 5,
                              "split": "train", "group_id": f"g{i // 3}",
                              "sha256": None} for i in range(16)}
        total = sum(adopt_shard(s, meta.get) for s in sorted(tmp_path.glob("*.tar")))
        assert total == 16

        state = scan_finalised(tmp_path)
        assert len(state["keys"]) == 16
        assert state["next_index"] == 2

        rebuild_index(tmp_path)
        index = ShardIndex(tmp_path)
        with ShardReader(index) as r:
            for row in range(len(index)):
                i = int(index.table.column("key")[row].as_py()[1:])
                assert r.read(row) == f"payload-{i:05d}".encode() * 4, (
                    "adopted offsets do not point at the right bytes"
                )
