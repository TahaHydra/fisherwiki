"""Portable, resumable shard storage for V2 training data.

Why shards at all
-----------------
V1 trained from 310,393 loose JPEGs in the content-addressed store. That works
at 300k files on a local SSD and falls apart at 3M across the storage this
machine actually has: the NAS does roughly 100 small files/second, so a single
epoch of 3M loose files would spend **over eight hours in filesystem overhead
alone**, before decoding a pixel.

A shard is a plain **tar** file holding a few thousand samples, plus a sidecar
index. Tar because it is the most portable container that exists - readable by
WebDataset, by `tar -x`, and by twenty lines of Python - which matters when the
final run may happen on a rented NVIDIA box. Nothing here is specific to this
machine, this GPU vendor, or this framework.

The index is what makes it fast *and* resumable
-----------------------------------------------
Each shard carries a sidecar `.idx` (Parquet) with one row per sample: byte
offset, length, class id, sha256, split, leak-group id. That gives three things
V1 could not do:

* **Sampling without touching the tars.** Class-balanced sampling, split
  filtering and epoch permutation all run over ~16 bytes/sample in RAM - about
  48 MB at 3M samples - instead of opening files.
* **Skip-ahead resume.** Resuming mid-epoch means skipping N samples of a
  deterministic stream. With offsets known, skipping costs a seek, not a
  decode, so a resume 80% through a 10-hour epoch is instant rather than eight
  hours of wasted work.
* **Verification.** Every sample's sha256 is recorded, so a shard can be
  checked against the provenance store without decoding it.

Access pattern, and why it is sequential
----------------------------------------
Samples are read **in shard order, sequentially**, through a seeded shuffle
buffer, rather than by random access across the whole dataset. Measured on this
machine, the difference is decisive: D: sustains ~120 MB/s sequentially but only
a few hundred random IOPS, and 3M prepared crops at ~100 KB will not fit in E:'s
229 GB. Sequential reads let the cold tier serve training directly; random
access would not.

Randomness is preserved by shuffling **which** shard comes next and buffering
within it, which is the WebDataset model. The stream is a pure function of
`(seed, epoch, world_size, rank)`, so it is identical on every machine and can
be replayed exactly - that is what makes resume correct rather than approximate.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Samples per shard. Large enough that tar overhead is negligible and
#: sequential runs are long; small enough that a resume replays little and a
#: single shard still fits comfortably in page cache.
DEFAULT_SHARD_SIZE = 2048

INDEX_COLUMNS = (
    "key", "shard", "offset", "length", "class_id", "taxon_id",
    "sha256", "split", "group_id",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Sample:
    key: str
    data: bytes
    class_id: int
    taxon_id: int
    sha256: str
    split: str
    group_id: str


def shard_sidecar(shard_path: Path) -> Path:
    """Durable per-shard index, written when that shard is finalised."""
    return Path(str(shard_path) + ".idx.parquet")


def _write_table(rows: list[dict], path: Path) -> None:
    """Write index rows atomically: temp file, then rename."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({c: pa.array([r[c] for r in rows]) for c in INDEX_COLUMNS})
    tmp = Path(str(path) + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def scan_finalised(out_dir: Path) -> dict:
    """What a previous run already finished in ``out_dir``.

    Only ``.tar`` files count. A ``.tar.tmp`` is by definition an interrupted
    shard and is never treated as work done - it is removed and its samples are
    prepared again, which costs at most one shard.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return {"shards": [], "keys": set(), "next_index": 0, "stale_tmp": []}
    shards = sorted(out_dir.glob("*.tar"))
    keys = set()
    for s in shards:
        side = shard_sidecar(s)
        if side.exists():
            import pyarrow.parquet as pq

            keys.update(pq.read_table(side, columns=["key"]).column("key").to_pylist())
    highest = -1
    for s in shards:
        try:
            highest = max(highest, int(s.name[:-4].rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return {
        "shards": shards,
        "keys": keys,
        "next_index": highest + 1,
        "stale_tmp": sorted(out_dir.glob("*.tar.tmp")),
    }


def adopt_shard(shard_path: Path, lookup) -> int:
    """Write a sidecar index for a finalised shard that has none.

    The first implementation kept every index row in RAM and wrote one combined
    ``index.parquet`` only when the writer closed, so an interrupted run left
    valid tar files that nothing could describe. Rather than discard hours of
    work, the index is reconstructed from the shard itself: a tar records every
    member's name and payload offset, and ``lookup`` supplies the labels for a
    key from the provenance store.

    The shards were always recoverable - but only because tar is a transparent
    format, which is not a property to rely on twice.
    """
    rows = []
    with tarfile.open(shard_path, "r") as tar:
        for info in tar:
            if not info.isfile():
                continue
            key = info.name.rsplit(".", 1)[0]
            meta = lookup(key)
            if meta is None:
                continue
            rows.append({
                "key": key,
                "shard": shard_path.name,
                "offset": info.offset_data,
                "length": info.size,
                "class_id": int(meta["class_id"]),
                "taxon_id": int(meta["taxon_id"]),
                "sha256": meta.get("sha256") or key,
                "split": meta["split"],
                "group_id": meta["group_id"],
            })
    if rows:
        _write_table(rows, shard_sidecar(shard_path))
    return len(rows)


def rebuild_index(out_dir: Path) -> Path:
    """Combine every sidecar into the ``index.parquet`` that readers expect."""
    out_dir = Path(out_dir)
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(shard_sidecar(s))
              for s in sorted(out_dir.glob("*.tar")) if shard_sidecar(s).exists()]
    path = out_dir / "index.parquet"
    if not tables:
        return path
    tmp = Path(str(path) + ".tmp")
    pq.write_table(pa.concat_tables(tables), tmp, compression="zstd")
    tmp.replace(path)
    return path


class ShardWriter:
    """Writes samples into tar shards, finalising each one durably.

    Every finalised shard gets its own sidecar index, written atomically as soon
    as the tar is renamed into place. That is what makes preparation resumable:
    the first implementation accumulated index rows in RAM and wrote one
    combined file at close, so stopping a long run near the end left valid tars
    and no index. Per-shard state means an interruption costs at most the shard
    in flight.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        prefix: str = "fw",
        shard_size: int = DEFAULT_SHARD_SIZE,
        start_index: int = 0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = int(shard_size)
        self.rows: list[dict] = []
        self._shard_rows: list[dict] = []
        self._tar = None
        self._tmp_path = None
        self._shard_name = None
        self._n_in_shard = 0
        self._shard_index = int(start_index)

    # -- shard lifecycle ------------------------------------------------
    def _open_shard(self) -> None:
        self._shard_name = f"{self.prefix}-{self._shard_index:05d}.tar"
        self._tmp_path = self.out_dir / (self._shard_name + ".tmp")
        self._tar = tarfile.open(self._tmp_path, "w")
        self._n_in_shard = 0
        self._shard_rows = []

    def _close_shard(self) -> None:
        if self._tar is None:
            return
        # Flush and fsync before the rename is allowed to make it look
        # finished. A rename is atomic, but renaming a file whose bytes are
        # still in the page cache only makes the truncation atomic too.
        try:
            self._tar.fileobj.flush()
            os.fsync(self._tar.fileobj.fileno())
        except (AttributeError, OSError):
            pass
        self._tar.close()
        final = self.out_dir / self._shard_name
        self._tmp_path.replace(final)
        if self._shard_rows:
            _write_table(self._shard_rows, shard_sidecar(final))
        self.rows.extend(self._shard_rows)
        self._shard_rows = []
        self._tar = None
        self._tmp_path = None
        self._shard_index += 1

    def write(self, sample: Sample) -> None:
        if self._tar is None:
            self._open_shard()

        info = tarfile.TarInfo(name=f"{sample.key}.jpg")
        info.size = len(sample.data)
        self._tar.addfile(info, io.BytesIO(sample.data))

        # Where the payload starts, which is what a reader seeks to.
        #
        # Not `info.offset_data`: addfile() works on a *copy* of the TarInfo, so
        # the caller's object is never populated and that field stays 0 - which
        # silently yields tar headers instead of images, caught by
        # test_roundtrip_bytes_are_exact. Deriving it from the archive offset
        # after the write is correct even when a long filename forces tarfile to
        # emit extra header blocks, which guessing "header start + 512" is not.
        padded = ((len(sample.data) + 511) // 512) * 512
        data_offset = self._tar.offset - padded

        self._shard_rows.append({
            "key": sample.key,
            "shard": self._shard_name,
            "offset": data_offset,
            "length": len(sample.data),
            "class_id": int(sample.class_id),
            "taxon_id": int(sample.taxon_id),
            "sha256": sample.sha256,
            "split": sample.split,
            "group_id": sample.group_id,
        })
        self._n_in_shard += 1
        if self._n_in_shard >= self.shard_size:
            self._close_shard()

    def close(self) -> dict:
        self._close_shard()
        # Rebuilt from the sidecars rather than from self.rows, so a resumed run
        # produces an index covering shards this process never wrote.
        index_path = rebuild_index(self.out_dir)
        return {
            "shards": self._shard_index,
            "samples": len(self.rows),
            "index": str(index_path),
        }

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ShardIndex:
    """The sample index, loaded once and queried in memory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        import pyarrow.parquet as pq

        self.table = pq.read_table(self.root / "index.parquet")
        self._cols = {c: self.table.column(c).to_pylist()
                      for c in ("shard", "offset", "length", "class_id", "split")}

    def __len__(self) -> int:
        return self.table.num_rows

    def rows_for_split(self, split: str) -> list[int]:
        return [i for i, s in enumerate(self._cols["split"]) if s == split]

    def class_ids(self, rows: list[int]) -> list[int]:
        cid = self._cols["class_id"]
        return [cid[i] for i in rows]

    def locate(self, row: int) -> tuple[str, int, int]:
        return (self._cols["shard"][row], self._cols["offset"][row],
                self._cols["length"][row])

    def shard_of(self, row: int) -> str:
        return self._cols["shard"][row]

    def group_rows_by_shard(self, rows: list[int]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        shard = self._cols["shard"]
        for r in rows:
            out.setdefault(shard[r], []).append(r)
        for v in out.values():
            v.sort(key=lambda r: self._cols["offset"][r])
        return out


class ShardStream:
    """A deterministic, resumable, rank-aware stream of sample rows.

    The order is a pure function of ``(seed, epoch, world_size, rank)``. Nothing
    about it depends on wall-clock time, dict iteration order, worker count or
    how far a previous run happened to get, which is what lets a resumed run
    continue the same epoch rather than an approximation of it.

    Resume is by **skip-ahead**: the stream is regenerated and the first
    ``skip`` rows are dropped. Dropping is free because it never opens a shard -
    the cost of resuming 80% into a ten-hour epoch is a few milliseconds, not
    eight hours of recomputation.
    """

    def __init__(
        self,
        index: ShardIndex,
        split: str,
        *,
        seed: int = 1337,
        shuffle: bool = True,
        buffer_size: int = 8192,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.index = index
        self.split = split
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.buffer_size = int(buffer_size)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self._rows = index.rows_for_split(split)
        self._by_shard = index.group_rows_by_shard(self._rows)
        self._shard_names = sorted(self._by_shard)

    def __len__(self) -> int:
        """Samples this rank sees per epoch.

        Every rank gets the same count - the shard list is truncated to a
        multiple of world_size - because an uneven split deadlocks DDP's
        gradient all-reduce when one rank runs out of batches early.
        """
        per_rank = len(self._shard_names) // self.world_size
        return sum(len(self._by_shard[s])
                   for s in self._shards_for_rank(0, per_rank))

    def _shards_for_rank(self, epoch: int, per_rank: int | None = None) -> list[str]:
        import random as _random

        names = list(self._shard_names)
        if self.shuffle:
            _random.Random(self.seed * 1_000_003 + epoch).shuffle(names)
        if per_rank is None:
            per_rank = len(names) // self.world_size
        if per_rank == 0:
            # Fewer shards than ranks: every rank reads everything rather than
            # some ranks idling. Wasteful, but it cannot deadlock.
            return names
        usable = per_rank * self.world_size
        return names[self.rank:usable:self.world_size]

    def epoch_rows(self, epoch: int, skip: int = 0) -> Iterator[int]:
        """Row indices for one epoch on this rank, after skipping ``skip``."""
        import random as _random

        rng = _random.Random(self.seed * 7_919_837 + epoch * 104_729 + self.rank)
        emitted = 0
        buffer: list[int] = []
        for shard in self._shards_for_rank(epoch):
            for row in self._by_shard[shard]:
                if not self.shuffle:
                    if emitted >= skip:
                        yield row
                    emitted += 1
                    continue
                buffer.append(row)
                if len(buffer) >= self.buffer_size:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    out = buffer.pop()
                    if emitted >= skip:
                        yield out
                    emitted += 1
        while buffer:
            j = rng.randrange(len(buffer))
            buffer[j], buffer[-1] = buffer[-1], buffer[j]
            out = buffer.pop()
            if emitted >= skip:
                yield out
            emitted += 1


class ShardReader:
    """Reads sample bytes by row index, keeping shard handles open."""

    def __init__(self, index: ShardIndex, root: Path | None = None) -> None:
        self.index = index
        self.root = Path(root or index.root)
        self._handles: dict[str, io.BufferedReader] = {}

    def read(self, row: int) -> bytes:
        shard, offset, length = self.index.locate(row)
        fh = self._handles.get(shard)
        if fh is None:
            fh = open(self.root / shard, "rb", buffering=1024 * 1024)
            self._handles[shard] = fh
        fh.seek(offset)
        return fh.read(length)

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def verify_shards(root: Path, *, limit: int | None = None, log=print) -> dict:
    """Re-hash stored samples and compare against the index.

    The path is not the hash here - unlike the CAS - so corruption during shard
    preparation, transfer to a rented machine, or a half-written file would
    otherwise be invisible until it showed up as unexplained accuracy loss.
    """
    index = ShardIndex(root)
    sha = index.table.column("sha256").to_pylist()
    n = len(index) if limit is None else min(limit, len(index))
    bad: list[str] = []
    with ShardReader(index) as reader:
        for row in range(n):
            if _sha256_bytes(reader.read(row)) != sha[row]:
                bad.append(index.table.column("key")[row].as_py())
    log(f"verified {n:,} samples, {len(bad)} mismatched")
    return {"checked": n, "mismatched": len(bad), "keys": bad[:20]}


def write_manifest(root: Path, meta: dict) -> Path:
    path = Path(root) / "shards.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
