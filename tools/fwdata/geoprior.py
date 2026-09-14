"""Builds the binary geographic occurrence prior shipped inside a pack.

The prior is a per-class histogram over equal-angle cells, quantised to one byte
per cell. For ~2,000 classes at 2-degree cells it comes to a few hundred KB,
which is small enough to ship and load eagerly on a phone.

Binary format
-------------
**Big-endian**, because the reader is Java's ``DataInputStream``, whose
``readInt``/``readFloat`` are network byte order. Getting this wrong produces a
file that parses without error and yields nonsense, so it is pinned by a
round-trip test that writes with this module and reads with the Kotlin reader.

::

    u32   magic       0x46574750 ("FWGP")
    u32   version     1
    f32   cellDegrees
    u32   numClasses
    repeat numClasses:
        u32  classTotal      total observations for this class
        u32  cellCount
        repeat cellCount:
            i32 packedCell   (latIdx << 16) | (lonIdx & 0xFFFF)
            u8  value        quantised log-frequency, 1..255

Quantisation
------------
Cell counts are extremely skewed: a species' modal cell may hold thousands of
records while its range edge holds one. Storing a linear count would let the
modal cell dominate and make every other cell indistinguishable from absent, so
we store ``log1p(count)`` scaled to 1..255 **per class**. Zero is reserved to
mean "not present in this file", which the reader treats as unknown rather than
absent - the distinction that keeps a vagrant catch identifiable.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

MAGIC = 0x46574750
VERSION = 1
DEFAULT_CELL_DEGREES = 2.0


@dataclass
class GeoPriorStats:
    num_classes: int = 0
    cell_degrees: float = DEFAULT_CELL_DEGREES
    classes_with_data: int = 0
    total_cells: int = 0
    total_observations: int = 0
    bytes_written: int = 0
    median_cells_per_class: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def pack_cell(lat: float, lon: float, cell_degrees: float) -> int:
    """Pack a coordinate into the same int key the Kotlin reader computes."""
    lat = max(-90.0, min(90.0, lat))
    while lon < -180.0:
        lon += 360.0
    while lon >= 180.0:
        lon -= 360.0
    lat_idx = int((lat + 90.0) / cell_degrees)
    lon_idx = int((lon + 180.0) / cell_degrees)
    return (lat_idx << 16) | (lon_idx & 0xFFFF)


def build(
    *,
    out_path: Path,
    corpus: str,
    provenance_db: Path,
    num_classes: int,
    cell_degrees: float = DEFAULT_CELL_DEGREES,
    use_all_observations: bool = True,
    log=print,
) -> GeoPriorStats:
    """Write the prior for ``corpus`` to ``out_path``.

    By default the histogram is built from **all** research-grade observations
    of each taxon, not only the images we downloaded. The training corpus is
    capped at 300 images per species and skewed by our own sampling, which would
    make the prior describe our sampling rather than the species' range.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(provenance_db), read_only=True)
    con.execute("PRAGMA threads=8")

    log(f"geo prior: {num_classes} classes at {cell_degrees} degree cells")

    if use_all_observations:
        # Join the full candidate table (every licensed photo we know of, not
        # just the ones we fetched) so the range is as complete as the data
        # allows. Coordinates come from the observation, so we count distinct
        # observations rather than photos.
        rows = con.execute(
            f"""
            SELECT m.class_id,
                   c.latitude, c.longitude,
                   count(DISTINCT c.group_key) AS n
            FROM corpus_members m
            JOIN candidates c USING (candidate_id)
            WHERE m.corpus = '{corpus}'
              AND c.latitude IS NOT NULL AND c.longitude IS NOT NULL
            GROUP BY m.class_id, c.latitude, c.longitude
            """
        ).fetchall()
    else:
        rows = con.execute(
            f"""
            SELECT m.class_id, p.latitude, p.longitude, count(*) AS n
            FROM corpus_members m
            JOIN provenance p USING (candidate_id)
            WHERE m.corpus = '{corpus}'
              AND p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            GROUP BY m.class_id, p.latitude, p.longitude
            """
        ).fetchall()
    con.close()

    per_class: list[dict[int, int]] = [dict() for _ in range(num_classes)]
    totals = [0] * num_classes
    for class_id, lat, lon, n in rows:
        if class_id is None or class_id >= num_classes or class_id < 0:
            continue
        key = pack_cell(float(lat), float(lon), cell_degrees)
        per_class[class_id][key] = per_class[class_id].get(key, 0) + int(n)
        totals[class_id] += int(n)

    stats = GeoPriorStats(num_classes=num_classes, cell_degrees=cell_degrees)
    cells_per_class = []

    with open(out_path, "wb") as fh:
        fh.write(struct.pack(">IIfI", MAGIC, VERSION, cell_degrees, num_classes))
        for c in range(num_classes):
            cells = per_class[c]
            fh.write(struct.pack(">II", totals[c], len(cells)))
            if not cells:
                cells_per_class.append(0)
                continue
            stats.classes_with_data += 1
            cells_per_class.append(len(cells))
            stats.total_cells += len(cells)
            stats.total_observations += totals[c]

            counts = np.fromiter(cells.values(), dtype=np.float64, count=len(cells))
            logs = np.log1p(counts)
            hi = float(logs.max())
            # Scale per class so a widespread species and a local one both use
            # the full byte range; the reader only ever compares within a class.
            scaled = np.clip(np.round(1.0 + 254.0 * (logs / hi if hi > 0 else logs)),
                             1, 255).astype(np.uint8)
            for key, val in zip(cells.keys(), scaled.tolist()):
                fh.write(struct.pack(">iB", key, int(val)))

    stats.bytes_written = out_path.stat().st_size
    stats.median_cells_per_class = float(np.median(cells_per_class)) if cells_per_class else 0.0

    log(f"  classes with data : {stats.classes_with_data}/{num_classes}")
    log(f"  cells             : {stats.total_cells:,} "
        f"(median {stats.median_cells_per_class:.0f}/class)")
    log(f"  observations       : {stats.total_observations:,}")
    log(f"  size               : {stats.bytes_written/1024:.0f} KB -> {out_path.name}")
    return stats


def read(path: Path) -> tuple[float, int, list[dict[int, int]], list[int]]:
    """Read back a prior. Used by tests and by the pack verifier."""
    with open(path, "rb") as fh:
        magic, version, cell_deg, n = struct.unpack(">IIfI", fh.read(16))
        if magic != MAGIC:
            raise ValueError(f"not a geo prior file (magic {magic:#x})")
        if version != VERSION:
            raise ValueError(f"unsupported geo prior version {version}")
        maps: list[dict[int, int]] = []
        totals: list[int] = []
        for _ in range(n):
            total, count = struct.unpack(">II", fh.read(8))
            totals.append(total)
            m: dict[int, int] = {}
            if count:
                buf = fh.read(5 * count)
                for i in range(count):
                    key, val = struct.unpack_from(">iB", buf, i * 5)
                    m[key] = val
            maps.append(m)
        return cell_deg, n, maps, totals
