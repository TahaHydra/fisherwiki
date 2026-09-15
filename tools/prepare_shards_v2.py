#!/usr/bin/env python
"""Build V2 training shards from the CAS originals and the V2 split assignment.

    python tools/prepare_shards_v2.py --out E:/FisherWiki/shards/v2
    python tools/prepare_shards_v2.py --out E:/FisherWiki/shards/v2 --limit 2000

Reads the immutable V2 split tables (``split_groups`` / ``split_group_members``)
as authority, pulls each image from the content-addressed store, produces a
training derivative, and writes tar shards plus an index. Originals are never
modified: this only ever reads them.

Why a derivative at all, and why this size
------------------------------------------
The CAS holds full-resolution originals averaging ~123 KB but up to several
megapixels. Decoding those every epoch wastes most of the CPU budget on pixels
that are immediately resized away - and at 4-8 GPUs it is the difference between
feeding them and starving them.

``--long-edge 512`` is the default because the training schedule tops out at
448: 512 leaves headroom for random-resized-crop without storing detail nothing
will ever read. Measured over the 67,584 samples prepared so far that is
**30 KB/sample**, so the whole 479k corpus is ~15 GB and a 3M-image corpus would
be ~90 GB - both of which fit on E: with room to spare. Re-deriving at a larger
size later is always possible because the originals are archived.

Crops
-----
If ``detections`` carries a box for an image it is used, expanded by
``--crop-pad``, before resizing; otherwise the whole frame is kept. Both paths
are deliberately represented in the shards so the model sees the uncropped case
during training too - a classifier trained only on tight crops degrades badly
when the on-device detector finds nothing, which is a train/serve skew this
project has been bitten by before.

Throughput
----------
Three things decide how fast this runs, and the first implementation got all
three wrong:

* **Read order.** Rows came back ``ORDER BY split, taxon, sha``. The CAS is laid
  out ``cas/ab/cd/<sha>``, so that walks 65,536 directories in random order -
  one seek per image on a spinning disk. Reading in ``cas_path`` order makes the
  pass near-sequential. It changes only the order samples are produced in, not
  which samples exist.
* **Overlap.** Decode, crop, resize and JPEG encode are pure CPU and ran on the
  same thread that was waiting for the disk. They now run in a bounded worker
  pool while the main thread does nothing but pull rows and write tars.
* **Decode size.** A 4000x3000 original was fully decoded and then thrown away
  down to 512. ``Image.draft`` lets the JPEG decoder drop to 1/2, 1/4 or 1/8
  scale in the DCT domain, which is most of the decode cost. The reduction is
  chosen from the *crop region* so the final LANCZOS resize is still downscaling
  rather than inventing detail.

Results stay deterministic under all of this: the pool preserves input order, so
shard membership, sample keys, class ids and the crop/whole-frame choice are the
same whatever ``--workers`` is set to.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwdata.config import PATHS  # noqa: E402
from fwdata.splits_v2 import DATASET_VERSION  # noqa: E402

SPLITS = ("train", "validation", "dev_test", "final_test")

# Worker-process configuration, set once by the pool initializer rather than
# shipped with every job. Windows uses spawn, so this module is re-imported in
# each worker and this really is per-process state.
_CFG: dict = {}


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _init_worker(cfg: dict) -> None:
    global _CFG
    _CFG = cfg


def _draft_divisor(box_w: int, box_h: int, long_edge: int,
                   full_w: int, full_h: int) -> int:
    """Largest JPEG DCT reduction that still leaves the crop big enough.

    The decoder can only halve, so the answer is 1, 2, 4 or 8. Taking it from
    the *crop* rather than from the frame matters: a fish occupying an eighth of
    a 24MP photo still has to come out at 512px, and drafting on frame size
    would have discarded those pixels before the crop ever happened.
    """
    allowed = max(box_w, box_h) / max(long_edge, 1)
    for s in (8, 4, 2):
        if s <= allowed and min(full_w, full_h) // s >= 1:
            return s
    return 1


def prepare_one(job):
    """Decode, crop, resize and encode one image. Runs in a worker process.

    Returns ``(sha, split, taxon_id, group_id, variant, data, timings)``, with
    ``split`` set to ``None`` when the image could not be produced. Timings are
    per-stage seconds that the caller accumulates for the benchmark; four
    ``perf_counter`` pairs per image is well below the noise floor.
    """
    import time as _t

    from PIL import Image

    from fwml.crop_v2 import expand_box, grow_box_to_min

    sha, split, taxon_id, group_id, cas_path, x0, y0, x1, y1 = job
    cfg = _CFG
    t_read = t_decode = t_geom = t_encode = 0.0

    if not cas_path:
        return (sha, None, 0, "", "missing", b"", (0.0, 0.0, 0.0, 0.0))
    src = cfg["cas_root"] / cas_path

    # Read the file in one go rather than letting PIL stream it. A single
    # sequential read of ~123 KB beats the decoder's seek-and-sip pattern on a
    # spinning disk, and it separates "waiting for D:" from "burning CPU",
    # which is exactly the split these measurements need.
    t = _t.perf_counter()
    try:
        raw = src.read_bytes()
    except OSError:
        return (sha, None, 0, "", "missing", b"", (0.0, 0.0, 0.0, 0.0))
    t_read = _t.perf_counter() - t

    keep_whole = (
        split == "train"
        and x0 is not None
        and (int(sha[:8], 16) % 1000) < int(cfg["whole_frame_fraction"] * 1000)
    )
    variant = "whole" if (x0 is None or keep_whole) else "crop"

    try:
        t = _t.perf_counter()
        with Image.open(io.BytesIO(raw)) as im:
            full_w, full_h = im.size
            if variant == "crop":
                box = expand_box((x0, y0, x1, y1), full_w, full_h,
                                 context=cfg["crop_pad"])
                box = grow_box_to_min(box, full_w, full_h, cfg["min_crop_px"])
            else:
                box = (0, 0, full_w, full_h)

            if cfg["decode_draft"]:
                s = _draft_divisor(box[2] - box[0], box[3] - box[1],
                                   cfg["long_edge"], full_w, full_h)
                if s > 1:
                    im.draft("RGB", (full_w // s, full_h // s))
            im = im.convert("RGB")
            t_decode = _t.perf_counter() - t

            t = _t.perf_counter()
            dw, dh = im.size
            if variant == "crop":
                # Boxes are in original-image coordinates and draft() changed
                # the coordinate system underneath us. Scale by the size it
                # actually produced, not by the divisor we asked for: draft
                # rounds up, so those are not always the same ratio.
                sx, sy = dw / full_w, dh / full_h
                dbox = (max(0, int(math.floor(box[0] * sx))),
                        max(0, int(math.floor(box[1] * sy))),
                        min(dw, int(math.ceil(box[2] * sx))),
                        min(dh, int(math.ceil(box[3] * sy))))
                if dbox[2] - dbox[0] >= 16 and dbox[3] - dbox[1] >= 16:
                    im = im.crop(dbox)
                else:
                    variant = "whole"       # degenerate box, keep the frame

            # Bound the LONG edge and preserve aspect. Never resize to a fixed
            # shape: stretching a disc-shaped fish into a wide rectangle is what
            # made Fishial call an ocean sunfish a remora at 98.4% confidence.
            # Letterboxing happens at train time, so nothing is padded on disk.
            w, h = im.size
            scale = cfg["long_edge"] / max(w, h)
            if scale < 1.0:
                im = im.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.LANCZOS,
                )
            t_geom = _t.perf_counter() - t

            t = _t.perf_counter()
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=cfg["jpeg_quality"],
                    optimize=cfg["jpeg_optimize"], progressive=False)
            data = buf.getvalue()
            t_encode = _t.perf_counter() - t
    except Exception:
        return (sha, None, 0, "", "failed", b"", (t_read, t_decode, t_geom, 0.0))

    return (sha, split, taxon_id, group_id, variant, data,
            (t_read, t_decode, t_geom, t_encode))


def corpus_rows(con, splits, *, only_detected: bool = False, limit=None,
                prefer_source: str | None = None):
    """Assigned, un-quarantined images with their primary box, in CAS order.

    ``prefer_source`` handles the re-fetch case. Pulling the corpus again at
    1024 px stores a second file for the same photograph - different bytes,
    different sha, same iNaturalist photo id - and both get assigned to the same
    observation group, because that is what keeps splits immutable. Preparing
    both would put two resolutions of one photo in the same split: not a leak,
    but redundant data the model sees twice. Naming the preferred source drops
    the superseded copy at selection time rather than after the pixels are
    written.
    """
    has_detections = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='detections'"
    ).fetchone()[0] > 0
    detect_join = ("LEFT JOIN detections d ON d.sha256 = m.sha256 AND d.is_primary"
                   if has_detections else "")
    detect_cols = (
        "any_value(d.x0) AS x0, any_value(d.y0) AS y0, any_value(d.x1) AS x1, "
        "any_value(d.y1) AS y1"
        if has_detections
        else "NULL AS x0, NULL AS y0, NULL AS x1, NULL AS y1")
    detect_filter = (
        "AND EXISTS (SELECT 1 FROM detections dd WHERE dd.sha256 = m.sha256)"
        if (only_detected and has_detections) else "")
    supersede = ""
    if prefer_source:
        supersede = f"""
          AND NOT EXISTS (
              SELECT 1 FROM provenance better
              WHERE better.source_record_id = p.source_record_id
                AND better.source_dataset = '{prefer_source}'
                AND p.source_dataset <> '{prefer_source}'
                AND EXISTS (SELECT 1 FROM split_group_members bm
                            WHERE bm.sha256 = better.sha256
                              AND bm.dataset_version = m.dataset_version))"""
    placeholders = ",".join("?" * len(splits))
    rows = con.execute(
        f"""
        SELECT m.sha256, g.split,
               -- The label is the image's OWN taxon, not its leak group's.
               -- Groups exist to stop a re-post crossing a split boundary, and
               -- 484 iNaturalist observations hold photos of several species;
               -- taking the group's taxon mislabelled 616 images, and split the
               -- affected species across spurious extra class ids. The split
               -- still comes from the group, which is the part that has to be
               -- leak-safe.
               coalesce(any_value(p.species_taxon_id), g.taxon_id) AS taxon_id,
               g.group_id,
               any_value(p.cas_path) AS cas_path,
               any_value(p.accepted_scientific_name) AS name,
               {detect_cols}
        FROM split_group_members m
        JOIN split_groups g USING (dataset_version, group_id)
        JOIN provenance p ON p.sha256 = m.sha256
        {detect_join}
        WHERE m.dataset_version = ?
          AND g.split IN ({placeholders})
          AND NOT EXISTS (
              SELECT 1 FROM split_quarantine q
              WHERE q.dataset_version = m.dataset_version AND q.sha256 = m.sha256)
          {detect_filter}{supersede}
        GROUP BY m.sha256, g.split, g.taxon_id, g.group_id
        -- On-disk order, not logical order. See the module docstring: the CAS
        -- is cas/ab/cd/<sha>, so split/taxon order is a seek per image while
        -- cas_path order is a near-sequential walk. Resumability is unaffected,
        -- because the pending set is defined by which keys are already inside a
        -- finalised shard and not by the order rows arrive in.
        ORDER BY any_value(p.cas_path)
        """,
        [DATASET_VERSION, *splits],
    ).fetchall()
    return rows[:limit] if limit else rows


def make_fingerprint(*, long_edge, quality, jpeg_optimize, crop_pad, min_crop_px,
                     whole_frame_fraction, decode_draft, shard_size) -> dict:
    from fwml.shards import PREP_IMPL_VERSION

    return {
        "prep_impl_version": PREP_IMPL_VERSION,
        "dataset_version": DATASET_VERSION,
        "long_edge": int(long_edge),
        "jpeg_quality": int(quality),
        "jpeg_optimize": bool(jpeg_optimize),
        "jpeg_progressive": False,
        "crop_pad": float(crop_pad),
        "min_crop_px": int(min_crop_px),
        "whole_frame_fraction": float(whole_frame_fraction),
        "resample": "LANCZOS",
        "decode_draft": bool(decode_draft),
        "shard_size": int(shard_size),
    }


def build(
    out_dir: Path,
    *,
    db_path: Path | None = None,
    long_edge: int = 512,
    quality: int = 90,
    jpeg_optimize: bool = False,
    decode_draft: bool = True,
    shard_size: int = 2048,
    limit: int | None = None,
    crop_pad: float = 0.25,
    whole_frame_fraction: float = 0.15,
    min_crop_px: int = 224,
    only_detected: bool = False,
    prefer_source: str | None = None,
    workers: int = 6,
    chunk: int = 16,
    adopt_fingerprint: bool = False,
    splits: tuple[str, ...] = SPLITS,
    rows=None,
    log=None,
) -> dict:
    import hashlib
    import multiprocessing as mp

    import duckdb

    from fwml.progress import Progress
    from fwml.shards import (
        ClassMap,
        Sample,
        ShardWriter,
        check_fingerprint,
        fingerprint_path,
        rebuild_index,
        relabel_sidecars,
        scan_finalised,
        write_fingerprint,
        write_manifest,
    )

    out_dir = Path(out_dir)
    fingerprint = make_fingerprint(
        long_edge=long_edge, quality=quality, jpeg_optimize=jpeg_optimize,
        crop_pad=crop_pad, min_crop_px=min_crop_px,
        whole_frame_fraction=whole_frame_fraction, decode_draft=decode_draft,
        shard_size=shard_size)

    # --- what a previous run already finished -----------------------------
    states = {s: scan_finalised(out_dir / s) for s in splits}
    has_work = any(st["shards"] for st in states.values())

    # The fingerprint gate runs before a single image is read and before any
    # `.tar.tmp` is deleted: a refused run leaves the directory exactly as it
    # found it.
    if adopt_fingerprint and has_work and not fingerprint_path(out_dir).exists():
        write_fingerprint(out_dir, fingerprint)
    check_fingerprint(out_dir, fingerprint, has_work=has_work)

    con = duckdb.connect(str(db_path or PATHS.provenance_db), read_only=True)
    try:
        if rows is None:
            rows = corpus_rows(con, splits, only_detected=only_detected,
                               limit=limit, prefer_source=prefer_source)
    finally:
        con.close()
    if not rows:
        raise SystemExit(
            "no V2 split assignments found - run "
            "`python tools/dataset.py v2-assign --batch batch0_v1_cas` first")

    # Class ids come from a persisted, append-only map rather than from whatever
    # taxa happen to be in this query. Re-deriving them, as the first version
    # did, silently relabels every shard already written the next time a species
    # is added to the corpus.
    class_map = ClassMap(out_dir)
    added = class_map.extend([(r[2], r[5]) for r in rows])

    total_rows = len(rows)
    done_keys: set[str] = set()
    start_index: dict[str, int] = {}
    for split in splits:
        st = states[split]
        start_index[split] = st["next_index"]
        done_keys |= st["keys"]
    already = len(done_keys)

    with Progress("prepare", total=total_rows, resumable=True,
                  already_done=already,
                  extra={"out_dir": str(out_dir), "workers": workers,
                         "fingerprint": fingerprint}) as prog:
        log = log or prog.log
        log(f"{total_rows:,} assigned images, {len(class_map):,} classes "
            f"({added:,} new)")
        class_map.save()
        write_fingerprint(out_dir, fingerprint)

        for split in splits:
            st = states[split]
            for stale in st["stale_tmp"]:
                log(f"  discarding interrupted shard {stale.name}")
                stale.unlink(missing_ok=True)
            if st["shards"]:
                log(f"  {split}: {len(st['shards'])} finalised shards, "
                    f"{len(st['keys']):,} samples done, "
                    f"next shard {start_index[split]:05d}")
        if done_keys:
            rows = [r for r in rows if r[0] not in done_keys]
            log(f"  resuming: {already:,} already prepared, "
                f"{len(rows):,} remaining")

        # Shards written before this run may carry class ids from a smaller (or
        # differently derived) class map. Their taxon ids are authoritative, so
        # the labels are re-derived rather than the shards rebuilt.
        moved = 0
        for split in splits:
            moved += relabel_sidecars(out_dir / split, class_map.by_taxon)
        if moved:
            log(f"  relabelled {moved} existing shard indexes from the "
                f"persisted class map")
            for split in splits:
                if (out_dir / split).exists():
                    rebuild_index(out_dir / split)

        cfg = {
            "cas_root": PATHS.cas, "long_edge": long_edge,
            "jpeg_quality": quality, "jpeg_optimize": jpeg_optimize,
            "decode_draft": decode_draft, "crop_pad": crop_pad,
            "min_crop_px": min_crop_px,
            "whole_frame_fraction": whole_frame_fraction,
        }
        jobs = [(r[0], r[1], int(r[2]), r[3], r[4], r[6], r[7], r[8], r[9])
                for r in rows]

        counts: dict[str, int] = {}
        variants: dict[str, int] = {}
        stage_s = [0.0, 0.0, 0.0, 0.0]
        out_bytes = 0
        failures = 0
        writers: dict[str, ShardWriter] = {}
        t0 = time.time()
        pool = None
        try:
            if workers > 0:
                pool = mp.Pool(workers, initializer=_init_worker,
                               initargs=(cfg,))
                # imap, not imap_unordered: results must arrive in input order
                # or shard membership depends on which worker won a race.
                results = pool.imap(prepare_one, jobs, chunksize=chunk)
            else:
                _init_worker(cfg)
                results = map(prepare_one, jobs)

            for i, (sha, split, taxon, group_id, variant, data,
                    timings) in enumerate(results):
                for k in range(4):
                    stage_s[k] += timings[k]
                if split is None:
                    failures += 1
                    prog.advance(failures=1)
                    continue
                variants[variant] = variants.get(variant, 0) + 1
                out_bytes += len(data)

                writer = writers.get(split)
                if writer is None:
                    writer = ShardWriter(out_dir / split, prefix=split,
                                         shard_size=shard_size,
                                         start_index=start_index.get(split, 0))
                    writers[split] = writer
                writer.write(Sample(
                    key=sha, data=data, class_id=class_map[taxon],
                    taxon_id=int(taxon), sha256=hashlib.sha256(data).hexdigest(),
                    split=split, group_id=group_id))
                counts[split] = counts.get(split, 0) + 1
                prog.advance()
                if (i + 1) % 20000 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    log(f"  {i + 1:,}/{len(jobs):,}  {rate:.0f} img/s  "
                        f"{failures} failed")
                    prog.checkpointed(f"{sum(counts.values()):,} samples written")
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
            summary = {s: w.close() for s, w in writers.items()}

        elapsed = time.time() - t0
        prepared = sum(counts.values())
        meta = {
            "fingerprint": fingerprint,
            "variants": variants,
            "classes": len(class_map),
            "counts": counts,
            "failures": failures,
            "shards": summary,
            "prepared_this_run": prepared,
            "seconds": round(elapsed, 1),
            "img_per_s": round(prepared / max(elapsed, 1e-9), 1),
            "mean_bytes": round(out_bytes / max(prepared, 1)),
            "worker_stage_seconds": {
                "read": round(stage_s[0], 1),
                "decode": round(stage_s[1], 1),
                "crop_resize": round(stage_s[2], 1),
                "jpeg_encode": round(stage_s[3], 1),
            },
            "class_map": str(class_map.path),
        }
        write_manifest(out_dir, meta)
        log(f"wrote {prepared:,} samples in {elapsed:.0f}s "
            f"({meta['img_per_s']} img/s, {failures} failed); "
            f"variants {variants}")
        for split, n in sorted(counts.items()):
            log(f"  {split:12} {n:>9,}")
        prog.note = f"{prepared:,} samples at {meta['img_per_s']} img/s"
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-edge", type=int, default=512)
    ap.add_argument("--only-detected", action="store_true",
                    help="prepare only images that already have a detection "
                         "row, for incremental runs while detection is running")
    ap.add_argument("--whole-frame-fraction", type=float, default=0.15,
                    help="deterministic share of TRAIN kept uncropped, so the "
                         "model still handles the no-detection fallback")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--jpeg-optimize", action="store_true",
                    help="re-optimize Huffman tables; a few percent smaller for "
                         "a large share of the encode budget, so off by default")
    ap.add_argument("--no-decode-draft", dest="decode_draft",
                    action="store_false",
                    help="decode JPEGs at full resolution instead of letting "
                         "the decoder downscale in the DCT domain - slower, and "
                         "a different preprocessing fingerprint")
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--crop-pad", type=float, default=0.25)
    ap.add_argument("--min-crop-px", type=int, default=224,
                    help="widen small boxes with real context instead of "
                         "upscaling a tiny crop at train time")
    ap.add_argument("--workers", type=int, default=6,
                    help="decode/encode processes; 0 runs in-process")
    ap.add_argument("--chunk", type=int, default=16,
                    help="jobs handed to a worker at once - keeps each worker "
                         "on a contiguous run of CAS paths rather than "
                         "scattering reads across the disk")
    ap.add_argument("--adopt-fingerprint", action="store_true",
                    help="declare that shards already in --out were produced "
                         "with these exact settings, for a directory written "
                         "before fingerprints existed")
    ap.add_argument("--prefer-source", default=None,
                    help="when the same source photo exists under two source "
                         "datasets (a 500px and a 1024px fetch of the same "
                         "iNaturalist photo), prepare only this one")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args(argv)

    build(
        Path(args.out), long_edge=args.long_edge, quality=args.quality,
        jpeg_optimize=args.jpeg_optimize, decode_draft=args.decode_draft,
        shard_size=args.shard_size, limit=args.limit, crop_pad=args.crop_pad,
        whole_frame_fraction=args.whole_frame_fraction,
        min_crop_px=args.min_crop_px, only_detected=args.only_detected,
        prefer_source=args.prefer_source,
        workers=args.workers, chunk=args.chunk,
        adopt_fingerprint=args.adopt_fingerprint,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
