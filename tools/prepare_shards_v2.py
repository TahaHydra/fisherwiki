#!/usr/bin/env python
"""Build V2 training shards from the CAS originals and the V2 split assignment.

    python tools/prepare_shards_v2.py --out E:/FisherWiki/shards --limit 2000
    python tools/prepare_shards_v2.py --out D:/fisherwiki-data/v2/shards

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

``--short-edge 512`` is the default because the training schedule tops out at
448: 512 leaves headroom for random-resized-crop without storing detail nothing
will ever read. It also decides whether the dataset fits on this machine at all.
At 3M images:

    short edge 512  ~= 100 KB/img  ->  ~300 GB   fits on D: (535 GB free)
    short edge 1080 ~= 350 KB/img  ->  ~1.0 TB   does not fit anywhere local

Storing 1080px derivatives, as originally proposed, would force the working set
onto the NAS, whose ~100 files/second would take over eight hours per epoch in
filesystem overhead alone. Re-deriving at a larger size later is always possible
because the originals are archived - that is what they are for.

Crops
-----
If ``detections`` carries a box for an image it is used, expanded by
``--crop-pad``, before resizing; otherwise the whole frame is kept. Both paths
are deliberately represented in the shards so the model sees the uncropped case
during training too - a classifier trained only on tight crops degrades badly
when the on-device detector finds nothing, which is a train/serve skew this
project has been bitten by before.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fwdata.config import PATHS  # noqa: E402
from fwdata.splits_v2 import DATASET_VERSION  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def build(
    out_dir: Path,
    *,
    db_path: Path | None = None,
    long_edge: int = 512,
    quality: int = 90,
    shard_size: int = 2048,
    limit: int | None = None,
    crop_pad: float = 0.25,
    whole_frame_fraction: float = 0.15,
    min_crop_px: int = 224,
    only_detected: bool = False,
    splits: tuple[str, ...] = ("train", "validation", "dev_test", "final_test"),
    log=_log,
) -> dict:
    import duckdb
    from PIL import Image

    from fwml.crop_v2 import expand_box, grow_box_to_min
    from fwml.shards import (
        Sample,
        ShardWriter,
        adopt_shard,
        scan_finalised,
        shard_sidecar,
        write_manifest,
    )

    con = duckdb.connect(str(db_path or PATHS.provenance_db), read_only=True)
    has_detections = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='detections'"
    ).fetchone()[0] > 0

    detect_join = (
        "LEFT JOIN detections d ON d.sha256 = m.sha256 AND d.is_primary"
        if has_detections else ""
    )
    detect_cols = (
        "any_value(d.x0) AS x0, any_value(d.y0) AS y0, any_value(d.x1) AS x1, "
        "any_value(d.y1) AS y1, any_value(d.score) AS score"
        if has_detections
        else "NULL AS x0, NULL AS y0, NULL AS x1, NULL AS y1, NULL AS score"
    )

    detect_filter = (
        "AND EXISTS (SELECT 1 FROM detections dd WHERE dd.sha256 = m.sha256)"
        if (only_detected and has_detections) else ""
    )
    placeholders = ",".join("?" * len(splits))
    rows = con.execute(
        f"""
        SELECT m.sha256, g.split, g.taxon_id, g.group_id,
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
          {detect_filter}
        GROUP BY m.sha256, g.split, g.taxon_id, g.group_id
        ORDER BY g.split, g.taxon_id, m.sha256
        """,
        [DATASET_VERSION, *splits],
    ).fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(
            "no V2 split assignments found - run "
            "`python tools/dataset.py v2-assign --batch batch0_v1_cas` first"
        )

    # Dense class ids over the taxa that actually appear, ordered by name so the
    # mapping is reproducible from the data rather than from insertion order.
    taxa = sorted({(r[2], r[5]) for r in rows}, key=lambda t: (t[1] or "", t[0]))
    class_of = {taxon: i for i, (taxon, _name) in enumerate(taxa)}
    log(f"{len(rows):,} images, {len(taxa):,} classes")

    cas_root = PATHS.cas
    counts: dict[str, int] = {}
    variants: dict[str, int] = {}
    failures = 0
    t0 = time.time()

    # --- resume -----------------------------------------------------------
    # Look at what a previous run finished before preparing anything. Finalised
    # shards are kept and their samples skipped; only a `.tar.tmp` - a shard
    # interrupted mid-write - is discarded, costing at most one shard.
    meta_by_key = {
        r[0]: {"class_id": class_of[r[2]], "taxon_id": int(r[2]),
               "split": r[1], "group_id": r[3], "sha256": r[0]}
        for r in rows
    }
    done_keys: set[str] = set()
    start_index: dict[str, int] = {}
    adopted = 0
    for split in splits:
        state = scan_finalised(out_dir / split)
        start_index[split] = state["next_index"]
        for stale in state["stale_tmp"]:
            log(f"  discarding interrupted shard {stale.name}")
            stale.unlink(missing_ok=True)
        # Shards written before per-shard indexes existed have valid tars and no
        # sidecar. Reconstruct the sidecar from the tar rather than redo the
        # work: a tar records every member name and payload offset, and the
        # labels come from the corpus rows already loaded above.
        for shard in state["shards"]:
            if not shard_sidecar(shard).exists():
                adopted += adopt_shard(shard, meta_by_key.get)
        if adopted:
            # Re-scan so the freshly written sidecars are picked up.
            state = scan_finalised(out_dir / split)
            start_index[split] = state["next_index"]
        done_keys |= state["keys"]
        if state["shards"]:
            log(f"  {split}: {len(state['shards'])} finalised shards, "
                f"{len(state['keys']):,} samples already done, "
                f"next shard {start_index[split]:05d}")
    if adopted:
        log(f"  adopted {adopted:,} samples from shards written before "
            f"per-shard indexes existed")
    if done_keys:
        before = len(rows)
        rows = [r for r in rows if r[0] not in done_keys]
        log(f"  resuming: {before - len(rows):,} samples already prepared, "
            f"{len(rows):,} remaining")

    writers: dict[str, ShardWriter] = {}
    try:
        for i, (sha, split, taxon, group_id, cas_path, _name,
                x0, y0, x1, y1, _score) in enumerate(rows):
            src = cas_root / cas_path if cas_path else None
            if src is None or not src.exists():
                failures += 1
                continue
            # Deterministic choice of crop vs whole frame, by content hash, so
            # a rebuild makes the same choice and a resumed run does not drift.
            keep_whole = (
                split == "train"
                and x0 is not None
                and (int(sha[:8], 16) % 1000) < int(whole_frame_fraction * 1000)
            )
            variant = "whole" if (x0 is None or keep_whole) else "crop"

            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if variant == "crop":
                        box = expand_box((x0, y0, x1, y1), w, h, context=crop_pad)
                        box = grow_box_to_min(box, w, h, min_crop_px)
                        if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
                            im = im.crop(box)
                        else:
                            variant = "whole"   # degenerate box, keep the frame
                    # Bound the LONG edge and preserve aspect. Never resize to a
                    # fixed shape: stretching a disc-shaped fish into a wide
                    # rectangle is what made Fishial call an ocean sunfish a
                    # remora at 98.4% confidence. Letterboxing happens at train
                    # time, so nothing is padded on disk.
                    w, h = im.size
                    scale = long_edge / max(w, h)
                    if scale < 1.0:
                        im = im.resize(
                            (max(1, round(w * scale)), max(1, round(h * scale))),
                            Image.LANCZOS,
                        )
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=quality,
                            optimize=True, progressive=False)
                    data = buf.getvalue()
            except Exception:
                failures += 1
                continue
            variants[variant] = variants.get(variant, 0) + 1

            writer = writers.get(split)
            if writer is None:
                writer = ShardWriter(out_dir / split, prefix=split,
                                     shard_size=shard_size,
                                     start_index=start_index.get(split, 0))
                writers[split] = writer
            import hashlib

            writer.write(Sample(
                key=sha, data=data, class_id=class_of[taxon], taxon_id=int(taxon),
                sha256=hashlib.sha256(data).hexdigest(),
                split=split, group_id=group_id,
            ))
            counts[split] = counts.get(split, 0) + 1
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                log(f"  {i + 1:,}/{len(rows):,}  {rate:.0f} img/s  {failures} failed")
    finally:
        summary = {s: w.close() for s, w in writers.items()}

    meta = {
        "dataset_version": DATASET_VERSION,
        "long_edge": long_edge,
        "whole_frame_fraction": whole_frame_fraction,
        "min_crop_px": min_crop_px,
        "variants": variants,
        "jpeg_quality": quality,
        "crop_pad": crop_pad,
        "used_detections": has_detections,
        "classes": len(taxa),
        "counts": counts,
        "failures": failures,
        "shards": summary,
        "class_map": [
            {"class_id": class_of[t], "taxon_id": t, "scientific_name": n}
            for t, n in taxa
        ],
    }
    write_manifest(out_dir, meta)
    log(f"wrote {sum(counts.values()):,} samples in "
        f"{time.time() - t0:.0f}s ({failures} failed); variants {variants}")
    for split, n in sorted(counts.items()):
        log(f"  {split:12} {n:>9,}")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--long-edge", type=int, default=512)
    ap.add_argument("--only-detected", action="store_true",
                    help="prepare only images that already have a detection row, "
                         "for incremental runs while detection is still going")
    ap.add_argument("--whole-frame-fraction", type=float, default=0.15,
                    help="deterministic share of TRAIN kept uncropped, so the "
                         "model still handles the no-detection fallback")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--crop-pad", type=float, default=0.25)
    ap.add_argument("--min-crop-px", type=int, default=224,
                    help="widen small boxes with real context instead of "
                         "upscaling a tiny crop at train time")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", default="train,validation,dev_test,final_test")
    args = ap.parse_args(argv)

    build(
        Path(args.out), long_edge=args.long_edge, quality=args.quality,
        shard_size=args.shard_size, limit=args.limit, crop_pad=args.crop_pad,
        whole_frame_fraction=args.whole_frame_fraction,
        min_crop_px=args.min_crop_px,
        only_detected=args.only_detected,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
