#!/usr/bin/env python
"""Build a pre-resized training cache on fast storage.

    python tools/prepare_cache.py --corpus global_v1 --cache-root E:/fisherwiki-cache

Why this exists
---------------
Measured on the reference machine: training ran at **63 images/second** while
the GPU benchmarks at 374. The bottleneck was neither the model nor the CPU -
it was the data root sitting on a **spinning disk**. Randomly reading 307,415
small files from an HDD costs ~12 ms of seek latency each, and no amount of
DataLoader workers fixes physics.

This pass does two things at once:

1. **Moves the working set to an SSD.** Random small-file reads are what HDDs
   are worst at and flash is best at.
2. **Shrinks the decode.** Source images are ~500 px on the long edge; training
   crops to 224. Re-encoding at a 256 px short edge cuts both bytes read and
   JPEG decode work by roughly 4x, with no effect on training quality because
   the image is downscaled to 224 anyway.

The cache is **derived data**, not a second copy of the corpus. The
content-addressed store remains the source of truth: the cache is keyed by the
same SHA-256, can be deleted at any time, and is rebuilt by re-running this.

Resumable: an existing cache entry of non-zero size is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageFile, ImageOps  # noqa: E402

from fwdata.config import PATHS, free_space_gb  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

#: Short-edge target. Training resizes to 256 then crops 224, so storing at 256
#: preserves the exact pipeline while removing all the wasted pixels.
DEFAULT_SHORT_EDGE = 256
DEFAULT_QUALITY = 90


def _cache_path(cache_root: Path, sha: str) -> Path:
    return cache_root / sha[:2] / sha[2:4] / f"{sha}.jpg"


def _one(args: tuple) -> tuple[int, int]:
    """Returns (written, bytes). Runs in a worker process."""
    src, dst, short_edge, quality = args
    src = Path(src)
    dst = Path(dst)
    try:
        if dst.exists() and dst.stat().st_size > 0:
            return (0, 0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            # draft() lets libjpeg decode at 1/2, 1/4 or 1/8 scale directly in
            # the DCT domain - far cheaper than decoding full size and then
            # resampling. Safe here because we are downscaling regardless.
            im.draft("RGB", (short_edge * 2, short_edge * 2))
            im = ImageOps.exif_transpose(im) or im
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) > short_edge:
                s = short_edge / min(w, h)
                im = im.resize(
                    (max(1, round(w * s)), max(1, round(h * s))),
                    Image.Resampling.BILINEAR,
                )
            tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
            im.save(tmp, format="JPEG", quality=quality, optimize=False)
            os.replace(tmp, dst)
            return (1, dst.stat().st_size)
    except Exception:
        # A file that will not decode here would also fail in training, where
        # the loader already drops it. Leave it absent rather than half-written.
        return (0, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="global_v1")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--short-edge", type=int, default=DEFAULT_SHORT_EDGE)
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8)))
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq

    manifest = PATHS.artifacts / args.corpus / "manifest.parquet"
    if not manifest.exists():
        raise SystemExit(f"{manifest} missing - build the corpus first")
    rows = pq.read_table(manifest, columns=["sha256", "cas_path"]).to_pylist()

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    # ~30 KB per cached image at 256px/q90; budget generously.
    need_gb = len(rows) * 40_000 / 1e9 + 2
    have = free_space_gb(cache_root)
    if have < need_gb:
        raise SystemExit(
            f"cache root {cache_root} has {have:.1f} GB free, needs ~{need_gb:.1f} GB"
        )

    print(f"corpus     : {args.corpus} ({len(rows):,} images)", flush=True)
    print(f"source     : {PATHS.cas}", flush=True)
    print(f"cache      : {cache_root}  ({have:.0f} GB free)", flush=True)
    print(f"short edge : {args.short_edge} px, JPEG q{args.quality}", flush=True)
    print(f"workers    : {args.workers}", flush=True)

    tasks = [
        (
            str(PATHS.cas / r["cas_path"]),
            str(_cache_path(cache_root, r["sha256"])),
            args.short_edge,
            args.quality,
        )
        for r in rows
    ]

    t0 = time.time()
    written = 0
    total_bytes = 0
    done = 0
    with Pool(processes=args.workers) as pool:
        for w, b in pool.imap_unordered(_one, tasks, chunksize=64):
            written += w
            total_bytes += b
            done += 1
            if done % 20000 == 0:
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                print(
                    f"  {done:,}/{len(tasks):,} ({100*done/len(tasks):.1f}%) "
                    f"{rate:.0f} img/s  {total_bytes/1e9:.2f} GB  "
                    f"eta {(len(tasks)-done)/max(rate,1e-6)/60:.1f}m",
                    flush=True,
                )

    el = time.time() - t0
    print("", flush=True)
    print(f"wrote {written:,} new entries in {el/60:.1f} min "
          f"({done/max(el,1e-6):.0f} img/s)", flush=True)
    print(f"cache size (new writes): {total_bytes/1e9:.2f} GB", flush=True)
    print(f"cache root: {cache_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
