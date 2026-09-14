#!/usr/bin/env python
"""Detect the fish in every stored image and record the box.

    python tools/detect_fish.py --limit 2000
    python tools/detect_fish.py                     # everything, resumable

Writes one row per image to ``detections``. Resumable by construction: images
that already have a row are skipped, so this can be interrupted and re-run for
as long as it takes without losing work or redoing any.

Why a detector at all
---------------------
Measured on our own test images, cropping to the fish before classification is
worth **+18.9 points** of top-1 (0.417 -> 0.606, same classifier, n=700), and on
the venomous *Trachinus draco* case it is worth 0.038 -> 0.538. No other change
available to this project comes close.

Which detector
--------------
Fishial's YOLO26-nano fish detector, which found a fish in 95% of our images at
the default threshold. It is MIT licensed, so it can be used and redistributed
with attribution. **That licence covers the repository; whether the
GCS-hosted weights inherit it has not been separately confirmed, and must be
before anything derived from it ships** - this project excluded FishNet
entirely over exactly this question, and the same standard applies here.
Detections are only *metadata*, so nothing about the licence blocks using them
to choose crops for our own training.

Boxes from a source that already has them are not overwritten - see
``--source-boxes``. FathomNet and the open bounding-box datasets publish real
annotations, and a human-drawn box beats an inferred one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from fwdata.config import PATHS  # noqa: E402

DETECTOR_DIR = Path(r"D:\fisherwiki-data\models\fishial\detector")
MODEL_NAME = "fishial_yolo26n_v3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    sha256      VARCHAR,
    x0          DOUBLE,
    y0          DOUBLE,
    x1          DOUBLE,
    y1          DOUBLE,
    score       DOUBLE,
    area_frac   DOUBLE,
    n_boxes     INTEGER,
    source      VARCHAR,      -- 'detector' | 'annotation'
    model       VARCHAR,
    is_primary  BOOLEAN,
    detected_at TIMESTAMP,
    PRIMARY KEY (sha256, model)
);
"""


def _letterbox_square(im, imgsz: int):
    """Pad to a uniform square so a real batch can be formed.

    Variable-size inputs make ultralytics fall back to batch-of-one: measured
    4 img/s against 87 for a uniform tensor batch. Padding rather than
    stretching also keeps the boxes honest - a stretched frame yields a box for
    a shape the fish never had.
    """
    from PIL import Image

    from fwml.crop_v2 import PAD_RGB

    w, h = im.size
    scale = imgsz / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (imgsz, imgsz), PAD_RGB)
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (0, 0))
    return canvas, scale


class _DetectSet:
    """Decodes and letterboxes one image. Module level so Windows `spawn` can
    pickle it into DataLoader workers."""

    def __init__(self, items, cas_root, imgsz=640):
        self.items = items
        self.cas_root = cas_root
        self.imgsz = imgsz

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import numpy as np
        import torch
        from PIL import Image

        sha, rel = self.items[i]
        fp = self.cas_root / rel if rel else None
        if fp is None or not fp.exists():
            return None
        try:
            with Image.open(fp) as im:
                # Decode straight to roughly the size we need. draft() lets the
                # JPEG decoder drop to 1/2, 1/4 or 1/8 scale in the DCT domain,
                # so a 4000x3000 original is decoded at 1000x750 rather than
                # decoded in full and then thrown away. It never goes below the
                # requested size, so the subsequent resize still does the exact
                # framing. The full size is captured *before* drafting, because
                # box coordinates must be reported against the original image.
                size = im.size
                im.draft("RGB", (self.imgsz, self.imgsz))
                im = im.convert("RGB")
                padded, scale = _letterbox_square(im, self.imgsz)
                # scale now maps drafted pixels -> 640; boxes need original ->
                # 640, so fold in whatever draft() already did.
                scale = scale * (im.size[0] / size[0])
            # uint8, converted to float on the GPU: sending float32 is 4x the
            # PCIe traffic and puts the divide on the main thread.
            arr = torch.from_numpy(np.asarray(padded).copy()).permute(2, 0, 1)
            return sha, size[0], size[1], scale, arr
        except Exception:
            return None


def _collate_detect(items):
    import torch

    items = [x for x in items if x is not None]
    if not items:
        return None
    meta = [(s, (w, h), sc) for s, w, h, sc, _ in items]
    return meta, torch.stack([a for *_, a in items])


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def run(
    *,
    limit: int | None = None,
    batch: int = 16,
    decode_workers: int = 8,
    conf: float = 0.25,
    iou: float = 0.45,
    db_path: Path | None = None,
    log=_log,
) -> dict:
    import duckdb
    import numpy as np  # noqa: F401  (used by the worker dataset)
    import torch
    from PIL import Image  # noqa: F401  (used by the worker dataset)
    from ultralytics import YOLO
    from ultralytics.utils import nms

    weights = DETECTOR_DIR / "model.pt"
    if not weights.exists():
        raise SystemExit(
            f"detector weights not found at {weights}\n"
            "Fetch once with:\n"
            "  python tools/detect_fish.py --fetch-weights"
        )

    con = duckdb.connect(str(db_path or PATHS.provenance_db))
    con.execute(_SCHEMA)

    rows = con.execute(
        """
        SELECT p.sha256, any_value(p.cas_path)
        FROM provenance p
        WHERE p.sha256 IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM detections d WHERE d.sha256 = p.sha256)
        GROUP BY p.sha256
        -- Read in on-disk order. The CAS lays files out as cas/ab/cd/<sha>, so
        -- hash order walks 65,536 directories at random, and on a spinning disk
        -- that is a seek per image. Sorting by path turns the pass into a
        -- near-sequential walk. Resumability is unaffected: the set of pending
        -- images is defined by the NOT EXISTS, not by the order they arrive in.
        ORDER BY any_value(p.cas_path)
        """
    ).fetchall()
    if limit:
        rows = rows[:limit]
    total = len(rows)
    log(f"{total:,} images without a detection")
    if not total:
        con.close()
        return {"processed": 0}

    # Fuse and call the network directly rather than going through
    # YOLO.predict(). Two reasons, one of which is not about speed:
    #
    # Fusing folds BatchNorm into the preceding convolutions, which is also the
    # only reason this runs on ROCm/Windows at all - an unfused forward hits the
    # MIOpen BatchNorm bug that env.py documents.
    #
    # predict() additionally copies the whole GPU batch back to CPU
    # (`ops.convert_torch2numpy_batch`) to build Results objects this pipeline
    # never looks at. Measured, that costs about 1%: 101 img/s through predict()
    # against 103 direct, because this detector is YOLO26-*medium* at 68.1
    # GFLOPs, so inference dwarfs a 39 MB transfer. Worth removing because it is
    # pointless work, not because it was the bottleneck.
    net = YOLO(str(weights)).model
    if torch.cuda.is_available():
        net = net.to("cuda")
    net = net.eval().fuse()
    cas = PATHS.cas
    done = found = 0
    pending: list[tuple] = []
    t0 = time.time()

    _flush_seconds = [0.0]

    def flush():
        """Bulk-insert via Arrow rather than row-at-a-time executemany.

        This was the actual bottleneck, and it hid well: instrumenting the loop
        showed 14s waiting on the loader and 47s on the GPU out of ~270s wall
        for 4,000 images. The missing 200s was here - `executemany` against a
        table with a composite primary key re-plans and re-indexes per row on a
        280 MB database. The same substitution was worth a similar amount in
        `splits_v2`.
        """
        nonlocal pending
        if not pending:
            return
        import time as _time

        import pyarrow as pa

        _t = _time.time()
        cols = list(zip(*pending))
        names = ["sha256", "x0", "y0", "x1", "y1", "score", "area_frac",
                 "n_boxes", "source", "model", "is_primary", "detected_at"]
        types = [pa.string(), pa.float64(), pa.float64(), pa.float64(),
                 pa.float64(), pa.float64(), pa.float64(), pa.int32(),
                 pa.string(), pa.string(), pa.bool_(), pa.timestamp("us")]
        tbl = pa.table({n: pa.array(c, type=ty)
                        for n, c, ty in zip(names, cols, types)})
        con.register("_det_rows", tbl)
        try:
            con.execute("INSERT OR REPLACE INTO detections SELECT * FROM _det_rows")
        finally:
            con.unregister("_det_rows")
        _flush_seconds[0] += _time.time() - _t
        pending = []

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    import numpy as np
    import torch

    from fwml.crop_v2 import PAD_RGB


    # Decode in worker *processes*, overlapped with inference.
    #
    # Profiled on cold images: decode 62 img/s, upload 526, inference 84. Run
    # strictly in sequence that is ~33 img/s at best, and the real run measured
    # 14-19 because the images are scattered across 480k files on a spinning
    # disk, which needs queue depth to go fast and gets none from one reader.
    # A thread pool inside the loop did not help (19 img/s) because decode and
    # inference still alternate - the GPU idles through every decode.
    #
    # A DataLoader gives both: several processes reading at once, so the disk
    # sees a real queue, and prefetch, so decode for the next batch overlaps
    # inference on the current one.
    from torch.utils.data import DataLoader

    loader = DataLoader(
        _DetectSet(rows, cas), batch_size=batch, shuffle=False,
        num_workers=decode_workers, collate_fn=_collate_detect,
        # 2, not 4: at 12 workers x 4 x batch 32 this queue held up to
        # 1,536 letterboxed images - about 1.9 GB - moving through Windows
        # multiprocessing IPC. Depth 2 is ample to hide the GPU step.
        prefetch_factor=2 if decode_workers else None,
    )

    _t_wait = _t_gpu = 0.0
    _last = time.time()
    for loaded in loader:
        _t_wait += time.time() - _last
        if loaded is None:
            _last = time.time()
            continue
        keep, stack = loaded
        _g = time.time()
        if torch.cuda.is_available():
            stack = stack.cuda(non_blocking=True)
        stack = stack.float().div_(255)

        with torch.no_grad():
            raw = net(stack)
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        dets = nms.non_max_suppression(raw, conf, iou, nc=1, max_det=10)
        # One transfer for the whole batch instead of a .tolist() per image,
        # each of which is its own device synchronisation.
        dets = [d.cpu().tolist() for d in dets]

        for (sha, (w, h), scale), boxes in zip(keep, dets):
            done += 1
            if not boxes:
                pending.append((sha, None, None, None, None, None, None, 0,
                                "detector", MODEL_NAME, False, now))
                continue
            best = max(boxes, key=lambda b: b[4])
            # Back out of letterbox space. Padding sits bottom-right, so only
            # the scale has to be undone, but the result is still clamped: a box
            # can legitimately touch the padded edge.
            x0 = max(0.0, min(w, best[0] / scale))
            y0 = max(0.0, min(h, best[1] / scale))
            x1 = max(0.0, min(w, best[2] / scale))
            y1 = max(0.0, min(h, best[3] / scale))
            if x1 - x0 < 2 or y1 - y0 < 2:
                pending.append((sha, None, None, None, None, None, None,
                                len(boxes), "detector", MODEL_NAME, False, now))
                continue
            area = ((x1 - x0) * (y1 - y0)) / float(max(1, w * h))
            pending.append((sha, x0, y0, x1, y1, best[4], area, len(boxes),
                            "detector", MODEL_NAME, True, now))
            found += 1

        _t_gpu += time.time() - _g
        if len(pending) >= 4000:
            flush()
        if done % 2000 < batch:
            rate = done / max(1e-6, time.time() - t0)
            eta = (total - done) / max(rate, 1e-6) / 60
            log(f"  {done:,}/{total:,}  {rate:.0f} img/s  found {found:,} "
                f"({100 * found / max(1, done):.0f}%)  eta {eta:.0f} min"
                f"  [loader {_t_wait:.0f}s, gpu {_t_gpu:.0f}s, db {_flush_seconds[0]:.0f}s]")
        _last = time.time()

    flush()
    stats = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE is_primary), "
        "avg(area_frac) FILTER (WHERE is_primary) FROM detections"
    ).fetchone()
    con.close()
    out = {
        "processed": done,
        "with_box": found,
        "total_rows": stats[0],
        "total_with_box": stats[1],
        "mean_area_fraction": round(stats[2] or 0.0, 4),
        "seconds": round(time.time() - t0, 1),
    }
    log(f"done: {done:,} processed, {found:,} with a box "
        f"({100 * found / max(1, done):.1f}%), mean box covers "
        f"{100 * (stats[2] or 0):.0f}% of frame")
    return out


def fetch_weights(log=_log) -> Path:
    """One-time download of the detector into the data root."""
    import urllib.request
    import zipfile

    DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DETECTOR_DIR / "detector_v26_n3.zip"
    if not (DETECTOR_DIR / "model.pt").exists():
        url = "https://storage.googleapis.com/fishial-ml-resources/detector_v26_n3.zip"
        log(f"downloading {url}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.endswith(".pt") and "MACOSX" not in name:
                    with z.open(name) as src, open(DETECTOR_DIR / "model.pt", "wb") as dst:
                        dst.write(src.read())
                elif name.endswith(".json") and "MACOSX" not in name:
                    with z.open(name) as src, open(DETECTOR_DIR / "info.json", "wb") as dst:
                        dst.write(src.read())
        zip_path.unlink(missing_ok=True)
    log(f"detector ready at {DETECTOR_DIR / 'model.pt'}")
    return DETECTOR_DIR / "model.pt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16,
                    help="16 measured fastest; 64 thrashes VRAM to 8 img/s")
    ap.add_argument("--decode-workers", type=int, default=4,
                    help="decode workers. Measured end to end: 4 fastest, 8 slower, "
                         "12 much slower - concurrent readers thrash an HDD")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--fetch-weights", action="store_true")
    args = ap.parse_args(argv)

    if args.fetch_weights:
        fetch_weights()
        return 0
    run(limit=args.limit, batch=args.batch, conf=args.conf, iou=args.iou,
        decode_workers=args.decode_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
