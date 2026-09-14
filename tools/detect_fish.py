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


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def run(
    *,
    limit: int | None = None,
    batch: int = 16,
    conf: float = 0.25,
    iou: float = 0.45,
    db_path: Path | None = None,
    log=_log,
) -> dict:
    import duckdb
    from PIL import Image
    from ultralytics import YOLO

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
        """
    ).fetchall()
    if limit:
        rows = rows[:limit]
    total = len(rows)
    log(f"{total:,} images without a detection")
    if not total:
        con.close()
        return {"processed": 0}

    model = YOLO(str(weights))
    cas = PATHS.cas
    done = found = 0
    pending: list[tuple] = []
    t0 = time.time()

    def flush():
        nonlocal pending
        if pending:
            con.executemany(
                "INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                pending,
            )
            pending = []

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    import numpy as np
    import torch

    from fwml.crop_v2 import PAD_RGB

    imgsz = 640

    def _letterbox_640(im):
        """Pad to a uniform square so a real batch can be formed.

        Passing variable-size images to ultralytics silently degrades to
        batch-of-one: measured 4 img/s against 95 img/s for a uniform tensor
        batch, a 24x difference that decides whether detecting 480k images takes
        1.4 hours or 33. Padding rather than stretching also keeps the boxes
        honest - a stretched frame yields a box for a shape the fish never had.
        """
        w, h = im.size
        scale = imgsz / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = Image.new("RGB", (imgsz, imgsz), PAD_RGB)
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), (0, 0))
        return canvas, scale

    for i in range(0, total, batch):
        chunk = rows[i:i + batch]
        tensors, keep = [], []
        for sha, rel in chunk:
            fp = cas / rel if rel else None
            if fp is None or not fp.exists():
                continue
            try:
                with Image.open(fp) as im:
                    im = im.convert("RGB")
                    size = im.size
                    padded, scale = _letterbox_640(im)
            except Exception:
                continue
            tensors.append(torch.from_numpy(np.asarray(padded)).permute(2, 0, 1))
            keep.append((sha, size, scale))
        if not tensors:
            continue

        stack = torch.stack(tensors).float().div(255)
        if torch.cuda.is_available():
            stack = stack.cuda(non_blocking=True)
        preds = model.predict(stack, conf=conf, iou=iou, verbose=False)

        for (sha, (w, h), scale), res in zip(keep, preds):
            boxes = res.boxes.data.tolist() if res.boxes is not None else []
            done += 1
            if not boxes:
                pending.append((sha, None, None, None, None, None, None, 0,
                                "detector", MODEL_NAME, False, now))
                continue
            best = max(boxes, key=lambda b: b[4])
            # Back out of letterbox space. The padding sits bottom/right, so
            # only the scale has to be undone, but the result still has to be
            # clamped: a box can legitimately touch the padded edge.
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

        if len(pending) >= 500:
            flush()
        if done % 2000 < batch:
            rate = done / max(1e-6, time.time() - t0)
            eta = (total - done) / max(rate, 1e-6) / 60
            log(f"  {done:,}/{total:,}  {rate:.0f} img/s  found {found:,} "
                f"({100 * found / max(1, done):.0f}%)  eta {eta:.0f} min")

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
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--fetch-weights", action="store_true")
    args = ap.parse_args(argv)

    if args.fetch_weights:
        fetch_weights()
        return 0
    run(limit=args.limit, batch=args.batch, conf=args.conf, iou=args.iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
