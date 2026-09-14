"""Real images for static INT8 quantisation calibration.

Static quantisation needs representative activation ranges. Calibrating on
Gaussian noise is easy and wrong: the activation distribution of a network fed
noise is not the one it sees on photographs, and calibrating on noise is a
well-known way to lose several points of accuracy for no reason at all.

So this draws from the **validation** split - never train (which the model has
memorised to some degree) and never test (which must stay untouched) - and
spreads the sample across classes rather than taking the first N rows, which
would calibrate on whichever species sorts first alphabetically.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from fwdata.config import PATHS

from .data import AugmentConfig, eval_transform, to_tensor


def load_calibration_tensors(
    corpus: str,
    size: int = 224,
    count: int = 200,
    split: str = "val",
    seed: int = 0,
) -> list[np.ndarray]:
    """Return ``count`` preprocessed NCHW float32 arrays from ``split``."""
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    manifest = PATHS.artifacts / corpus / "manifest.parquet"
    if not manifest.exists():
        raise SystemExit(f"{manifest} missing - build the corpus first")

    rows = [r for r in pq.read_table(manifest).to_pylist() if r["split"] == split]
    if not rows:
        raise SystemExit(f"no rows in split {split!r}")

    # Stratify: at most a couple per class, so a 300-image species does not
    # dominate the calibration set the way it would in a flat random sample.
    by_class: dict[int, list] = {}
    for r in rows:
        by_class.setdefault(r["class_id"], []).append(r)

    rng = random.Random(seed)
    picked: list = []
    per_class = max(1, count // max(1, len(by_class)))
    for cid in sorted(by_class):
        rng.shuffle(by_class[cid])
        picked.extend(by_class[cid][:per_class])
    rng.shuffle(picked)
    picked = picked[:count]

    cfg = AugmentConfig.eval_only(size)
    out: list[np.ndarray] = []
    for r in picked:
        path = Path(PATHS.cas) / r["cas_path"]
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im) or im
                img = eval_transform(im.convert("RGB"), cfg.size)
                t = to_tensor(img).unsqueeze(0).numpy().astype(np.float32)
                out.append(t)
        except Exception:
            continue
    if not out:
        raise SystemExit("no calibration images could be decoded")
    return out
