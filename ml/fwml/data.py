"""Dataset, augmentation and sampling for fish classification.

Augmentation policy
-------------------
Augmentations are chosen to reproduce the conditions our users actually shoot
in, and to avoid destroying the features that distinguish species.

**Used**, because they mirror real angler photographs:

* horizontal flip - a fish can face either way
* random resized crop (scale 0.55-1.0) - framing varies enormously
* small rotation (+/-12 degrees) - hand-held, fish held at an angle
* brightness/contrast jitter - direct sun to dusk to headtorch
* mild colour-temperature shift - daylight vs tungsten vs LED
* motion/defocus blur - wet lens, wriggling fish, one-handed shot
* JPEG recompression - phone pipelines and messaging apps
* random erasing - fingers, net mesh and grass occlude parts of the fish

**Deliberately limited**:

* *Saturation and hue* jitter is kept small (0.15 / 0.02). Colour pattern is
  diagnostic for many of our species - the red fins of a rudd versus a roach,
  the flank spots of a brown versus a rainbow trout. Aggressive hue jitter is a
  standard ImageNet recipe and here it would actively teach the model to ignore
  the most reliable field mark. This is the augmentation most worth ablating.
* *Vertical flip* is not used. Fish are photographed the right way up, dorsal
  fin uppermost, essentially always; training upside-down fish spends capacity
  on a pose that never occurs.
* *Grayscale* is not used, for the same reason as hue.

Class balance
-------------
Image counts span 40 to 300 per class after capping. Plain shuffling lets the
common species dominate. We use a square-root-frequency weighted sampler, which
is a middle ground between natural frequency (biased) and full balancing
(over-samples 40-image classes 7x and overfits them).
"""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFile, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class AugmentConfig:
    size: int = 224
    scale_min: float = 0.55
    scale_max: float = 1.0
    rotation_degrees: float = 12.0
    brightness: float = 0.30
    contrast: float = 0.30
    saturation: float = 0.15     # kept low: colour pattern is diagnostic
    hue_shift: float = 0.02      # kept very low, same reason
    blur_prob: float = 0.20
    blur_max_radius: float = 1.8
    jpeg_prob: float = 0.25
    jpeg_quality_min: int = 35
    jpeg_quality_max: int = 90
    erase_prob: float = 0.25
    erase_max_fraction: float = 0.18
    hflip_prob: float = 0.5

    @staticmethod
    def eval_only(size: int = 224) -> "AugmentConfig":
        return AugmentConfig(
            size=size, scale_min=1.0, scale_max=1.0, rotation_degrees=0.0,
            brightness=0.0, contrast=0.0, saturation=0.0, hue_shift=0.0,
            blur_prob=0.0, jpeg_prob=0.0, erase_prob=0.0, hflip_prob=0.0,
        )


def _resize_shorter(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    s = target / min(w, h)
    return img.resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR
    )


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def eval_transform(img: Image.Image, size: int) -> Image.Image:
    """Resize-shorter-to-256/224 then centre crop.

    This must stay byte-for-byte equivalent to
    ``com.fisherwiki.core.infer.Preprocessor`` on the device side; the parity is
    pinned by a test that compares tensors produced by both implementations.
    """
    resize_to = round(size * 256 / 224)
    return center_crop(_resize_shorter(img, resize_to), size)


def train_transform(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> Image.Image:
    w, h = img.size

    # --- random resized crop ------------------------------------------------
    area = w * h
    for _ in range(10):
        target_area = area * rng.uniform(cfg.scale_min, cfg.scale_max)
        aspect = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw = int(round(math.sqrt(target_area * aspect)))
        ch = int(round(math.sqrt(target_area / aspect)))
        if cw <= w and ch <= h:
            x0 = rng.randint(0, w - cw)
            y0 = rng.randint(0, h - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            break
    else:
        img = center_crop(_resize_shorter(img, min(w, h)), min(w, h))

    if cfg.rotation_degrees > 0 and rng.random() < 0.5:
        img = img.rotate(
            rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees),
            resample=Image.Resampling.BILINEAR,
            expand=False,
        )

    img = img.resize((cfg.size, cfg.size), Image.Resampling.BILINEAR)

    if rng.random() < cfg.hflip_prob:
        img = ImageOps.mirror(img)

    return photometric(img, cfg, rng)


def photometric(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> Image.Image:
    """Colour, blur and recompression augmentation.

    Split out of :func:`train_transform` so the V2 crop path
    (:mod:`fwml.crop_v2`) can share it rather than growing a second, slowly
    diverging copy. The order of `rng` calls is unchanged, so V1 runs reproduce
    byte for byte - pinned by `test_v1_photometric_is_unchanged_by_the_refactor`.
    """
    if cfg.brightness:
        img = ImageEnhance.Brightness(img).enhance(
            1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
        )
    if cfg.contrast:
        img = ImageEnhance.Contrast(img).enhance(
            1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        )
    if cfg.saturation:
        img = ImageEnhance.Color(img).enhance(
            1.0 + rng.uniform(-cfg.saturation, cfg.saturation)
        )
    if cfg.hue_shift and rng.random() < 0.3:
        img = _shift_color_temperature(img, rng.uniform(-cfg.hue_shift, cfg.hue_shift))

    if cfg.blur_prob and rng.random() < cfg.blur_prob:
        from PIL import ImageFilter

        img = img.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.3, cfg.blur_max_radius))
        )

    if cfg.jpeg_prob and rng.random() < cfg.jpeg_prob:
        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=rng.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max),
        )
        buf.seek(0)
        img = Image.open(buf)
        img.load()

    return img


def _shift_color_temperature(img: Image.Image, amount: float) -> Image.Image:
    """Warm/cool shift by scaling R and B, preserving luminance better than hue."""
    arr = np.asarray(img, dtype=np.float32)
    arr[..., 0] *= 1.0 + amount * 4.0
    arr[..., 2] *= 1.0 - amount * 4.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def to_tensor(img: Image.Image, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def random_erase(t: torch.Tensor, cfg: AugmentConfig, rng: random.Random) -> torch.Tensor:
    """Occlusion, standing in for fingers, net mesh and grass."""
    if cfg.erase_prob <= 0 or rng.random() >= cfg.erase_prob:
        return t
    _, h, w = t.shape
    frac = rng.uniform(0.02, cfg.erase_max_fraction)
    eh = max(1, int(h * math.sqrt(frac)))
    ew = max(1, int(w * math.sqrt(frac)))
    y0 = rng.randint(0, max(0, h - eh))
    x0 = rng.randint(0, max(0, w - ew))
    # A seeded Generator, not torch.randn's default global RNG. The region
    # (y0, x0, eh, ew) above is already deterministic in (seed, epoch, idx)
    # via `rng`; the *fill* was not, because torch's global RNG state
    # advances with every call regardless of which item is being processed,
    # so two calls for the same item could still produce different noise.
    # Drawing the seed from `rng` keeps the whole augmentation - region and
    # fill both - inside the same reproducible chain.
    fill_gen = torch.Generator().manual_seed(rng.randrange(2**31))
    t[:, y0:y0 + eh, x0:x0 + ew] = torch.randn(3, eh, ew, generator=fill_gen) * 0.25
    return t


class FishDataset(Dataset):
    """Reads images from the content-addressed store, or a pre-resized cache.

    ``cache_root``, when given, points at the output of
    ``tools/prepare_cache.py``: the same images re-encoded at a 256 px short
    edge on fast storage. This is not an optimisation detail, it is the
    difference between a feasible training run and an infeasible one - see that
    script for the measurement. Falls back to the content-addressed store per
    image when a cache entry is missing, so a partial cache still trains.
    """

    def __init__(
        self,
        manifest,                 # pandas/pyarrow-backed list of dicts
        cas_root: Path,
        cfg: AugmentConfig,
        train: bool,
        seed: int = 0,
        cache_root: Path | None = None,
        emit_index: bool = False,
    ) -> None:
        self.rows = manifest
        self.cas_root = Path(cas_root)
        self.cache_root = Path(cache_root) if cache_root else None
        self.cfg = cfg
        self.train = train
        self.seed = seed
        # Emit the manifest row index alongside each sample. Evaluation needs
        # it to attribute a prediction back to its row: decode failures are
        # dropped from the batch, so position in the output is *not* position
        # in the manifest, and any per-row breakdown computed by zipping the
        # two would silently misalign after the first unreadable file.
        self.emit_index = emit_index
        # Shared across worker processes -- this is load-bearing, not a style
        # choice. `train.py` runs with `persistent_workers=True`, so worker
        # processes are spawned once and reused for every epoch; a plain
        # Python attribute set on the main-process dataset object after that
        # point would never reach them; DataLoader only re-pickles the dataset
        # at worker *startup*. `multiprocessing.Value` is backed by real
        # shared memory, so a mutation in the main process (see `set_epoch`)
        # is visible to already-running workers, including under Windows'
        # `spawn` start method. See `set_epoch` for why this exists at all.
        self._epoch = mp.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch, or every item gets identical augmentation
        on every epoch.

        Before this existed, `__getitem__`'s RNG was seeded from `(seed, idx)`
        only, despite a comment on that line claiming `(seed, epoch, idx)` --
        the epoch was never actually in the formula, and nothing threaded an
        epoch number into this class at all. The effect: image #18273 got the
        exact same random crop, flip, brightness, contrast, colour shift, blur,
        JPEG recompression and erase rectangle on epoch 1, 2, 3, ... 30. Not a
        crash, not a metric that looks wrong -- augmentation still ran, the
        loss curve still looked like augmentation was happening -- just a
        quieter, harder-to-notice kind of overfitting to 30 fixed variants of
        each photo instead of the many the config was written to produce.
        """
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row) -> Path:
        if self.cache_root is not None:
            sha = row.get("sha256")
            if sha:
                p = self.cache_root / sha[:2] / sha[2:4] / f"{sha}.jpg"
                if p.exists():
                    return p
        return self.cas_root / row["cas_path"]

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        # Per-item RNG so a worker's output depends only on (seed, epoch, idx)
        # -- actually depends on it now; see `set_epoch`. Deterministic in all
        # three, which keeps runs reproducible across different worker counts
        # and makes a resumed run's augmentation match a from-scratch one.
        rng = random.Random(
            (self.seed * 1_000_003 + self._epoch.value * 7_919_837 + idx) & 0x7FFFFFFF
        )
        try:
            with Image.open(self._path(row)) as im:
                # Decode at reduced scale directly in the DCT domain where the
                # JPEG allows it. Free, and meaningful when the source is a
                # 500 px original rather than a cache entry.
                im.draft("RGB", (self.cfg.size * 2, self.cfg.size * 2))
                im = ImageOps.exif_transpose(im) or im
                im = im.convert("RGB")
                img = (
                    train_transform(im, self.cfg, rng)
                    if self.train
                    else eval_transform(im, self.cfg.size)
                )
                t = to_tensor(img)
        except Exception:
            # A handful of files will be unreadable. Returning a zero tensor
            # with the true label would teach the model that grey means this
            # species, so we return a sentinel label of -1 and the training
            # loop drops those rows from the batch.
            if self.emit_index:
                return torch.zeros(3, self.cfg.size, self.cfg.size), -1, idx
            return torch.zeros(3, self.cfg.size, self.cfg.size), -1

        if self.train:
            t = random_erase(t, self.cfg, rng)
        if self.emit_index:
            return t, int(row["class_id"]), idx
        return t, int(row["class_id"])


def sqrt_balanced_sampler(class_ids: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Sample with weight proportional to 1/sqrt(class frequency).

    Full inverse-frequency balancing over-samples a 40-image class about 7x per
    epoch relative to a 300-image class, which overfits the rare classes we are
    least able to afford to overfit. The square root halves that pressure while
    still lifting the tail well above natural frequency.
    """
    counts = np.bincount(class_ids, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    per_class_weight = 1.0 / np.sqrt(counts)
    weights = per_class_weight[class_ids]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(class_ids),
        replacement=True,
    )


def collate_drop_failures(batch):
    """Drop items whose image failed to decode (label == -1)."""
    keep = [(x, y) for x, y in batch if y >= 0]
    if not keep:
        return None
    xs = torch.stack([x for x, _ in keep])
    ys = torch.tensor([y for _, y in keep], dtype=torch.long)
    return xs, ys


def collate_drop_failures_indexed(batch):
    """As :func:`collate_drop_failures`, but carries the manifest row index.

    Pair with ``FishDataset(..., emit_index=True)`` when a caller needs to join
    predictions back to manifest columns (observer, licence, geography). The
    index is what makes that join sound; zipping predictions against the row
    list positionally is only correct while nothing fails to decode, which is
    not a property any caller should depend on.
    """
    keep = [(x, y, i) for x, y, i in batch if y >= 0]
    if not keep:
        return None
    xs = torch.stack([x for x, _, _ in keep])
    ys = torch.tensor([y for _, y, _ in keep], dtype=torch.long)
    idx = torch.tensor([i for _, _, i in keep], dtype=torch.long)
    return xs, ys, idx
