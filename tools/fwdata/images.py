"""Image decoding, hashing and quality measurement.

Everything here operates on bytes or PIL images and has no network or database
dependency, so it is straightforward to test.

Hashes
------
``sha256``   exact-duplicate detection and content addressing.
``dhash``    64-bit difference hash (each pixel vs its right neighbour).
``phash``    64-bit DCT hash thresholded at the median coefficient.

Both are computed on a normalised greyscale thumbnail and compared by Hamming
distance. :func:`is_probably_duplicate` matches on **either**, which is not
belt-and-braces but a measured necessity:

=====================  =========  =========
same pixels, PNG->JPEG  phash dist  dhash dist
=====================  =========  =========
low-texture image              22          0
lightly textured               10          0
noisy synthetic                 6          2
=====================  =========  =========

A median-thresholded DCT hash is *unreliable on low-texture images*: when an
image has little high-frequency energy every coefficient sits near the median,
so trivial re-encoding noise flips a third of the bits. dhash does not have
this failure mode. Do not "simplify" this by dropping dhash and keeping only
phash - that is the intuitive choice and it is the wrong one.

Quality signals
---------------
The measurements here are deliberately *descriptive*, not prescriptive. They
are recorded per image and thresholds are applied later, because the right
threshold is a product decision that differs per corpus. A blurry photo of a
fish in a landing net at dusk is a **valuable** training example, not garbage -
see `docs/DATASETS.md`. Only genuinely unusable images (undecodable, tiny,
single-colour) are hard-rejected.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

# Truncated JPEGs are common in any large scrape-free corpus too; decoding what
# we can and flagging it beats discarding the record entirely.
ImageFile.LOAD_TRUNCATED_IMAGES = True

#: Below this on the short edge an image cannot support a 224px crop without
#: heavy upsampling.
MIN_SHORT_EDGE = 96

#: Aspect ratios beyond this are usually banners, collages or scans.
MAX_ASPECT_RATIO = 4.0


class ImageRejected(ValueError):
    """Image cannot be used at all (not merely low quality)."""


@dataclass
class ImageFacts:
    """Measured facts about one image."""

    sha256: str
    nbytes: int
    width: int
    height: int
    mode: str
    format: str | None
    dhash: str
    phash: str
    mean_luma: float
    #: Variance of the Laplacian; low means blurry. Scale depends on resolution,
    #: so compare within a corpus rather than against an absolute constant.
    blur_score: float
    #: Fraction of pixels that are pure black or pure white.
    clipped_fraction: float
    #: Standard deviation of colour saturation; ~0 means greyscale/specimen scan.
    saturation_std: float
    flags: list[str] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float:
        lo, hi = sorted((self.width, self.height))
        return hi / max(1, lo)

    def as_row(self) -> dict:
        d = self.__dict__.copy()
        d["flags"] = ",".join(self.flags)
        d["aspect_ratio"] = round(self.aspect_ratio, 4)
        return d


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_gray_array(img: Image.Image, size: int) -> np.ndarray:
    g = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(g, dtype=np.float64)


def dhash(img: Image.Image, size: int = 8) -> str:
    """64-bit difference hash: compare each pixel with its right neighbour."""
    a = _to_gray_array(img, size + 1)[:, : size + 1]
    diff = a[:size, 1:] > a[:size, :size]
    return _bits_to_hex(diff.flatten())


def phash(img: Image.Image, size: int = 8, factor: int = 4) -> str:
    """64-bit perceptual hash from the low-frequency DCT coefficients."""
    n = size * factor
    a = _to_gray_array(img, n)
    d = _dct2(a)[:size, :size]
    # Exclude the DC term from the median so a uniform brightness shift does
    # not flip every bit.
    flat = d.flatten()
    med = np.median(flat[1:])
    return _bits_to_hex(flat > med)


def _dct1(a: np.ndarray) -> np.ndarray:
    """Type-II DCT along the last axis (no SciPy dependency)."""
    n = a.shape[-1]
    k = np.arange(n)
    # Orthonormal-ish basis; absolute scale is irrelevant for a median compare.
    basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    return a @ basis.T


def _dct2(a: np.ndarray) -> np.ndarray:
    return _dct1(_dct1(a).T).T


def _bits_to_hex(bits: Iterable[bool]) -> str:
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return f"{v:016x}"


def hamming(a: str, b: str) -> int:
    """Hamming distance between two hex hash strings."""
    if not a or not b:
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian - the standard cheap focus measure."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    # Valid-mode 2-D convolution via stride tricks (small, fixed kernel).
    sub = np.lib.stride_tricks.sliding_window_view(gray, (3, 3))
    lap = np.einsum("ijkl,kl->ij", sub, k)
    return float(lap.var())


def measure(data: bytes, *, thumb: int = 256) -> ImageFacts:
    """Decode ``data`` and compute all hashes and quality signals.

    Raises :class:`ImageRejected` only for images that cannot be used at all.
    """
    digest = sha256_bytes(data)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected(f"undecodable: {exc}") from exc

    fmt = img.format
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ImageRejected("zero-sized image")

    flags: list[str] = []
    if min(w, h) < MIN_SHORT_EDGE:
        flags.append("too_small")

    rgb = img.convert("RGB")
    small = rgb.resize(
        (min(thumb, max(1, w)), min(thumb, max(1, h))), Image.Resampling.BILINEAR
    )
    arr = np.asarray(small, dtype=np.float64) / 255.0
    gray = arr.mean(axis=2)

    mean_luma = float(gray.mean())
    blur = _blur_score(gray * 255.0)
    clipped = float(((gray <= 0.004) | (gray >= 0.996)).mean())

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    sat_std = float(sat.std())

    # Reject only genuinely flat images. Using the standard deviation here is a
    # trap: a real night-time photo of a fish under a headtorch has a small std
    # simply because it is dark, and rejecting it would throw away exactly the
    # hard, valuable examples this app has to handle. Peak-to-peak range
    # separates "dark but textured" from "one solid colour".
    if float(gray.max() - gray.min()) < 0.01:
        raise ImageRejected("single-colour image")

    facts = ImageFacts(
        sha256=digest,
        nbytes=len(data),
        width=w,
        height=h,
        mode=img.mode,
        format=fmt,
        dhash=dhash(rgb),
        phash=phash(rgb),
        mean_luma=round(mean_luma, 5),
        blur_score=round(blur, 4),
        clipped_fraction=round(clipped, 5),
        saturation_std=round(sat_std, 5),
        flags=flags,
    )

    if facts.aspect_ratio > MAX_ASPECT_RATIO:
        facts.flags.append("extreme_aspect")
    if sat_std < 0.02:
        facts.flags.append("near_greyscale")
    if clipped > 0.35:
        facts.flags.append("heavily_clipped")
    if mean_luma < 0.06:
        facts.flags.append("very_dark")
    elif mean_luma > 0.94:
        facts.flags.append("very_bright")
    return facts


def is_probably_duplicate(a: ImageFacts, b: ImageFacts, *, threshold: int = 6) -> bool:
    """Perceptual near-duplicate test used for dedupe and leakage checks."""
    if a.sha256 == b.sha256:
        return True
    return (
        hamming(a.phash, b.phash) <= threshold
        or hamming(a.dhash, b.dhash) <= threshold
    )


def exif_transposed(img: Image.Image) -> Image.Image:
    """Apply the EXIF orientation tag so downstream code sees upright pixels.

    Phone cameras overwhelmingly store landscape sensor data plus a rotation
    tag. Training on un-transposed pixels while the app transposes (or the
    reverse) is a silent train/serve skew, so both sides go through this.
    """
    from PIL import ImageOps

    return ImageOps.exif_transpose(img) or img
