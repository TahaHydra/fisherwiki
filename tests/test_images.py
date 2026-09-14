"""Image hashing and quality measurement, tested on synthesised images."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from fwdata.images import (
    ImageFacts,
    ImageRejected,
    dhash,
    hamming,
    is_probably_duplicate,
    measure,
    phash,
)


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _encode(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def _fishy(seed: int = 0, w: int = 480, h: int = 320) -> Image.Image:
    """A deterministic, structured, non-uniform test image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        128
        + 60 * np.sin(xx / 23.0)
        + 40 * np.cos(yy / 17.0)
        + rng.normal(0, 8, size=(h, w))
    )
    arr = np.stack([base, base * 0.85 + 20, base * 0.6 + 40], axis=2)
    return _img(np.clip(arr, 0, 255))


class TestMeasureBasics:
    def test_returns_facts_for_a_normal_image(self):
        f = measure(_encode(_fishy()))
        assert isinstance(f, ImageFacts)
        assert f.width == 480 and f.height == 320
        assert len(f.sha256) == 64
        assert len(f.phash) == 16 and len(f.dhash) == 16
        assert f.format == "JPEG"

    def test_sha256_is_of_the_exact_bytes(self):
        import hashlib

        data = _encode(_fishy())
        assert measure(data).sha256 == hashlib.sha256(data).hexdigest()

    def test_png_and_jpeg_of_same_pixels_have_different_sha_same_perceptual(self):
        img = _fishy(1)
        a = measure(_encode(img, "PNG"))
        b = measure(_encode(img, "JPEG", quality=95))
        assert a.sha256 != b.sha256
        # Re-encoding must not defeat perceptual matching - this is the whole
        # reason we do not rely on sha256 alone for dedupe. The contract is
        # is_probably_duplicate(), not a particular bit distance: _fishy()
        # carries a deliberate high-frequency noise component that JPEG smooths
        # away, so a few DCT bits legitimately flip.
        assert is_probably_duplicate(a, b)
        assert hamming(a.dhash, b.dhash) <= 4

    def test_low_texture_image_still_deduplicates_via_dhash(self):
        # Regression guard for a measured pHash failure mode: on a low-texture
        # image every DCT coefficient sits near the median, so re-encoding
        # flips ~a third of the phash bits (measured distance 22/64) while
        # dhash is unaffected (distance 0). This is why is_probably_duplicate
        # ORs the two hashes; dropping dhash would break dedupe on exactly the
        # smooth, evenly-lit photos a studio-style fish shot produces.
        h, w = 320, 480
        yy, xx = np.mgrid[0:h, 0:w]
        base = 128 + 60 * np.sin(xx / 40.0) + 40 * np.cos(yy / 30.0)
        img = _img(np.clip(np.stack([base, base * 0.8 + 20, base * 0.6 + 40], 2), 0, 255))
        a = measure(_encode(img, "PNG"))
        b = measure(_encode(img, "JPEG", quality=90))

        assert hamming(a.dhash, b.dhash) <= 2
        assert is_probably_duplicate(a, b)


class TestRejection:
    def test_undecodable_bytes_rejected(self):
        with pytest.raises(ImageRejected):
            measure(b"this is definitely not an image")

    def test_empty_bytes_rejected(self):
        with pytest.raises(ImageRejected):
            measure(b"")

    def test_single_colour_image_rejected(self):
        flat = _img(np.full((200, 200, 3), 140))
        with pytest.raises(ImageRejected):
            measure(_encode(flat, "PNG"))

    def test_truncated_jpeg_does_not_crash(self):
        data = _encode(_fishy(3))
        truncated = data[: int(len(data) * 0.75)]
        # Either it decodes (PIL fills the rest) or it is cleanly rejected;
        # what must not happen is an unhandled exception type.
        try:
            f = measure(truncated)
            assert f.width > 0
        except ImageRejected:
            pass


class TestQualityFlags:
    def test_tiny_image_flagged(self):
        f = measure(_encode(_fishy(4, w=60, h=40), "PNG"))
        assert "too_small" in f.flags

    def test_extreme_aspect_flagged(self):
        f = measure(_encode(_fishy(5, w=1200, h=100), "PNG"))
        assert "extreme_aspect" in f.flags
        assert f.aspect_ratio > 4.0

    def test_greyscale_flagged(self):
        g = _fishy(6).convert("L").convert("RGB")
        f = measure(_encode(g, "PNG"))
        assert "near_greyscale" in f.flags

    def test_very_dark_flagged(self):
        arr = np.asarray(_fishy(7), dtype=np.float64) * 0.04
        f = measure(_encode(_img(arr), "PNG"))
        assert "very_dark" in f.flags

    def test_blur_score_orders_sharp_above_blurred(self):
        from PIL import ImageFilter

        sharp = _fishy(8)
        blurred = sharp.filter(ImageFilter.GaussianBlur(radius=5))
        fs = measure(_encode(sharp, "PNG"))
        fb = measure(_encode(blurred, "PNG"))
        assert fs.blur_score > fb.blur_score

    def test_normal_image_has_no_hard_flags(self):
        f = measure(_encode(_fishy(9)))
        assert "too_small" not in f.flags
        assert "extreme_aspect" not in f.flags


class TestPerceptualHashing:
    def test_hash_is_stable_for_identical_input(self):
        img = _fishy(10)
        assert phash(img) == phash(img)
        assert dhash(img) == dhash(img)

    def test_different_images_differ(self):
        a, b = _fishy(11), _fishy(12)
        assert hamming(phash(a), phash(b)) > 6

    def test_mild_rescale_preserves_hash(self):
        img = _fishy(13)
        small = img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)
        assert hamming(dhash(img), dhash(small)) <= 6

    def test_jpeg_quality_drop_preserves_hash(self):
        img = _fishy(14)
        a = measure(_encode(img, "JPEG", quality=95))
        b = measure(_encode(img, "JPEG", quality=35))
        assert is_probably_duplicate(a, b)

    def test_hamming_of_equal_hashes_is_zero(self):
        assert hamming("abcd1234abcd1234", "abcd1234abcd1234") == 0

    def test_hamming_handles_missing_hash(self):
        assert hamming("", "abcd1234abcd1234") == 64


class TestExifOrientation:
    def test_exif_transpose_rotates_portrait_photo(self):
        from fwdata.images import exif_transposed

        img = _fishy(15, w=400, h=300)
        buf = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6  # Orientation: rotate 90 CW
        img.save(buf, format="JPEG", exif=exif)
        loaded = Image.open(io.BytesIO(buf.getvalue()))
        out = exif_transposed(loaded)
        # A tagged-rotated landscape image must come back portrait, otherwise
        # training and inference disagree about which way up a fish is.
        assert out.size == (300, 400)
