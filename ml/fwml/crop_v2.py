"""V2 geometry: tight fish crops that preserve aspect ratio.

Why letterbox rather than resize-to-square
------------------------------------------
Measured, not assumed. Fishial's classifier resizes every input to a fixed
434x154 (2.82:1). Given a correctly detected ocean sunfish - a disc, 1044x1044,
detector confidence 0.89 - that stretch turns the disc into a torpedo, and the
model returns *Echeneis naucrates*, a remora, at **98.4% confidence**. The true
species was in its class list and dropped to 0.0%. The crop was perfect; the
resize destroyed the body shape, and body shape is the primary diagnostic for
most fish.

V1 has a milder version of the same flaw: ``train_transform`` finishes with
``img.resize((size, size))``, which stretches whatever the random-resized-crop
produced. The crop bounds aspect to 3:4-4:3 so the distortion is small, but it
is still there, and a tight detector crop makes it much worse - fish are far
from square once you cut the background away.

So V2 resizes on the **long** edge and pads the short one. Nothing is stretched
and nothing is cut off, which matters for both a 4:1 eel and a 1:1 sunfish.
The padding costs some pixels; being able to trust body shape is worth more.
"""

from __future__ import annotations

import random

from PIL import Image

#: Pad colour, ImageNet mean in 0-255. A neutral grey-brown sits closer to the
#: normalised zero than black does, so the padding contributes less activation
#: than a hard border would.
PAD_RGB = (124, 116, 104)

#: Context kept around a detector box, as a fraction of box size. A tight box
#: cuts the fins, and fin shape separates species that body colour does not;
#: too much context and the crop stops being a crop. 0.25 keeps the whole
#: animal plus a little water.
DEFAULT_CONTEXT = 0.25


def expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    context: float = DEFAULT_CONTEXT,
) -> tuple[int, int, int, int]:
    """Grow a detection box by ``context`` on each side, clamped to the image."""
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    px, py = bw * context, bh * context
    return (
        max(0, int(round(x0 - px))),
        max(0, int(round(y0 - py))),
        min(width, int(round(x1 + px))),
        min(height, int(round(y1 + py))),
    )


def grow_box_to_min(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    min_px: int,
) -> tuple[int, int, int, int]:
    """Widen a small box until its long edge reaches ``min_px``, if the image allows.

    A tight box around a distant fish can be 60 px across. Cropping to it and
    letting the trainer resize to 448 is a 7x upscale - it does not recover
    detail, it invents it, and the model learns from blur. Measured on the first
    2,000 prepared images, 10% of boxes had a long edge under 160 px and 3%
    under 96.

    Growing the box keeps the fish centred while filling the frame with **real**
    pixels instead of interpolated ones. The fish ends up smaller within the
    crop, which is honest: that is genuinely all the evidence the photograph
    contains.
    """
    x0, y0, x1, y1 = box
    cur = max(x1 - x0, y1 - y0)
    if cur >= min_px:
        return box
    target = min(min_px, width, height)
    grow = (target - cur) / 2.0
    nx0, ny0 = x0 - grow, y0 - grow
    nx1, ny1 = x1 + grow, y1 + grow
    # Shift back inside the image rather than clipping, so the box keeps its
    # size when the fish sits near an edge.
    if nx0 < 0:
        nx1 -= nx0
        nx0 = 0
    if ny0 < 0:
        ny1 -= ny0
        ny0 = 0
    if nx1 > width:
        nx0 -= nx1 - width
        nx1 = width
    if ny1 > height:
        ny0 -= ny1 - height
        ny1 = height
    return (max(0, int(nx0)), max(0, int(ny0)),
            min(width, int(nx1)), min(height, int(ny1)))


def letterbox(img: Image.Image, size: int, fill: tuple[int, int, int] = PAD_RGB) -> Image.Image:
    """Resize on the long edge and pad to ``size`` x ``size``. Never stretches."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), fill)
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    if nw == size and nh == size:
        return resized
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def eval_transform_v2(img: Image.Image, size: int) -> Image.Image:
    """Deterministic: letterbox only. No crop, so nothing is ever cut off."""
    return letterbox(img, size)


def train_transform_v2(img, cfg, rng: random.Random):
    """Geometric jitter around an already-cropped fish, then letterbox.

    Deliberately gentler than V1's random-resized-crop. V1 crops 55-100% of a
    whole scene, which is a reasonable way to find the fish when nothing else
    has. Here the fish has already been found, so an aggressive crop mostly
    amputates it - the jitter exists to make the model robust to a sloppy
    detector box, not to search the frame.
    """
    from .data import photometric

    w, h = img.size

    # Scale/translate jitter standing in for detector box error.
    if cfg.scale_max > cfg.scale_min:
        keep = rng.uniform(max(0.6, cfg.scale_min), min(1.0, cfg.scale_max))
        cw, ch = max(8, int(w * keep)), max(8, int(h * keep))
        x0 = rng.randint(0, max(0, w - cw))
        y0 = rng.randint(0, max(0, h - ch))
        img = img.crop((x0, y0, x0 + cw, y0 + ch))

    if cfg.rotation_degrees > 0 and rng.random() < 0.5:
        img = img.rotate(
            rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees),
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=PAD_RGB,
        )

    img = letterbox(img, cfg.size)

    if rng.random() < cfg.hflip_prob:
        from PIL import ImageOps

        img = ImageOps.mirror(img)

    return photometric(img, cfg, rng)
