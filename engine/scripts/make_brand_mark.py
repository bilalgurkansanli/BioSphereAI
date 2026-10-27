"""Draw the brand mark the renderer composites onto both poles.

The zenith and nadir are the two directions a three-row sweep never photographs, so
`poles.fill_uncovered` invents them and `render.py` then covers the invention with this
mark. That makes the mark a deliverable, not a watermark: it is the only thing a viewer
sees when they look straight up or straight down, and it should read as *our badge* --
name and logo, plainly legible -- rather than as a pale film over a smear.

Two properties follow from that, and both were wrong in the first placeholder:

* **Opaque.** The disc used to sit at 88 % alpha, so the synthesised ceiling showed
  through it and, on a blown-out white ceiling, the mark all but vanished. Its whole job
  is to *replace* those pixels. Only the outermost pixels fade, and only so the badge has
  no aliased edge.
* **Filled.** The wordmark used to span a third of the disc, leaving the rest empty. The
  cap it lands on is small in angular terms (a 42-degree nadir, 30-degree zenith), so the
  art has to fill the disc to be readable at all from the middle of the room.

The disc stays inside the caps `empty_room` protects from a second synthesis: at 0.90 of
the frame it reaches 19.0 degrees from the nadir and 13.5 from the zenith, against the
20 and 14 those guards allow. RAISING ``DISC_RADIUS_FRAC`` WITHOUT MOVING THOSE GUARDS
would let the furniture remover inpaint over our own logo.

Regenerate with::

    python engine/scripts/make_brand_mark.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
#: Everything is drawn this many times larger and boxed back down. The mark is all thin
#: strokes and curves; PIL does not antialias them, and the pole warp magnifies whatever
#: jaggedness survives.
SUPERSAMPLE = 4

#: Disc radius, as a fraction of the half-frame. See the module docstring before raising.
DISC_RADIUS_FRAC = 0.90
#: How far the alpha ramps in from the disc edge, in output pixels. Enough to kill the
#: staircase, small enough that the badge still has an edge.
FEATHER_PX = 22.0

DISC = (245, 245, 245)
INK = (46, 52, 58)
RULE = (198, 205, 211)

NAME = "BioSphere AI"
TAGLINE = "360° SANAL TUR"

#: Bold grotesques, best first. The mark is a logo, so a fallback that changes its shape is
#: worse than a loud failure -- hence the raise at the end of ``_font``.
_FONT_FILES = (
    "Inter-Bold.ttf",
    "seguisb.ttf",  # Segoe UI Semibold
    "segoeuib.ttf",  # Segoe UI Bold
    "arialbd.ttf",
    "DejaVuSans-Bold.ttf",
)
_FONT_DIRS = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
)


def _font_path() -> Path:
    for name in _FONT_FILES:
        for d in _FONT_DIRS:
            p = d / name
            if p.exists():
                return p
    raise SystemExit(f"no usable font found; looked for {_FONT_FILES} in {_FONT_DIRS}")


def _fit(path: Path, text: str, target_w: float, tracking: float = 0.0) -> ImageFont.FreeTypeFont:
    """The largest size at which ``text`` is ``target_w`` wide, tracking included."""
    lo, hi = 4, 4000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        f = ImageFont.truetype(str(path), mid)
        if _text_w(f, text, tracking * mid) <= target_w:
            lo = mid
        else:
            hi = mid - 1
    return ImageFont.truetype(str(path), lo)


def _text_w(font: ImageFont.FreeTypeFont, text: str, extra: float) -> float:
    w = font.getlength(text)
    return w + extra * max(0, len(text) - 1)


def _draw_tracked(
    d: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    tracking: float,
) -> None:
    """Centred text with letter spacing. PIL has no tracking, so the glyphs are placed one
    at a time; the tagline needs it (a wide-tracked caption reads as a caption) and the
    wordmark does not."""
    total = _text_w(font, text, tracking)
    x = xy[0] - total / 2.0
    for ch in text:
        d.text((x, xy[1]), ch, font=font, fill=fill, anchor="lm")
        x += font.getlength(ch) + tracking


def _globe(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, stroke: float) -> None:
    """The product mark from `web/src/components/dashboard/Logo.tsx`, at any size.

    That component draws in a 24-unit box: a circle of radius 9, a meridian ellipse of
    half-width 4.1, and two parallels 2.6 above and below centre. Those ratios are the
    logo; only the scale changes here.
    """
    w = max(1, int(round(stroke)))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=INK, width=w)
    rx = r * (4.1 / 9.0)
    d.ellipse((cx - rx, cy - r, cx + rx, cy + r), outline=INK, width=w)
    for dy in (-r * (2.6 / 9.0), r * (2.6 / 9.0)):
        # Chord ends, not the SVG's fixed 3.2..20.8: at this size an overshoot past the
        # circle is a visible spur rather than a rounded cap.
        half = math.sqrt(max(0.0, r * r - dy * dy))
        d.line((cx - half, cy + dy, cx + half, cy + dy), fill=INK, width=w)


def build() -> Image.Image:
    s = SIZE * SUPERSAMPLE
    c = s / 2.0
    R = c * DISC_RADIUS_FRAC

    # The disc colour fills the whole frame; alpha, applied after the downscale, is what
    # makes it a disc. Drawing a hard-edged circle here and feathering it later would leave
    # a fringe of frame colour along the ramp.
    img = Image.new("RGB", (s, s), DISC)
    d = ImageDraw.Draw(img)

    ring = R * 0.945
    d.ellipse((c - ring, c - ring, c + ring, c + ring), outline=RULE, width=max(1, int(R * 0.006)))

    _globe(d, c, c - 0.33 * R, 0.36 * R, 0.36 * R * 0.115)

    font_path = _font_path()
    name_font = _fit(font_path, NAME, 1.40 * R)
    _draw_tracked(d, (c, c + 0.34 * R), NAME, name_font, INK, 0.0)

    tag_font = _fit(font_path, TAGLINE, 0.70 * R, tracking=0.22)
    _draw_tracked(d, (c, c + 0.66 * R), TAGLINE, tag_font, INK, 0.22 * tag_font.size)

    img = img.resize((SIZE, SIZE), Image.LANCZOS)

    half = SIZE / 2.0
    r_out = half * DISC_RADIUS_FRAC
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    rad = np.hypot(yy + 0.5 - half, xx + 0.5 - half)
    t = np.clip((r_out - rad) / FEATHER_PX, 0.0, 1.0)
    alpha = (t * t * (3.0 - 2.0 * t) * 255.0).round().astype(np.uint8)

    out = Image.fromarray(np.dstack([np.asarray(img, dtype=np.uint8), alpha]), "RGBA")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "biosphere" / "assets" / "brand_placeholder.png",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build().save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
