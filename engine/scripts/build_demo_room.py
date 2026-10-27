"""Render the landing page's demo room — a synthetic one, on purpose.

The page shows a room the engine itself rendered: the procedural room of the synthetic
quality gate (``biosphere.synth``), the same ground truth every stitch is scored against. It
used to show a real captured bedroom, which put somebody's home, belongings and screen into a
public repository; a rendered room shows the player just as well and belongs to nobody.

Two outputs:

*The equirect texture* the sphere is built from. 3072x1536 is the smallest size that still
looks sharp at a 75 degree field of view on a 600 px orb — the visible slice is about a
fifth of the width, so 3072 puts roughly 640 real pixels across the viewport and 2048 puts
420, which reads soft on a retina screen.

*A perspective poster*, which is what the page paints before any JavaScript runs. It has to
be a real perspective view rather than a crop of the equirect: a raw equirect in a circular
mask looks like a smeared strip and would undo the very impression the hero is there to make.

Usage:  python scripts/build_demo_room.py [--seed 7] [--style rich]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "web" / "public" / "demo"
sys.path.insert(0, str(REPO / "engine"))

#: Seed 7 was chosen by looking at seeds 3, 7, 11 and 21: the window sits mid-frame, the
#: wardrobe and bed read as furniture at a glance, and the door is in view from the start.
DEFAULT_SEED = 7

TEXTURE_WIDTH = 3072
POSTER_SIZE = 1400
POSTER_FOV_DEG = 76.0
#: Where the poster looks, and where the sphere starts: the web component starts its camera
#: at the same angle, or the poster would be replaced by a different view the moment WebGL
#: takes over.
POSTER_YAW_DEG = 50.0
POSTER_PITCH_DEG = -6.0


def perspective_from_equirect(
    equirect: np.ndarray, *, size: int, fov_deg: float, yaw_deg: float, pitch_deg: float
) -> np.ndarray:
    """One rectilinear view out of the sphere, by the project's own convention.

    Rays are built in camera space, rotated into world space by yaw then pitch, and read back
    as spherical coordinates: ``theta`` across the width, ``phi`` down from the zenith. The
    sampling is ``cv2.remap`` with ``BORDER_WRAP`` so a view that straddles the seam does not
    show it.
    """
    height, width = equirect.shape[:2]
    focal = (size / 2) / np.tan(np.radians(fov_deg) / 2)

    xs, ys = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    x = (xs - size / 2) / focal
    y = (ys - size / 2) / focal
    z = np.ones_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    pitch = np.radians(pitch_deg)
    y, z = y * np.cos(pitch) - z * np.sin(pitch), y * np.sin(pitch) + z * np.cos(pitch)
    yaw = np.radians(yaw_deg)
    x, z = x * np.cos(yaw) + z * np.sin(yaw), -x * np.sin(yaw) + z * np.cos(yaw)

    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1.0, 1.0))
    map_x = ((theta / (2 * np.pi)) + 0.5) * width - 0.5
    map_y = ((phi / np.pi) + 0.5) * height - 0.5
    return cv2.remap(
        equirect,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )


def _encode_webp(image: np.ndarray, path: Path, quality: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"could not write {path}")
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--style", default="rich", help="rich | low_texture | specular")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--quality", type=int, default=82)
    args = parser.parse_args()

    from biosphere.synth.render import render_equirect
    from biosphere.synth.room import Room

    room = Room.random(args.seed, args.style)
    image = render_equirect(room, TEXTURE_WIDTH, TEXTURE_WIDTH // 2, supersample=args.supersample)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    total = _encode_webp(image, OUT / "room.webp", args.quality)
    print(f"  room         {TEXTURE_WIDTH}x{TEXTURE_WIDTH // 2}  {total / 1024:6.0f} KB")
    poster = perspective_from_equirect(
        image, size=POSTER_SIZE, fov_deg=POSTER_FOV_DEG, yaw_deg=POSTER_YAW_DEG, pitch_deg=POSTER_PITCH_DEG
    )
    size = _encode_webp(poster, OUT / "poster.webp", args.quality)
    total += size
    print(f"  poster       {POSTER_SIZE}x{POSTER_SIZE}  {size / 1024:6.0f} KB")
    print(f"\n{len(list(OUT.iterdir()))} files, {total / 1024:.0f} KB total, in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
