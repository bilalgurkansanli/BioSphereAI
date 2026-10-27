"""Delivery: the resolution pyramid, the cube faces and the listing thumbnail.

A viewer cannot stream an 8K JPEG before showing anything, and a listing portal will not
accept one either.  What ships is a pyramid: a 512 px preview that decodes in a blink and
is what the user sees while the rest downloads, then progressively better rungs.  Two
details are easy to get wrong and both are visible:

*Area-downscaling softens.*  ``INTER_AREA`` is the correct box filter for minification --
it is the only one that does not alias -- but a box filter costs perceived sharpness.  A
very mild unsharp mask on the intermediate rungs restores the apparent detail without
introducing halos.  The native rung is never sharpened: it was not resampled, so there is
nothing to compensate for and sharpening it would only amplify blending noise.

*Never upscale.*  A "4K" rung generated from a 2048-wide render is a lie that costs
bandwidth and gains nothing, so a rung is emitted only when it is strictly smaller than
the native width.  That also guarantees the emitted rungs have strictly descending widths
and file sizes, which is what a client-side pyramid loader assumes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .types import PanoLevel, PipelineConfig
from .warp import to_uint8

__all__ = ["LEVEL_WIDTHS", "write_levels", "write_cubemap", "write_thumbnail", "unsharp"]

_LOG = logging.getLogger(__name__)

#: Fixed rungs; ``uhd`` is whatever the render's native width happens to be.
LEVEL_WIDTHS: dict[str, int | None] = {
    "preview": 512,
    "sd": 2048,
    "hd": 4096,
    "uhd": None,
}

#: Rungs that get the compensating unsharp mask (radius 1.0, amount 0.35).
_SHARPEN_LEVELS = ("sd", "hd")


def unsharp(img: np.ndarray, radius: float = 1.0, amount: float = 0.35) -> np.ndarray:
    """Mild unsharp mask, applied in float so the subtraction cannot clip mid-way."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    out = cv2.addWeighted(img.astype(np.float32), 1.0 + amount, blurred.astype(np.float32), -amount, 0.0)
    del blurred
    return to_uint8(out)


def _jpeg_params(quality: int) -> list[int]:
    return [
        int(cv2.IMWRITE_JPEG_QUALITY),
        int(np.clip(quality, 1, 100)),
        int(cv2.IMWRITE_JPEG_PROGRESSIVE),
        1,
        int(cv2.IMWRITE_JPEG_OPTIMIZE),
        1,
    ]


def write_levels(pano: np.ndarray, out_dir: Path, config: PipelineConfig) -> list[PanoLevel]:
    """Write every requested rung as a progressive JPEG; also a WebP for the preview."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    native_w = int(pano.shape[1])
    levels: list[PanoLevel] = []

    for name in ("preview", "sd", "hd", "uhd"):
        if name not in tuple(config.emit_levels):
            continue
        target = LEVEL_WIDTHS[name]
        width = native_w if target is None else int(target)
        if name != "uhd" and width >= native_w:
            _LOG.info("output: skipping %s (%d px would not be a downscale of %d)", name, width, native_w)
            continue
        height = width // 2
        if width == native_w and height == pano.shape[0]:
            img = pano
        else:
            img = cv2.resize(pano, (width, height), interpolation=cv2.INTER_AREA)
        if name in _SHARPEN_LEVELS:
            img = unsharp(img)
        path = out_dir / f"{name}.jpg"
        if not cv2.imwrite(str(path), img, _jpeg_params(config.jpeg_quality)):
            raise OSError(f"could not write {path}")
        levels.append(
            PanoLevel(
                name=name,
                width=width,
                height=height,
                path=path,
                bytes=int(path.stat().st_size),
            )
        )
        if name == "preview":
            webp = out_dir / "preview.webp"
            cv2.imwrite(str(webp), img, [int(cv2.IMWRITE_WEBP_QUALITY), 88])
        if img is not pano:
            del img

    _LOG.info("output: wrote %d levels to %s", len(levels), out_dir)
    return levels


# ======================================================================================
# Cube map
# ======================================================================================

#: three.js ``CubeTexture`` face order.
CUBE_FACES: tuple[str, ...] = ("px", "nx", "py", "ny", "pz", "nz")


def _face_rays(face: str, size: int) -> np.ndarray:
    """Unit rays for one cube face, in **this engine's** world frame.

    Convention, stated precisely because it is the single most common source of a
    mirrored or upside-down sky:

    * The faces follow the OpenGL / WebGL cube-map table, which is what three.js'
      ``CubeTexture`` uploads verbatim.  With ``S = 2*(col+0.5)/N - 1`` and
      ``T = 2*(row+0.5)/N - 1`` (so ``T`` grows *downward* in the stored image), the
      direction in three.js' own coordinate frame is
      ``px:(1,-T,-S)  nx:(-1,-T,S)  py:(S,1,T)  ny:(S,-1,-T)  pz:(S,-T,1)  nz:(-S,-T,-1)``.
    * three.js is Y-up and its default camera looks down -Z; this engine is Y-**down** and
      the panorama's ``theta = 0`` meridian looks along +Z.  The two frames are therefore
      related by a 180-degree rotation about X: ``(x, y, z)_engine = (X, -Y, -Z)_three``.
      That maps the panorama's forward direction onto the direction a freshly created
      three.js camera is already looking, so a scene using these faces as its background
      opens facing the same way the operator started the sweep.
    """
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    S = np.broadcast_to(t[None, :], (size, size))
    T = np.broadcast_to(t[:, None], (size, size))
    one = np.ones((size, size), dtype=np.float32)
    if face == "px":
        d = np.stack([one, -T, -S], axis=-1)
    elif face == "nx":
        d = np.stack([-one, -T, S], axis=-1)
    elif face == "py":
        d = np.stack([S, one, T], axis=-1)
    elif face == "ny":
        d = np.stack([S, -one, -T], axis=-1)
    elif face == "pz":
        d = np.stack([S, -T, one], axis=-1)
    elif face == "nz":
        d = np.stack([-S, -T, -one], axis=-1)
    else:
        raise ValueError(f"unknown cube face {face!r}")
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    # three.js frame -> engine frame (Y and Z flipped; a proper rotation, det = +1).
    return np.stack([d[..., 0], -d[..., 1], -d[..., 2]], axis=-1).astype(np.float32)


def write_cubemap(pano: np.ndarray, out_dir: Path, face_size: int) -> dict[str, Path]:
    """Render and write the six cube faces.  Returns ``{face_name: path}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = pano.shape[:2]
    faces: dict[str, Path] = {}
    for face in CUBE_FACES:
        rays = _face_rays(face, int(face_size))
        u, v = g.equirect_from_rays(rays, w, h)
        del rays
        img = g.sample_equirect(pano, u, v, interpolation=cv2.INTER_LANCZOS4)
        del u, v
        path = out_dir / f"{face}.jpg"
        if not cv2.imwrite(str(path), img, _jpeg_params(92)):
            raise OSError(f"could not write {path}")
        faces[face] = path
        del img
    _LOG.info("output: wrote %d cube faces at %d px", len(faces), face_size)
    return faces


# ======================================================================================
# Thumbnail
# ======================================================================================


def write_thumbnail(
    pano: np.ndarray, path: Path, *, size: tuple[int, int] = (640, 360), yaw_deg: float = 90.0
) -> np.ndarray:
    """A 640x360 crop of the horizon band -- what a listing portal actually shows.

    Cropping rather than squashing the whole equirect matters: a 2:1 panorama letterboxed
    into 16:9 reads as a distorted smear, whereas a horizon crop looks like a photograph.
    """
    h, w = pano.shape[:2]
    out_w, out_h = size
    crop_w = max(16, int(round(w * yaw_deg / 360.0)))
    crop_h = max(16, int(round(crop_w * out_h / out_w)))
    crop_h = min(crop_h, h)
    x0 = (w - crop_w) // 2
    y0 = max(0, (h - crop_h) // 2)
    crop = pano[y0 : y0 + crop_h, x0 : x0 + crop_w]
    thumb = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    thumb = unsharp(thumb, 1.0, 0.30)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), thumb, _jpeg_params(90)):
        raise OSError(f"could not write thumbnail to {path}")
    return thumb
