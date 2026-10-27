"""Views out of the synthetic room, and the camera response that dirties them.

Three things live here, in increasing order of dishonesty:

1. :func:`render_equirect` — the **ground truth**. A perfect, band-limited
   equirectangular panorama traced from a single point. This is the image the stitcher
   is ultimately trying to reproduce, and every dB in the report is measured against it.

2. :func:`render_view` — one video frame. Same tracer, same room, but seen through a
   pinhole camera that is *not* at the panorama's centre. Because the tracer is
   re-entrant from any origin, the parallax between a frame and the reference panorama
   is exact physics rather than a warp bolted on afterwards. Motion blur and rolling
   shutter are integrated the same way a sensor integrates them: by sampling the
   trajectory at several instants and averaging (blur) or by giving each horizontal band
   its own readout instant (rolling shutter).

3. :func:`apply_camera_response` — gain, vignette, shot + read noise, JPEG. In that
   order, because that is the order the physical signal chain applies them, and applying
   noise before gain (say) would produce a noise floor that no real sensor exhibits.

:func:`reproject_to_equirect` is the harness's own deliberately naive un-stitcher. It
does no seam finding, no exposure matching and no blending — it just backward-warps every
frame onto the sphere using a rotation. That makes it useless as a product but perfect as
an instrument: if a nodal, noise-free capture does not reproject onto its own ground
truth to within a fraction of a pixel, then the simulator and
:mod:`biosphere.pano.geometry` disagree about what "equirectangular" means, and nothing
downstream can be trusted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from biosphere.pano import geometry as g
from biosphere.pano.types import CameraPose, Intrinsics

from .room import Room, pixel_angle_equirect, pixel_angle_pinhole

__all__ = [
    "render_equirect",
    "render_view",
    "apply_camera_response",
    "reproject_to_equirect",
    "to_bgr8",
]

_TARGET_RAYS_PER_BLOCK = 1 << 21


def to_bgr8(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB in ``[0, 1+]`` -> clipped uint8 **BGR**, the engine's image currency."""
    out = np.clip(rgb, 0.0, 1.0) * np.float32(255.0)
    return np.ascontiguousarray(np.rint(out).astype(np.uint8)[..., ::-1])


def _resolve_rotation(pose: CameraPose | np.ndarray) -> np.ndarray:
    R = pose.R if isinstance(pose, CameraPose) else pose
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation, got {R.shape}")
    return R


# ======================================================================================
# Ground truth
# ======================================================================================


def render_equirect(
    room: Room,
    width: int,
    height: int,
    origin: np.ndarray | None = None,
    *,
    supersample: int = 1,
    chunk: int = 1 << 20,
) -> np.ndarray:
    """Trace a full equirectangular panorama of ``room`` from ``origin``.

    ``origin`` defaults to the room's eye point, which is also the point every
    :class:`~biosphere.synth.capture.Trajectory` orbits — so the panorama really is the
    image a nodal capture should reconstruct.

    ``supersample`` renders at ``s x s`` and box-averages. The tracer is already
    footprint-filtered, so this is belt-and-braces rather than a necessity; it matters
    when comparing against renders whose own footprint is finer.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width != 2 * height:
        raise ValueError(f"an equirect canvas must be 2:1, got {width}x{height}")
    s = max(int(supersample), 1)
    origin = room.eye if origin is None else np.asarray(origin, dtype=np.float32).reshape(3)

    big_w, big_h = width * s, height * s
    pa = pixel_angle_equirect(height, s)
    out = np.empty((height, width, 3), dtype=np.float32)

    rows = max(s, (_TARGET_RAYS_PER_BLOCK // max(big_w, 1)) or s)
    rows -= rows % s
    rows = max(rows, s)
    for v0 in range(0, big_h, rows):
        h = min(rows, big_h - v0)
        theta, phi = g.equirect_angles(big_w, big_h, v0=v0, h=h)
        rays = g.rays_from_equirect(theta, phi).reshape(-1, 3)
        rgb, _, _ = room.trace(origin, rays, pixel_angle=pa, chunk=chunk)
        block = rgb.reshape(h, big_w, 3)
        if s > 1:
            block = block.reshape(h // s, s, width, s, 3).mean(axis=(1, 3))
        out[v0 // s : v0 // s + h // s] = block
    return to_bgr8(out)


# ======================================================================================
# One video frame
# ======================================================================================


def render_view(
    room: Room,
    pose: CameraPose | np.ndarray,
    intrinsics: Intrinsics,
    *,
    origin: np.ndarray,
    subsamples: int = 1,
    exposure_s: float = 0.0,
    rolling_shutter_s: float = 0.0,
    pose_at: Callable[[float], np.ndarray] | None = None,
    origin_at: Callable[[float], np.ndarray] | None = None,
    t0: float = 0.0,
    supersample: int = 1,
    bands: int = 16,
    chunk: int = 1 << 20,
) -> np.ndarray:
    """Render one pinhole frame of ``room``, optionally blurred and rolling-shuttered.

    ``pose``/``origin`` describe the nominal instant ``t0``. When ``subsamples > 1`` the
    exposure interval ``[t0, t0 + exposure_s)`` is sampled that many times and averaged,
    which is a real box-integration of the motion rather than a post-hoc directional
    blur — so the smear correctly follows the curved path of a hand-held sweep. When
    ``rolling_shutter_s > 0`` the image is rendered in ``bands`` horizontal strips, each
    at its own readout instant, reproducing the shear that makes a fast pan lean.

    ``pose_at(t) -> R`` and ``origin_at(t) -> (3,)`` supply the continuous trajectory;
    without them the pose is held constant and only the static image is produced.
    """
    R0 = _resolve_rotation(pose)
    o0 = np.asarray(origin, dtype=np.float32).reshape(3)
    s = max(int(supersample), 1)
    intr = intrinsics if s == 1 else intrinsics.scaled(float(s))
    w, h = int(intr.width), int(intr.height)
    n_sub = max(int(subsamples), 1)
    n_bands = max(int(bands), 1) if rolling_shutter_s > 0.0 else 1
    n_bands = min(n_bands, h)

    rays_cam = g.camera_rays(w, h, intr.K, intr.k1, intr.k2)
    pa = pixel_angle_pinhole(float(intr.focal))
    acc = np.zeros((h, w, 3), dtype=np.float32)

    edges = np.linspace(0, h, n_bands + 1).round().astype(int)
    for b in range(n_bands):
        r0, r1 = int(edges[b]), int(edges[b + 1])
        if r1 <= r0:
            continue
        rs_t = rolling_shutter_s * ((0.5 * (r0 + r1)) / float(h)) if rolling_shutter_s > 0.0 else 0.0
        band_rays = rays_cam[r0:r1].reshape(-1, 3)
        band = np.zeros((r1 - r0, w, 3), dtype=np.float32)
        for i in range(n_sub):
            frac = (i + 0.5) / n_sub if n_sub > 1 else 0.0
            t = t0 + rs_t + frac * exposure_s
            R_t = _resolve_rotation(pose_at(t)) if pose_at is not None else R0
            o_t = (
                np.asarray(origin_at(t), dtype=np.float32).reshape(3)
                if origin_at is not None
                else o0
            )
            rays_world = band_rays @ R_t.astype(np.float32)  # R.T @ v, vectorised
            rgb, _, _ = room.trace(o_t, rays_world, pixel_angle=pa, chunk=chunk)
            band += rgb.reshape(r1 - r0, w, 3)
        acc[r0:r1] = band / np.float32(n_sub)

    if s > 1:
        acc = acc.reshape(h // s, s, w // s, s, 3).mean(axis=(1, 3))
    return to_bgr8(acc)


# ======================================================================================
# Camera response
# ======================================================================================


def apply_camera_response(
    img: np.ndarray,
    gain: float | Sequence[float] = 1.0,
    vignette_strength: float = 0.0,
    noise_sigma: float = 0.0,
    jpeg_quality: int = 0,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Push a clean render through a phone's signal chain.

    Order is physical and deliberate: **gain -> vignette -> noise -> JPEG**.

    * ``gain`` is the auto-exposure multiplier (scalar, or per-channel for a white
      balance wobble).
    * ``vignette_strength`` is the fractional fall-off at the image corner, applied as
      ``1 - s * r^2`` with ``r`` normalised to the corner radius.
    * ``noise_sigma`` is the read-noise standard deviation in 8-bit counts. Shot noise is
      folded in by scaling it with ``sqrt(signal)``, which is what makes shadows noisier
      than highlights — the single most recognisable signature of a phone sensor.
    * ``jpeg_quality`` in ``1..99`` runs a real ``cv2`` encode/decode round-trip so the
      stitcher sees genuine block artefacts and chroma subsampling. ``0`` skips it.
    """
    x = np.asarray(img).astype(np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) image, got {x.shape}")
    h, w = x.shape[:2]

    gain_arr = np.asarray(gain, dtype=np.float32).reshape(-1)
    if gain_arr.size == 1:
        x = x * gain_arr[0]
    elif gain_arr.size == 3:
        x = x * gain_arr[None, None, :]
    else:
        raise ValueError("gain must be a scalar or a 3-vector")

    if vignette_strength != 0.0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / np.float32(cx * cx + cy * cy + 1e-6)
        x = x * (np.float32(1.0) - np.float32(vignette_strength) * r2)[:, :, None]

    if noise_sigma > 0.0:
        gen = rng if rng is not None else np.random.default_rng(0)
        signal = np.clip(x, 0.0, 255.0)
        sigma = np.float32(noise_sigma) * (
            np.float32(0.40) + np.float32(0.60) * np.sqrt(signal * np.float32(1.0 / 255.0))
        )
        x = x + gen.standard_normal(x.shape, dtype=np.float32) * sigma

    out = np.rint(np.clip(x, 0.0, 255.0)).astype(np.uint8)

    if 0 < int(jpeg_quality) < 100:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed while simulating the JPEG stage")
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


# ======================================================================================
# Reference reprojection (the harness's own instrument)
# ======================================================================================


def _column_spans(u0: int, count: int, width: int) -> list[tuple[int, int, int, int]]:
    """Split a cyclic column range into at most two contiguous ``(dst0, dst1, src0, src1)``."""
    u0 = int(u0) % width
    count = min(int(count), width)
    if u0 + count <= width:
        return [(u0, u0 + count, 0, count)]
    first = width - u0
    return [(u0, width, 0, first), (0, count - first, first, count)]


def reproject_to_equirect(
    frames: Sequence[np.ndarray],
    rotations: Sequence[np.ndarray],
    intrinsics: Intrinsics | Sequence[Intrinsics],
    width: int,
    height: int,
    *,
    interpolation: int = cv2.INTER_CUBIC,
    border_px: int = 3,
    gains: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-warp frames onto an equirect canvas assuming a common optical centre.

    Returns ``(panorama_uint8_bgr, weight)`` where ``weight`` counts how many frames
    contributed to each pixel; overlapping frames are simply averaged. There is no
    blending and no seam finding on purpose — this is a measuring device, not a renderer,
    and any cleverness here would hide exactly the disagreements it exists to expose.

    ``gains`` optionally divides out a known per-frame exposure multiplier, which is how
    the harness checks geometry independently of photometry.
    """
    if len(frames) != len(rotations):
        raise ValueError("frames and rotations must be the same length")
    if width != 2 * height:
        raise ValueError(f"an equirect canvas must be 2:1, got {width}x{height}")
    intr_list = (
        [intrinsics] * len(frames) if isinstance(intrinsics, Intrinsics) else list(intrinsics)
    )
    acc = np.zeros((height, width, 3), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)

    for idx, (frame, R, intr) in enumerate(zip(frames, rotations, intr_list, strict=True)):
        R = _resolve_rotation(R)
        img = np.asarray(frame, dtype=np.float32)
        if gains is not None:
            gv = float(gains[idx])
            if gv > 1e-6:
                img = img / np.float32(gv)

        yaw0, yaw1, pit0, pit1 = g.fov_footprint_deg(intr, R)
        pad = 4
        u0 = int(math.floor((yaw0 + 180.0) / 360.0 * width)) - pad
        u1 = int(math.ceil((yaw1 + 180.0) / 360.0 * width)) + pad
        v0 = max(0, int(math.floor((90.0 - pit1) / 180.0 * height)) - pad)
        v1 = min(height, int(math.ceil((90.0 - pit0) / 180.0 * height)) + pad)
        count = min(u1 - u0, width)
        if count <= 0 or v1 <= v0:
            continue

        theta, phi = g.equirect_angles(width, height, u0=u0, v0=v0, w=count, h=v1 - v0)
        rays = g.rays_from_equirect(theta, phi).reshape(-1, 3)
        cam = rays @ R.T.astype(np.float32)  # R @ d_world
        x, y, valid = g.project_rays(cam, intr.K, intr.k1, intr.k2)
        bh, bw = v1 - v0, count
        x = x.reshape(bh, bw)
        y = y.reshape(bh, bw)
        m = float(border_px)
        inside = (
            valid.reshape(bh, bw)
            & (x >= m)
            & (x <= intr.width - 1 - m)
            & (y >= m)
            & (y <= intr.height - 1 - m)
        )
        if not bool(inside.any()):
            continue
        # `project_rays` works in the "pixel i spans [i, i+1]" convention that
        # `pixel_grid` sets up, so a pixel *centre* comes back as ``i + 0.5``. cv2.remap
        # addresses pixel centres by their integer index instead. Skipping this half-pixel
        # conversion costs about 6 dB and looks exactly like a soft lens.
        mx = np.where(inside, x - np.float32(0.5), np.float32(-1.0))
        my = np.where(inside, y - np.float32(0.5), np.float32(-1.0))
        patch = cv2.remap(
            img, mx, my, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0)
        )
        wt = inside.astype(np.float32)
        patch *= wt[:, :, None]
        for d0, d1, s0, s1 in _column_spans(u0, count, width):
            acc[v0:v1, d0:d1] += patch[:, s0:s1]
            wsum[v0:v1, d0:d1] += wt[:, s0:s1]

    safe = np.maximum(wsum, 1e-6)
    pano = np.clip(acc / safe[:, :, None], 0.0, 255.0)
    pano[wsum <= 0.0] = 0.0
    return np.rint(pano).astype(np.uint8), wsum
