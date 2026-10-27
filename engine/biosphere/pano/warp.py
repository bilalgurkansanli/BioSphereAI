"""Projection of solved cameras onto the equirectangular canvas.

Warping is where a *rotation* becomes *pixels*, and it is the one stage whose errors
cannot be recovered later: a half-pixel bias here shows up as a doubled edge after
blending, and a mask that is one pixel too generous drags Lanczos ringing from the frame
border straight into the seam.  Three decisions therefore deserve their justification:

*Inverse mapping, never forward.*  We iterate over the *destination* window and ask which
source pixel feeds it.  Forward-splatting a source image onto the sphere leaves holes
wherever the equirect stretches (i.e. everywhere above ~45 deg elevation).

*Bounding boxes, never full canvases.*  A 70 deg phone frame covers under 4 % of an 8K
canvas.  Allocating a full 8192x4096 buffer per camera would cost 400 MB each; we compute
the angular footprint and touch only those rows and columns.  The two exceptions are a
footprint that reaches a pole (where every meridian converges, so the whole width is
needed) and a footprint that straddles the cyclic seam.  The latter is deliberately rare:
:func:`biosphere.pano.render.choose_wrap_offset` rotates the world so that *exactly one*
camera straddles, and that camera is the one whose continuity makes the wrap invisible.

*Photometry in float, before quantisation.*  Gain and vignette corrections are applied to
the resampled float patch and only then rounded to uint8, so a 1.35x gain does not
amplify quantisation steps into visible banding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .types import CameraPose, Frame, Intrinsics, PipelineConfig, ProgressReporter

__all__ = [
    "WarpedPatch",
    "Overlap",
    "INTERPOLATION",
    "interpolation_flag",
    "to_uint8",
    "load_frame_image",
    "warp_camera",
    "warp_all",
    "patch_overlap",
    "iter_overlaps",
]

INTERPOLATION: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}

#: How far the resampling kernel reaches, in source pixels.  The source is padded by this
#: much with BORDER_REPLICATE so that BORDER_CONSTANT black never bleeds into valid pixels.
_KERNEL_PAD: dict[int, int] = {
    cv2.INTER_NEAREST: 1,
    cv2.INTER_LINEAR: 2,
    cv2.INTER_CUBIC: 3,
    cv2.INTER_LANCZOS4: 5,
}

#: Rows of the destination window built per chunk; caps the transient ray buffer.
_CHUNK_PIXELS = 2_000_000


def interpolation_flag(name: str) -> int:
    """Map a :class:`PipelineConfig.interpolation` string onto an OpenCV flag."""
    return INTERPOLATION.get(str(name).lower(), cv2.INTER_LANCZOS4)


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Quantise a float image to uint8 by *rounding*, never by truncating.

    ``ndarray.astype(np.uint8)`` truncates toward zero, which is not a rounding error you
    can wave away as half a grey level.  Apply a correction that is a hair below 1.0 --
    exactly what a well-fitted vignette model is over most of the frame -- and every pixel
    whose float value sits just under an integer drops a whole level, while every pixel
    where the correction is a hair above 1.0 does not.  The result is a hard one-level
    contour wherever the correction crosses unity: a ring burned into every frame, which on
    a smooth wall or an overcast sky is exactly the banding a viewer notices first.
    """
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ======================================================================================
# Patch container
# ======================================================================================


@dataclass(slots=True)
class WarpedPatch:
    """One source frame resampled onto an axis-aligned window of the equirect canvas.

    ``corner`` is the ``(x, y)`` canvas position of ``image[0, 0]``.  The window is always
    fully inside ``[0, canvas_w) x [0, canvas_h)``; seam-straddling and pole-touching
    footprints are represented as full-canvas-width windows with a sparse mask rather than
    as two pieces, which keeps every downstream stage working on plain rectangles.
    """

    index: int
    frame_index: int
    image: np.ndarray  # (h, w, 3) uint8 BGR
    mask: np.ndarray  # (h, w) uint8, 255 = this camera contributed here
    corner: tuple[int, int]
    canvas_w: int
    canvas_h: int
    pose: CameraPose

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` canvas rectangle, half-open."""
        x0, y0 = self.corner
        return x0, y0, x0 + self.width, y0 + self.height

    def nbytes(self) -> int:
        return int(self.image.nbytes + self.mask.nbytes)


@dataclass(slots=True)
class Overlap:
    """The shared canvas rectangle of two patches, plus the slices that address it."""

    i: int
    j: int
    rect: tuple[int, int, int, int]  # canvas x0, y0, x1, y1
    si: tuple[slice, slice]
    sj: tuple[slice, slice]

    @property
    def shape(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.rect
        return y1 - y0, x1 - x0


def patch_overlap(a: WarpedPatch, b: WarpedPatch) -> Overlap | None:
    """Rectangular intersection of two patches, or ``None`` when they are disjoint."""
    ax0, ay0, ax1, ay1 = a.rect
    bx0, by0, bx1, by1 = b.rect
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Overlap(
        i=a.index,
        j=b.index,
        rect=(x0, y0, x1, y1),
        si=(slice(y0 - ay0, y1 - ay0), slice(x0 - ax0, x1 - ax0)),
        sj=(slice(y0 - by0, y1 - by0), slice(x0 - bx0, x1 - bx0)),
    )


def iter_overlaps(patches: list[WarpedPatch], *, min_pixels: int = 256, only: int | None = None):
    """Yield every ``(a, b, Overlap)`` whose masks actually share ``min_pixels`` pixels.

    ``only`` restricts the pairs to those involving that patch index; the order of the pairs
    that remain is unchanged.
    """
    for ia in range(len(patches)):
        for ib in range(ia + 1, len(patches)):
            if only is not None and only not in (ia, ib):
                continue
            a, b = patches[ia], patches[ib]
            ov = patch_overlap(a, b)
            if ov is None:
                continue
            both = (a.mask[ov.si] > 0) & (b.mask[ov.sj] > 0)
            if int(both.sum()) < min_pixels:
                del both
                continue
            yield ia, ib, ov, both


# ======================================================================================
# Source images
# ======================================================================================


def load_frame_image(frame: Frame, scale: float = 1.0) -> np.ndarray:
    """Read a frame's pixels, optionally area-downscaled by ``scale``.

    An in-memory array parked on ``Frame._cache['image']`` wins over the file, which lets
    the synthetic harness (and any caller that already holds the decoded frame) avoid a
    pointless round trip through the disk.
    """
    cached = frame._cache.get("image") if frame._cache is not None else None
    if cached is not None:
        img = np.asarray(cached)
    else:
        path = Path(frame.path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read frame image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if scale < 0.999:
        w = max(16, int(round(img.shape[1] * scale)))
        h = max(16, int(round(img.shape[0] * scale)))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _match_intrinsics(intr: Intrinsics, image: np.ndarray) -> Intrinsics:
    """Rescale intrinsics to whatever resolution the caller actually handed us."""
    ih, iw = image.shape[:2]
    if iw == intr.width and ih == intr.height:
        return intr
    if intr.width <= 0:
        raise ValueError("intrinsics have zero width")
    return intr.scaled(iw / float(intr.width))


# ======================================================================================
# Footprint -> canvas window
# ======================================================================================


def _pole_inside(pose: CameraPose, intr: Intrinsics, pad_fraction: float) -> tuple[bool, bool]:
    """``(zenith_inside, nadir_inside)``: does the pole's direction land inside the image?

    World +Y points down, so the zenith is ``(0, -1, 0)``. The test is done in normalised
    image coordinates against the half-widths of the frame, widened by ``pad_fraction`` so
    a pole sitting exactly on the border counts as inside rather than as a hairline miss.
    """
    R = np.asarray(pose.R, dtype=np.float64)
    tan_h = (intr.width / 2.0) / float(intr.focal) * (1.0 + 2.0 * pad_fraction)
    tan_v = (intr.height / 2.0) / float(intr.focal) * (1.0 + 2.0 * pad_fraction)
    out: list[bool] = []
    for d in ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)):
        cam = R @ np.asarray(d)
        if cam[2] <= 1e-9:
            out.append(False)
            continue
        out.append(bool(abs(cam[0] / cam[2]) <= tan_h and abs(cam[1] / cam[2]) <= tan_v))
    return out[0], out[1]


def footprint_window(
    pose: CameraPose, canvas_w: int, canvas_h: int, *, pad_fraction: float = 0.01
) -> tuple[int, int, int, int]:
    """``(u0, v0, w, h)`` canvas window that contains this camera's angular footprint.

    The window degenerates to the full canvas width when the footprint straddles the
    cyclic seam or reaches within half a vertical FOV of a pole -- in both cases a
    rectangle in ``(yaw, pitch)`` is simply not a rectangle in ``(u, v)``.
    """
    intr = pose.intrinsics
    y0, y1, p0, p1 = g.fov_footprint_deg(intr, pose.R)
    yspan = max(y1 - y0, 1e-6)
    pspan = max(p1 - p0, 1e-6)
    y0 -= pad_fraction * yspan
    y1 += pad_fraction * yspan
    p0 -= pad_fraction * pspan
    p1 += pad_fraction * pspan

    # A pole is inside the frame only if its direction projects inside the image. This used
    # to be guessed from "within half a vertical field of view of the pole", which fires 20
    # degrees early: a down-row view pitched to -40 with a 59-degree vertical field reaches
    # -70 and was handed the whole 8192-wide canvas as its window with 12.5 % of it valid --
    # five times the warp cost, a third more blend work, and a seam-finder graph so large the
    # graph-cut was refused and the seams fell back to dynamic programming at 512 px.
    near_zenith, near_nadir = _pole_inside(pose, intr, pad_fraction)

    u_lo = (math.radians(y0) + math.pi) / (2.0 * math.pi) * canvas_w - 0.5
    u_hi = (math.radians(y1) + math.pi) / (2.0 * math.pi) * canvas_w - 0.5
    u0 = int(math.floor(u_lo))
    u1 = int(math.ceil(u_hi)) + 1
    if near_zenith or near_nadir or u0 < 0 or u1 > canvas_w or (u1 - u0) >= canvas_w:
        u0, u1 = 0, canvas_w

    v_lo = (0.5 * math.pi - math.radians(p1)) / math.pi * canvas_h - 0.5
    v_hi = (0.5 * math.pi - math.radians(p0)) / math.pi * canvas_h - 0.5
    v0 = int(math.floor(v_lo))
    v1 = int(math.ceil(v_hi)) + 1
    if near_zenith:
        v0 = 0
    if near_nadir:
        v1 = canvas_h
    v0 = max(0, min(canvas_h - 1, v0))
    v1 = max(v0 + 1, min(canvas_h, v1))
    return u0, v0, u1 - u0, v1 - v0


# ======================================================================================
# The warp itself
# ======================================================================================


#: A view's local alignment: ``(grid, cell, (gu0, gv0))`` -- a ``(gh, gw, 2)`` float32 grid
#: of canvas-pixel displacements whose vertex ``(r, c)`` sits at canvas index coordinates
#: ``(gu0 + c * cell, gv0 + r * cell)``.  See :mod:`biosphere.pano.mesh`.
Displacement = tuple[np.ndarray, float, tuple[int, int]]


def _displaced_rays(
    displacement: Displacement,
    canvas_w: int,
    canvas_h: int,
    u0: int,
    v0: int,
    w: int,
    h: int,
) -> np.ndarray:
    """World rays for a canvas window whose content has been moved by a mesh displacement.

    The warp is an inverse map: destination pixel ``(u, v)`` shows the source at ``(u - du,
    v - dv)``, so the content itself moves by ``+d``.  The grid is evaluated bilinearly at the
    destination pixel; a grid that spans the whole canvas width is treated cyclically, which
    is what lets the seam-straddling view be re-rendered rolled by half a turn.
    """
    grid, cell, (gu0, gv0) = displacement
    gh, gw = grid.shape[:2]
    xs = np.arange(u0, u0 + w, dtype=np.float32)
    ys = np.arange(v0, v0 + h, dtype=np.float32)
    gx = xs - float(gu0)
    if (gw - 1) * cell >= canvas_w - 1e-6:
        gx = np.mod(gx, float(canvas_w))
    gx = (gx / float(cell)).astype(np.float32)
    gy = ((ys - float(gv0)) / float(cell)).astype(np.float32)
    mx = np.ascontiguousarray(np.broadcast_to(gx[None, :], (h, w)))
    my = np.ascontiguousarray(np.broadcast_to(gy[:, None], (h, w)))
    disp = cv2.remap(grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    del mx, my
    uu = xs[None, :] - disp[..., 0]
    vv = ys[:, None] - disp[..., 1]
    del disp
    theta = (2.0 * np.pi * (uu + 0.5) / float(canvas_w) - np.pi).astype(np.float32)
    phi = (0.5 * np.pi - np.pi * (vv + 0.5) / float(canvas_h)).astype(np.float32)
    del uu, vv
    rays = np.empty((h, w, 3), dtype=np.float32)
    cp = np.cos(phi)
    rays[..., 0] = cp * np.sin(theta)
    rays[..., 1] = -np.sin(phi)
    rays[..., 2] = cp * np.cos(theta)
    return rays


def warp_camera(
    image: np.ndarray,
    pose: CameraPose,
    canvas_w: int,
    canvas_h: int,
    config: PipelineConfig,
    *,
    index: int = 0,
    frame_index: int | None = None,
    displacement: Displacement | None = None,
) -> WarpedPatch:
    """Resample one source image onto its equirect footprint.

    ``image`` may already be downscaled; the intrinsics are rescaled to match so the
    caller never has to keep two copies of the calibration in sync.

    ``displacement`` is the view's local mesh alignment (see :mod:`biosphere.pano.mesh`);
    the footprint window is widened by the largest displacement so moved content is not
    clipped.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")
    intr = _match_intrinsics(pose.intrinsics, image)
    ih, iw = image.shape[:2]

    u0, v0, w, h = footprint_window(pose, canvas_w, canvas_h)
    if displacement is not None and displacement[0].size:
        reach = int(math.ceil(float(np.abs(displacement[0]).max()))) + 2
        if w < canvas_w:
            nu0 = max(0, u0 - reach)
            nu1 = min(canvas_w, u0 + w + reach)
            u0, w = nu0, nu1 - nu0
        nv0 = max(0, v0 - reach)
        nv1 = min(canvas_h, v0 + h + reach)
        v0, h = nv0, nv1 - nv0
    interp = interpolation_flag(config.interpolation)
    pad = _KERNEL_PAD.get(interp, 5)

    mapx = np.empty((h, w), dtype=np.float32)
    mapy = np.empty((h, w), dtype=np.float32)
    valid = np.empty((h, w), dtype=bool)

    K = intr.K
    k1, k2 = float(intr.k1), float(intr.k2)
    Rt = np.ascontiguousarray(np.asarray(pose.R, dtype=np.float32).T)

    rows = max(1, min(h, _CHUNK_PIXELS // max(1, w)))
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        if displacement is not None and displacement[0].size:
            rays = _displaced_rays(displacement, canvas_w, canvas_h, u0, v0 + r0, w, r1 - r0)
        else:
            theta, phi = g.equirect_angles(canvas_w, canvas_h, u0=u0, v0=v0 + r0, w=w, h=r1 - r0)
            rays = g.rays_from_equirect(theta, phi)  # (rows, w, 3) world directions
        # p_cam = R @ d_world; for row vectors that is d_world @ R.T.
        cam = (rays.reshape(-1, 3) @ Rt).reshape(rays.shape)
        del rays
        # project_rays speaks *continuous* pixel coordinates (centre of pixel i at i+0.5,
        # matching Intrinsics' cx = width/2); the validity test below is evaluated in that
        # same space.  cv2.remap addresses pixel i by the number i, so the maps -- and only
        # the maps -- are converted.  Skipping that conversion shifts every sample half a
        # pixel: nothing misaligns, but every tap lands at the worst interpolation phase
        # and the whole panorama comes out soft.
        x, y, ok = g.project_rays(cam, K, k1, k2)
        del cam
        inside = ok & (x >= 0.5) & (x <= iw - 1.5) & (y >= 0.5) & (y <= ih - 1.5)
        mx, my = g.to_remap_coords(x, y)
        del x, y
        np.copyto(mapx[r0:r1], np.where(inside, mx + pad, -1.0).astype(np.float32))
        np.copyto(mapy[r0:r1], np.where(inside, my + pad, -1.0).astype(np.float32))
        valid[r0:r1] = inside
        del mx, my, ok, inside

    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    patch = cv2.remap(padded, mapx, mapy, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    del padded

    rendered = np.where(valid, np.uint8(255), np.uint8(0))
    del valid
    # Two pixels of erosion keep the resampling kernel's border ringing out of the blend.
    mask = cv2.erode(rendered, np.ones((5, 5), np.uint8), iterations=1)

    _apply_photometry(patch, mapx, mapy, intr, pose, pad)
    del mapx, mapy
    extend_border(patch, rendered, _border_reach(config))
    del rendered

    return WarpedPatch(
        index=index,
        frame_index=int(pose.frame_index if frame_index is None else frame_index),
        image=patch,
        mask=mask,
        corner=(u0, v0),
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        pose=pose,
    )


def _border_reach(config: PipelineConfig) -> int:
    """How far real colour is grown past the footprint, in canvas pixels.

    Matched to the coarsest blend band's reach (``2^bands``), with the same factor of four
    the tiled blender uses for its strip overlap.
    """
    bands = int(np.clip(config.blend_bands, 1, 8))
    return int(np.clip(4 * (1 << bands), 16, 192))


def extend_border(image: np.ndarray, rendered: np.ndarray, reach: int) -> None:
    """Grow each patch's colour outward into the margin it did not render, in place.

    ``cv2.detail_MultiBandBlender`` builds its Laplacian pyramid from the *image*; the mask
    only weights the accumulation.  Everything outside a camera's footprint is black, and
    at coarse pyramid levels that black is averaged into pixels that are inside it.  In the
    interior of the panorama a neighbouring camera covers the same ground and the effect
    cancels, but along the *outer* rim of the whole capture there is no neighbour -- so the
    top and bottom few degrees of a single-row sweep come out measurably dimmed, a dark
    fringe exactly where the eye travels from captured wall to synthesised ceiling.

    Replacing the black with the nearest rendered colour removes the gradient the pyramid
    was reacting to.  It cannot leak into the result on its own: these pixels have zero
    blend weight, so they only ever influence the low-frequency bands, which is precisely
    the influence we are trying to make benign.  A single labelled distance transform gives
    every margin pixel its nearest real neighbour in one pass.
    """
    if reach <= 0:
        return
    outside = rendered == 0
    if not outside.any() or not (rendered > 0).any():
        return
    src = np.where(rendered > 0, np.uint8(0), np.uint8(255))
    dist, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    del src
    fill = outside & (dist <= float(reach))
    del dist, outside
    if not fill.any():
        del fill, labels
        return
    known = rendered > 0
    lookup = np.zeros((int(labels.max()) + 1, image.shape[2]), dtype=image.dtype)
    lookup[labels[known]] = image[known]
    image[fill] = lookup[labels[fill]]
    del lookup, labels, fill, known


def _apply_photometry(
    patch: np.ndarray,
    mapx: np.ndarray,
    mapy: np.ndarray,
    intr: Intrinsics,
    pose: CameraPose,
    pad: int,
) -> None:
    """In-place gain + vignette correction, evaluated in float32 and re-quantised once.

    Deliberately touches every pixel of the patch, including the two-pixel guard band the
    mask erosion excluded from the blend.  Those pixels hold real resampled colour and the
    multi-band blender builds its Laplacian pyramid from the image itself -- the mask only
    weights the accumulation -- so blacking them out would put a hard edge exactly where
    every seam runs and let the pyramid smear it back inward as a dark halo.  Uncovered
    canvas pixels are zeroed once, at the end of blending, where it is safe.
    """
    gain = np.asarray(pose.gain, dtype=np.float32).reshape(-1)
    if gain.size == 1:
        gain = np.repeat(gain, 3)
    a1, a2, a3 = (float(v) for v in pose.vignette)
    flat_gain = bool(np.allclose(gain, 1.0, atol=1e-4))
    flat_vign = a1 == 0.0 and a2 == 0.0 and a3 == 0.0
    if flat_gain and flat_vign:
        return

    cx, cy = float(intr.cx), float(intr.cy)
    rn2 = float(intr.width * intr.width + intr.height * intr.height) * 0.25
    h = patch.shape[0]
    rows = max(1, min(h, _CHUNK_PIXELS // max(1, patch.shape[1])))
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        block = patch[r0:r1].astype(np.float32)
        if not flat_gain:
            block *= gain[None, None, :]
        if not flat_vign:
            # The maps hold remap indices; the vignette model is radial in the *continuous*
            # image frame where the principal point lives, so convert back before measuring.
            sx, sy = g.from_remap_coords(mapx[r0:r1] - pad, mapy[r0:r1] - pad)
            dx = sx - cx
            dy = sy - cy
            del sx, sy
            r2 = (dx * dx + dy * dy) / rn2
            del dx, dy
            v = 1.0 + r2 * (a1 + r2 * (a2 + r2 * a3))
            del r2
            np.clip(v, 0.35, 3.0, out=v)
            block /= v[..., None]
            del v
        patch[r0:r1] = to_uint8(block)
        del block


def warp_all(
    frames: list[Frame],
    poses: list[CameraPose],
    canvas_w: int,
    canvas_h: int,
    config: PipelineConfig,
    *,
    source_scale: float = 1.0,
    progress: ProgressReporter | None = None,
    stage: str = "warp",
    stage_span: tuple[float, float] = (0.0, 1.0),
    displacements: list[Displacement | None] | None = None,
) -> list[WarpedPatch]:
    """Warp every pose in turn, holding at most one full-resolution source frame at a time.

    Sequential on purpose: the per-camera cost is dominated by ``cv2.remap``, which is
    already internally threaded, and a process pool would have to ship whole patches back
    through a pipe.  Releasing each source image before loading the next is what keeps a
    90-keyframe 8K render inside a 16 GB machine.

    ``displacements`` holds one mesh alignment per pose (or ``None``), in pose order.
    """
    by_index = {f.index: f for f in frames}
    patches: list[WarpedPatch] = []
    n = len(poses)
    lo, hi = stage_span
    for i, pose in enumerate(poses):
        frame = by_index.get(pose.frame_index)
        if frame is None:
            if i >= len(frames):
                raise KeyError(f"no frame for pose {i} (frame_index={pose.frame_index})")
            frame = frames[i]
        image = load_frame_image(frame, source_scale)
        disp = displacements[i] if displacements is not None and i < len(displacements) else None
        patch = warp_camera(
            image, pose, canvas_w, canvas_h, config, index=i, frame_index=frame.index,
            displacement=disp,
        )
        del image
        patches.append(patch)
        if progress is not None:
            progress.stage(stage, lo + (hi - lo) * (i + 1) / max(1, n), f"warped {i + 1}/{n}")
    return patches
