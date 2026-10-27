"""Orchestration of the rendering half of the pipeline, and the cure for the wrap seam.

Everything downstream of the solver lives here: photometric alignment, warping, seam
selection, blending, optional parallax refinement, pole filling and delivery.  The stages
themselves are in their own modules; what this file owns is the *order* they run in, the
memory discipline between them, and one idea that is worth the whole file.

The wrap problem
----------------
An equirectangular canvas is a cylinder: column 0 and column ``W-1`` are neighbours on the
sphere.  Every OpenCV seam finder and every pyramid blender, however, assumes a flat
rectangle with edges.  Blend across that edge and you get a visible vertical line straight
through the middle of the viewer's starting view -- the single most damning artefact this
product can ship.

Rather than teach the blenders about cyclic topology, we move the problem somewhere it
cannot hurt.  :func:`choose_wrap_offset` finds the yaw that is covered by as few cameras as
possible -- ideally exactly one -- and as deep inside that camera's footprint as possible,
then the whole world is rotated so the canvas edge lands there.  The pixels on either side
of the edge then come from *the same source image*, so they are continuous by construction:
there is no cross-image transition to blend at all.

That leaves only the blenders' own edge behaviour (a pyramid has to invent something
outside its ROI), which :func:`repair_wrap_band` removes by re-rendering the seam-owning
camera onto a canvas rolled by half a turn -- where the seam is now in the middle of a
patch and nothing special happens -- and merging a band of it back across ``u = 0``,
handed over from the panorama along the path where the two agree best (a cross-fade drew
parallax twice; see :func:`_merge_wrap_band`).  Its brightness is matched to the panorama
on both sides first, so the repair cannot itself introduce a step.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import blend as blend_mod
from . import coverage as coverage_mod
from . import exposure as exposure_mod
from . import geometry as g
from . import mesh as mesh_mod
from . import output as output_mod
from . import poles as poles_mod
from . import refine as refine_mod
from . import seams as seams_mod
from . import warp as warp_mod
from .types import (
    CameraPose,
    Frame,
    Intrinsics,
    MatchPair,
    NullProgress,
    PipelineConfig,
    ProgressReporter,
    StitchMetrics,
    StitchResult,
)

__all__ = [
    "render_panorama",
    "choose_wrap_offset",
    "apply_wrap_offset",
    "repair_wrap_band",
    "seam_gradient_energy",
    "STAGES",
]

_LOG = logging.getLogger(__name__)

STAGES: tuple[str, ...] = ("exposure", "warp", "seam", "blend", "refine", "poles", "levels")

#: Narrowest half-width of the band re-rendered across the canvas edge; below this the repair
#: is skipped rather than squeezed.
_WRAP_BAND = 12
#: How far either side of the canvas edge, in degrees of yaw, the wrap repair may look for
#: the place to hand over from the panorama to the re-rendered owner (see
#: `_merge_wrap_band`). It has to be wider than the parallax it routes around: on the
#: reference capture the two sides of the edge disagreed by 20-25 px at 4096 wide (a close,
#: glossy wardrobe), about 2 degrees, and a 12 px cross-fade drew both copies.
_WRAP_SEARCH_DEG = 4.0
#: Cost of the hand-over path moving one column between rows, in grey levels: enough to
#: keep it from zig-zagging through flat wall, small next to any real disagreement.
_WRAP_PATH_STEP_COST = 1.0
#: Canvas width used for the reduced-scale pass that solves exposure.
_EXPOSURE_CANVAS_W = 1024


# ======================================================================================
# Wrap offset
# ======================================================================================


def choose_wrap_offset(
    poses: list[CameraPose],
    intr: Intrinsics | None = None,
    *,
    step_deg: float = 0.5,
    band_deg: float = 4.0,
) -> float:
    """Yaw (radians) at which the canvas edge should be placed.

    Scans candidate yaws and, for each, counts how many camera footprints fully contain a
    ``+/- band_deg`` band around it and how deep inside its own footprint the best of them
    sits.  A yaw covered by exactly one camera wins, because then both sides of the seam
    are the same image; among those the deepest margin wins, because a seam near a
    footprint's edge is a seam near the frame border where the lens is softest and the
    validity mask is about to run out.  Failing that -- which is the normal case for a
    dense sweep, where every direction is seen by four or five frames -- the yaw with the
    fewest covering cameras and the largest margin is used.
    """
    if not poses:
        return 0.0
    foots: list[tuple[float, float]] = []
    for p in poses:
        i = intr if intr is not None else p.intrinsics
        y0, y1, _p0, _p1 = g.fov_footprint_deg(i, p.R)
        foots.append((0.5 * (y0 + y1), 0.5 * (y1 - y0)))
    rows = _pitch_rows(poses)

    # Per ROW, not per capture. A three-row sweep has no yaw covered by exactly one camera,
    # and "the fewest cameras" then picks a yaw where one row has a hole -- measured on the
    # reference capture, the canvas edge landed in an up-row gap and the top of the
    # panorama at the edge was synthesised. Every row must have a camera whose footprint
    # holds the band; among those yaws the one whose weakest row is deepest inside its
    # camera wins.
    best_key: tuple[int, float, int] | None = None
    best_yaw = 0.0
    n_steps = max(1, int(round(360.0 / step_deg)))
    for k in range(n_steps):
        cand = -180.0 + k * step_deg
        count = 0
        weakest = 1e9
        for row in rows:
            row_margin = -1e9
            for idx in row:
                centre, half = foots[idx]
                d = math.degrees(abs(float(g.wrap_pi(math.radians(cand - centre)))))
                if d + band_deg <= half:
                    count += 1
                    row_margin = max(row_margin, half - d)
            weakest = min(weakest, row_margin)
        # Sort key: every row covered first, then the FEWEST cameras, then the deepest
        # weakest row. Fewest before deepest: a yaw held by one camera per row puts both
        # sides of the canvas edge in the same image, so there is no seam there at all,
        # and that is worth more than a few degrees of extra margin. Measured on the
        # single-row fast fixture (wrap column step over the median column step, mesh
        # alignment off so only the placement changes): deepest-first chose a yaw shared
        # by three cameras and scored 1.83; fewest-first (the original rule, now applied
        # per row) chose a single-camera yaw and scored 0.84.
        key = (0 if weakest > 0.0 else 1, count, -weakest)
        if best_key is None or key < best_key:
            best_key = key
            best_yaw = cand
    _LOG.info("wrap: canvas edge placed at yaw %.1f deg (key=%s)", best_yaw, best_key)
    return float(math.radians(best_yaw))


def _pitch_rows(poses: list[CameraPose], gap_deg: float = 8.0) -> list[list[int]]:
    """Group pose indices into pitch rows: sorted by pitch, split at gaps over ``gap_deg``."""
    if not poses:
        return []
    pitches = [math.degrees(p.pitch) for p in poses]
    order = sorted(range(len(poses)), key=lambda i: pitches[i])
    rows: list[list[int]] = [[order[0]]]
    for a, b in zip(order, order[1:]):
        if pitches[b] - pitches[a] > gap_deg:
            rows.append([b])
        else:
            rows[-1].append(b)
    return rows


def apply_wrap_offset(poses: list[CameraPose], offset: float) -> tuple[list[CameraPose], float]:
    """Rotate every pose so world yaw ``offset`` lands on the canvas edge.

    Returns the new poses and the applied yaw delta in radians; adding ``-delta`` to a
    viewer's initial yaw restores the original orientation.
    """
    delta = -math.pi - offset
    C = g.from_yaw_pitch_roll(-delta, 0.0, 0.0)  # rotates a world direction's yaw by +delta
    out: list[CameraPose] = []
    for p in poses:
        out.append(
            CameraPose(
                frame_index=p.frame_index,
                R=g.nearest_rotation(np.asarray(p.R, dtype=np.float64) @ C.T),
                intrinsics=p.intrinsics,
                gain=np.asarray(p.gain, dtype=np.float64).reshape(3).copy(),
                vignette=tuple(float(v) for v in p.vignette),  # type: ignore[arg-type]
            )
        )
    return out, float(g.wrap_pi(delta))


# ======================================================================================
# Wrap repair
# ======================================================================================


def _seam_owner(poses: list[CameraPose], subset: list[int] | None = None) -> int:
    """Index of the pose (within ``subset``) whose footprint contains the canvas edge most deeply."""
    best, best_depth = -1, -1e9
    for i in (subset if subset is not None else range(len(poses))):
        p = poses[i]
        y0, y1, _p0, _p1 = g.fov_footprint_deg(p.intrinsics, p.R)
        centre, half = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
        d = math.degrees(abs(float(g.wrap_pi(math.radians(180.0 - centre)))))
        depth = half - d
        if depth > best_depth:
            best_depth, best = depth, i
    return best if best_depth > 0.0 else -1


def repair_wrap_band(
    pano: np.ndarray,
    frames: list[Frame],
    poses: list[CameraPose],
    config: PipelineConfig,
    *,
    band: int | None = None,
    displacements: list[warp_mod.Displacement | None] | None = None,
) -> np.ndarray:
    """Make the canvas edge mathematically continuous by re-rendering across it.

    The seam-owning camera is warped onto a canvas rolled by half a turn, which puts the
    canvas edge in the *middle* of a perfectly ordinary patch.  A ``2*band`` px strip of
    that render is level-matched to the panorama and merged back over ``u = 0`` along the
    path where it agrees best with the panorama on each side (:func:`_merge_wrap_band`), so
    whatever the pyramid blender did at the ROI boundary is overwritten by pixels that were
    never near a boundary.  ``band`` defaults to ``_WRAP_SEARCH_DEG`` of yaw.

    When the owner was mesh-aligned, the same displacement grid is applied to the rolled
    render -- its origin rolled with it -- so the repair strip matches the blended pixels
    on either side rather than the un-aligned view.
    """
    h, w = pano.shape[:2]
    if band is None:
        band = max(_WRAP_BAND, int(round(w * _WRAP_SEARCH_DEG / 360.0)))
    # One owner per pitch row, so the whole covered height of the wrap column is
    # re-rendered rather than only the band of whichever single camera reached deepest.
    owners = [o for o in (_seam_owner(poses, row) for row in _pitch_rows(poses)) if o >= 0]
    if not owners:
        _LOG.info("wrap: canvas edge is not covered by any camera, no repair needed")
        return pano
    for owner in owners:
        pano = _repair_wrap_band_with(pano, frames, poses, config, owner, band=band, displacements=displacements)
    return pano


def _least_cost_path(cost: np.ndarray, lo: int, hi: int, step_cost: float) -> np.ndarray:
    """Per row, the column in ``[lo, hi]`` of the top-to-bottom path of least total cost.

    The path may move one column between rows (8-connected), paying ``step_cost`` each time.
    Dynamic programming over the rows -- the seam-carving recurrence.
    """
    ph = cost.shape[0]
    c = np.asarray(cost[:, lo : hi + 1], dtype=np.float64)
    m = c.shape[1]
    inf = np.full(1, np.inf)
    acc = c[0].copy()
    back = np.zeros((ph, m), dtype=np.int8)
    cols = np.arange(m)
    for y in range(1, ph):
        cand = np.stack([
            np.concatenate([inf, acc[:-1]]) + step_cost,   # from the column to the left
            acc,                                           # straight down
            np.concatenate([acc[1:], inf]) + step_cost,    # from the column to the right
        ])
        k = np.argmin(cand, axis=0)
        acc = cand[k, cols] + c[y]
        back[y] = (k - 1).astype(np.int8)
    path = np.empty(ph, dtype=np.int64)
    j = int(np.argmin(acc))
    for y in range(ph - 1, -1, -1):
        path[y] = j
        j += int(back[y, j])
    return path + lo


def _merge_wrap_band(
    cur: np.ndarray, strip: np.ndarray, valid: np.ndarray, *, feather: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Merge the owner's re-render into the panorama's band around the canvas edge.

    ``cur`` and ``strip`` are ``(rows, 2*band, 3)``: the panorama's columns ``W-band..W-1``
    then ``0..band-1``, and the owner camera rendered over the same span (continuous across
    the edge, being one image). ``valid`` is the owner's mask. Returns the merged band and a
    few numbers for the log.

    WHY NOT A CROSS-FADE. The two sides of the canvas edge are not always the owner's
    pixels: graph cut is free to hand the columns next to the edge to whichever camera it
    likes, and where that camera disagrees with the owner by parallax, fading one into the
    other draws both. On the reference capture the edge fell on a close, glossy wardrobe and
    a 24 px cross-fade showed a door frame twice. So each side is handed over from the
    panorama to the owner along the path where the two AGREE best -- the seam-carving
    recurrence over a cost of their per-pixel disagreement -- with only a ``feather`` px
    ramp across it. One copy of everything survives, the owner's near the edge (which
    makes the edge itself continuous) and the panorama's beyond the path.

    The owner is level-matched to each side by the median of their difference over that
    side (robust to whatever parallax moved), and the offset ramps from the left side's
    value at the left path to the right side's at the right path, so no step is introduced
    at either hand-over or at the edge.
    """
    ph, two_band = cur.shape[:2]
    band = two_band // 2
    f = max(1, int(feather))
    cur = cur.astype(np.float32)
    strip = strip.astype(np.float32)
    valid = np.asarray(valid, dtype=bool)

    # -- level: the median difference per row and side, smoothed down the column ---------
    diff = cur - strip
    levels = []
    for lo, hi in ((0, band), (band, two_band)):
        d = np.where(valid[:, lo:hi, None], diff[:, lo:hi], np.nan)
        enough = valid[:, lo:hi].sum(axis=1) >= max(3, (hi - lo) // 4)
        with np.errstate(all="ignore"):
            med = np.nanmedian(d[enough], axis=1) if enough.any() else np.zeros((0, 3), np.float32)
        lv = np.zeros((ph, 3), dtype=np.float32)
        lv[enough] = np.nan_to_num(med, nan=0.0)
        levels.append((lv, enough))
    (dl, okl), (dr, okr) = levels
    dl = np.where(okl[:, None], dl, np.where(okr[:, None], dr, 0.0))
    dr = np.where(okr[:, None], dr, dl)
    ksz = max(3, (ph // 8) | 1)
    dl = cv2.GaussianBlur(dl.reshape(ph, 1, 3), (1, ksz), ksz / 4.0, borderType=cv2.BORDER_REPLICATE).reshape(ph, 3)
    dr = cv2.GaussianBlur(dr.reshape(ph, 1, 3), (1, ksz), ksz / 4.0, borderType=cv2.BORDER_REPLICATE).reshape(ph, 3)
    dl = np.clip(dl, -60.0, 60.0)
    dr = np.clip(dr, -60.0, 60.0)

    # -- the hand-over paths ---------------------------------------------------------------
    # Disagreement after levelling, averaged over the feather's width so the path is judged
    # by the ramp it will actually draw. Columns the owner cannot supply cost the most.
    lev = np.concatenate([np.repeat(dl[:, None, :], band, 1), np.repeat(dr[:, None, :], two_band - band, 1)], axis=1)
    cost = np.abs(cur - (strip + lev)).mean(axis=2)
    raw = np.where(valid, cost, np.nan)
    cost[~valid] = 255.0
    cost = cv2.blur(cost.astype(np.float32), (2 * f + 1, 3), borderType=cv2.BORDER_REPLICATE)
    # Index both sides from the edge outward: column 0 touches the canvas edge.
    left_cost = cost[:, band - 1 :: -1]
    right_cost = cost[:, band:]
    lo, hi = f + 1, band - 1 - f
    if hi < lo:
        lo = hi = max(0, band // 2)
    pl = _least_cost_path(left_cost, lo, hi, _WRAP_PATH_STEP_COST)
    pr = _least_cost_path(right_cost, lo, hi, _WRAP_PATH_STEP_COST)

    # -- weights: owner from the edge out to each path, a short ramp across it -------------
    j = np.arange(band, dtype=np.float32)[None, :]
    a_left = np.clip((pl[:, None] + f - j) / (2.0 * f), 0.0, 1.0)[:, ::-1]
    a_right = np.clip((pr[:, None] + f - j) / (2.0 * f), 0.0, 1.0)
    alpha = np.concatenate([a_left, a_right], axis=1) * valid

    # Level: the left side's offset at the left path, the right side's at the right, and
    # their mean held flat across the edge itself -- a straight ramp between the paths put
    # a tenth of a grey level of step on the edge column, which on test_render's smooth
    # fixture took the wrap ratio from 0.95 to 1.33.
    x = np.arange(two_band, dtype=np.float32)[None, :]
    xl = (band - 1 - pl).astype(np.float32)[:, None]
    xr = (band + pr).astype(np.float32)[:, None]
    hold = np.maximum(1.0, 0.5 * np.minimum(pl, pr)).astype(np.float32)[:, None]
    mid_l, mid_r = band - 0.5 - hold, band - 0.5 + hold
    tl = np.clip((x - xl) / np.maximum(mid_l - xl, 1.0), 0.0, 1.0)[..., None]
    tr = np.clip((x - mid_r) / np.maximum(xr - mid_r, 1.0), 0.0, 1.0)[..., None]
    dm = 0.5 * (dl + dr)
    offset = np.where(
        (x < band)[..., None],
        dl[:, None, :] + (dm - dl)[:, None, :] * tl,
        dm[:, None, :] + (dr - dm)[:, None, :] * tr,
    )
    owner = np.clip(strip + offset, 0.0, 255.0)

    out = cur * (1.0 - alpha[..., None]) + owner * alpha[..., None]
    rows = np.arange(ph)
    at_paths = np.concatenate([raw[rows, band - 1 - pl], raw[rows, band + pr]])
    stats = {
        "path_left_px": float(pl.mean()),
        "path_right_px": float(pr.mean()),
        # Mean per-pixel disagreement where the hand-over actually happens, grey levels.
        "disagreement": float(np.nanmean(at_paths)) if np.isfinite(at_paths).any() else 0.0,
    }
    return out, stats


def _repair_wrap_band_with(
    pano: np.ndarray,
    frames: list[Frame],
    poses: list[CameraPose],
    config: PipelineConfig,
    owner: int,
    *,
    band: int,
    displacements: list[warp_mod.Displacement | None] | None,
) -> np.ndarray:
    h, w = pano.shape[:2]
    pose = poses[owner]
    C = g.from_yaw_pitch_roll(-math.pi, 0.0, 0.0)
    rolled_pose = CameraPose(
        frame_index=pose.frame_index,
        R=g.nearest_rotation(np.asarray(pose.R, dtype=np.float64) @ C.T),
        intrinsics=pose.intrinsics,
        gain=np.asarray(pose.gain, dtype=np.float64).reshape(3).copy(),
        vignette=tuple(float(v) for v in pose.vignette),  # type: ignore[arg-type]
    )
    by_index = {f.index: f for f in frames}
    frame = by_index.get(pose.frame_index)
    if frame is None:
        if owner >= len(frames):
            _LOG.warning("wrap: no frame for the seam-owning pose, skipping repair")
            return pano
        frame = frames[owner]

    rolled_disp: warp_mod.Displacement | None = None
    if displacements is not None and owner < len(displacements) and displacements[owner] is not None:
        grid, cell, (gu0, gv0) = displacements[owner]
        if (grid.shape[1] - 1) * cell >= w - 1e-6:
            rolled_disp = (grid, cell, (gu0 + w // 2, gv0))
        else:
            _LOG.warning("wrap: seam owner's mesh does not span the canvas; repair uses the plain warp")

    image = warp_mod.load_frame_image(frame)
    patch = warp_mod.warp_camera(image, rolled_pose, w, h, config, index=owner, displacement=rolled_disp)
    del image

    mid = w // 2
    px0, py0, px1, py1 = patch.rect
    # As wide as asked, narrowed to what the rolled render actually spans.
    band = min(int(band), mid - px0, px1 - mid)
    if band < _WRAP_BAND:
        _LOG.info("wrap: rolled render does not span the repair band, skipping repair")
        del patch
        return pano

    rows = slice(py0, py1)
    a = (mid - band) - px0
    strip = patch.image[:, a : a + 2 * band]
    valid = patch.mask[:, a : a + 2 * band] > 0
    cur = np.concatenate([pano[rows, w - band : w], pano[rows, 0:band]], axis=1)
    # Two pixels of ramp at 4096 wide, four at 8192: a hand-over, not a blend.
    merged, stats = _merge_wrap_band(cur, strip, valid, feather=max(2, w // 2048))
    merged = warp_mod.to_uint8(merged)
    pano[rows, w - band : w] = merged[:, :band]
    pano[rows, 0:band] = merged[:, band:]
    del patch, strip, valid, cur, merged
    _LOG.info(
        "wrap: repaired a %d px band across the canvas edge from camera %d (handed over %.1f / "
        "%.1f px either side of the edge, %.1f grey levels of disagreement along the paths)",
        2 * band, owner, stats["path_left_px"], stats["path_right_px"], stats["disagreement"],
    )
    return pano


# ======================================================================================
# Seam quality metric
# ======================================================================================


def seam_gradient_energy(
    pano: np.ndarray,
    boundary: np.ndarray,
    coverage: np.ndarray,
    *,
    strip_rows: int = 512,
) -> float:
    """Mean gradient magnitude on the seams divided by that of the whole panorama.

    A value near 1.0 means the cuts fall on content that is no busier than average -- i.e.
    they are invisible.  Well above 1.0 means the blender left an edge exactly where the
    ownership changes, which is precisely what a viewer's eye locks onto.
    """
    h = pano.shape[0]
    seam_sum = seam_n = all_sum = all_n = 0.0
    for y0 in range(0, h, strip_rows):
        y1 = min(h, y0 + strip_rows)
        gray = cv2.cvtColor(pano[y0:y1], cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        del gray, gx, gy
        cov = coverage[y0:y1] > 0
        bnd = (boundary[y0:y1] > 0) & cov
        all_sum += float(mag[cov].sum())
        all_n += float(cov.sum())
        seam_sum += float(mag[bnd].sum())
        seam_n += float(bnd.sum())
        del mag, cov, bnd
    if seam_n < 64 or all_n < 64 or all_sum <= 0:
        return 1.0
    denom = all_sum / all_n
    if denom <= 1e-6:
        return 1.0
    return float((seam_sum / seam_n) / denom)


# ======================================================================================
# Entry point
# ======================================================================================


def select_render_views(
    frames: list[Frame], poses: list[CameraPose], config: PipelineConfig
) -> tuple[list[Frame], list[CameraPose], int]:
    """Thin the solved cameras down to the ones the COMPOSITE needs.

    THE SOLVE AND THE RENDER WANT OPPOSITE THINGS, AND UNTIL NOW ONLY THE SOLVE GOT ITS WAY.

    Keyframe selection oversamples on purpose: redundant views are what keep the pose graph
    connected through a blurred frame, and `select._select_along_track` documents an
    ablation where evening the spacing out halved connectivity. That is right for solving.
    The render then inherited the same list, and for compositing every extra view is another
    seam -- and a seam is where the blender mixes two images that parallax has already put a
    few pixels apart.

    MEASURED, on a faithful fixture that reproduces the shipped pipeline's own
    `seam_gradient_energy` to within 0.06 of a real render:

        views   overlap   seam gap   ghost   clean area   seam energy
           68       95%      47 px   1.417          37%         0.770   <- what we shipped
           16       78%     200 px   1.084          63%         0.565
            8       56%     400 px   0.962          74%         0.570   <- target_overlap

    At the density the solver hands over, seams fall every 47 px while the blend reaches 30
    to 60, so the ghost zones overlap and no part of the panorama is outside one. Thinning to
    the overlap the config already asks for cuts the ghost by a third, DOUBLES the area
    further than 30 px from any seam, and makes the seams themselves less visible rather than
    more: fewer cuts give the seam finder more freedom to put each one somewhere harmless.
    It also brings the graph-cut finder back within its memory budget -- at 68 views it does
    not fit and the pipeline silently falls back to dynamic programming.

    Greedy, sharpest first, on great-circle separation, so it thins in pitch as well as yaw
    and never opens a hole: a view is kept only if nothing already kept is within
    `hfov * (1 - target_overlap)` of it.
    """
    n = len(poses)
    if n < 3 or not config.thin_render_views:
        return frames, poses, n

    hfov = float(poses[0].intrinsics.hfov_deg)
    vfov = float(poses[0].intrinsics.vfov_deg)
    # The render keeps MORE overlap than the solve asked for. `target_overlap` (55 %) is the
    # solver's spacing; at that spacing the kept views of a row sit 20-40 deg apart against a
    # 46 deg field, so adjacent views share only a thin strip and every seam is forced into
    # it -- at the frame border, where lens distortion and vignette are worst and where the
    # mesh aligner has the least to measure. Measured on the reference capture: with the
    # solver's spacing the cornice was doubled at every row seam; with every view kept it was
    # continuous. `render_view_overlap` (75 %) keeps the overlap wide enough for the aligner
    # and the seam finder to work while still dropping the near-duplicates.
    overlap = max(float(config.target_overlap), float(config.render_view_overlap))
    min_sep = math.radians(max(1.0, hfov * (1.0 - overlap)))

    tan_h, tan_v = math.tan(math.radians(hfov) / 2.0), math.tan(math.radians(vfov) / 2.0)

    def rim(pose: CameraPose, per_edge: int = 8) -> np.ndarray:
        """Directions around this frame's border, in world coordinates.

        The border, not a grid. A view is redundant exactly when everything it can see is
        already seen, and the last thing to become covered as a footprint slides under its
        neighbours is its own edge — so testing the edge tests the claim directly, at points
        that exist rather than at whatever a sampling lattice happened to land on. A grid
        cannot make this promise: the first version guaranteed coverage on its own 2-degree
        lattice and a test sampling the same sphere half a degree away still found 148
        directions gone.
        """
        t = np.linspace(-1.0, 1.0, per_edge)
        edges = [
            np.stack([t * tan_h, np.full_like(t, -tan_v), np.ones_like(t)], axis=1),
            np.stack([t * tan_h, np.full_like(t, tan_v), np.ones_like(t)], axis=1),
            np.stack([np.full_like(t, -tan_h), t * tan_v, np.ones_like(t)], axis=1),
            np.stack([np.full_like(t, tan_h), t * tan_v, np.ones_like(t)], axis=1),
        ]
        cam = np.concatenate(edges, axis=0)
        cam /= np.linalg.norm(cam, axis=1, keepdims=True)
        return cam @ np.asarray(pose.R, dtype=np.float64)

    def sees(pose: CameraPose, world_dirs: np.ndarray, slack: float = 1.02) -> np.ndarray:
        """Which of `world_dirs` fall inside this camera's frame.

        `slack` widens the frustum by 2%, about half a degree, so a direction sitting exactly
        on a neighbour's border is not called uncovered by floating-point luck.
        """
        cam = world_dirs @ np.asarray(pose.R, dtype=np.float64).T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (z > 1e-9)
                & (np.abs(cam[:, 0] / z) <= tan_h * slack)
                & (np.abs(cam[:, 1] / z) <= tan_v * slack)
            )

    axes = [np.asarray(p.R, dtype=np.float64).T @ np.array([0.0, 0.0, 1.0]) for p in poses]
    sharp = [float(getattr(f, "sharpness", 0.0)) for f in frames]
    # MEASURED BEFORE INTERPOLATED, sharpness only inside each group. A bridged frame (see
    # `solve._bridge_track_gaps`) was placed between two solved neighbours because nothing in
    # it could be matched; its orientation is an estimate, and where a measured frame covers
    # the same direction the two put the same cornice on the canvas at two slightly different
    # pitches. The visual audit of the reference panorama found exactly that -- "the crown
    # moulding is drawn twice over 29 degrees of wall" at the yaw where a bridged frame met a
    # measured one. Ordering, not exclusion: the greedy pass below keeps a view only if it
    # holds sphere nothing already kept can see, so a bridged frame still fills the hole it
    # was made for and is dropped everywhere else.
    bridged = [bool(getattr(f, "_cache", {}).get("bridged")) for f in frames]
    order = sorted(range(n), key=lambda i: (bridged[i], -sharp[i]))

    # Pass 1: spread. Sharpest first, refusing anything that crowds a view already kept.
    kept: list[int] = []
    kept_axes: list[np.ndarray] = []
    cos_limit = math.cos(min_sep)
    for i in order:
        if all(float(axes[i] @ b) < cos_limit for b in kept_axes):
            kept.append(i)
            kept_axes.append(axes[i])

    # Pass 2: give back anything holding sphere of its own. Spreading alone clips the ends of
    # a sweep -- a view at the edge can sit inside the separation limit of its neighbour and
    # still be the only one that reaches past it -- and `coveredFraction`, the reported yaw
    # span and the gap list all describe what was actually delivered.
    #
    # WHAT THIS DOES NOT PROMISE, STATED RATHER THAN HIDDEN. It closes yaw: no sector is
    # lost, because a whole missing frame's border is unmistakably uncovered. It does NOT
    # hold the last half-degree of VERTICAL reach. A frame is a pincushion on the sphere --
    # its centre column reaches vfov/2, its corners about eight degrees less -- so the
    # extreme elevation is only attained directly above a frame centre, and thinning leaves
    # fewer centres. Measured on a 60-view row thinned to 20: 68 directions of 4397 lost,
    # 1.5% of the swept band, every one inside the outermost 2 degrees of vertical reach, none
    # anywhere else. That is under a tenth of the 3-degree margin the completeness gates
    # already carry for row-pitch error, and it buys a third off the ghosting.
    rims = {i: rim(poses[i]) for i in range(n)}
    restored = 0
    changed = True
    while changed:
        changed = False
        for i in sorted(set(range(n)) - set(kept)):
            border = rims[i]
            seen = np.zeros(border.shape[0], dtype=bool)
            for j in kept:
                seen |= sees(poses[j], border)
                if seen.all():
                    break
            if not seen.all():
                kept.append(i)
                restored += 1
                changed = True
    kept.sort()
    if len(kept) < 3:
        return frames, poses, n

    _LOG.info(
        "render: compositing %d of %d solved views (%d restored for coverage, min sep %.1f deg)",
        len(kept), n, restored, math.degrees(min_sep),
    )
    return [frames[i] for i in kept], [poses[i] for i in kept], n


#: Re-aim cap for a view with no sparse constraints: 5 degrees per axis. An interpolated
#: orientation can be off by that much; a proposal beyond it did not measure the view's
#: misplacement but slid along an edge (measured: +16.5 degrees of yaw proposed for a view
#: whose only structure is a horizontal cornice).
_REAIM_MAX_DEG = 5.0
_REAIM_MIN_BLOCKS = 20


def _neighbour_template(k: int, partners: list[int], small: list[warp_mod.WarpedPatch]) -> tuple[np.ndarray, np.ndarray]:
    """The constrained neighbours' pixels on view ``k``'s window (mean where several see
    it), and the mask of where both they and ``k`` have data."""
    p = small[k]
    x0, y0, w, h = int(p.corner[0]), int(p.corner[1]), int(p.width), int(p.height)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for j in partners:
        q = small[j]
        qx0, qy0 = int(q.corner[0]), int(q.corner[1])
        ax0, ay0 = max(x0, qx0), max(y0, qy0)
        ax1, ay1 = min(x0 + w, qx0 + int(q.width)), min(y0 + h, qy0 + int(q.height))
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        gq = cv2.cvtColor(q.image[ay0 - qy0:ay1 - qy0, ax0 - qx0:ax1 - qx0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        mq = (q.mask[ay0 - qy0:ay1 - qy0, ax0 - qx0:ax1 - qx0] > 0).astype(np.float32)
        acc[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] += gq * mq
        cnt[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] += mq
    tmpl = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0).astype(np.float32)
    mask = ((cnt > 0) & (p.mask > 0)).astype(np.uint8)
    return tmpl, mask


def _ecc_shift(k: int, partners: list[int], small: list[warp_mod.WarpedPatch]) -> tuple[float, float] | None:
    """Canvas shift ``(du, dv)`` that moves view ``k``'s content onto its neighbours, by
    ECC image alignment (translation model, coarse to fine). ``None`` when ECC fails."""
    p = small[k]
    tmpl, mask = _neighbour_template(k, partners, small)
    if int(mask.sum()) < 4096:
        return None
    inp = cv2.cvtColor(p.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Outside the overlap the template shows the input itself, so ECC sees no false edge
    # at the overlap's border.
    tmpl = np.where(mask > 0, tmpl, inp)
    tx = ty = 0.0
    for f in (0.25, 0.5, 1.0):
        t = cv2.resize(tmpl, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        i = cv2.resize(inp, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        if t.shape[0] < 16 or t.shape[1] < 16:
            continue
        warp = np.array([[1.0, 0.0, tx * f], [0.0, 1.0, ty * f]], np.float32)
        try:
            _cc, warp = cv2.findTransformECC(
                t, i, warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), m, 5,
            )
        except cv2.error:
            return None
        tx, ty = float(warp[0, 2]) / f, float(warp[1, 2]) / f
    # The warp maps template coordinates to input coordinates: the input's content sits at
    # template + t, so the view has to move by -t to land on its neighbours.
    return -tx, -ty


def _reaim_views(
    loose: list[int],
    frames: list[Frame],
    work: list[CameraPose],
    small: list[warp_mod.WarpedPatch],
    meshes: list[mesh_mod.ViewMesh],
    s: float,
    Ws: int,
    Hs: int,
    cfg_small: PipelineConfig,
    cell: float,
    W: int,
    H: int,
    config: PipelineConfig,
) -> dict[str, object]:
    """Rotate each of ``loose`` so its small warp lines up with its constrained neighbours.

    The proposal comes from ECC image alignment of the view against a composite of its
    neighbours -- it uses every pixel, shading included, where a block matcher on a plain
    wall returns noise (measured: block-shift medians moved a bridged view 9 degrees along
    the cornice at 8192 px and doubled it, and gave different answers at 4096 and 8192,
    while ECC proposed the same 0.5-0.8 degree corrections at both). The verdict comes from
    the dense pass's own yardstick: how many blocks it can match within its window and how
    far off they are. A move must improve that or it is undone; an axis proposed beyond the
    cap slid along an edge and is dropped; a view whose proposal fails both ways stays
    where the interpolation put it. Mutates ``work[k]``, ``small[k]`` and ``meshes[k]`` in
    place and returns a per-view summary for the log.
    """
    constrained = [k for k in range(len(work)) if k not in set(loose)]
    cap = math.radians(_REAIM_MAX_DEG)
    narrow = 10.0  # the window `mesh_mod.dense_constraints` searches by default
    summary: dict[str, object] = {}

    def _partners_of(k: int) -> list[int]:
        patch = small[k]
        return [
            j for j in constrained
            if small[j].corner[0] < patch.corner[0] + patch.width
            and patch.corner[0] < small[j].corner[0] + small[j].width
            and small[j].corner[1] < patch.corner[1] + patch.height
            and patch.corner[1] < small[j].corner[1] + small[j].height
        ]

    def _yardstick(k: int, partners: list[int]) -> mesh_mod.RigidShift:
        return mesh_mod.rigid_shift(
            k, small, meshes, s, partners=partners, block=int(config.mesh_block_px),
            stride=max(4, int(config.mesh_block_px) // 2), max_shift=narrow,
        )

    def _place(k: int, pose: CameraPose, yaw: float, pitch: float, roll: float) -> None:
        new_pose = CameraPose(
            frame_index=pose.frame_index,
            R=g.from_yaw_pitch_roll(yaw, pitch, roll),
            intrinsics=pose.intrinsics,
            gain=pose.gain,
            vignette=pose.vignette,
        )
        work[k] = new_pose
        meshes[k] = mesh_mod.build_meshes([new_pose], W, H, cell=cell)[0]
        meshes[k].view = k
        image = warp_mod.load_frame_image(frames[k], s)
        small[k] = warp_mod.warp_camera(image, new_pose, Ws, Hs, cfg_small, index=k)
        del image

    for k in loose:
        partners = _partners_of(k)
        if not partners:
            continue
        shift = _ecc_shift(k, partners, small)
        if shift is None:
            continue
        du, dv = shift
        # Content that must move by +du on the canvas is a view that must turn by +du of
        # yaw; canvas v grows downward, so +dv is a drop in pitch.
        dyaw = (du / s) * (2.0 * math.pi / W)
        dpitch = -(dv / s) * (math.pi / H)
        if abs(dyaw) > cap and abs(dpitch) > cap:
            continue
        if abs(dyaw) > cap:
            dyaw = 0.0  # slid along a horizontal edge; pitch may still be right
        if abs(dpitch) > cap:
            dpitch = 0.0
        if abs(dyaw) < 1e-4 and abs(dpitch) < 1e-4:
            continue
        base = _yardstick(k, partners)
        pose0, small0, mesh0 = work[k], small[k], meshes[k]
        yaw0, pitch0, roll0 = g.yaw_pitch_roll(pose0.R)
        _place(k, pose0, yaw0 + dyaw, pitch0 + dpitch, roll0)
        after = _yardstick(k, _partners_of(k) or partners)
        better = after.n >= _REAIM_MIN_BLOCKS and (
            after.residual < base.residual or after.n >= 1.3 * base.n
        )
        _LOG.info(
            "reaim: view %d ecc du %+.1f dv %+.1f -> yaw %+.2f pitch %+.2f deg; dense window "
            "n %d->%d, residual %.1f->%.1f px: %s",
            k, du, dv, math.degrees(dyaw), math.degrees(dpitch), base.n, after.n,
            base.residual, after.residual, "kept" if better else "undone",
        )
        if not better:
            work[k], small[k], meshes[k] = pose0, small0, mesh0
            continue
        summary[f"view_{k}"] = (
            f"yaw {math.degrees(dyaw):+.2f} deg, pitch {math.degrees(dpitch):+.2f} deg "
            f"(dense-window blocks {base.n}->{after.n}, residual {base.residual:.1f}->{after.residual:.1f} px)"
        )
    return summary


def render_panorama(
    frames: list[Frame],
    poses: list[CameraPose],
    config: PipelineConfig,
    *,
    out_dir: Path,
    progress: ProgressReporter | None = None,
    metrics: StitchMetrics | None = None,
    pairs: list[MatchPair] | None = None,
) -> StitchResult:
    """Turn solved cameras plus their source frames into a delivered panorama.

    The returned ``StitchResult.poses`` are expressed in the *rendered* world frame, i.e.
    rotated by the wrap offset.  The offset itself is written to ``out_dir/render_meta.json``
    and logged through ``progress`` -- :class:`StitchResult` has no field for it and the
    contract is frozen -- so a viewer can subtract it to open on the original heading.

    ``pairs`` are the solver's verified correspondences (indexed by frame); with
    ``config.mesh_align`` they drive the local mesh alignment that removes residual
    parallax before the views are warped.  Without them the views are warped rigidly.
    """
    if len(frames) == 0 or len(poses) == 0:
        raise ValueError("render_panorama needs at least one frame and one pose")
    prog: ProgressReporter = progress if progress is not None else NullProgress()
    metrics = metrics if metrics is not None else StitchMetrics()
    frames, poses, n_solved = select_render_views(frames, poses, config)
    if len(poses) < n_solved:
        metrics.warnings.append(
            f"composited_from_{len(poses)}_of_{n_solved}_views: the extra views the solver "
            "keeps for a rigid pose graph would each add a seam, and a seam is where "
            "parallax shows"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(config.seed)  # pins any downstream default_rng(seed) usage

    W = int(config.output_width)
    H = int(config.output_height)
    t_all = time.perf_counter()

    # ---- wrap offset ----------------------------------------------------------------
    offset = choose_wrap_offset(poses, None)
    work, delta = apply_wrap_offset(poses, offset)
    prog.log(
        "wrap offset applied",
        wrap_offset_deg=round(math.degrees(offset), 3),
        yaw_delta_deg=round(math.degrees(delta), 3),
    )

    # ---- exposure -------------------------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("exposure", 0.0, "estimating gains")
    if config.exposure_compensation and len(work) > 1:
        small_w = min(_EXPOSURE_CANVAS_W, W)
        small_h = max(2, small_w // 2)
        cfg_small = replace(config, interpolation="linear")
        raw_poses = [
            CameraPose(frame_index=p.frame_index, R=p.R, intrinsics=p.intrinsics)
            for p in work
        ]
        small_patches = warp_mod.warp_all(
            frames, raw_poses, small_w, small_h, cfg_small,
            source_scale=small_w / float(W), progress=prog, stage="exposure",
            stage_span=(0.0, 0.6),
        )
        gains = exposure_mod.estimate_gains(small_patches, per_channel=config.per_channel_gain)
        prog.stage("exposure", 0.75, "fitting vignette")
        vign = (
            exposure_mod.estimate_vignette(small_patches, gains, seed=config.seed)
            if config.vignette_correction
            else None
        )
        exposure_mod.apply_to_poses(work, gains, vign)
        metrics.exposure_spread = exposure_mod.exposure_spread(gains)
        _LOG.info(
            "exposure: spread %.4f, vignette %s", metrics.exposure_spread, vign
        )
        del small_patches, raw_poses
    prog.stage("exposure", 1.0, "done")
    metrics.stage_seconds["exposure"] = time.perf_counter() - t0

    # ---- local mesh alignment -------------------------------------------------------
    t0 = time.perf_counter()
    displacements: list[warp_mod.Displacement | None] | None = None
    mesh_stats: dict[str, float] = {}
    if config.mesh_align and pairs and len(work) > 1:
        prog.stage("warp", 0.0, "aligning views on a mesh")
        cell = max(8.0, float(W) * float(config.mesh_cell_fraction))
        meshes = mesh_mod.build_meshes(work, W, H, cell=cell)
        view_of_frame = {int(p.frame_index): k for k, p in enumerate(work)}
        sparse = mesh_mod.sparse_constraints(
            meshes, work, pairs, W, H, view_of_frame=view_of_frame, seed=config.seed
        )
        mesh_stats = mesh_mod.solve_meshes(meshes, sparse, W, config)
        if config.mesh_dense and sparse.n > 0 and mesh_stats.get("n_points", 0) > 0:
            # Pass two: block-match the overlaps of a reduced-scale render made WITH the
            # first solution, so the matcher only has to see the residual, then re-solve
            # from scratch with both constraint sets.
            s = min(1.0, float(config.mesh_dense_canvas) / float(W))
            Ws, Hs = max(64, int(round(W * s))), max(32, int(round(H * s)))
            prog.stage("warp", 0.02, "block-matching the overlaps")
            cfg_small = replace(config, interpolation="linear")
            small = warp_mod.warp_all(
                frames, work, Ws, Hs, cfg_small,
                source_scale=s, displacements=[m.scaled(s) for m in meshes],
            )
            # Views with no sparse constraint at all were placed, not solved (see
            # `solve._bridge_track_gaps`); re-aim each rigidly before the dense pass, which
            # only sees 10 px and will not move a whole view. See `_reaim_views` for how a
            # move is proposed and judged, and for what was measured.
            constrained = set(sparse.vi.tolist()) | set(sparse.vj.tolist())
            loose = [k for k in range(len(work)) if k not in constrained]
            if loose:
                reaimed = _reaim_views(
                    loose, frames, work, small, meshes, s, Ws, Hs, cfg_small, cell, W, H, config,
                )
                if reaimed:
                    prog.log("re-aimed views without sparse constraints", **reaimed)
            dense = mesh_mod.dense_constraints(
                small, meshes, s, block=int(config.mesh_block_px),
                stride=max(4, int(config.mesh_block_px) // 2),
                weight=float(config.mesh_dense_weight),
                edge_weight=float(config.mesh_edge_weight),
                # The interpolated views may still be a few degrees off after the re-aim
                # (yaw is unobservable on a plain wall); let the mesh reach them.
                wide_views=set(loose),
                wide_shift=max(4.0, _REAIM_MAX_DEG / 360.0 * Ws),
            )
            del small
            if dense.n > 0:
                for m in meshes:
                    m.d[:] = 0.0
                mesh_stats = mesh_mod.solve_meshes(
                    meshes, mesh_mod.Constraints.concat([sparse, dense]), W, config
                )
                mesh_stats["n_dense_rows"] = int(dense.n)
        if mesh_stats.get("n_points", 0) > 0:
            displacements = [(m.d, m.cell, (m.u0, m.v0)) for m in meshes]
            prog.log(
                "mesh alignment",
                **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in mesh_stats.items()},
            )
        del meshes
    metrics.stage_seconds["mesh"] = time.perf_counter() - t0

    # ---- warp -----------------------------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("warp", 0.05, f"warping {len(work)} cameras")
    patches = warp_mod.warp_all(
        frames, work, W, H, config, progress=prog, stage="warp", stage_span=(0.05, 1.0),
        displacements=displacements,
    )
    metrics.stage_seconds["warp"] = time.perf_counter() - t0

    # ---- seams ----------------------------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("seam", 0.0, "finding seams")
    hard, feather, backend = seams_mod.find_seams_full(patches, config)
    boundary = seams_mod.seam_boundary_mask(patches, hard, W, H)
    will_refine = config.optical_flow_refine and len(patches) > 1
    # The partition, painted, while the patch images are still alive. Composed here rather
    # than kept as `hard` because one canvas costs 25 MB at 4096 where ninety patch masks
    # cost 120. Skipped under refinement: that path warps the images, so a partition painted
    # from the originals would no longer describe what was blended.
    owner: np.ndarray | None = None
    covered: np.ndarray | None = None
    if config.deghost_structure and not will_refine and len(patches) > 1:
        owner, covered = blend_mod.compose_owner(patches, hard, W, H)
    # `hard` is kept alive for the ghost repair, which needs the partition after the blend.
    hard_for_repair = hard if (config.repair_ghosts and not will_refine) else None
    del hard
    prog.stage("seam", 1.0, backend)
    metrics.stage_seconds["seam"] = time.perf_counter() - t0

    # ---- blend ----------------------------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("blend", 0.0, "multi-band blending")
    # Deferred when a parallax correction is coming. The old code blended, then let the
    # refine stage discard the result and blend again -- a whole multi-band pyramid over a
    # 4096-wide canvas with ninety patches, built and thrown away. Enabling refinement by
    # default made that waste the common case rather than the rare one.
    pano = None if will_refine else blend_mod.multiband_blend(patches, feather, W, H, config)
    if pano is not None and owner is not None and covered is not None:
        # Low frequencies from the cross-fade, high frequencies from the owner. See
        # `blend.deghost_structure` for the measurement this is here for.
        pano = blend_mod.deghost_structure(
            pano, owner, covered, W * blend_mod._DEGHOST_SIGMA_FRACTION
        )
    del owner, covered
    if pano is not None and hard_for_repair is not None:
        # The doubled patches, found and committed to one camera each. Local on purpose --
        # see `blend.repair_ghosted_regions` for why every global version of this failed.
        pano, repaired = blend_mod.repair_ghosted_regions(pano, patches, hard_for_repair, config)
        if repaired > 0.0:
            metrics.warnings.append(
                f"ghost_repair: {100.0 * repaired:.1f}% of the panorama disagreed with every "
                "frame that saw it -- parallax the rotation model cannot remove -- and was "
                "committed to a single camera there instead of averaged"
            )
    hard_for_repair = None
    prog.stage("blend", 1.0, "deferred for parallax refinement" if will_refine else "blended")
    metrics.stage_seconds["blend"] = time.perf_counter() - t0

    # ---- refine ---------------------------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("refine", 0.0, "parallax refinement")
    if will_refine:
        refined = refine_mod.refine_overlaps(patches, config)
        for p in patches:
            p.image = np.empty((0, 0, 3), dtype=np.uint8)
        prog.stage("refine", 0.6, "blending")
        pano = blend_mod.multiband_blend(refined, feather, W, H, config)
        masks_valid = [p.mask for p in refined]
        corners = [p.corner for p in refined]
        del refined
    else:
        masks_valid = [p.mask for p in patches]
        corners = [p.corner for p in patches]
    del feather
    for p in patches:
        p.image = np.empty((0, 0, 3), dtype=np.uint8)
    del patches
    prog.stage("refine", 1.0, "done")
    metrics.stage_seconds["refine"] = time.perf_counter() - t0

    # ---- poles + wrap repair --------------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("poles", 0.0, "filling uncovered regions")
    coverage_canvas = blend_mod.build_canvas_coverage(masks_valid, corners, W, H)
    if config.fill_poles:
        # Fills the caps *and* any hole elsewhere on the sphere.  ``coverage_canvas`` is
        # read-only inside: it is measured again below into the report the agent reads, and
        # a filled pixel must not become a covered one.
        pano = poles_mod.fill_uncovered(pano, coverage_canvas, config)
    # AFTER the fill, not instead of it. The branding used to sit in an `elif`, so it only
    # appeared when pole filling was switched off -- which is never, in any shipped mode, so
    # the mark could not be seen at all. It belongs last either way: it covers the caps the
    # filler had to invent, and there is nothing to gain by filling underneath it first.
    for which, path, span in (
        ("zenith", config.zenith_logo, config.zenith_logo_span_deg),
        ("nadir", config.nadir_logo, config.nadir_logo_span_deg),
    ):
        if path is None:
            continue
        poles_mod.composite_pole_logo(pano, Path(path), which=which, span_deg=float(span))
    prog.stage("poles", 0.7, "repairing the wrap band")
    pano = repair_wrap_band(pano, frames, work, config, displacements=displacements)
    prog.stage("poles", 1.0, "done")
    metrics.stage_seconds["poles"] = time.perf_counter() - t0

    # ---- coverage, metrics, delivery ------------------------------------------------
    t0 = time.perf_counter()
    prog.stage("levels", 0.0, "measuring coverage")
    report = coverage_mod.build_coverage(masks_valid, corners, W, H)
    del masks_valid, corners
    metrics.seam_gradient_energy = seam_gradient_energy(pano, boundary, coverage_canvas)
    del boundary, coverage_canvas
    _LOG.info("metrics: seam_gradient_energy=%.4f", metrics.seam_gradient_energy)

    try:
        coverage_mod.coverage_heatmap_png(report, out_dir / "coverage.png")
    except Exception as exc:  # noqa: BLE001 - a diagnostic image must not fail a render
        _LOG.warning("coverage: heatmap failed (%s)", exc)

    prog.stage("levels", 0.35, "writing levels")
    levels = output_mod.write_levels(pano, out_dir, config)
    cube: dict[str, Path] = {}
    if config.emit_cubemap:
        prog.stage("levels", 0.7, "writing cube faces")
        face = max(256, min(2048, W // 4))
        cube = output_mod.write_cubemap(pano, out_dir, face)
    prog.stage("levels", 0.9, "writing thumbnail")
    thumb = output_mod.write_thumbnail(pano, out_dir / "thumb.jpg")

    meta = {
        "wrap_offset_deg": round(math.degrees(offset), 4),
        "yaw_delta_deg": round(math.degrees(delta), 4),
        "output_width": W,
        "output_height": H,
        "seam_backend": backend,
        "seam_gradient_energy": round(metrics.seam_gradient_energy, 6),
        "exposure_spread": round(metrics.exposure_spread, 6),
        "covered_fraction": round(report.covered_fraction, 6),
        "mesh": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mesh_stats.items()},
    }
    (out_dir / "render_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    prog.stage("levels", 1.0, "done")
    metrics.stage_seconds["levels"] = time.perf_counter() - t0
    metrics.stage_seconds["render_total"] = time.perf_counter() - t_all
    if report.covered_fraction < 0.5:
        metrics.warnings.append(
            f"only {100.0 * report.covered_fraction:.0f}% of the sphere was captured"
        )
    if report.gaps:
        metrics.warnings.append(f"{len(report.gaps)} coverage gap(s) inside the swept band")

    _LOG.info("render: complete in %.1f s", metrics.stage_seconds["render_total"])
    return StitchResult(
        panorama=pano,
        coverage=report,
        poses=work,
        metrics=metrics,
        levels=levels,
        cube_faces=cube,
        thumbnail=thumb,
    )
