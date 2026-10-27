"""Deciding *which* camera owns each pixel before anything is blended.

Blending hides a small photometric step.  It cannot hide a chair that appears twice, a
door frame that bends, or a person who walks between two exposures.  The only defence
against those is to route the boundary between two images *around* the offending content,
which is exactly what a seam finder does: it cuts along a path where the two images
already agree, so the transition lands on a wall or a patch of floor rather than through
the middle of an object.

Three properties matter here and none of them are free:

*Scale.*  Graph cuts are super-linear in pixel count and a full 8K canvas would take
minutes.  The cut is therefore computed at ``seam_scale_width`` and upsampled.  A seam
that is one canvas pixel off is invisible; a seam through a chandelier is not.

*Totality.*  Every covered pixel must belong to exactly one image.  A nearest-neighbour
upsample of the label map plus the per-patch validity mask can orphan a thin rim of
pixels, and an orphaned pixel blends to black.  Orphans are therefore hunted down and
re-attached to the nearest claiming image.

*Softness.*  A hard 0/255 label handed to a multi-band blender makes the top pyramid
level switch abruptly, which rings.  Feathering the labels over roughly ``canvas_w/400``
pixels gives the blender a gradient to work with, and costs nothing.

*Where the cut is allowed to go, not just where it is cheap.*  Agreement alone is a bad
guide in a room whose walls are flat: the cheapest path is one that follows a crown
moulding or a door frame, because a line is self-similar along itself and two views that
are misregistered *along* it still agree pixel for pixel there.  Cutting there is the worst
thing the finder can do, and it is what produced the delivered panorama's headline
artefacts.  :func:`_line_penalty` and :func:`_run_graphcut_edge_aware` add a second,
image-independent term to the cost that keeps the cut off strong coherent lines and stops
it ruling a straight line across a low-texture surface.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .types import PipelineConfig
from .warp import WarpedPatch

__all__ = ["find_seams", "find_seams_full", "seam_boundary_mask", "as_numpy_mask"]

_LOG = logging.getLogger(__name__)


def as_numpy_mask(m) -> np.ndarray:
    """OpenCV 5's seam finders hand back ``cv2.UMat``; every caller needs plain numpy."""
    if isinstance(m, cv2.UMat):
        m = cv2.UMat.get(m)
    arr = np.asarray(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.ascontiguousarray(arr.astype(np.uint8))


# ======================================================================================
# Seam finding back-ends
# ======================================================================================


#: Graph-cut edge cost. COST_COLOR, not COST_COLOR_GRAD: the latter divides the colour
#: disagreement by the gradient magnitude, which is exactly what a *misaligned* edge
#: maximises -- so it routes the cut along the cornice or the wardrobe bar it should be
#: avoiding. Measured on the mesh-aligned reference capture at 2048 px: disagreement of the
#: two views on the seam pixels 16.4 -> 14.8 grey levels, structure on the seams 42.6 ->
#: 36.6, seam_gradient_energy 0.93 -> 0.74, and the cut runs 2.5x faster.
_GRAPHCUT_COST = "COST_COLOR"


def _run_graphcut(images: list[np.ndarray], corners: list[tuple[int, int]], masks: list[np.ndarray]):
    finder = cv2.detail_GraphCutSeamFinder(_GRAPHCUT_COST)
    return finder.find(images, corners, [m.copy() for m in masks])


def _run_dp(images: list[np.ndarray], corners: list[tuple[int, int]], masks: list[np.ndarray]):
    finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    return finder.find(images, corners, [m.copy() for m in masks])


def _voronoi_labels(
    masks: list[np.ndarray], corners: list[tuple[int, int]], cw: int, ch: int
) -> np.ndarray:
    """Distance-transform partition: each pixel goes to whichever image contains it most deeply.

    This is the unconditional fallback.  It never fails, never allocates a graph, and its
    seams -- the medial axis between two footprints -- are at least always as far from
    both frame borders as possible.
    """
    best = np.full((ch, cw), -1.0, dtype=np.float32)
    owner = np.full((ch, cw), -1, dtype=np.int32)
    for idx, (m, (x0, y0)) in enumerate(zip(masks, corners, strict=True)):
        h, w = m.shape[:2]
        x1, y1 = min(cw, x0 + w), min(ch, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = m[: y1 - y0, : x1 - x0]
        padded = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        d = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        region = best[y0:y1, x0:x1]
        take = (sub > 0) & (d > region)
        region[take] = d[take]
        owner[y0:y1, x0:x1][take] = idx
        del padded, d, take
    return owner


# ======================================================================================
# Label bookkeeping
# ======================================================================================


def _labels_from_masks(
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    cw: int,
    ch: int,
) -> np.ndarray:
    """Composite per-image seam masks into one ``int32`` ownership map (-1 = unclaimed)."""
    owner = np.full((ch, cw), -1, dtype=np.int32)
    for idx, (m, (x0, y0)) in enumerate(zip(masks, corners, strict=True)):
        h, w = m.shape[:2]
        x1, y1 = min(cw, x0 + w), min(ch, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = m[: y1 - y0, : x1 - x0] > 0
        region = owner[y0:y1, x0:x1]
        region[sub & (region < 0)] = idx
    return owner


def _repair_orphans(owner: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Re-attach every covered-but-unclaimed pixel to the nearest claiming image."""
    orphan = covered & (owner < 0)
    n_orphan = int(orphan.sum())
    if n_orphan == 0:
        return owner
    claimed = owner >= 0
    if not claimed.any():
        return owner
    # distanceTransform measures distance *to the nearest zero pixel*, so claimed pixels
    # are the zeros.  DIST_LABEL_PIXEL gives every zero pixel its own label, which turns
    # the label image into a "nearest claimed pixel" index.
    src = np.where(claimed, np.uint8(0), np.uint8(255))
    _dist, lab = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    lookup = np.full(int(lab.max()) + 1, -1, dtype=np.int32)
    lookup[lab[claimed]] = owner[claimed]
    owner[orphan] = lookup[lab[orphan]]
    _LOG.debug("seams: re-attached %d orphaned pixels", n_orphan)
    del src, _dist, lab, lookup, orphan, claimed
    return owner


def _split_labels(
    owner: np.ndarray, shapes: list[tuple[int, int]], corners: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Inverse of :func:`_labels_from_masks`."""
    out: list[np.ndarray] = []
    ch, cw = owner.shape[:2]
    for idx, ((h, w), (x0, y0)) in enumerate(zip(shapes, corners, strict=True)):
        m = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = min(cw, x0 + w), min(ch, y0 + h)
        if x1 > x0 and y1 > y0:
            m[: y1 - y0, : x1 - x0] = np.where(owner[y0:y1, x0:x1] == idx, np.uint8(255), np.uint8(0))
        out.append(m)
    return out


# ======================================================================================
# Public API
# ======================================================================================


# Node budget for the graph-cut max-flow solve, in overlap pixels summed over every
# overlapping patch pair. OpenCV's GCGraph carries roughly a hundred bytes per node once
# edges and residuals are counted, so ~8 M nodes is already close to a gigabyte.
#
# This has to be a *pre-flight* estimate rather than a try/except, because exceeding the
# budget does not raise: `cv2.detail_GraphCutSeamFinder.find` allocates inside C++ and
# takes the whole interpreter down with it. A real 101-second three-row capture (78
# keyframes onto a 4096-wide canvas) killed the process with exit code 4 and no
# traceback, which no Python handler can catch.
_GRAPHCUT_NODE_BUDGET = 8_000_000  # noqa: E501 - see the module docstring on scale
#: Time budget: the sum of every pair's overlap, in scaled pixels. OpenCV cuts pair by pair
#: and frees each graph before the next, so the SUM bounds wall clock, not memory; it was
#: previously held to the memory budget, which on a three-row capture refused graph-cut
#: outright and handed the seams to dynamic programming at 512 px.
_GRAPHCUT_TOTAL_BUDGET = 60_000_000
_MIN_SEAM_SCALE_FRACTION = 0.25


def _pairwise_overlap_px(patches: list[WarpedPatch], scale: float) -> tuple[int, int]:
    """``(total, largest)`` overlap area in scaled pixels over every intersecting patch pair.

    ``largest`` is what one max-flow graph is built over -- OpenCV's pairwise finder
    allocates per pair and frees between pairs -- and ``total`` is what the whole stage
    costs in time. Patch *count* alone is a poor predictor of either.
    """
    boxes = [
        (
            p.corner[0] * scale,
            p.corner[1] * scale,
            (p.corner[0] + p.width) * scale,
            (p.corner[1] + p.height) * scale,
        )
        for p in patches
    ]
    total = 0.0
    largest = 0.0
    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            w = min(ax1, bx1) - max(ax0, bx0)
            if w <= 0:
                continue
            h = min(ay1, by1) - max(ay0, by0)
            if h > 0:
                total += w * h
                largest = max(largest, w * h)
    return int(total), int(largest)


def _fit_seam_scale(
    patches: list[WarpedPatch], canvas_w: int, scale: float
) -> tuple[float, bool]:
    """Shrink the seam-search canvas until graph-cut fits the budgets; report if it can't.

    Reducing the scale is strongly preferable to switching backend: the seam only has to
    be topologically right, and graph-cut placing it at half resolution still routes it
    through the same low-gradient region. Dropping to dynamic programming changes where
    the seam goes, not merely how precisely it is drawn.
    """
    total, largest = _pairwise_overlap_px(patches, scale)
    if largest <= _GRAPHCUT_NODE_BUDGET and total <= _GRAPHCUT_TOTAL_BUDGET:
        return scale, True

    floor = scale * _MIN_SEAM_SCALE_FRACTION
    # Both costs are quadratic in scale, so solve directly rather than stepping.
    fitted = scale * min(
        math.sqrt(_GRAPHCUT_NODE_BUDGET / float(max(largest, 1))),
        math.sqrt(_GRAPHCUT_TOTAL_BUDGET / float(max(total, 1))),
    )
    if fitted >= floor:
        _LOG.info(
            "seams: graph-cut overlap %.1f Mpx total / %.1f Mpx largest over budget; seam "
            "canvas %d -> %d px",
            total / 1e6,
            largest / 1e6,
            int(canvas_w * scale),
            int(canvas_w * fitted),
        )
        return fitted, True

    total_floor, largest_floor = _pairwise_overlap_px(patches, floor)
    _LOG.warning(
        "seams: graph-cut needs %.1f Mpx total / %.1f Mpx largest even at the %d px floor; "
        "using dynamic programming instead so the process survives",
        total_floor / 1e6,
        largest_floor / 1e6,
        int(canvas_w * floor),
    )
    return floor, False


#: How much softer than the best available a patch may be and still be allowed to own a
#: region. A ratio, on local high-frequency energy.
#:
#: 0.35 is deliberately permissive. Sharpness measured this way is noisy, and the cost of
#: being wrong is asymmetric: wrongly excluding a patch risks a hole, wrongly including one
#: costs a soft rectangle. Only a patch carrying less than a third of the detail its
#: neighbours have is refused, which on a well-shot capture excludes nobody at all.
_SHARPNESS_FLOOR_RATIO = 0.35

#: Below this absolute detail level the comparison is meaningless — a genuinely blank white
#: wall carries no high frequencies from ANY camera, and dividing noise by noise would start
#: excluding patches at random.
_SHARPNESS_NOISE_FLOOR = 2.0


def _detail_map(small_img: np.ndarray) -> np.ndarray:
    """Local high-frequency energy of a seam-scale patch, as a smooth field.

    The Laplacian's local variance is the same statistic `extract.score_frames` ranks whole
    frames by, computed here per region instead: a motion-blurred frame is not uniformly
    soft, it is soft along the direction it moved and at the scales the blur crossed.
    """
    grey = cv2.cvtColor(np.clip(small_img, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    # Local mean of the squared response, i.e. variance about a locally-zero mean.
    return cv2.blur(lap * lap, (17, 17))


def _mask_out_the_soft(
    small_imgs: list[np.ndarray],
    small_masks: list[np.ndarray],
    small_corners: list[tuple[int, int]],
    cw: int,
    ch: int,
) -> list[np.ndarray]:
    """Refuse a patch the right to own a region a much sharper patch also covers.

    WHY THIS EXISTS. On a real capture the seam finder chose between frames purely on how
    well they agreed, which on a plain ceiling is "equally well" — there is nothing there to
    disagree about. So ownership of large flat areas fell to whichever camera happened to
    win the cut, and roughly a third of the frames in that capture were motion-blurred by a
    40 ms exposure. Measured across the delivered panorama's ceiling, tile to tile, LOCAL
    SHARPNESS varied 7.4x more than brightness did: the rectangular patches a viewer reads
    as a "shift" are not exposure steps at all, they are a sharp frame and a smeared one
    meeting on a surface with no texture to hide the change in grain.

    Agreement cannot see that, because both frames agree about a blank ceiling. Detail can.

    The mask is only ever SHRUNK, and only where another patch still covers the pixel, so
    this cannot open a hole: the worst it can do is hand a region to a neighbour.
    """
    n = len(small_imgs)
    if n < 2:
        return small_masks

    details = [_detail_map(img) for img in small_imgs]

    # Best detail available anywhere on the canvas, and how many patches cover each pixel.
    best = np.zeros((ch, cw), dtype=np.float32)
    cover = np.zeros((ch, cw), dtype=np.int16)
    for det, m, (sx, sy) in zip(details, small_masks, small_corners, strict=True):
        h, w = m.shape[:2]
        sl = (slice(sy, sy + h), slice(sx, sx + w))
        on = m > 0
        np.maximum(best[sl], np.where(on, det, 0.0), out=best[sl])
        cover[sl] += on.astype(np.int16)

    out: list[np.ndarray] = []
    trimmed = 0
    for det, m, (sx, sy) in zip(details, small_masks, small_corners, strict=True):
        h, w = m.shape[:2]
        sl = (slice(sy, sy + h), slice(sx, sx + w))
        local_best = best[sl]
        # Too soft to justify owning this pixel...
        soft = det < np.maximum(local_best * _SHARPNESS_FLOOR_RATIO, _SHARPNESS_NOISE_FLOOR)
        # ...but only where somebody else can take it, and only where there is real detail
        # to be lost. A blank wall nobody resolves stays with whoever had it.
        drop = soft & (cover[sl] > 1) & (local_best > _SHARPNESS_NOISE_FLOOR) & (m > 0)
        if not drop.any():
            out.append(m)
            continue
        # Only coherent soft REGIONS are surrendered. Speckled exclusions carve the
        # eligibility into hundreds of islands (measured: 647 ownership islands with the raw
        # mask against 96 without it on the reference capture), and every island boundary is
        # another seam to hide. An opening at about a degree removes the speckle and keeps
        # the smeared-frame patches the stage exists for.
        k = max(3, int(round(cw / 360.0 * 1.5)) | 1)
        drop = cv2.morphologyEx(drop.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
        if not drop.any():
            out.append(m)
            continue
        kept = m.copy()
        kept[drop] = 0
        # A patch reduced to nothing helps no one; keep the original rather than vanish.
        if int((kept > 0).sum()) < int((m > 0).sum()) // 4:
            out.append(m)
            continue
        trimmed += int(drop.sum())
        out.append(kept)

    if trimmed:
        _LOG.info(
            "seams: excluded %d soft px across %d patches from seam eligibility", trimmed, n
        )
    return out


# ======================================================================================
# Edge-aware seam cost
# ======================================================================================
#
# WHAT THE GRAPH CUT ACTUALLY MINIMISES, and why that is a problem in a bedroom.
#
# OpenCV's `GraphCutSeamFinder` with COST_COLOR gives the edge between neighbouring pixels
# p and q the weight ``d(p) + d(q) + 1``, where ``d`` is the SQUARED colour distance between
# the two images at that pixel (`normL2(Point3f, Point3f)` in the stitching module is the
# squared norm) and 1 is `weight_eps`.  Horizontal and vertical edges use the same formula,
# so the whole cost is a single scalar field ``d`` over the overlap and the cut is a
# minimum-length geodesic through it.  Two consequences follow, and both are visible in the
# delivered panorama:
#
#  * A wall with a moulding on it has ``d`` near zero ON the moulding whenever the two views
#    are displaced ALONG it -- the aperture problem: a line slid along itself looks
#    identical.  So the cheapest path in a flat room runs *down the middle of the crown
#    moulding*, and the two sides hand over a structure that is a few pixels apart
#    perpendicular to the line.  Measured on the reference capture that is one moulding
#    drawn as five overlapping runs across 15 deg of wall (yaw 253-290 in the audited
#    render), a cornice that passes over itself instead of mitring at a corner, and a
#    continuous white door edge cut into detached bars.
#  * Where ``d`` is genuinely flat -- a blackout curtain, a bare ceiling -- the geodesic
#    through a constant field is a straight line, and the finder draws one: a dead-straight
#    horizontal boundary with a 1 DN tone step and a fold pattern that does not continue
#    across it.  A 1 DN step is nothing; a 1 DN step ruled perfectly straight for 30 deg is
#    the first thing a human eye finds.
#
# Neither is a defect in the max-flow.  Both are the honest answer to a cost that only knows
# about agreement.  The fix is a second term that does not depend on the pair at all.
#
# HOW A PAIR-INDEPENDENT TERM IS INJECTED.  There is no API for a cost map: the only lever
# is the pixels handed to `find`.  Because the edge weight is exactly ``d(p) + d(q) + 1``, a
# per-pixel additive penalty ``P(p)`` is equivalent to raising ``d`` by ``P``, and ``d`` is
# ``||I_a(p) - I_b(p)||^2``.  So for ONE PAIR we add ``+a(p)n(p)`` to the first image and
# ``-a(p)n(p)`` to the second, where ``n`` is a unit vector chosen ORTHOGONAL to the pair's
# own colour difference at that pixel.  Orthogonality makes the cross term vanish exactly:
#
#     ||(I_a + a n) - (I_b - a n)||^2 = ||I_a - I_b||^2 + 4 a^2
#
# -- the real colour disagreement is preserved in full, not traded away, and the penalty is
# exact rather than approximate.  This only works two images at a time (for N images there
# is no set of N vectors that is equidistant in three channels), so `find` is driven one
# pair at a time instead of once over the whole set.  That is not a re-implementation:
# OpenCV's `PairwiseSeamFinder::run` is itself a double loop over rect-overlapping pairs
# that mutates the two masks in place, and driving it from Python in the same order over the
# same pairs reproduces its output bit for bit -- `test_seams.py` asserts exactly that with
# the penalty set to zero.
#
# Penalties are quoted in GREY LEVELS so they can be compared against the numbers the rest
# of this file measures in: a penalty of g grey levels costs the same as the two views
# disagreeing by g on every channel, i.e. 3 g^2, so ``4 a^2 = 3 g^2``.


#: Converts a penalty expressed in grey levels into the per-image channel offset that
#: produces it: ``4 a^2 = 3 g^2``.
_PENALTY_TO_CHANNEL = math.sqrt(3.0) / 2.0

#: Cost of cutting on the strongest coherent line in the room, in grey levels.
#:
#: The scale to judge this against is the disagreement the seams already carry: ~15 grey
#: levels per channel on the reference capture (see `_GRAPHCUT_COST`), i.e. ~675 in cost
#: units. 22 grey levels is 1452 -- enough to buy a detour of hundreds of pixels around a
#: moulding, and still well under OpenCV's `terminal_cost_` of 10000, which is what pins a
#: pixel to the only camera that covers it. Pushing past the terminal cost would let the
#: max-flow prefer abandoning a singly-covered pixel over paying the penalty, which shows up
#: as orphan pixels rather than as a better seam.
_STRUCTURE_PENALTY_GREY = 22.0

#: The luminance gradient, in grey levels per seam-scale pixel, at which a line is treated
#: as "as strong as architecture gets". A white moulding against a grey wall runs 40-60 DN
#: over two or three pixels at seam scale; sensor noise on a flat wall runs 1-2. 12 saturates
#: on the former and leaves the latter at a tenth of the penalty.
_STRUCTURE_FULL_GRADIENT_DN = 12.0

#: Structure-tensor integration window, in degrees of yaw, with a hard floor in pixels.
#: Coherence is meaningless at a single pixel -- it needs a neighbourhood long enough to tell
#: a line from a corner and from noise. 1.4 deg is ~8 px at the default seam scale, several
#: moulding widths. The 5 px floor is what keeps the estimator honest on a small canvas:
#: coherence of a 3x3 window is estimated from nine gradients and is high by luck alone.
_STRUCTURE_WINDOW_DEG = 1.4
_STRUCTURE_WINDOW_FLOOR_PX = 5

#: Derivative scale for the structure tensor, in degrees of yaw, floored at 0.8 px.
#:
#: A structure tensor differentiates at one scale and integrates at a larger one; taking a
#: raw Sobel is the degenerate case where the derivative scale is a single pixel, and on a
#: real capture that is dominated by sensor noise. MEASURED on a fixture of a 4 px moulding
#: (65 DN of contrast) beside a patch of sigma-26 noise standing in for a carpet: with a raw
#: Sobel the two scored 25.4 and 20.8 grey levels of penalty -- the carpet was almost as
#: expensive to cut as the cornice, which would have pushed the seam out of the one place it
#: is safe. With pre-smoothing and the squared coherence below, the STRUCTURE term alone
#: scores 20.2 on the moulding and 2.7 on the carpet. 0.2 deg is ~1.1 px at the default seam
#: scale, which is below a moulding's width there and well above the noise correlation length
#: of a demosaiced frame.
_STRUCTURE_DERIVATIVE_DEG = 0.2
_STRUCTURE_DERIVATIVE_FLOOR_PX = 0.8

#: How far either side of a line the penalty reaches is NOT a constant here: it is the
#: coarsest blend band's own reach, ``2 ** config.blend_bands`` canvas pixels, converted to
#: seam pixels by the seam scale. `find_seams_full` computes it and passes it in.
#:
#: WHY IT IS THAT AND NOT SOMETHING SMALLER, measured the hard way. The first version used a
#: fixed 0.6 deg, about 3 seam pixels. It fixed the ruled curtain cut and thinned the crown
#: moulding fan, and it made the cornice corner at yaw 18-29 MEASURABLY WORSE: the two runs
#: went from one faint ghost to a plainly doubled mitre. The reason is that
#: `cv2.detail_MultiBandBlender` builds a pyramid whose coarsest level mixes over
#: ``2 ** blend_bands`` = 64 canvas pixels at the balanced settings. Moving the cut three
#: pixels off a moulding takes it out of the place where the two views agree by the aperture
#: problem and leaves it well inside the zone where the pyramid draws BOTH views' mouldings.
#: The cut has to clear the whole band or it should not move at all.
#:
#: The penalty is spread as a linear CONE, not a plateau: at the blender's reach a flat band
#: would cover 70% of the canvas (measured on the reference capture) and a uniform field
#: steers nothing -- the max-flow would be back to drawing straight lines through it. A cone
#: keeps a gradient everywhere, so the cut settles on the ridge that is furthest from every
#: line it can see, which is the same objective the Voronoi fallback has always used.
_STRUCTURE_CONE_STEPS = 4

#: Cost, in grey levels, of cutting across the smooth part of a textured surface rather than
#: along one of its creases. An eighth of the structure penalty: this term exists to SHAPE a
#: cut that is already free, not to route it.
#:
#: 3, not 6, and the difference is measurable. At 6 the term is strong enough to pull seams
#: out of flat regions and into textured ones, which is not what it is for; on the reference
#: capture that cost `seam_gradient_energy` 0.6955 -> 0.7075 while the ruled step across the
#: curtain came out slightly WORSE (ruledness 1.86 -> 1.96, see below). At 3 it still buys
#: the whole meander: 3 grey levels is 27 cost units per pixel against OpenCV's
#: `weight_eps` of 1, so a free cut still has 27x more reason to follow the field than to go
#: straight.
_RULING_PENALTY_GREY = 3.0

#: The half-period of the meander this buys, in degrees of yaw. It is the radius over which
#: local texture relief is measured, so it sets the distance the cut will stray to reach a
#: crease. 2.5 deg breaks the collinearity a viewer reads as a ruled line while staying far
#: below the 30 deg runs the audit found.
_RULING_WAVELENGTH_DEG = 2.5

#: Texture relief, in grey levels of local Laplacian amplitude, that saturates the ruling
#: term. Measured on the reference capture's blackout curtain, at seam scale, over the
#: region the audit named: the smooth panel between folds runs 3.3 DN (10th percentile) to
#: 4.1 (25th), a crease 9.1 (75th) and up. 4 is the step between the two, so a crease
#: saturates the term and the panel does not.
_RULING_CONTRAST_DN = 4.0

#: Amplitude of the deterministic dither mixed into the texture field before relief is
#: taken, in the same grey levels.
#:
#: WHY A DITHER IS NEEDED AT ALL, and this is the part that is easy to get wrong. Relief in
#: the real texture can only steer the cut ALONG THE DIRECTION IT VARIES IN. A curtain's
#: folds are vertical, so its relief is a function of x alone; a horizontal seam crossing
#: them sees the same cost in every row and has no reason to prefer one over another. The
#: min-cut is then degenerate and max-flow returns one of the many equal-cost paths -- in
#: practice the straight one. Sized to `_RULING_CONTRAST_DN` so a region with no texture at
#: all still gets a full-amplitude maze to thread; where real texture exists it dominates.
_RULING_DITHER_DN = 4.0

#: Fixed seed for that dither. It must be deterministic -- two renders of the same capture
#: have to agree, and a bug reproduced once has to reproduce again -- and the value itself
#: carries no meaning.
_RULING_DITHER_SEED = 0x5EA45

#: Master switch, kept so the whole stage can be A/B'd against the plain cost from a bench
#: script without a config field.
#:
#: WHAT IT BUYS, on the cached three-row bedroom capture rendered at 4096 through the same
#: solve, everything else held fixed:
#:
#:     variant                       seam_gradient_energy   ruled step   doubled lines   s
#:     off (plain graph-cut)                       0.7249         2.44             495  191
#:     structure + ruling (this)                   0.6955         1.86             479  164
#:     structure only, ruling 0                    0.6757         1.99             502  168
#:     ruling only, structure 0                    0.8100         2.33             490  281
#:
#: "ruled step" is the worst row's mean vertical difference divided by the median row's, over
#: the curtain the audit named -- collinearity, independent of how big the step is. "doubled
#: lines" counts pairs of long segments that are parallel, overlapping and 4-45 px apart over
#: 48 rectilinear views: one moulding drawn twice. Lower is better in every column.
#:
#: NOT LISTED, deliberately: `straightness_px`. Re-rendering this configuration with nothing
#: changed but `_RULING_DITHER_SEED` moved it 3.23 -> 3.73, while `seam_gradient_energy` moved
#: 0.7075 -> 0.7089. Every seam in the panorama is re-routed by any change to the cost, and
#: `straightness_px` is a mean over per-view medians of however many Hough segments happen to
#: be detected (33 to 43 across these runs), so it cannot resolve a change of this size. The
#: doubled-line count has a seed-to-seed spread of about 21, so its 495 -> 479 is about one
#: noise width and is reported as suggestive, not proven. The other two are stable.
#:
#: The ruling term costs `seam_gradient_energy` on its own and that is expected rather than
#: alarming: the metric is gradient magnitude measured ON the seam pixels, and a term whose
#: whole purpose is to move the seam into texture where the step is masked must raise it.
#: It is kept because it is the only thing that removes the ruled cut across the curtain, and
#: at 3 grey levels it costs 0.02 of the 0.049 the structure term wins.
_EDGE_AWARE_SEAM_COST = True


def _ruling_dither(cw: int, ch: int, wave: int) -> np.ndarray:
    """A smooth deterministic field in ``[0, 1]`` with blobs about ``wave`` pixels across.

    Built at one sample per ``wave`` pixels and bicubically enlarged rather than blurred
    from white noise: enlarging keeps the full amplitude at the blob scale, whereas blurring
    noise down to that scale costs an order of magnitude of amplitude and would need
    re-normalising.  The field does not wrap at the cyclic canvas edge; the discontinuity
    there is a step in a cost, not in an image, and the wrap-owning camera's seam is
    re-rendered by `render.repair_wrap_band` regardless.
    """
    rng = np.random.default_rng(_RULING_DITHER_SEED)
    gw = max(2, int(round(cw / max(1, wave))))
    gh = max(2, int(round(ch / max(1, wave))))
    coarse = rng.random((gh, gw), dtype=np.float32)
    field = cv2.resize(coarse, (cw, ch), interpolation=cv2.INTER_CUBIC)
    # Bicubic overshoots at a sharp step between two samples; the clip keeps the dither
    # inside the amplitude its constant advertises.
    return np.clip(field, 0.0, 1.0)


def _cone_spread(field: np.ndarray, reach: int, steps: int = _STRUCTURE_CONE_STEPS) -> np.ndarray:
    """Grey dilation by a cone: a peak of ``v`` leaves ``v * (1 - d / reach)`` at distance d.

    OpenCV has no non-flat structuring element, so the cone is built from ``steps`` flat
    dilations at rising radii and falling weights and then smoothed.  The distance is
    Chebyshev rather than Euclidean because a rectangular kernel dilates separably and a
    circular one does not; at these radii the corner error is a fraction of a degree and the
    field is a soft prior, not a boundary.
    """
    if reach < 1:
        return field
    out = field.copy()
    for k in range(1, steps):
        r = max(1, int(round(reach * k / steps)))
        np.maximum(out, cv2.dilate(field, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) * (1.0 - k / steps), out=out)
    smooth = max(3, (reach // steps) | 1)
    return cv2.blur(out, (smooth, smooth))


def _line_penalty(
    small_imgs: list[np.ndarray],
    small_masks: list[np.ndarray],
    small_corners: list[tuple[int, int]],
    cw: int,
    ch: int,
    reach: int,
) -> np.ndarray:
    """Canvas-space cost, in grey levels, of putting a seam pixel here.

    Two terms are summed.

    STRUCTURE, from the structure tensor of each patch's luminance.  A structure tensor was
    chosen over a line-segment detector and over long oriented kernels because it is dense
    (every pixel gets a cost, so the max-flow has a smooth field to descend rather than a
    set of hard obstacles), it costs four box filters per patch, and it separates *how
    strong* a gradient is from *how coherent* it is -- which is the whole distinction we
    need.  A moulding scores on both; sensor noise on a wall scores on neither; a busy
    carpet is strong but incoherent and is deliberately left cheap, because a seam through
    a carpet is invisible and a seam through a cornice is not.  An LSD would have given a
    sparse set of segments with no cost anywhere else, and would have missed the soft
    3 DN cornice shadow entirely.

    The two patches are combined by MAXIMUM, not by average: a line that is sharp in one
    view and smeared in the other is still a line, and averaging would let a motion-blurred
    frame vote the penalty away exactly where it is most needed.

    RULING, the straightness term.  A pairwise graph cut cannot express curvature -- the
    energy has no term that sees three consecutive seam pixels -- so a straightness penalty
    cannot be written directly.  What *can* be written is a cost field whose minimum path is
    not a straight line, because a straight cut is only optimal while the field is uniform.
    So the term is the local RELIEF of a texture field: how much more texture there is
    within `_RULING_WAVELENGTH_DEG` of a pixel than at the pixel itself.  The cut pays to
    cross the smooth panel between two folds and travels free along a crease, which is
    exactly "hide in the folds", and the relief construction guarantees the cheapest place
    in every neighbourhood costs nothing.

    Relief, not the texture level itself, and that distinction is load-bearing: a penalty
    proportional to *flatness* would add a near-constant to every pixel of a flat region,
    which is arithmetically identical to raising `weight_eps` and makes the cut SHORTER, i.e.
    straighter -- the opposite of the intent.  Only the variation may cost anything.

    The texture field carries a deterministic dither before relief is taken; see
    `_RULING_DITHER_DN` for why relief in the real texture alone is not enough.
    """
    win = max(_STRUCTURE_WINDOW_FLOOR_PX, int(round(cw / 360.0 * _STRUCTURE_WINDOW_DEG)))
    sigma = max(_STRUCTURE_DERIVATIVE_FLOOR_PX, cw / 360.0 * _STRUCTURE_DERIVATIVE_DEG)
    struct = np.zeros((ch, cw), dtype=np.float32)
    texture = np.zeros((ch, cw), dtype=np.float32)

    for img, m, (sx, sy) in zip(small_imgs, small_masks, small_corners, strict=True):
        grey = cv2.cvtColor(np.clip(img, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(grey, (0, 0), sigma)
        # 1/8 undoes the 3x3 Sobel kernel's gain, so gx is in grey levels per pixel and the
        # thresholds above can be reasoned about as image contrast rather than as filter units.
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) * 0.125
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) * 0.125
        del smooth
        jxx = cv2.blur(gx * gx, (win, win))
        jyy = cv2.blur(gy * gy, (win, win))
        jxy = cv2.blur(gx * gy, (win, win))
        del gx, gy
        trace = jxx + jyy
        # spread = l1 - l2; coherence = (l1 - l2) / (l1 + l2); l1 = (trace + spread) / 2.
        spread = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
        del jxx, jyy, jxy
        coherence = spread / np.maximum(trace, 1e-6)
        strength = np.sqrt(np.maximum(0.5 * (trace + spread), 0.0))
        del trace, spread
        # SQUARED coherence. A tensor integrated over a finite window has a finite number of
        # independent gradients in it, so isotropic noise still comes out somewhat coherent by
        # luck -- about 0.4 for this window. Squaring leaves a real line (coherence ~1)
        # untouched and drops that floor to 0.16. Measured on the moulding/carpet fixture in
        # test_seams.py, structure term alone: carpet 3.9 -> 2.7 grey levels of penalty, while
        # the moulding stayed at 20.2.
        s = coherence * coherence * np.clip(strength / _STRUCTURE_FULL_GRADIENT_DN, 0.0, 1.0)
        del coherence, strength

        # The same Laplacian statistic `_detail_map` uses, but as an amplitude in grey levels
        # so `_RULING_CONTRAST_DN` can be quoted as image contrast.
        lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
        # The clamp is not paranoia: a box filter over a float32 field of squares returns a
        # value a few ULPs below zero on a perfectly uniform patch, and sqrt of that is NaN,
        # which then propagates through the maximum into the whole penalty map.
        t = np.sqrt(np.maximum(cv2.blur(lap * lap, (win, win)), 0.0))
        del lap, grey

        h, w = m.shape[:2]
        sl = (slice(sy, sy + h), slice(sx, sx + w))
        on = m > 0
        np.maximum(struct[sl], np.where(on, s, 0.0), out=struct[sl])
        np.maximum(texture[sl], np.where(on, t, 0.0), out=texture[sl])
        del s, t, on

    struct = _cone_spread(struct, reach)

    wave = max(1, int(round(cw / 360.0 * _RULING_WAVELENGTH_DEG)))
    texture += _RULING_DITHER_DN * _ruling_dither(cw, ch, wave)
    relief = cv2.dilate(texture, np.ones((2 * wave + 1, 2 * wave + 1), np.uint8)) - texture
    del texture
    ruling = np.clip(relief / _RULING_CONTRAST_DN, 0.0, 1.0)
    del relief

    out = _STRUCTURE_PENALTY_GREY * struct + _RULING_PENALTY_GREY * ruling
    del struct, ruling
    return out.astype(np.float32)


def _orthogonal_bump(a: np.ndarray, b: np.ndarray, grey: np.ndarray) -> np.ndarray:
    """Per-pixel offset to add to ``a`` and subtract from ``b`` to cost ``grey`` grey levels.

    The offset points along a unit vector orthogonal to ``a - b``, which is what makes the
    added cost exactly ``3 * grey**2`` instead of interfering with the colour disagreement
    the finder is there to measure.  Two candidate orthogonals are built by crossing the
    difference with the blue and green axes and the longer one is taken, so the construction
    never degenerates on a grey difference (the common case indoors, and the one that would
    collapse a single fixed choice).
    """
    d = (a - b).astype(np.float32)
    d0, d1, d2 = d[..., 0], d[..., 1], d[..., 2]
    zero = np.zeros_like(d0)
    n1 = np.stack([zero, d2, -d1], axis=-1)  # d x (1, 0, 0)
    n2 = np.stack([-d2, zero, d0], axis=-1)  # d x (0, 1, 0)
    take = (d1 * d1 + d2 * d2) >= (d0 * d0 + d2 * d2)
    n = np.where(take[..., None], n1, n2)
    del d, d0, d1, d2, zero, n1, n2, take
    norm = np.sqrt((n * n).sum(axis=-1))
    # Where the two views agree exactly there is no difference to be orthogonal to; any unit
    # vector is correct to within the rounding of a float32 subtraction.
    flat = norm < 1e-6
    n[flat] = (1.0, 0.0, 0.0)
    n /= np.where(flat, np.float32(1.0), norm)[..., None]
    del norm, flat
    n *= (grey * _PENALTY_TO_CHANNEL).astype(np.float32)[..., None]
    return n


def _run_graphcut_edge_aware(
    images: list[np.ndarray],
    corners: list[tuple[int, int]],
    masks: list[np.ndarray],
    penalty: np.ndarray,
) -> list[np.ndarray]:
    """Graph-cut, driven one pair at a time so the pair-independent penalty can be applied.

    Mirrors ``cv2::detail::PairwiseSeamFinder::run``: the same ``i < j`` order over the same
    rect-overlapping pairs, mutating the two masks in place and carrying them into the next
    pair.  With ``penalty`` all zero this is bit-for-bit the single call it replaces.

    ``images`` is modified in place and restored, rather than copied per pair: a
    canvas-width patch is 25 MB at seam scale and there are hundreds of pairs, so copying
    would move gigabytes for nothing.  The restore is a float32 subtract of the same array,
    accurate to ~1e-5 grey levels.
    """
    finder = cv2.detail_GraphCutSeamFinder(_GRAPHCUT_COST)
    out = [m.copy() for m in masks]
    n = len(images)
    for i in range(n - 1):
        ax0, ay0 = corners[i]
        ah, aw = images[i].shape[:2]
        for j in range(i + 1, n):
            bx0, by0 = corners[j]
            bh, bw = images[j].shape[:2]
            x0, y0 = max(ax0, bx0), max(ay0, by0)
            x1, y1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
            if x1 <= x0 or y1 <= y0:
                continue  # OpenCV's overlapRoi skips these too; skipping keeps parity
            si = (slice(y0 - ay0, y1 - ay0), slice(x0 - ax0, x1 - ax0))
            sj = (slice(y0 - by0, y1 - by0), slice(x0 - bx0, x1 - bx0))
            # Rectangles that touch but masks that do not. OpenCV builds and solves a graph
            # over the whole rectangle anyway, and the cut can change nothing: every pixel
            # of the rectangle is in at most one of the two masks, its terminal weight pins
            # it to that side, and `findInPair` only ever clears a pixel the OTHER mask
            # claims. So skipping is exact (`test_seams` pins it). The views that straddle
            # the canvas edge get canvas-wide windows, so each rect-overlaps every other view
            # while touching a handful: on the reference capture 798 rectangle pairs, of
            # which 380 footprints touch. Together with the crop below, measured at 4096 on
            # its real patches: 133.5 -> 124.6 s, partition identical to the last pixel. A
            # modest saving -- the graphs that remain are the ones that genuinely overlap.
            if not np.any((out[i][si] > 0) & (out[j][sj] > 0)):
                continue
            bump = _orthogonal_bump(images[i][si], images[j][sj], penalty[y0:y1, x0:x1])
            images[i][si] += bump
            images[j][sj] -= bump
            # Hand OpenCV only the overlap plus its `_GC_GAP` border, not the two patches.
            # `find` copies both images whole and runs a Sobel over each before it looks at
            # the overlap -- gradients the COST_COLOR graph never reads -- so a canvas-wide
            # patch paid for its full width on every one of its pairs. The graph itself reads
            # only the overlap and `_GC_GAP` pixels around it, the crops carry exactly those
            # pixels (the overlap of two crops is the overlap of the two patches), and
            # nothing outside the overlap is written: the cut is bit-identical.
            ci, crop_i = _gc_crop(corners[i], images[i].shape, x0, y0, x1, y1)
            cj, crop_j = _gc_crop(corners[j], images[j].shape, x0, y0, x1, y1)
            try:
                res = finder.find(
                    [images[i][crop_i], images[j][crop_j]],
                    [ci, cj],
                    [out[i][crop_i].copy(), out[j][crop_j].copy()],
                )
                out[i][crop_i] = as_numpy_mask(res[0])
                out[j][crop_j] = as_numpy_mask(res[1])
                del res
            finally:
                images[i][si] -= bump
                images[j][sj] += bump
                del bump
    return out


#: The border OpenCV's `GraphCutSeamFinder` builds around every pair's overlap
#: (`findInPair`'s `const int gap = 10`).
_GC_GAP = 10


def _gc_crop(
    corner: tuple[int, int], shape: tuple[int, ...], x0: int, y0: int, x1: int, y1: int
) -> tuple[tuple[int, int], tuple[slice, slice]]:
    """A patch cropped to the overlap ``[x0, x1) x [y0, y1)`` plus the graph's border,
    clipped to the patch: ``(new_corner, slices)``."""
    px0, py0 = corner
    ph, pw = int(shape[0]), int(shape[1])
    cx0, cy0 = max(px0, x0 - _GC_GAP), max(py0, y0 - _GC_GAP)
    cx1, cy1 = min(px0 + pw, x1 + _GC_GAP), min(py0 + ph, y1 + _GC_GAP)
    return (cx0, cy0), (slice(cy0 - py0, cy1 - py0), slice(cx0 - px0, cx1 - px0))


def find_seams_full(
    patches: list[WarpedPatch], config: PipelineConfig
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    """Return ``(hard_masks, feathered_masks, backend_name)`` at full patch resolution.

    ``hard_masks`` is a strict partition of the covered area and is what the seam-quality
    metric measures; ``feathered_masks`` is what the blender consumes.
    """
    n = len(patches)
    if n == 0:
        return [], [], "none"
    canvas_w = patches[0].canvas_w
    canvas_h = patches[0].canvas_h
    if n == 1:
        hard = [patches[0].mask.copy()]
        return hard, [patches[0].mask.copy()], "single"

    scale = min(1.0, float(config.seam_scale_width) / float(canvas_w))
    scale, graphcut_affordable = _fit_seam_scale(patches, canvas_w, scale)
    cw = max(8, int(round(canvas_w * scale)))
    ch = max(4, int(round(canvas_h * scale)))

    small_imgs: list[np.ndarray] = []
    small_masks: list[np.ndarray] = []
    small_corners: list[tuple[int, int]] = []
    for p in patches:
        sw = max(2, int(round(p.width * scale)))
        sh = max(2, int(round(p.height * scale)))
        sx = int(np.clip(round(p.corner[0] * scale), 0, max(0, cw - sw)))
        sy = int(np.clip(round(p.corner[1] * scale), 0, max(0, ch - sh)))
        # 0..255 FLOAT, NOT 0..1. OpenCV's graph-cut adds a constant `weight_eps = 1.0` to
        # every edge and measures disagreement as a squared colour difference: on a 0..1
        # image a full-contrast disagreement weighs ~3 against that epsilon, so the cut
        # degenerates into the shortest path and runs straight through misaligned objects.
        # Measured on a 12 px-misaligned fixture with a smooth detour available: 150/300 rows
        # cut through texture at 0..1, 24/300 at 0..255. OpenCV's own stitcher feeds
        # `convertTo(CV_32F)` with no rescale.
        small_imgs.append(cv2.resize(p.image, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32))
        m = cv2.resize(p.mask, (sw, sh), interpolation=cv2.INTER_AREA)
        small_masks.append(np.where(m > 200, np.uint8(255), np.uint8(0)))
        small_corners.append((sx, sy))

    # Detail-aware ownership, applied BEFORE the cut: the seam finder decides where to cut,
    # this decides who is eligible to be cut to. See `_mask_out_the_soft`.
    if config.prefer_sharp_contributor:
        small_masks = _mask_out_the_soft(small_imgs, small_masks, small_corners, cw, ch)

    requested = str(config.seam_finder).lower()
    order = {
        "graphcut": [("graphcut", _run_graphcut), ("dp", _run_dp)],
        "dp": [("dp", _run_dp), ("graphcut", _run_graphcut)],
        "voronoi": [],
    }.get(requested, [("graphcut", _run_graphcut), ("dp", _run_dp)])

    if not graphcut_affordable:
        # Not a preference — a hard exclusion. See _fit_seam_scale for why this cannot be
        # left to the try/except below.
        order = [(name, fn) for name, fn in order if name != "graphcut"]

    # The edge-aware cost is tried FIRST and the plain graph-cut stays behind it, so a
    # failure in the pairwise driver costs the extra cut and nothing else.
    #
    # It is graph-cut only. Dynamic programming scans column by column and cannot detour
    # around an obstacle at all, so a penalty it cannot route around would only corrupt the
    # colour term it does use.
    if _EDGE_AWARE_SEAM_COST and any(k == "graphcut" for k, _ in order):
        # The cut has to clear the blender's coarsest band or moving it off a line only
        # trades one artefact for another; see the comment on `_STRUCTURE_CONE_STEPS`.
        reach = max(1, int(round((1 << int(np.clip(config.blend_bands, 1, 8))) * scale)))
        penalty = _line_penalty(small_imgs, small_masks, small_corners, cw, ch, reach)
        _LOG.info(
            "seams: line penalty reach %d px, over 4 grey levels on %.1f%% of the canvas, "
            "peak %.1f",
            reach,
            100.0 * float((penalty > 4.0).mean()),
            float(penalty.max()),
        )

        def _edge_aware(imgs, corners_, masks_, _p=penalty):
            return _run_graphcut_edge_aware(imgs, corners_, masks_, _p)

        order = [("graphcut+lines", _edge_aware), *order]

    backend = "voronoi"
    owner: np.ndarray | None = None
    for name, fn in order:
        try:
            result = fn(small_imgs, small_corners, small_masks)
            cut = [as_numpy_mask(m) for m in result]
            if len(cut) != n:
                raise RuntimeError(f"{name} returned {len(cut)} masks for {n} images")
            owner = _labels_from_masks(cut, small_corners, cw, ch)
            backend = name
            del cut, result
            break
        except Exception as exc:  # noqa: BLE001 - any backend failure must be survivable
            _LOG.warning("seams: %s finder failed (%s), falling back", name, exc)
            owner = None
    if owner is None:
        owner = _voronoi_labels(small_masks, small_corners, cw, ch)
    _LOG.info("seams: using %s backend at %dx%d", backend, cw, ch)

    covered_small = np.zeros((ch, cw), dtype=bool)
    for m, (x0, y0) in zip(small_masks, small_corners, strict=True):
        h, w = m.shape[:2]
        x1, y1 = min(cw, x0 + w), min(ch, y0 + h)
        if x1 > x0 and y1 > y0:
            covered_small[y0:y1, x0:x1] |= m[: y1 - y0, : x1 - x0] > 0
    owner = _repair_orphans(owner, covered_small)
    small_cut = _split_labels(owner, [m.shape[:2] for m in small_masks], small_corners)
    del owner, covered_small, small_imgs

    hard: list[np.ndarray] = []
    for p, sm in zip(patches, small_cut, strict=True):
        up = cv2.resize(sm, (p.width, p.height), interpolation=cv2.INTER_NEAREST)
        np.bitwise_and(up, p.mask, out=up)
        hard.append(up)
    del small_cut, small_masks

    _repair_orphans_fullres(patches, hard, canvas_w, canvas_h)
    feather = _feather(patches, hard, canvas_w, canvas_h)
    return hard, feather, backend


def find_seams(patches: list[WarpedPatch], config: PipelineConfig) -> list[np.ndarray]:
    """Feathered per-patch ownership masks, ready to hand to the blender."""
    return find_seams_full(patches, config)[1]


def _repair_orphans_fullres(
    patches: list[WarpedPatch], hard: list[np.ndarray], canvas_w: int, canvas_h: int
) -> None:
    """Second orphan pass at full resolution.

    The nearest-neighbour upsample plus the AND with each patch's validity mask can leave
    a one-pixel rim unclaimed exactly where two footprints meet.  Those pixels are grown
    back into the nearest claiming mask by successive dilations, which is both cheaper and
    more local than another global distance transform on an 8K canvas.
    """
    claimed = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    covered = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for p, m in zip(patches, hard, strict=True):
        x0, y0, x1, y1 = p.rect
        np.bitwise_or(claimed[y0:y1, x0:x1], m, out=claimed[y0:y1, x0:x1])
        np.bitwise_or(covered[y0:y1, x0:x1], p.mask, out=covered[y0:y1, x0:x1])
    orphan = (covered > 0) & (claimed == 0)
    del claimed, covered
    remaining = int(orphan.sum())
    if remaining == 0:
        del orphan
        return
    _LOG.debug("seams: %d full-resolution orphan pixels", remaining)
    for k in (3, 5, 9, 17):
        kernel = np.ones((k, k), np.uint8)
        for p, m in zip(patches, hard, strict=True):
            x0, y0, x1, y1 = p.rect
            sub = orphan[y0:y1, x0:x1]
            if not sub.any():
                continue
            grown = cv2.dilate(m, kernel)
            take = sub & (grown > 0) & (p.mask > 0)
            del grown
            if not take.any():
                del take
                continue
            m[take] = 255
            sub[take] = False
            del take
        if not orphan.any():
            break
    if orphan.any():
        for p, m in zip(patches, hard, strict=True):
            x0, y0, x1, y1 = p.rect
            sub = orphan[y0:y1, x0:x1]
            take = sub & (p.mask > 0)
            m[take] = 255
            sub[take] = False
            del take
    del orphan


#: Feather radius of the blender weights, in canvas pixels, independent of canvas size.
#:
#: This used to be ``canvas_w / 400`` -- 20 px at 8192 -- and that was the doubled edges.
#: ``cv2.detail_MultiBandBlender`` uses the mask it is fed as the *finest* band's weight and
#: only pyramid-downsamples it for the coarser bands, so a 20 px feather cross-fades the
#: highest frequencies over a 32 px zone (1%..99%); any structure the two views place more
#: than a couple of pixels apart inside that zone is drawn twice. The pyramid already does
#: the wide low-frequency cross-fade on its own; the fine band wants a hard hand-over.
#: Measured on a 6 px bar misaligned by 6 px across the seam: hard mask -> one bar with a
#: break, 20 px feather -> one smeared bar at 218/240 contrast; at 12-20 px misalignment the
#: feather gives two bars at partial contrast. Two pixels keeps the hand-over free of the
#: nearest-neighbour label staircase without opening a mixing zone.
_FEATHER_RADIUS_PX = 2


def _feather(
    patches: list[WarpedPatch], hard: list[np.ndarray], canvas_w: int, canvas_h: int
) -> list[np.ndarray]:
    """Blur the hard partition into blender weights, then prove no pixel lost its weight."""
    radius = _FEATHER_RADIUS_PX
    ksize = radius * 2 + 1
    sigma = max(0.8, ksize / 4.0)
    feather: list[np.ndarray] = []
    for p, m in zip(patches, hard, strict=True):
        f = cv2.GaussianBlur(m, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
        # Feathering must never spill outside the pixels this camera actually rendered.
        f[p.mask == 0] = 0
        feather.append(f)

    total = np.zeros((canvas_h, canvas_w), dtype=np.uint16)
    covered = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for p, f in zip(patches, feather, strict=True):
        x0, y0, x1, y1 = p.rect
        total[y0:y1, x0:x1] += f
        np.bitwise_or(covered[y0:y1, x0:x1], p.mask, out=covered[y0:y1, x0:x1])
    starved = (covered > 0) & (total == 0)
    del total, covered
    if starved.any():
        _LOG.debug("seams: restoring %d starved pixels to full weight", int(starved.sum()))
        for p, m, f in zip(patches, hard, feather, strict=True):
            x0, y0, x1, y1 = p.rect
            sub = starved[y0:y1, x0:x1]
            take = sub & (m > 0)
            f[take] = 255
            sub[take] = False
            del take
    del starved
    return feather


def seam_boundary_mask(
    patches: list[WarpedPatch], hard: list[np.ndarray], canvas_w: int, canvas_h: int
) -> np.ndarray:
    """Canvas mask (uint8, 255) of the pixels where ownership changes.

    Used by the seam-quality metric: gradient energy measured *on* these pixels versus the
    panorama as a whole tells you whether the cuts are visible.
    """
    owner = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for idx, (p, m) in enumerate(zip(patches, hard, strict=True)):
        x0, y0, x1, y1 = p.rect
        region = owner[y0:y1, x0:x1]
        region[(m > 0) & (region == 0)] = idx + 1
    grad_x = np.zeros_like(owner, dtype=bool)
    grad_y = np.zeros_like(owner, dtype=bool)
    grad_x[:, :-1] = (owner[:, :-1] != owner[:, 1:]) & (owner[:, :-1] > 0) & (owner[:, 1:] > 0)
    grad_y[:-1, :] = (owner[:-1, :] != owner[1:, :]) & (owner[:-1, :] > 0) & (owner[1:, :] > 0)
    boundary = (grad_x | grad_y).astype(np.uint8) * 255
    del owner, grad_x, grad_y
    return cv2.dilate(boundary, np.ones((3, 3), np.uint8))
