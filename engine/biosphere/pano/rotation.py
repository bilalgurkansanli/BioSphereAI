"""From verified homographies to a first guess at focal length and camera orientations.

Bundle adjustment is a local optimiser: it polishes, it does not search. Everything here
exists to hand it a starting point inside the basin of the right answer.

Self-calibration from homographies
----------------------------------
Under pure rotation the homography between two views is ``H = K_j R K_i^-1``. Because
``R`` is orthonormal, the entries of ``H`` are over-determined given ``K``, and the
constraints can be solved for the focal length in closed form — this is the classic
Szeliski / Hartley autocalibration used by OpenCV's ``autocalib.cpp``. Each pair yields
two independent estimates (one per image). Any single one is fragile: it divides by
quantities that vanish whenever the relative rotation is close to a pure pan about the
optical axis, and it inherits the homography's noise. The *median* over every pair in the
capture, however, is both cheap and very stable, which is why we never trust one pair.

Spanning-tree initialisation
----------------------------
Chaining relative rotations accumulates drift, so the chain we pick matters. Traversing a
*maximum* spanning tree — weights being the AutoStitch pair confidence — minimises the
number of low-confidence hops between any camera and the reference, which keeps the
initial poses close enough that the bundle adjuster converges in a handful of iterations
instead of wandering. The tree is only ever a starting point: the loop-closure edges it
necessarily leaves out are exactly what BA then uses to distribute the residual drift.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import structlog

from . import geometry as geom
from .types import (
    InsufficientOverlapError,
    Intrinsics,
    MatchPair,
)

__all__ = [
    "focal_from_homography",
    "estimate_focal",
    "relative_rotations",
    "initial_poses",
    "MIN_HFOV_DEG",
    "MAX_HFOV_DEG",
]

log = structlog.get_logger(__name__)

_EPS = 1e-9

#: Sanity envelope for the self-calibrated field of view. Anything outside is an
#: estimation failure rather than an exotic lens, and clamping beats propagating nonsense
#: into the solve.
#:
#: These are deliberately wide. The bound is there to catch a *degenerate* estimate, not
#: to encode what a phone looks like, and a bound tight enough to do the latter starts
#: biting real hardware: a portrait capture measures its horizontal field across the
#: short edge, which for a 26 mm-equivalent main camera is about 43 degrees. The previous
#: 45-degree floor clamped a genuine 45.0-degree capture — it happened to land on the true
#: value there, but only by luck, and on a 42-degree phone it would have injected a real
#: error that the solve would then have to fight.
MIN_HFOV_DEG = 30.0
MAX_HFOV_DEG = 110.0


# ======================================================================================
# Self-calibration
# ======================================================================================


def _pick_focal_square(v1: float | None, v2: float | None, d1: float, d2: float) -> float | None:
    """Choose between the two focal-square candidates as ``autocalib.cpp`` does.

    When both constraints are usable they should agree; where they disagree the one whose
    denominator was larger in magnitude is the better conditioned of the two, so it wins.
    """
    ok1 = v1 is not None and v1 > 0.0
    ok2 = v2 is not None and v2 > 0.0
    if ok1 and ok2:
        return float(v1) if abs(d1) > abs(d2) else float(v2)  # type: ignore[arg-type]
    if ok1:
        return float(v1)  # type: ignore[arg-type]
    if ok2:
        return float(v2)  # type: ignore[arg-type]
    return None


def focal_from_homography(
    H: np.ndarray,
    size_i: tuple[int, int],
    size_j: tuple[int, int],
) -> tuple[float | None, float | None]:
    """Estimate ``(focal_i, focal_j)`` in pixels from a pure-rotation homography.

    ``size_*`` are ``(width, height)``. The estimator assumes the principal point sits at
    the origin, so the homography is first conjugated into centred coordinates:
    ``H' = T_j^-1 H T_i`` with ``T = [[1,0,cx],[0,1,cy],[0,0,1]]``. Returns ``None`` for a
    component whose constraint was degenerate.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return None, None

    def _T(size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return np.array([[1.0, 0.0, w / 2.0], [0.0, 1.0, h / 2.0], [0.0, 0.0, 1.0]])

    Hc = np.linalg.inv(_T(size_j)) @ H @ _T(size_i)
    if not np.isfinite(Hc).all():
        return None, None
    h = Hc.ravel()

    # --- focal of the second image ---------------------------------------------------
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1 if abs(d1) > _EPS else None
    v2 = (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2 if abs(d2) > _EPS else None
    vj = _pick_focal_square(v1, v2, d1, d2)

    # --- focal of the first image ----------------------------------------------------
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
    v1 = -(h[2] * h[5]) / d1 if abs(d1) > _EPS else None
    v2 = (h[5] ** 2 - h[2] ** 2) / d2 if abs(d2) > _EPS else None
    vi = _pick_focal_square(v1, v2, d1, d2)

    fi = math.sqrt(vi) if vi is not None and vi > 0.0 else None
    fj = math.sqrt(vj) if vj is not None and vj > 0.0 else None
    if fi is not None and not math.isfinite(fi):
        fi = None
    if fj is not None and not math.isfinite(fj):
        fj = None
    return fi, fj


def estimate_focal(
    pairs: list[MatchPair],
    width: int,
    height: int,
    prior_px: float | None = None,
) -> float:
    """Median self-calibrated focal in pixels, clamped to a plausible phone field of view.

    Falls back to ``prior_px`` (typically EXIF) and finally to a 67-degree assumption.
    The path taken is logged because a wrong focal is the single most common cause of a
    panorama that stitches "almost" right, and it is invisible in any later metric.
    """
    estimates: list[float] = []
    for pair in pairs:
        if pair.H is None:
            continue
        fi, fj = focal_from_homography(pair.H, (width, height), (width, height))
        for f in (fi, fj):
            if f is not None and math.isfinite(f) and f > 0.0:
                estimates.append(float(f))

    if estimates:
        focal = float(np.median(estimates))
        source = "homography"
    elif prior_px is not None and prior_px > 0.0:
        focal = float(prior_px)
        source = "prior"
    else:
        focal = float(Intrinsics.from_hfov(width, height, 67.0).focal)
        source = "default"

    hfov = math.degrees(2.0 * math.atan((width / 2.0) / focal))
    clamped = False
    if hfov > MAX_HFOV_DEG:
        focal = float(Intrinsics.from_hfov(width, height, MAX_HFOV_DEG).focal)
        clamped = True
    elif hfov < MIN_HFOV_DEG:
        focal = float(Intrinsics.from_hfov(width, height, MIN_HFOV_DEG).focal)
        clamped = True

    log.info(
        "estimate_focal",
        source=source,
        n_estimates=len(estimates),
        focal_px=round(focal, 2),
        hfov_deg=round(math.degrees(2.0 * math.atan((width / 2.0) / focal)), 2),
        clamped=clamped,
        prior_px=None if prior_px is None else round(float(prior_px), 2),
    )
    return focal


# ======================================================================================
# Relative and absolute rotations
# ======================================================================================


def relative_rotations(pairs: list[MatchPair], intr: Intrinsics) -> None:
    """Fill in ``pair.R`` (camera_i -> camera_j) for every pair.

    Fitted directly to the correspondences (Kabsch on the unprojected rays) whenever a pair
    carries enough of them, and only projected out of the homography as a fallback. The
    homography of a hand-held pair absorbs parallax in its projective terms, and the
    rotation nearest to ``K^-1 H K`` inherits that: measured on the cross-row pairs of the
    reference capture it left 64-142 px of residual where the direct fit left 17-32 px.
    ``pair.R`` seeds the spanning tree and is what the loop-closure statistic compares
    against, so both were being judged against a distorted measurement.
    """
    from .bundle import pair_rotation_kabsch  # noqa: PLC0415 - avoid a circular import

    K = intr.K
    n_ok = n_fit = 0
    for pair in pairs:
        if pair.pts_i is not None and pair.pts_i.shape[0] >= 8:
            pair.R, _med = pair_rotation_kabsch(pair.pts_i, pair.pts_j, K)
            n_ok += 1
            n_fit += 1
            continue
        if pair.H is None:
            pair.R = None
            continue
        pair.R = geom.rotation_from_homography(pair.H, K, K)
        n_ok += 1
    log.info("relative_rotations", n_pairs=len(pairs), n_with_rotation=n_ok, n_fitted=n_fit)


def initial_poses(n: int, pairs: list[MatchPair], intr: Intrinsics) -> list[np.ndarray]:
    """Chain relative rotations outward from the best-connected camera.

    Raises :class:`InsufficientOverlapError` naming the stranded frames when the pose
    graph is not connected — that is a capture problem (the user jumped, or covered a
    blank wall) and no amount of optimisation can invent the missing constraint.
    """
    if n <= 0:
        raise InsufficientOverlapError("no keyframes to solve")
    poses: list[np.ndarray | None] = [None] * n
    if n == 1:
        return [np.eye(3)]

    usable = [p for p in pairs if p.R is not None and 0 <= p.i < n and 0 <= p.j < n and p.i != p.j]
    if not usable:
        raise InsufficientOverlapError(
            "no frame pair produced a usable relative rotation; the sweep has no overlap"
        )

    adjacency: dict[int, list[tuple[float, int, np.ndarray]]] = {i: [] for i in range(n)}
    total_conf = np.zeros(n, dtype=np.float64)
    for p in usable:
        R = np.asarray(p.R, dtype=np.float64)
        adjacency[p.i].append((float(p.confidence), p.j, R))
        adjacency[p.j].append((float(p.confidence), p.i, R.T))
        total_conf[p.i] += float(p.confidence)
        total_conf[p.j] += float(p.confidence)

    ref = int(np.argmax(total_conf))
    poses[ref] = np.eye(3)

    # Prim over a *maximum* spanning tree: negate the weight for the min-heap.
    heap: list[tuple[float, int, int]] = []
    edge_lookup: dict[tuple[int, int], np.ndarray] = {}
    for conf, other, R in adjacency[ref]:
        heapq.heappush(heap, (-conf, ref, other))
        edge_lookup[(ref, other)] = R
    visited = {ref}
    while heap:
        neg_conf, src, dst = heapq.heappop(heap)
        if dst in visited:
            continue
        R_rel = edge_lookup[(src, dst)]
        base = poses[src]
        assert base is not None  # src is visited, so it has a pose
        poses[dst] = geom.nearest_rotation(R_rel @ base)
        visited.add(dst)
        for conf, other, R in adjacency[dst]:
            if other not in visited:
                heapq.heappush(heap, (-conf, dst, other))
                edge_lookup[(dst, other)] = R

    missing = [i for i in range(n) if poses[i] is None]
    if missing:
        raise InsufficientOverlapError(
            "keyframes "
            + ", ".join(str(i) for i in missing[:12])
            + (" ..." if len(missing) > 12 else "")
            + f" could not be linked to the rest of the capture (reference frame {ref}, "
            + f"{len(visited)}/{n} frames connected)"
        )

    log.info(
        "initial_poses",
        n=n,
        reference=ref,
        n_edges=len(usable),
        hfov_deg=round(intr.hfov_deg, 2),
    )
    return [np.asarray(R, dtype=np.float64) for R in poses]  # type: ignore[arg-type]
