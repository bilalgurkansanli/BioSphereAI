"""Global rotation refinement — the stage that decides whether the seams disappear.

The spanning-tree initialisation in :mod:`biosphere.pano.rotation` produces poses that are
individually plausible and collectively wrong: every hop adds a fraction of a degree of
drift, and by the time the chain gets back round to where it started the accumulated error
is several degrees. A few degrees is invisible in any single pair and catastrophic at the
loop-closure seam. Bundle adjustment is what spreads that error over all the cameras
instead of dumping it in one place.

Design choices that matter
--------------------------
*Residuals in pixels, in both directions.* Minimising the angle between rotation matrices
would be cheaper but weights every pair equally regardless of how many points support it,
and it cannot use the intrinsics. Reprojecting the actual correspondences is the only
objective that matches what the renderer will do. Both directions are included because the
one-directional error is not symmetric under a focal change, and a focal that is too short
can otherwise be traded against a rotation error without penalty.

*Camera 0 is frozen.* The objective is invariant to a global rotation of the world, so the
normal equations are rank-deficient by three without a gauge fix. Freezing one camera is
the cheapest fix and keeps the Jacobian sparsity trivially block-structured.

*Our own Levenberg-Marquardt, on the normal equations, with an analytic Jacobian.* The
stage used to hand SciPy's trust-region solver a finite-difference Jacobian and let its
default ``lsmr`` sub-solver take *inexact* Gauss-Newton steps (no ``tr_options``, so
``atol = btol = 1e-6`` and ``maxiter = n_params``). Measured on the reference 89-camera
capture that meant ~385 iterations per round, every round stopped by the clock, 289-341 s
of bundle time and a result that depended on where the clock stopped (focal 1261.7 in one
run, 1298.9 in the next, same input). The parameter count here is tiny — three per camera
plus a focal — so the normal matrix is a few hundred square and a Cholesky solve is
microseconds; what costs is forming the Jacobian, and that is closed form: the residual's
derivative with respect to a left perturbation of either rotation and to the log-focal is
a handful of ``einsum`` calls. The same capture now converges in about five seconds.

*Local parametrisation.* Each iteration solves for a small rotation ``delta`` per camera
and applies ``R <- exp(delta) R``. There is no rotation-vector branch cut to fall over, no
SO(3) Jacobian chain rule, and the linearisation is always taken at the current pose.

*Rotation averaging first.* Before any pixel residual is minimised, each pair's best
rotation-only fit (Kabsch on the unprojected correspondences) is turned into a relative
rotation measurement and the poses are averaged over the whole graph, loop edges included.
The spanning tree ignores every closing edge by construction; averaging uses them all and
hands the pixel solve an initialisation that is already loop-consistent to well under a
degree. It costs milliseconds.

*An annealed robust loss and a data-driven rejection.* Huber's gradient saturates at the
scale, so a solve that starts several pixels out with a 2 px scale creeps; the first round
therefore uses a scale set from the initial residual median and tightens to the configured
value. The final round hard-rejects correspondences whose residual exceeds
``max(3 * f_scale, 3 * 1.4826 * MAD)`` and refits. The MAD term is the difference between
this and the old fixed ``3 * f_scale``: on a hand-held capture the residual floor is set by
parallax rather than by noise, and a fixed 6 px cutoff was measured to discard 54-61 % of
the correspondences — the whole near field — and then anchor the fit on the far wall.
Both the survivor statistics and the statistics over EVERY correspondence are reported;
the latter is what says how far the capture is from a rotation, which the former hides.

*Residuals in float64.* A float32 pixel coordinate near 1000 px is quantised at ~1e-4 px,
enough to bias a Jacobian; ``_Problem`` stores float64 and the geometry helpers preserve
it. ``test_solver`` pins the dtype path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import structlog

from . import geometry as geom
from .types import Intrinsics, MatchPair, PipelineConfig

__all__ = [
    "bundle_adjust",
    "loop_closure_residual",
    "loop_closure_stats",
    "LoopClosure",
    "finalise_poses",
    "reprojection_errors",
    "pair_rotation_kabsch",
    "rotation_average",
    "MAX_CORRESPONDENCES_PER_PAIR",
]

log = structlog.get_logger(__name__)

#: Per pair, per direction. More than this buys nothing: the pair's rotation has three
#: degrees of freedom and 100 points already over-determine it by a factor of thirty.
MAX_CORRESPONDENCES_PER_PAIR = 100

#: Penalty (in pixel units) for a correspondence that projects behind the camera. Large
#: enough to dominate any real residual, finite so the solver keeps a usable objective.
_INVALID_PENALTY = 1.0e3

#: Kept for callers and tests that reference the finite-difference era; the analytic
#: Jacobian does not use it, the distortion columns (studio mode) still do.
_DIFF_STEP = 1.0e-5

#: Relative cost decrease below which a round is converged. Residuals are pixels and the
#: target is hundredths of one; tighter than this spends iterations on the flat valley a
#: single-row sweep has between focal and yaw spacing.
_SOLVER_TOL = 1.0e-6

#: Floor on a single round's slice of `ba_max_seconds`. An LM iteration on a dense capture
#: is a tenth of a second, so two seconds is a dozen iterations -- enough to make progress
#: rather than return the vector it started with.
_MIN_ROUND_SECONDS = 2.0

#: Multiple of the initial median residual used as the first (loosest) robust scale.
_ANNEAL_START_MULTIPLE = 4.0

#: Each annealing step tightens by this factor.
_ANNEAL_RATIO = 4.0

#: Hard cap on the annealing schedule so a catastrophic initialisation cannot turn into
#: a dozen full solves.
_ANNEAL_MAX_STEPS = 3

#: Iteration cap per round. Convergence normally ends a round in 10-40.
_MAX_ITERATIONS_PER_ROUND = 80

#: Rotation-averaging robust threshold, radians (1.5 deg): a pair whose measured relative
#: rotation disagrees with the averaged poses by more than this is an outlier pair.
_ROTAVG_HUBER_RAD = math.radians(1.5)


# ======================================================================================
# The flattened problem
# ======================================================================================


@dataclass(slots=True)
class _Problem:
    """Every correspondence in the capture, flattened into parallel arrays.

    The residual vector is laid out as four contiguous segments of length ``M`` —
    forward-x, forward-y, backward-x, backward-y — so each pair still owns one contiguous
    row range per segment.
    """

    xi: np.ndarray  # (M,) pixel x in the first camera of each correspondence
    yi: np.ndarray  # (M,) pixel y in the first camera
    xj: np.ndarray  # (M,) the matching pixel x in the second camera
    yj: np.ndarray  # (M,) the matching pixel y
    ci: np.ndarray  # (M,) int32 camera index of the first camera
    cj: np.ndarray  # (M,) int32 camera index of the second camera
    spans: list[tuple[int, int, int, int]]  # per pair: (cam_i, cam_j, start, length)

    @property
    def m(self) -> int:
        return int(self.xi.size)

    @property
    def n_rows(self) -> int:
        return 4 * self.m


def _empty_problem() -> _Problem:
    z = np.zeros(0, dtype=np.float64)
    return _Problem(z, z, z, z, np.zeros(0, np.int32), np.zeros(0, np.int32), [])


def _build_problem(pairs: list[MatchPair], n_cameras: int, seed: int) -> _Problem:
    """Subsample each pair's inliers deterministically and flatten them.

    The subsample is seeded from ``(seed, i, j)`` rather than from a running counter, so
    it depends on a pair's identity instead of its position in the list — a run stays
    reproducible even if pair ordering changes.
    """
    xi: list[np.ndarray] = []
    yi: list[np.ndarray] = []
    xj: list[np.ndarray] = []
    yj: list[np.ndarray] = []
    ci: list[np.ndarray] = []
    cj: list[np.ndarray] = []
    spans: list[tuple[int, int, int, int]] = []
    cursor = 0

    for pair in pairs:
        if not (0 <= pair.i < n_cameras and 0 <= pair.j < n_cameras) or pair.i == pair.j:
            continue
        m = int(pair.pts_i.shape[0])
        if m < 4:
            continue
        pts_i = np.asarray(pair.pts_i, dtype=np.float64).reshape(-1, 2)
        pts_j = np.asarray(pair.pts_j, dtype=np.float64).reshape(-1, 2)
        if m > MAX_CORRESPONDENCES_PER_PAIR:
            rng = np.random.default_rng([int(seed), int(pair.i), int(pair.j)])
            sel = np.sort(rng.choice(m, size=MAX_CORRESPONDENCES_PER_PAIR, replace=False))
            pts_i, pts_j = pts_i[sel], pts_j[sel]
            m = MAX_CORRESPONDENCES_PER_PAIR
        xi.append(pts_i[:, 0])
        yi.append(pts_i[:, 1])
        xj.append(pts_j[:, 0])
        yj.append(pts_j[:, 1])
        ci.append(np.full(m, pair.i, dtype=np.int32))
        cj.append(np.full(m, pair.j, dtype=np.int32))
        spans.append((int(pair.i), int(pair.j), cursor, m))
        cursor += m

    if not spans:
        return _empty_problem()
    return _Problem(
        xi=np.concatenate(xi),
        yi=np.concatenate(yi),
        xj=np.concatenate(xj),
        yj=np.concatenate(yj),
        ci=np.concatenate(ci),
        cj=np.concatenate(cj),
        spans=spans,
    )


def _filter_problem(problem: _Problem, keep: np.ndarray) -> tuple[_Problem, int]:
    """Drop the masked-out correspondences and renumber the pair spans.

    A pair left with fewer than four survivors is dropped entirely: losing almost all of
    a pair's points means the pair itself was wrong, and letting the handful that
    happened to survive anchor it would keep a bad constraint alive under a deceptively
    small residual.
    """
    keep = np.asarray(keep, dtype=bool)
    spans: list[tuple[int, int, int, int]] = []
    pieces: list[np.ndarray] = []
    cursor = 0
    n_dropped = 0
    for i, j, start, length in problem.spans:
        local = keep[start : start + length]
        n_survivors = int(local.sum())
        if n_survivors < 4:
            n_dropped += length
            continue
        n_dropped += length - n_survivors
        pieces.append(np.flatnonzero(local) + start)
        spans.append((i, j, cursor, n_survivors))
        cursor += n_survivors

    if not spans:
        return _empty_problem(), n_dropped
    idx = np.concatenate(pieces)
    filtered = _Problem(
        xi=problem.xi[idx],
        yi=problem.yi[idx],
        xj=problem.xj[idx],
        yj=problem.yj[idx],
        ci=problem.ci[idx],
        cj=problem.cj[idx],
        spans=spans,
    )
    return filtered, n_dropped


# ======================================================================================
# Parameter packing (kept for callers that inspect the global parametrisation)
# ======================================================================================


def _pack(poses: list[np.ndarray], intr: Intrinsics, config: PipelineConfig) -> np.ndarray:
    parts = [geom.rotvec(R) for R in poses[1:]]
    x = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    extra: list[float] = []
    if config.refine_focal:
        extra.append(math.log(max(float(intr.focal), 1e-6)))
    if config.refine_distortion:
        extra.extend([float(intr.k1), float(intr.k2)])
    return np.concatenate([x, np.asarray(extra, dtype=np.float64)])


def _unpack(
    x: np.ndarray,
    R0: np.ndarray,
    n_cameras: int,
    intr: Intrinsics,
    config: PipelineConfig,
) -> tuple[list[np.ndarray], np.ndarray, float, float, float]:
    """Parameter vector -> ``(rotations, K, focal, k1, k2)``."""
    n_rot = 3 * (n_cameras - 1)
    poses = [np.asarray(R0, dtype=np.float64)]
    for c in range(n_cameras - 1):
        poses.append(geom.rodrigues(x[3 * c : 3 * c + 3]))
    cursor = n_rot
    focal = float(intr.focal)
    if config.refine_focal:
        focal = float(math.exp(float(x[cursor])))
        cursor += 1
    k1, k2 = float(intr.k1), float(intr.k2)
    if config.refine_distortion:
        k1, k2 = float(x[cursor]), float(x[cursor + 1])
        cursor += 2
    K = np.array(
        [[focal, 0.0, intr.cx], [0.0, focal, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return poses, K, focal, k1, k2


def _n_intrinsic_params(config: PipelineConfig) -> int:
    return int(config.refine_focal) + 2 * int(config.refine_distortion)


# ======================================================================================
# Residuals
# ======================================================================================


def _residual_vector(
    problem: _Problem,
    poses: list[np.ndarray],
    K: np.ndarray,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Symmetric reprojection residual for every correspondence, in pixels.

    ``R_i.T @ ray`` becomes the einsum ``'mba,mb->ma'`` (contracting R's *first* index)
    and ``R_j @ world`` becomes ``'mab,mb->ma'`` — the stacked-per-correspondence form of
    exactly the products :mod:`biosphere.pano.geometry` documents.

    Every array entering :func:`~biosphere.pano.geometry.unproject_pixels` and
    :func:`~biosphere.pano.geometry.project_rays` is float64, which those functions
    preserve.
    """
    if problem.m == 0:
        return np.zeros(0, dtype=np.float64)
    R = np.stack([np.asarray(P, dtype=np.float64) for P in poses])
    R_i = R[problem.ci]
    R_j = R[problem.cj]

    # Forward: pixel in i -> world ray -> pixel in j.
    rays_i = geom.unproject_pixels(problem.xi, problem.yi, K, k1, k2)
    world = np.einsum("mba,mb->ma", R_i, rays_i)
    cam_j = np.einsum("mab,mb->ma", R_j, world)
    px, py, valid_f = geom.project_rays(cam_j, K, k1, k2)
    fx = np.where(valid_f, px - problem.xj, _INVALID_PENALTY)
    fy = np.where(valid_f, py - problem.yj, _INVALID_PENALTY)

    # Backward: the same the other way round. Not redundant — the one-directional error
    # is not symmetric under a change of focal, so a too-short focal could otherwise be
    # traded against a rotation error at no cost.
    rays_j = geom.unproject_pixels(problem.xj, problem.yj, K, k1, k2)
    world_b = np.einsum("mba,mb->ma", R_j, rays_j)
    cam_i = np.einsum("mab,mb->ma", R_i, world_b)
    qx, qy, valid_b = geom.project_rays(cam_i, K, k1, k2)
    bx = np.where(valid_b, qx - problem.xi, _INVALID_PENALTY)
    by = np.where(valid_b, qy - problem.yi, _INVALID_PENALTY)

    return np.concatenate([fx, fy, bx, by])


def _residuals(
    x: np.ndarray,
    problem: _Problem,
    R0: np.ndarray,
    n_cameras: int,
    intr: Intrinsics,
    config: PipelineConfig,
) -> np.ndarray:
    """Residuals of the global (rotation-vector) parametrisation; kept for diagnostics."""
    poses, K, _focal, k1, k2 = _unpack(x, R0, n_cameras, intr, config)
    return _residual_vector(problem, poses, K, k1, k2)


def reprojection_errors(
    problem: _Problem,
    poses: list[np.ndarray],
    K: np.ndarray,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Per-correspondence pixel error, taken as the worse of the two directions."""
    m = problem.m
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    r = _residual_vector(problem, poses, K, k1, k2)
    fwd = np.hypot(r[0:m], r[m : 2 * m])
    bwd = np.hypot(r[2 * m : 3 * m], r[3 * m : 4 * m])
    return np.maximum(fwd, bwd)


# ======================================================================================
# Analytic Jacobian (local parametrisation)
# ======================================================================================


def _skew(v: np.ndarray) -> np.ndarray:
    """(M,3) -> (M,3,3) cross-product matrices."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def _pixel_jacobian(
    problem: _Problem,
    R: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    refine_focal: bool,
    n_params: int,
    n_rot: int,
) -> sp.csr_matrix:
    """Sparse ``(4M, n_params)`` derivative of :func:`_residual_vector`.

    Columns ``3(c-1) .. 3(c-1)+2`` hold the derivative with respect to a left perturbation
    ``delta_c`` of camera ``c`` (``R_c <- exp(delta_c) R_c``); camera 0 owns no columns.
    Column ``n_rot`` is the log-focal. Distortion is not differentiated here (the caller
    adds finite-difference columns when it refines ``k1, k2``), and the rotation columns
    assume an undistorted pinhole — with distortion refined the Jacobian is slightly
    inexact, which costs Levenberg-Marquardt iterations, not correctness.

    The projection is scale invariant, so the unnormalised ray ``[(x-cx)/f, (y-cy)/f, 1]``
    is used for the derivative; validity follows :func:`_residual_vector`.
    """
    M = problem.m
    ci, cj = problem.ci, problem.cj
    R_i, R_j = R[ci], R[cj]
    f = float(focal)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def _direction(xa, ya, Ra, Rb, ca, cb, seg_x, seg_y):
        ra = np.stack([(xa - cx) / f, (ya - cy) / f, np.ones_like(xa)], axis=-1)
        Rba = np.einsum("mab,mcb->mac", Rb, Ra)  # R_b @ R_a^T per correspondence
        c = np.einsum("mab,mb->ma", Rba, ra)
        cz = c[:, 2]
        valid = cz > 1e-6
        czs = np.where(valid, cz, 1.0)
        px = f * c[:, 0] / czs + cx
        py = f * c[:, 1] / czs + cy
        dpdc = np.zeros((M, 2, 3), dtype=np.float64)
        dpdc[:, 0, 0] = f / czs
        dpdc[:, 0, 2] = -f * c[:, 0] / (czs * czs)
        dpdc[:, 1, 1] = f / czs
        dpdc[:, 1, 2] = -f * c[:, 1] / (czs * czs)
        # Perturbing camera b: c' = exp(delta_b) c  ->  dc/d delta_b = -[c]x
        dp_db = np.einsum("mab,mbc->mac", dpdc, -_skew(c))
        # Perturbing camera a: R_a^T <- R_a^T exp(-delta_a) -> dc/d delta_a = R_ba [ra]x
        dp_da = np.einsum("mab,mbc->mac", dpdc, np.einsum("mab,mbc->mac", Rba, _skew(ra)))
        # log-focal: through the unprojected ray and through the projection scale.
        dra = -ra.copy()
        dra[:, 2] = 0.0
        dp_dlf = np.einsum("mab,mb->ma", dpdc, np.einsum("mab,mb->ma", Rba, dra))
        dp_dlf[:, 0] += px - cx
        dp_dlf[:, 1] += py - cy
        inv = ~valid
        dp_db[inv] = 0.0
        dp_da[inv] = 0.0
        dp_dlf[inv] = 0.0
        m_idx = np.arange(M)
        for comp, seg in ((0, seg_x), (1, seg_y)):
            rbase = seg * M + m_idx
            for cam, block in ((ca, dp_da), (cb, dp_db)):
                use = cam > 0
                if not use.any():
                    continue
                col0 = 3 * (cam[use] - 1)
                for k in range(3):
                    rows.append(rbase[use])
                    cols.append(col0 + k)
                    vals.append(block[use, comp, k])
            if refine_focal:
                rows.append(rbase)
                cols.append(np.full(M, n_rot))
                vals.append(dp_dlf[:, comp])

    _direction(problem.xi, problem.yi, R_i, R_j, ci, cj, 0, 1)
    _direction(problem.xj, problem.yj, R_j, R_i, cj, ci, 2, 3)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(4 * M, n_params),
    )


def _edge_block(
    edges: list[tuple[int, int, np.ndarray, float]],
    R: np.ndarray,
    n_params: int,
    *,
    with_jacobian: bool = True,
) -> tuple[np.ndarray, sp.csr_matrix | None]:
    """Rotation-edge residuals ``log(R_meas^T R_b R_a^T) / sigma`` and their Jacobian.

    ``edges`` are ``(a, b, R_ab_measured, sigma_rad)``; the measurement maps camera ``a``
    to camera ``b``. For small residuals the derivative with respect to ``delta_b`` is
    ``R_meas^T`` and with respect to ``delta_a`` is ``-E`` (with ``E`` the residual
    rotation), both divided by sigma. ``with_jacobian=False`` returns ``(r, None)``.
    """
    if not edges:
        return np.zeros(0, dtype=np.float64), (sp.csr_matrix((0, n_params)) if with_jacobian else None)
    a = np.array([e[0] for e in edges], dtype=np.int64)
    b = np.array([e[1] for e in edges], dtype=np.int64)
    Rm = np.stack([np.asarray(e[2], dtype=np.float64) for e in edges])
    sig = np.array([float(e[3]) for e in edges], dtype=np.float64)
    Rrel = np.einsum("mab,mcb->mac", R[b], R[a])  # R_b R_a^T
    E = np.einsum("mba,mbc->mac", Rm, Rrel)  # Rm^T Rrel
    r = _rotvec_batch(E) / sig[:, None]
    if not with_jacobian:
        return r.ravel(), None
    Jb = np.transpose(Rm, (0, 2, 1)) / sig[:, None, None]
    Ja = -E / sig[:, None, None]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    m_idx = np.arange(len(edges))
    for cam, block in ((a, Ja), (b, Jb)):
        use = cam > 0
        if not use.any():
            continue
        for comp in range(3):
            for k in range(3):
                rows.append(3 * m_idx[use] + comp)
                cols.append(3 * (cam[use] - 1) + k)
                vals.append(block[use, comp, k])
    J = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * len(edges), n_params),
    )
    return r.ravel(), J


def _rotvec_batch(E: np.ndarray) -> np.ndarray:
    """(M,3,3) -> (M,3) rotation vectors (log map), numerically safe near identity."""
    c = np.clip((np.trace(E, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)
    v = np.stack(
        [E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], axis=1
    )
    s = np.sin(th)
    scale = np.where(th < 1e-6, 0.5, th / np.maximum(2.0 * s, 1e-12))
    out = v * scale[:, None]
    big = th > math.pi - 1e-3
    for k in np.flatnonzero(big):
        out[k] = geom.rotvec(E[k])
    return out


# ======================================================================================
# Levenberg-Marquardt
# ======================================================================================


def _solve_damped(A: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Solve the damped normal equations ``A x = g`` (symmetric positive definite).

    By Cholesky through SciPy, NOT ``np.linalg.solve``, and the reason is a measurement, not
    a preference. NumPy's bundled OpenBLAS switches to its threaded path somewhere between a
    90- and a 128-square system, and on the reference laptop (hybrid i5-13450HX, Windows) that
    path costs 114 ms at 128 and 337 ms at 265 -- the size of an 89-camera solve -- against
    0.5 ms on one thread. It was 37.5 of the bundle's 48 seconds on the reference capture:
    112 solves of a 265-square matrix. SciPy's own OpenBLAS build does not fall into it
    (1.1 ms unloaded). The matrix is SPD by construction (``J^T W J`` plus positive damping),
    so Cholesky is also the right algorithm; the two agree to 1e-15 relative.
    """
    import scipy.linalg as sla  # noqa: PLC0415 - scipy is already loaded by scipy.sparse

    try:
        factor = sla.cho_factor(A, check_finite=False)
    except sla.LinAlgError as exc:  # not positive definite at this damping
        raise np.linalg.LinAlgError(str(exc)) from exc
    return sla.cho_solve(factor, g, check_finite=False)


@dataclass(slots=True)
class _LMResult:
    R: np.ndarray  # (n,3,3)
    focal: float
    k1: float
    k2: float
    cost: float
    cost0: float
    iterations: int
    accepted: int
    deadline_hit: bool
    converged: bool


def _robust_lm(
    R0: np.ndarray,
    intr: Intrinsics,
    focal: float,
    k1: float,
    k2: float,
    *,
    problem: _Problem | None,
    sigma_px: float,
    edges: list[tuple[int, int, np.ndarray, float]] | None,
    config: PipelineConfig,
    deadline_s: float,
    max_iter: int = _MAX_ITERATIONS_PER_ROUND,
    on_iteration=None,
) -> _LMResult:
    """Minimise the Huber cost of the sigma-normalised residuals over local rotations."""
    R = np.array(R0, dtype=np.float64, copy=True)
    n = R.shape[0]
    n_rot = 3 * (n - 1)
    refine_focal = bool(config.refine_focal)
    refine_dist = bool(config.refine_distortion) and problem is not None and problem.m > 0
    n_params = n_rot + int(refine_focal) + 2 * int(refine_dist)
    cx, cy = float(intr.cx), float(intr.cy)
    edges = edges or []
    t0 = time.perf_counter()
    lam = 1e-3
    # The lens's own tolerance, as an interval the focal may not leave. See
    # `PipelineConfig.focal_prior_px` for why a one-sided bias is answered with a bound
    # rather than with a penalty term: a single prior residual against thousands of pixel
    # rows cannot move the answer at all (measured: it would need a sigma of 0.07% to
    # compete), while the bound costs nothing and is exactly the statement we can defend --
    # this lens is 26 mm, give or take.
    focal_band: tuple[float, float] | None = None
    prior_px = getattr(config, "focal_prior_px", None)
    if refine_focal and prior_px is not None and float(prior_px) > 0.0:
        k = max(0.5, float(getattr(config, "focal_prior_tolerance", 2.0)))
        sig = max(0.01, float(getattr(config, "focal_prior_sigma", 0.10)))
        focal_band = (float(prior_px) * max(0.1, 1.0 - k * sig), float(prior_px) * (1.0 + k * sig))
        focal = min(max(float(focal), focal_band[0]), focal_band[1])

    def _K(f: float) -> np.ndarray:
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # THE JACOBIAN IS BUILT ONLY FOR A STEP THAT IS KEPT. Each iteration may try several
    # damping values before one lowers the cost, and every try used to pay for the full
    # sparse Jacobian of the point it was only testing -- the most expensive object in the
    # loop, thrown away on every rejection. Residuals decide acceptance; the Jacobian is
    # needed only at the point the next iteration linearises about, so it is formed once
    # per accepted step. Same arithmetic in the same order: the result is bit-identical.
    def residuals(Rc: np.ndarray, f: float, k1c: float, k2c: float):
        parts_r: list[np.ndarray] = []
        pixel_r: np.ndarray | None = None
        if problem is not None and problem.m > 0:
            pixel_r = _residual_vector(problem, [Rc[c] for c in range(n)], _K(f), k1c, k2c)
            parts_r.append(pixel_r / sigma_px)
        if edges:
            parts_r.append(_edge_block(edges, Rc, n_params, with_jacobian=False)[0])
        r = np.concatenate(parts_r) if parts_r else np.zeros(0)
        ar = np.abs(r)
        w = np.where(ar <= 1.0, 1.0, 1.0 / np.maximum(ar, 1e-12))
        cost = float(np.sum(np.where(ar <= 1.0, 0.5 * r * r, ar - 0.5)))
        return r, w, cost, pixel_r

    def jacobian(Rc: np.ndarray, f: float, k1c: float, k2c: float, pixel_r: np.ndarray | None):
        parts_J: list[sp.csr_matrix] = []
        if problem is not None and problem.m > 0:
            J = _pixel_jacobian(problem, Rc, f, cx, cy, refine_focal, n_params, n_rot)
            if refine_dist:
                # Two finite-difference columns for the distortion terms.
                poses = [Rc[c] for c in range(n)]
                col = n_rot + int(refine_focal)
                extra_cols = []
                for which in (0, 1):
                    step = 1e-4
                    d1 = k1c + (step if which == 0 else 0.0)
                    d2 = k2c + (step if which == 1 else 0.0)
                    rd = _residual_vector(problem, poses, _K(f), d1, d2)
                    extra_cols.append((rd - pixel_r) / step)
                J = J.tolil()
                J[:, col] = extra_cols[0][:, None]
                J[:, col + 1] = extra_cols[1][:, None]
                J = J.tocsr()
            parts_J.append(J / sigma_px)
        if edges:
            parts_J.append(_edge_block(edges, Rc, n_params)[1])
        return sp.vstack(parts_J).tocsr() if len(parts_J) > 1 else parts_J[0]

    r, w, cost, pixel_r = residuals(R, focal, k1, k2)
    J = jacobian(R, focal, k1, k2, pixel_r)
    cost0 = cost
    it = accepted = 0
    deadline_hit = converged = False
    for it in range(1, max_iter + 1):
        if time.perf_counter() - t0 > deadline_s:
            deadline_hit = True
            break
        JW = J.multiply(w[:, None]).tocsr()
        A = (J.T @ JW).toarray()
        g = J.T @ (w * r)
        diag = np.maximum(np.diag(A), 1e-9)
        step_ok = False
        rel = 0.0
        for _ in range(12):
            try:
                delta = -_solve_damped(A + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            Rt = R.copy()
            for c in range(1, n):
                Rt[c] = geom.nearest_rotation(geom.rodrigues(delta[3 * (c - 1) : 3 * c]) @ R[c])
            ft = focal * math.exp(float(delta[n_rot])) if refine_focal else focal
            if refine_focal and focal_band is not None:
                ft = min(max(ft, focal_band[0]), focal_band[1])
            col = n_rot + int(refine_focal)
            k1t = k1 + float(delta[col]) if refine_dist else k1
            k2t = k2 + float(delta[col + 1]) if refine_dist else k2
            rt, wt, ct, pixel_rt = residuals(Rt, ft, k1t, k2t)
            if ct < cost:
                rel = (cost - ct) / max(cost, 1e-12)
                R, focal, k1, k2, r, w, cost = Rt, ft, k1t, k2t, rt, wt, ct
                J = jacobian(R, focal, k1, k2, pixel_rt)
                lam = max(lam / 3.0, 1e-9)
                step_ok = True
                accepted += 1
                break
            lam *= 5.0
        if on_iteration is not None:
            on_iteration(it, cost)
        if not step_ok or rel < _SOLVER_TOL:
            converged = True
            break
    return _LMResult(
        R=R, focal=float(focal), k1=float(k1), k2=float(k2), cost=cost, cost0=cost0,
        iterations=it, accepted=accepted, deadline_hit=deadline_hit, converged=converged,
    )


# ======================================================================================
# Pairwise rotations and rotation averaging
# ======================================================================================


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation ``R`` minimising ``|R a - b|`` over unit rays (rows)."""
    Hm = a.T @ b
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def pair_rotation_kabsch(
    pts_i: np.ndarray, pts_j: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, float]:
    """Best rotation-only fit ``camera_i -> camera_j`` of a correspondence set, and its
    median pixel residual.

    Preferred over ``nearest_rotation(K^-1 H K)``: a homography fitted to a hand-held pair
    absorbs parallax in its projective terms and the rotation projected out of it can be
    tens of degrees off (measured 64-142 px of residual on cross-row pairs of the
    reference capture, against 17-32 px for this fit).
    """
    pi = np.asarray(pts_i, dtype=np.float64).reshape(-1, 2)
    pj = np.asarray(pts_j, dtype=np.float64).reshape(-1, 2)
    ri = geom.unproject_pixels(pi[:, 0], pi[:, 1], K)
    rj = geom.unproject_pixels(pj[:, 0], pj[:, 1], K)
    R = _kabsch(ri, rj)
    px, py, _ok = geom.project_rays(ri @ R.T, K)
    err = np.hypot(px - pj[:, 0], py - pj[:, 1])
    return R, float(np.median(err)) if err.size else 0.0


def _kabsch_edges(
    problem: _Problem, K: np.ndarray
) -> list[tuple[int, int, np.ndarray, float]]:
    edges: list[tuple[int, int, np.ndarray, float]] = []
    f = float(K[0, 0])
    for i, j, start, length in problem.spans:
        sl = slice(start, start + length)
        pi = np.stack([problem.xi[sl], problem.yi[sl]], axis=1)
        pj = np.stack([problem.xj[sl], problem.yj[sl]], axis=1)
        R, med = pair_rotation_kabsch(pi, pj, K)
        # Uncertainty of the pair's rotation: its residual in radians, shrunk by the
        # support, floored so a lucky pair cannot dominate.
        sigma = max(math.radians(0.05), 4.0 * (med / f) / math.sqrt(max(length, 4)))
        edges.append((i, j, R, sigma))
    return edges


def rotation_average(
    poses: list[np.ndarray],
    edges: list[tuple[int, int, np.ndarray, float]],
    *,
    deadline_s: float = 10.0,
    max_iter: int = 60,
) -> tuple[list[np.ndarray], _LMResult]:
    """Robust rotation averaging over relative-rotation measurements, camera 0 fixed."""
    R0 = np.stack([np.asarray(P, dtype=np.float64) for P in poses])
    intr = Intrinsics(focal=1.0, cx=0.0, cy=0.0, width=1, height=1)
    cfg = PipelineConfig(refine_focal=False, refine_distortion=False)
    # Sigma-normalise with a Huber at 1.0 <=> Huber at _ROTAVG_HUBER_RAD in angle.
    scaled = [(a, b, Rm, max(float(s), 1e-9) * (1.0 / _ROTAVG_HUBER_RAD) * _ROTAVG_HUBER_RAD)
              for (a, b, Rm, s) in edges]
    # The edge sigma is already in radians; the Huber corner is applied on r/sigma = 1, so
    # rescale sigma so that 1.0 corresponds to _ROTAVG_HUBER_RAD of disagreement for a
    # median pair.
    med_sigma = float(np.median([s for (_a, _b, _R, s) in scaled])) if scaled else 1.0
    factor = _ROTAVG_HUBER_RAD / max(med_sigma, 1e-9)
    scaled = [(a, b, Rm, s * factor) for (a, b, Rm, s) in scaled]
    res = _robust_lm(
        R0, intr, 1.0, 0.0, 0.0, problem=None, sigma_px=1.0, edges=scaled, config=cfg,
        deadline_s=deadline_s, max_iter=max_iter,
    )
    return [res.R[c] for c in range(res.R.shape[0])], res


# ======================================================================================
# Loop closure
# ======================================================================================


#: A pair only measures *loop closure* if it spans a decent arc of the sweep. Below this
#: fraction of the keyframe count a pair is just an ordinary medium-baseline constraint.
LOOP_SPAN_FRACTION = 0.25


@dataclass(slots=True)
class LoopClosure:
    """How far the solved poses sit from the pairwise measurements that close the loop."""

    n_pairs: int  # every pair flagged as a loop closure
    n_span_pairs: int  # those that actually span a quarter-turn or more
    robust_deg: float  # the headline
    p90_deg: float  # inlier-weighted 90th percentile over all loop pairs
    max_deg: float  # worst single pair, kept so a blow-up stays visible
    worst_pair: tuple[int, int] | None = None
    worst_inliers: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_span_pairs": self.n_span_pairs,
            "robust_deg": round(self.robust_deg, 4),
            "p90_deg": round(self.p90_deg, 4),
            "max_deg": round(self.max_deg, 4),
            "worst_pair": self.worst_pair,
            "worst_inliers": self.worst_inliers,
        }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile of ``values`` under ``weights``, without interpolation."""
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = float(w.sum())
    if total <= 0.0:
        return float(np.quantile(v, q))
    cum = np.cumsum(w) / total
    return float(v[int(np.searchsorted(cum, q, side="left"))])


def loop_closure_stats(poses: list[np.ndarray], pairs: list[MatchPair]) -> LoopClosure:
    """Disagreement, in degrees, between long-baseline pairs and the solved poses.

    A pair's measurement is ``R_ij``; the solve says the same quantity is ``R_j R_i^T``.
    How far those disagree is how much the capture had to be bent to close up.

    The headline is the inlier-weighted median over the pairs that actually *span* the
    sweep (a quarter turn or more); the inlier-weighted p90 over every loop-flagged pair is
    the fallback when nothing spans the loop; ``max_deg`` is kept in diagnostics so a
    genuine single-pair blow-up stays visible without reaching the user.
    """
    n = len(poses)
    residuals: list[float] = []
    weights: list[float] = []
    spans: list[int] = []
    ident: list[tuple[int, int]] = []
    for pair in pairs:
        if not pair.is_loop_closure or pair.R is None:
            continue
        if not (0 <= pair.i < n and 0 <= pair.j < n):
            continue
        predicted = poses[pair.j] @ poses[pair.i].T
        residuals.append(math.degrees(geom.angular_distance(np.asarray(pair.R), predicted)))
        weights.append(float(max(pair.n_inliers, 1)))
        spans.append(abs(int(pair.j) - int(pair.i)))
        ident.append((pair.i, pair.j))

    if not residuals:
        return LoopClosure(n_pairs=0, n_span_pairs=0, robust_deg=0.0, p90_deg=0.0, max_deg=0.0)

    values = np.asarray(residuals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    sep = np.asarray(spans, dtype=np.int64)
    worst = int(np.argmax(values))

    span_cut = max(4, int(math.ceil(LOOP_SPAN_FRACTION * n)))
    spanning = sep >= span_cut
    p90 = _weighted_quantile(values, w, 0.9)
    robust = (
        _weighted_quantile(values[spanning], w[spanning], 0.5) if spanning.any() else p90
    )
    return LoopClosure(
        n_pairs=values.size,
        n_span_pairs=int(spanning.sum()),
        robust_deg=robust,
        p90_deg=p90,
        max_deg=float(values.max()),
        worst_pair=ident[worst],
        worst_inliers=int(w[worst]),
    )


def loop_closure_residual(poses: list[np.ndarray], pairs: list[MatchPair]) -> float:
    """The headline loop-closure figure: see :func:`loop_closure_stats`."""
    return loop_closure_stats(poses, pairs).robust_deg


# ======================================================================================
# The solve
# ======================================================================================


def _schedule(problem: _Problem, poses: list[np.ndarray], intr: Intrinsics, config: PipelineConfig) -> list[float]:
    """Annealing schedule for the robust scale, ending at the configured value."""
    target = float(config.ba_f_scale)
    if problem.m == 0:
        return [target]
    err = reprojection_errors(problem, poses, intr.K, intr.k1, intr.k2)
    finite = err[err < _INVALID_PENALTY]
    median = float(np.median(finite)) if finite.size else 0.0
    scale = max(target, _ANNEAL_START_MULTIPLE * median)
    scale = min(scale, target * _ANNEAL_RATIO ** (_ANNEAL_MAX_STEPS - 1))
    schedule: list[float] = []
    while scale > 1.5 * target:
        schedule.append(scale)
        scale /= _ANNEAL_RATIO
    schedule.append(target)
    return schedule


def bundle_adjust(
    poses: list[np.ndarray],
    pairs: list[MatchPair],
    intr: Intrinsics,
    config: PipelineConfig,
    *,
    progress: Any | None = None,
    span: tuple[float, float] = (0.35, 0.9),
    prior_edges: list[tuple[int, int, np.ndarray, float]] | None = None,
) -> tuple[list[np.ndarray], Intrinsics, dict[str, Any]]:
    """Refine every camera rotation (and optionally the intrinsics) against all pairs.

    Returns the refined rotations, the refined intrinsics, and a metrics dictionary
    holding the reprojection statistics (over the survivors *and* over every
    correspondence), the loop-closure residual before and after, and the per-round
    timings.

    ``prior_edges`` are optional extra relative-rotation measurements ``(i, j, R_ij,
    sigma_rad)`` -- a gyroscope track, say -- added to the pixel objective in every round.

    `progress` is not decoration: this used to be the longest call in the pipeline.
    """
    t_start = time.perf_counter()
    n_cameras = len(poses)
    poses = [np.asarray(R, dtype=np.float64) for R in poses]
    metrics: dict[str, Any] = {
        "n_cameras": n_cameras,
        "n_pairs": len(pairs),
        "n_loop_closures": sum(1 for p in pairs if p.is_loop_closure),
        "loop_closure_residual_deg_before": loop_closure_residual(poses, pairs),
    }

    problem = _build_problem(pairs, n_cameras, int(config.seed))
    full_problem = problem
    if n_cameras < 2 or problem.m == 0:
        out_intr = Intrinsics(
            focal=float(intr.focal), cx=intr.cx, cy=intr.cy,
            width=intr.width, height=intr.height, k1=intr.k1, k2=intr.k2,
        )
        metrics.update(
            mean_reprojection_error_px=0.0,
            max_reprojection_error_px=0.0,
            loop_closure_residual_deg=metrics["loop_closure_residual_deg_before"],
            focal_px=float(intr.focal),
            hfov_deg=float(out_intr.hfov_deg),
            n_correspondences=0,
            n_outliers_dropped=0,
            n_rounds=0,
            bundle_deadline_hit=False,
            bundle_budget_seconds=float(config.ba_max_seconds),
            round_seconds=[],
            seconds=time.perf_counter() - t_start,
        )
        log.warning("bundle_adjust.skipped", n_cameras=n_cameras, n_correspondences=problem.m)
        return poses, out_intr, metrics

    deadline = float(config.ba_max_seconds)
    lo, hi = float(span[0]), float(span[1])
    round_seconds: list[float] = []
    hit_deadline = False

    def _left() -> float:
        if deadline <= 0.0:
            return float("inf")
        return deadline - (time.perf_counter() - t_start)

    # ---- rotation averaging -------------------------------------------------------------
    t_round = time.perf_counter()
    edges = _kabsch_edges(problem, intr.K)
    if prior_edges:
        edges = edges + list(prior_edges)
    ra_budget = min(10.0, max(_MIN_ROUND_SECONDS, 0.1 * _left())) if deadline > 0 else 10.0
    averaged, ra = rotation_average(poses, edges, deadline_s=ra_budget)
    loop_avg = loop_closure_stats(averaged, pairs)
    loop_init = loop_closure_stats(poses, pairs)
    # Keep the averaged poses only if they agree with the loop pairs at least as well as
    # the chain did; an averaging run that diverged (a pathological edge set) must not
    # replace a sound initialisation.
    if loop_avg.p90_deg <= loop_init.p90_deg + 1e-9 or ra.cost < ra.cost0:
        poses = averaged
    metrics["rotation_average"] = {
        "iterations": ra.iterations,
        "cost_before": round(ra.cost0, 3),
        "cost_after": round(ra.cost, 3),
        "loop_before_deg": round(loop_init.robust_deg, 4),
        "loop_after_deg": round(loop_avg.robust_deg, 4),
        "seconds": round(time.perf_counter() - t_round, 3),
    }
    if progress is not None:
        progress.stage("bundle", lo + (hi - lo) * 0.05, f"rotation averaging ({n_cameras} views)")

    # ---- pixel rounds --------------------------------------------------------------------
    schedule = _schedule(problem, poses, intr, config)
    plan = [(s, False) for s in schedule] + [(schedule[-1], True)]
    metrics["f_scale_schedule"] = [round(s, 3) for s in schedule]

    R = np.stack(poses)
    focal, k1, k2 = float(intr.focal), float(intr.k1), float(intr.k2)
    n_dropped = 0
    total_rounds = len(plan)
    for round_idx, (f_scale, is_rejection_round) in enumerate(plan):
        t_round = time.perf_counter()
        rounds_left = total_rounds - round_idx
        round_budget = max(_MIN_ROUND_SECONDS, _left() / rounds_left) if deadline > 0 else float("inf")
        base_frac = lo + (hi - lo) * (0.1 + 0.85 * round_idx / total_rounds)
        span_frac = (hi - lo) * 0.85 / total_rounds

        def _report(it: int, cost: float, _base=base_frac, _span=span_frac) -> None:
            if progress is not None and it % 5 == 0:
                progress.stage(
                    "bundle",
                    _base + _span * min(0.95, it / _MAX_ITERATIONS_PER_ROUND),
                    f"refining orientations ({n_cameras} views, {problem.m} correspondences)",
                )

        res = _robust_lm(
            R, intr, focal, k1, k2, problem=problem, sigma_px=float(f_scale),
            edges=list(prior_edges) if prior_edges else None, config=config,
            deadline_s=round_budget, on_iteration=_report,
        )
        R, focal, k1, k2 = res.R, res.focal, res.k1, res.k2
        hit_deadline = hit_deadline or res.deadline_hit
        round_seconds.append(time.perf_counter() - t_round)
        K = np.array([[focal, 0.0, intr.cx], [0.0, focal, intr.cy], [0.0, 0.0, 1.0]])
        poses_now = [R[c] for c in range(n_cameras)]
        err = reprojection_errors(problem, poses_now, K, k1, k2)
        log.info(
            "bundle_adjust.round",
            round=round_idx,
            f_scale=round(float(f_scale), 3),
            rejection_round=is_rejection_round,
            n_correspondences=problem.m,
            iterations=res.iterations,
            accepted=res.accepted,
            converged=res.converged,
            deadline_hit=res.deadline_hit,
            cost=round(res.cost, 3),
            mean_err_px=round(float(err.mean()) if err.size else 0.0, 4),
            median_err_px=round(float(np.median(err)) if err.size else 0.0, 4),
            max_err_px=round(float(err.max()) if err.size else 0.0, 4),
            focal_px=round(float(focal), 2),
            seconds=round(round_seconds[-1], 2),
        )
        if is_rejection_round:
            break
        if round_idx != len(plan) - 2:
            continue  # still annealing; only the final pre-rejection fit prunes

        # Hard-reject what the robust loss merely tolerated, with a cutoff that follows the
        # data: noise-limited captures keep the classic 3 * f_scale, parallax-limited ones
        # keep the near field rather than throwing half the capture away.
        finite = err[err < _INVALID_PENALTY]
        mad = float(np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
        cutoff = max(3.0 * float(config.ba_f_scale), 3.0 * 1.4826 * mad)
        filtered, dropped = _filter_problem(problem, err <= cutoff)
        n_dropped += dropped
        metrics["rejection_cutoff_px"] = round(cutoff, 3)
        if filtered.m == 0:
            log.warning("bundle_adjust.all_correspondences_rejected")
            break
        problem = filtered

    poses_out = [geom.nearest_rotation(R[c]) for c in range(n_cameras)]
    K = np.array([[focal, 0.0, intr.cx], [0.0, focal, intr.cy], [0.0, 0.0, 1.0]])
    all_err = reprojection_errors(problem, poses_out, K, k1, k2)
    every_err = reprojection_errors(full_problem, poses_out, K, k1, k2)
    every_finite = every_err[every_err < _INVALID_PENALTY]

    out_intr = Intrinsics(
        focal=float(focal),
        cx=float(intr.cx),
        cy=float(intr.cy),
        width=int(intr.width),
        height=int(intr.height),
        k1=float(k1),
        k2=float(k2),
    )
    # THE LOOP IS MEASURED IN THE SOLVE'S OWN UNITS, OR IT MEASURES THE FOCAL INSTEAD.
    #
    # `pair.R` is a rotation fitted to pixel correspondences, and how many degrees a pixel
    # displacement is worth depends on the focal it is read at. Until now it kept the value
    # fitted at the PRE-solve focal while the poses it is compared against live at the
    # SOLVED one, so every degree the bundle moved the focal came back as "loop error".
    # Measured on the reference three-row capture: focal 1157 -> 1284 px (11 %), loop
    # residual reported 3.19 deg (p90 4.29) and a `loop_did_not_close` warning; the same
    # poses against pair rotations re-fitted at 1284 px close to 0.83 deg (p90 1.76). The
    # panorama is unaffected -- only the statistic, and the warning it drives, were wrong.
    # So were comparisons made with it between solves that end at different focals.
    from .rotation import relative_rotations  # noqa: PLC0415 - rotation imports this module

    relative_rotations(pairs, out_intr)
    final_loop = loop_closure_stats(poses_out, pairs)
    metrics.update(
        mean_reprojection_error_px=float(all_err.mean()) if all_err.size else 0.0,
        max_reprojection_error_px=float(all_err.max()) if all_err.size else 0.0,
        # Over EVERY correspondence, not just the survivors: this is the number that says
        # how far the capture is from a pure rotation.
        mean_reprojection_error_all_px=float(every_finite.mean()) if every_finite.size else 0.0,
        median_reprojection_error_all_px=float(np.median(every_finite)) if every_finite.size else 0.0,
        p90_reprojection_error_all_px=float(np.percentile(every_finite, 90)) if every_finite.size else 0.0,
        frac_over_6px_all=float(np.mean(every_finite > 6.0)) if every_finite.size else 0.0,
        loop_closure_residual_deg=final_loop.robust_deg,
        loop_closure=final_loop.as_dict(),
        focal_px=float(focal),
        hfov_deg=float(out_intr.hfov_deg),
        k1=float(k1),
        k2=float(k2),
        n_correspondences=int(problem.m),
        n_correspondences_total=int(full_problem.m),
        n_pairs_used=len(problem.spans),
        n_outliers_dropped=int(n_dropped),
        n_rounds=len(round_seconds),
        bundle_deadline_hit=bool(hit_deadline),
        bundle_budget_seconds=float(deadline),
        round_seconds=[round(s, 3) for s in round_seconds],
        seconds=time.perf_counter() - t_start,
    )
    log.info(
        "bundle_adjust.done",
        mean_err_px=round(metrics["mean_reprojection_error_px"], 4),
        max_err_px=round(metrics["max_reprojection_error_px"], 4),
        mean_all_px=round(metrics["mean_reprojection_error_all_px"], 4),
        median_all_px=round(metrics["median_reprojection_error_all_px"], 4),
        frac_over_6px_all=round(metrics["frac_over_6px_all"], 4),
        loop_before_deg=round(metrics["loop_closure_residual_deg_before"], 4),
        loop_after_deg=round(metrics["loop_closure_residual_deg"], 4),
        loop_after_p90_deg=round(final_loop.p90_deg, 4),
        loop_after_max_deg=round(final_loop.max_deg, 4),
        loop_worst_pair=final_loop.worst_pair,
        focal_px=round(float(focal), 2),
        dropped=n_dropped,
        seconds=round(metrics["seconds"], 3),
    )
    return poses_out, out_intr, metrics


# ======================================================================================
# Gauge fixing for the renderer
# ======================================================================================


def finalise_poses(poses: list[np.ndarray], config: PipelineConfig) -> list[np.ndarray]:
    """Put the solved poses into the world frame the renderer expects.

    Two gauge choices, in this order:

    1. Level the horizon, so the equator of the equirectangular canvas is the real
       horizon rather than wherever the phone happened to be tilted.
    2. Rotate about gravity so the *first* keyframe sits at yaw 0. The panorama then opens
       facing the direction the user was pointing when they pressed record, which is the
       only orientation a viewer finds unsurprising.
    """
    if not poses:
        return []
    poses = [np.asarray(R, dtype=np.float64) for R in poses]

    if config.level_horizon:
        C = geom.align_horizon(poses)
        poses = [R @ C.T for R in poses]

    yaw0 = geom.yaw_pitch_roll(poses[0])[0]
    Y = geom.from_yaw_pitch_roll(yaw0, 0.0, 0.0)
    poses = [geom.nearest_rotation(R @ Y.T) for R in poses]

    log.info(
        "finalise_poses",
        n=len(poses),
        levelled=bool(config.level_horizon),
        first_yaw_deg=round(math.degrees(geom.yaw_pitch_roll(poses[0])[0]), 4),
    )
    return poses
