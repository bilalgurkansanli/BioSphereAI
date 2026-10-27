"""Photometric alignment of the source frames before they are ever blended.

A phone re-meters continuously through a sweep.  Between the first and last keyframe of a
single 360 turn it is routine to see a 30 % exposure change plus a white-balance drift,
and lens vignetting adds another 20-30 % falloff from centre to corner.  None of that is
geometry, so no amount of seam finding or blending removes it: a multi-band blender hides
a *step*, but a slow ramp across ten overlapping frames comes out as visible banding, and
vignetting comes out as a row of dark scallops along the horizon.

The fix is the classic Brown & Lowe formulation.  Every overlapping pair contributes a
constraint "the same patch of world must have the same brightness in both frames", the
whole set is a linear system in the per-image gains, and a prior pulls the solution toward
1 so that an unconstrained image cannot run away.  Vignetting is fitted afterwards from
the *residual* radial dependence of the same pairs: two frames see a shared point at two
different image radii, so the ratio of their intensities measures the ratio of the
vignette model at those radii, with the unknown scene radiance cancelling exactly.

Both models are deliberately global and low-order.  A per-frame free-form correction would
fit the noise and produce a panorama whose brightness wobbles frame by frame; a single
radial polynomial shared by every frame cannot.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from . import geometry as g
from .types import CameraPose, Intrinsics
from .warp import WarpedPatch, iter_overlaps

__all__ = [
    "estimate_gains",
    "estimate_vignette",
    "apply_to_poses",
    "exposure_spread",
    "vignette_curve",
]

_LOG = logging.getLogger(__name__)

#: Never trust a pair that shares less than this many pixels -- the mean is pure noise.
_MIN_OVERLAP_PX = 400
#: Cap on the samples examined per pair; means converge long before we run out of pixels.
_MAX_PAIR_SAMPLES = 250_000
#: Pixels outside this window are either sensor-noise dominated or clipped, and a clipped
#: pixel carries no photometric information at all.
_LO, _HI = 6.0, 249.0


# ======================================================================================
# Gains
# ======================================================================================


#: Ceiling on the measured prior. Past this the gains are free enough that a chain of weak
#: overlaps can walk a whole sector into a different exposure, which is a worse artefact than
#: the step it would have removed.
_MAX_SIGMA_G = 0.45


def _pair_means(
    patches: list[WarpedPatch],
) -> list[tuple[int, int, float, np.ndarray, np.ndarray]]:
    """``(i, j, N_ij, mean of i over the overlap, mean of j over the overlap)`` per pair."""
    out: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []
    for ia, ib, ov, both in iter_overlaps(patches, min_pixels=_MIN_OVERLAP_PX):
        a, b = patches[ia], patches[ib]
        area = int(both.size)
        step = max(1, int(math.sqrt(area / _MAX_PAIR_SAMPLES)) + 1) if area > _MAX_PAIR_SAMPLES else 1
        sub = both[::step, ::step]
        n = int(sub.sum())
        if n < _MIN_OVERLAP_PX // max(1, step * step):
            del both, sub
            continue
        ai = a.image[ov.si][::step, ::step]
        bj = b.image[ov.sj][::step, ::step]
        pi = ai[sub].astype(np.float64)
        pj = bj[sub].astype(np.float64)
        # Clipped or near-black pixels lie about the ratio; drop them from both sides.
        keep = (pi.min(axis=1) > _LO) & (pi.max(axis=1) < _HI)
        keep &= (pj.min(axis=1) > _LO) & (pj.max(axis=1) < _HI)
        if int(keep.sum()) < 64:
            keep = np.ones(pi.shape[0], dtype=bool)
        mi = pi[keep].mean(axis=0)
        mj = pj[keep].mean(axis=0)
        # N is the true overlap area, not the subsampled count: it weights the pair.
        out.append((ia, ib, float(int(both.sum())), mi, mj))
        del both, sub, pi, pj, keep
    return out


def _solve_channel(
    pairs: list[tuple[int, int, float, np.ndarray, np.ndarray]],
    n: int,
    channel: int | None,
    current: np.ndarray,
    sn2: float,
    sg2: float,
) -> np.ndarray:
    """One Gauss-Newton style solve of the Brown & Lowe normal equations.

    Differentiating
    ``E = 1/2 sum_{i,j} N_ij [ (g_i I_ij - g_j I_ji)^2/sn^2 + (1 - g_i)^2/sg^2 ]``
    over *ordered* pairs gives the data term twice (once as ``(i,j)`` and once as
    ``(j,i)``) while the prior appears once, hence the factor 2 on the data blocks:

        ``g_k sum_j N_kj (2 I_kj^2/sn^2 + 1/sg^2) - 2 sum_j N_kj I_kj I_jk g_j / sn^2
         = sum_j N_kj / sg^2``
    """
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for i, j, nij, mi, mj in pairs:
        if channel is None:
            Iij, Iji = float(mi.mean()) * current[i], float(mj.mean()) * current[j]
        else:
            Iij, Iji = float(mi[channel]) * current[i], float(mj[channel]) * current[j]
        A[i, i] += nij * (2.0 * Iij * Iij / sn2 + 1.0 / sg2)
        A[j, j] += nij * (2.0 * Iji * Iji / sn2 + 1.0 / sg2)
        A[i, j] -= 2.0 * nij * Iij * Iji / sn2
        A[j, i] -= 2.0 * nij * Iij * Iji / sn2
        b[i] += nij / sg2
        b[j] += nij / sg2
    # An image with no usable overlap leaves an empty row; pin it to unity gain.
    isolated = np.diag(A) <= 0.0
    if isolated.any():
        A[isolated, :] = 0.0
        A[:, isolated] = 0.0
        A[isolated, isolated] = 1.0
        b[isolated] = 1.0
    A[np.diag_indices(n)] += 1e-9 * max(1.0, float(np.abs(A).max()))
    try:
        gc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        _LOG.warning("exposure: singular gain system, falling back to lstsq")
        gc = np.linalg.lstsq(A, b, rcond=None)[0]
    del A, b
    return np.clip(gc, 0.4, 2.5)


def _observed_gain_spread(
    pairs: list[tuple[int, int, float, np.ndarray, np.ndarray]],
) -> float:
    """How much the camera's exposure ACTUALLY moved, read off the overlaps themselves.

    Every overlapping pair reports the same scene radiance twice, so ``log(I_ij / I_ji)`` is
    the exposure difference between those two frames and nothing else -- the scene cancels.
    The robust spread of those log-ratios is therefore a direct measurement of how far the
    gains have to travel, before a single gain is solved for.

    WHY THIS IS NEEDED. ``sigma_g`` is a PRIOR BELIEF about that spread, and the shipped 0.12
    is Brown & Lowe's own figure, tuned in 2007 on handfuls of outdoor photographs shot at a
    fixed exposure. A phone sweeping an interior for three minutes on continuous
    auto-exposure, past a window and into a dark corner, is not that. On a real capture the
    prior carried about a fifth of the total weight and dragged the answer a third of the way
    back towards "no correction at all", leaving 0.10 of exposure spread standing in the
    delivered panorama. That residue is exactly what a viewer sees as flat rectangular
    patches of slightly different brightness across a plain ceiling, where there is no
    texture to hide a step.

    A measurement is a better prior than a decade-old default, and it degrades gracefully: a
    capture whose exposure really was steady measures a small spread and keeps the tight
    prior it deserves.

    MAD rather than a standard deviation, because a few pairs straddling a lamp or a window
    would otherwise set the belief for the whole capture.
    """
    ratios: list[float] = []
    for _, _, _, mi, mj in pairs:
        a = float(np.mean(mi))
        b = float(np.mean(mj))
        if a > 1.0 and b > 1.0:
            ratios.append(math.log(a / b))
    if len(ratios) < 4:
        return 0.0
    r = np.asarray(ratios, dtype=np.float64)
    mad = float(np.median(np.abs(r - np.median(r))))
    return float(1.4826 * mad)


def estimate_gains(
    patches: list[WarpedPatch],
    per_channel: bool = True,
    sigma_n: float = 10.0,
    sigma_g: float = 0.12,
    *,
    iterations: int = 4,
) -> np.ndarray:
    """Brown & Lowe gain compensation.  Returns ``(n, 3)`` multiplicative BGR gains.

    Minimises ``sum_ij N_ij [ (g_i I_ij - g_j I_ji)^2/sigma_n^2 + (1 - g_i)^2/sigma_g^2 ]``
    as a linear system in ``g``, solved exactly per channel with ``np.linalg.solve``.

    It is then solved *again* on the corrected intensities, four times in total.  This
    matters, and is the one place we go beyond the textbook single solve.  The prior is a
    genuine MAP term, not just a rank fix: with ``sigma_g = 0.12`` it asserts that gains
    lie within about 12 % of 1, so a phone that actually swung its exposure by 40 % across
    the sweep comes back shrunk by roughly a third -- measurably, 0.09 of recovered spread
    against 0.19 injected.  Iterating re-applies the same well-conditioned step to the
    *residual* error, which converges on the maximum-likelihood answer wherever overlaps
    constrain it while still pinning an unconstrained image at 1.  In practice the
    difference is about 5 dB of panorama fidelity on a badly metered capture.

    The result is renormalised so the overlap-weighted mean gain is 1: without that, a
    chain of pairwise constraints is free to slide the whole panorama darker.
    """
    n = len(patches)
    gains = np.ones((n, 3), dtype=np.float64)
    if n == 0:
        return gains
    pairs = _pair_means(patches)
    if not pairs:
        _LOG.warning("exposure: no usable overlaps, leaving gains at 1.0")
        return gains

    # The prior is widened to whatever this capture's own overlaps say the exposure did, and
    # never narrowed below the caller's value. See `_observed_gain_spread`.
    observed = _observed_gain_spread(pairs)
    sigma_g_used = max(float(sigma_g), min(observed, _MAX_SIGMA_G))
    if sigma_g_used > float(sigma_g):
        _LOG.info(
            "exposure: overlaps measure %.3f of gain spread, widening the prior %.3f -> %.3f",
            observed,
            float(sigma_g),
            sigma_g_used,
        )

    sn2 = float(sigma_n) ** 2
    sg2 = float(sigma_g_used) ** 2
    weight = np.zeros(n, dtype=np.float64)
    for i, j, nij, _, _ in pairs:
        weight[i] += nij
        weight[j] += nij
    wsum = float(weight.sum())

    channels: tuple[int | None, ...] = (0, 1, 2) if per_channel else (None,)
    solved = np.ones((n, len(channels)), dtype=np.float64)
    for ci, c in enumerate(channels):
        acc = np.ones(n, dtype=np.float64)
        weighted = pairs
        for it in range(max(1, iterations)):
            step = _solve_channel(weighted, n, c, acc, sn2, sg2)
            acc = np.clip(acc * step, 0.25, 4.0)
            if wsum > 0:
                mean_g = float((weight * acc).sum() / wsum)
                if mean_g > 1e-6:
                    acc = acc / mean_g
            # Re-weight the pairs by how well the current gains explain them (Huber on the
            # log-ratio residual, scale from the MAD). A handful of pairs that straddle a
            # lamp, a window or a parallax-misaligned edge disagree by factors of several
            # and, un-weighted, drag their neighbours' gains by up to 10 % -- a flat step on
            # a plain wall that no seam can hide. Measured on the reference capture: 9 of
            # 214 pairs beyond 3 MAD, moving individual gains by up to 0.109.
            resid = []
            for i, j, _nij, mi, mj in pairs:
                a = float(mi.mean()) if c is None else float(mi[c])
                b = float(mj.mean()) if c is None else float(mj[c])
                resid.append(math.log(max(a * acc[i], 1e-6) / max(b * acc[j], 1e-6)))
            r = np.asarray(resid)
            scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.02)
            hub = np.where(np.abs(r) <= 2.0 * scale, 1.0, 2.0 * scale / np.maximum(np.abs(r), 1e-9))
            weighted = [(i, j, nij * float(h), mi, mj) for (i, j, nij, mi, mj), h in zip(pairs, hub)]
            if float(np.abs(step - 1.0).max()) < 1e-4 and it > 0:
                break
        solved[:, ci] = acc

    gains = solved if per_channel else np.repeat(solved[:, :1], 3, axis=1)
    return np.clip(gains, 0.25, 4.0)


# ======================================================================================
# Vignette
# ======================================================================================


def vignette_curve(coeffs: tuple[float, float, float], r: np.ndarray | float) -> np.ndarray:
    """Evaluate ``1 + a1 r^2 + a2 r^4 + a3 r^6`` (``r`` normalised to 1 at the corner)."""
    a1, a2, a3 = (float(v) for v in coeffs)
    r2 = np.asarray(r, dtype=np.float64) ** 2
    return 1.0 + r2 * (a1 + r2 * (a2 + r2 * a3))


def _image_radius(
    pose: CameraPose, canvas_w: int, canvas_h: int, us: np.ndarray, vs: np.ndarray
) -> np.ndarray:
    """Normalised image radius at which ``pose`` observes the given canvas pixels."""
    intr: Intrinsics = pose.intrinsics
    theta = (2.0 * np.pi * (us.astype(np.float64) + 0.5) / canvas_w - np.pi).astype(np.float32)
    phi = (0.5 * np.pi - np.pi * (vs.astype(np.float64) + 0.5) / canvas_h).astype(np.float32)
    cp = np.cos(phi)
    rays = np.stack([cp * np.sin(theta), -np.sin(phi), cp * np.cos(theta)], axis=-1)
    cam = rays @ np.asarray(pose.R, dtype=np.float32).T
    x, y, _ = g.project_rays(cam, intr.K, intr.k1, intr.k2)
    rn = math.hypot(intr.width * 0.5, intr.height * 0.5)
    dx = x.astype(np.float64) - intr.cx
    dy = y.astype(np.float64) - intr.cy
    return np.sqrt(dx * dx + dy * dy) / max(rn, 1e-6)


def estimate_vignette(
    patches: list[WarpedPatch],
    gains: np.ndarray | None = None,
    *,
    seed: int = 1729,
    max_samples_per_pair: int = 4000,
) -> tuple[float, float, float]:
    """Fit one radial falloff model shared by every frame.

    For a world point seen by both frames, ``g_i I_i / g_j I_j == V(r_i) / V(r_j)``: the
    scene radiance cancels.  Taking logs makes that linear in the coefficients, so the
    whole thing is a single least-squares problem over every sampled pixel pair.  The fit
    is then clamped to stay inside ``[0.5, 1.3]`` over ``r in [0, 1]`` -- an over-confident
    vignette model burns a bright halo into the middle of every frame, which is far worse
    than leaving the mild real falloff in place.
    """
    if len(patches) < 2:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    gains_arr = np.ones((len(patches), 3)) if gains is None else np.asarray(gains, dtype=np.float64)

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ia, ib, ov, both in iter_overlaps(patches, min_pixels=_MIN_OVERLAP_PX):
        a, b = patches[ia], patches[ib]
        ys, xs = np.nonzero(both)
        del both
        if ys.size > max_samples_per_pair:
            pick = rng.choice(ys.size, size=max_samples_per_pair, replace=False)
            ys, xs = ys[pick], xs[pick]
        x0, y0 = ov.rect[0], ov.rect[1]
        us = xs.astype(np.float64) + x0
        vs = ys.astype(np.float64) + y0

        ai = a.image[ov.si][ys, xs].astype(np.float64)
        bj = b.image[ov.sj][ys, xs].astype(np.float64)
        keep = (ai.min(axis=1) > _LO) & (ai.max(axis=1) < _HI)
        keep &= (bj.min(axis=1) > _LO) & (bj.max(axis=1) < _HI)
        if int(keep.sum()) < 32:
            continue
        li = (ai[keep] * gains_arr[ia][None, :]).mean(axis=1)
        lj = (bj[keep] * gains_arr[ib][None, :]).mean(axis=1)
        ri = _image_radius(a.pose, a.canvas_w, a.canvas_h, us[keep], vs[keep])
        rj = _image_radius(b.pose, b.canvas_w, b.canvas_h, us[keep], vs[keep])
        # Only pairs sampled at genuinely different radii carry vignette information.
        useful = np.abs(ri - rj) > 0.03
        useful &= (li > _LO) & (lj > _LO) & (ri < 1.35) & (rj < 1.35)
        if int(useful.sum()) < 32:
            continue
        ri, rj = ri[useful], rj[useful]
        d = np.log(li[useful]) - np.log(lj[useful])
        rows.append(
            np.stack([ri**2 - rj**2, ri**4 - rj**4, ri**6 - rj**6], axis=1)
        )
        targets.append(d)

    if not rows:
        _LOG.info("vignette: not enough radially separated samples, skipping")
        return (0.0, 0.0, 0.0)
    A = np.concatenate(rows, axis=0)
    y = np.concatenate(targets, axis=0)
    if A.shape[0] < 200:
        _LOG.info("vignette: only %d samples, skipping", A.shape[0])
        return (0.0, 0.0, 0.0)

    # Robust fit. The per-pixel log-ratios are dominated by misalignment and parallax --
    # measured on the reference capture the radial model explained 0.3 % of their variance
    # and 12.5 % of samples were gross outliers -- and plain least squares on that returned
    # a NON-MONOTONIC curve (0.959 at r = 0.7, 1.018 at the corner) that was then stamped
    # as a ring into every frame. Huber IRLS with the scale from the MAD lets the majority
    # decide; a mild ridge keeps r^4 and r^6, nearly collinear on [0, 1], from fighting.
    w = np.ones(A.shape[0], dtype=np.float64)
    coeffs = np.zeros(3)
    for _ in range(6):
        Aw = A * w[:, None]
        ATA = Aw.T @ A + 1e-3 * np.eye(3) * float(A.shape[0])
        ATy = Aw.T @ y
        try:
            coeffs = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            return (0.0, 0.0, 0.0)
        if not np.all(np.isfinite(coeffs)):
            return (0.0, 0.0, 0.0)
        res = y - A @ coeffs
        scale = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 1e-3)
        w = np.where(np.abs(res) <= 1.5 * scale, 1.0, 1.5 * scale / np.maximum(np.abs(res), 1e-9))

    # The fit describes log V; the renderer applies ``1 + a1 r^2 + a2 r^4 + a3 r^6``.
    # Convert rather than mislabel: for a strong falloff the two differ by up to 15 % at
    # the corner, which came out as a bright halo on every frame.
    r = np.linspace(0.0, 1.0, 101)
    basis = np.stack([r**2, r**4, r**6], axis=1)
    V = np.exp(basis @ coeffs)
    coeffs = np.linalg.lstsq(basis, V - 1.0, rcond=None)[0]

    # Physics check: a lens vignette darkens toward the corner and never brightens again.
    # A fit that fails this is describing a lamp or a window inside the window, not the
    # lens, and is discarded rather than applied.
    curve = vignette_curve(tuple(coeffs), r)
    if coeffs[0] > 1e-4 or np.any(np.diff(curve) > 1e-3):
        _LOG.info(
            "vignette: fit is not a monotone falloff (a=(%.4f, %.4f, %.4f)), leaving it out",
            *coeffs,
        )
        return (0.0, 0.0, 0.0)

    coeffs = _clamp_vignette(coeffs)
    _LOG.info("vignette: a=(%.4f, %.4f, %.4f) from %d samples", *coeffs, A.shape[0])
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))


def _clamp_vignette(coeffs: np.ndarray, lo: float = 0.5, hi: float = 1.3) -> np.ndarray:
    """Scale the fit toward "no vignette" until the curve stays inside ``[lo, hi]``."""
    r = np.linspace(0.0, 1.0, 101)
    scale = 1.0
    for _ in range(40):
        v = vignette_curve(tuple(coeffs * scale), r)
        if float(v.min()) >= lo and float(v.max()) <= hi:
            break
        scale *= 0.8
        if scale < 1e-3:
            return np.zeros(3)
    return coeffs * scale


# ======================================================================================
# Write-back
# ======================================================================================


def apply_to_poses(
    poses: list[CameraPose],
    gains: np.ndarray,
    vignette: tuple[float, float, float] | None = None,
) -> None:
    """Store the solved corrections on the poses so :mod:`warp` picks them up."""
    for i, pose in enumerate(poses):
        if i < len(gains):
            pose.gain = np.asarray(gains[i], dtype=np.float64).reshape(3).copy()
        if vignette is not None:
            pose.vignette = (float(vignette[0]), float(vignette[1]), float(vignette[2]))


def exposure_spread(gains: np.ndarray) -> float:
    """Std of the per-image luminance gain -- how hard the compensator had to work."""
    gains = np.asarray(gains, dtype=np.float64)
    if gains.size == 0:
        return 0.0
    return float(np.std(gains.mean(axis=1)))
