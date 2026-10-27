"""Local mesh alignment: bend each view a little so the overlaps agree.

Why this exists
---------------
A hand-held sweep is not a rotation about the lens. The phone rides on an arm, the arm on
a body that shifts its weight, and between the level row and the rows above and below it
the camera has usually moved by a few centimetres. Measured on the reference three-row
capture, the best *rotation-only* alignment of the verified feature pairs leaves a median
of 4 px of disagreement, 10 px between rows, and up to 30 px on the pairs that tie the
sweep back together -- residual parallax that no choice of rotations can remove, because
near and far surfaces genuinely shifted by different amounts. At the canvas that is the
doubled wardrobe bars, the smeared chair and the cornice that bobs up and down between the
row above and the row below.

The global rotation solve therefore stops at "as consistent as a rotation can be", and
this module takes over *locally*: every view gets a coarse mesh of displacement vectors on
the equirectangular canvas, and the meshes of all views are solved *jointly* so that
matching content of overlapping views lands on the same canvas pixel. Three terms compete:

* **data** -- a correspondence between views ``i`` and ``j`` says the displaced canvas
  positions must coincide, either in both directions (a feature match) or only across an
  edge (a block on a cornice or a door frame, where the shift *along* the edge is
  unobservable). Robust (Huber), because a match on a chair leg at one metre cannot agree
  with the wall behind it and the majority should win.
* **prior** -- each vertex prefers zero displacement, so a view with no correspondences,
  or a region far from any overlap, stays exactly where the rotation put it.
* **stiffness** -- first differences (neighbouring vertices move together) and second
  differences (the displacement is locally affine). An affine warp maps straight lines to
  straight lines, so this is the term that keeps door frames straight; it is what the
  earlier dense optical-flow attempt (``refine.py``) lacked and why that bent geometry.

Two passes. The first uses the solver's verified SIFT matches, which are precise but live
only where there are corners. The second re-warps the views at a reduced scale with that
first solution and block-matches every overlap by phase correlation, which sees the
things SIFT does not -- the cornice, the skirting, a plain door frame -- and turns each
block into an edge-aware constraint. Everything is linear in the vertex displacements, so
each pass is one sparse least-squares system re-weighted a handful of times.

Displacements are in canvas pixels and are consumed by
:func:`biosphere.pano.warp.warp_camera`, which samples the source at ``(u - du, v - dv)``
instead of ``(u, v)``.

Coordinates
-----------
A mesh vertex ``(r, c)`` of a view sits at canvas *index* coordinates
``(u0 + c * cell, v0 + r * cell)`` where ``(u0, v0)`` is the view's footprint window. The
data term and the warp both evaluate the grid by bilinear interpolation at
``(u - u0) / cell``; keeping that single convention in one place is what makes the two
agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import structlog

from . import geometry as geom
from .types import CameraPose, MatchPair, PipelineConfig

__all__ = [
    "ViewMesh",
    "Constraints",
    "build_meshes",
    "sparse_constraints",
    "dense_constraints",
    "solve_meshes",
    "canvas_coords",
]

log = structlog.get_logger(__name__)

#: Hard cap on correspondences taken from any one pair, so a 1500-inlier pair does not
#: out-vote the twenty pairs around it.
_MAX_PER_PAIR = 250

#: Normalised-correlation peak below which a block match is not believed.
_MIN_RESPONSE = 0.55

#: Ratio of structure-tensor eigenvalues below which a block is an edge, not a corner, and
#: only the shift across the edge is trusted.
_EDGE_RATIO = 0.12

#: Structure-tensor energy (mean squared gradient) below which a block is blank.
_MIN_TEXTURE = 4.0


@dataclass(slots=True)
class ViewMesh:
    """The displacement grid of one view, over its canvas footprint window."""

    view: int  # position in the render's pose list
    u0: int
    v0: int
    w: int
    h: int
    cell: float
    d: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 2), np.float32))

    @property
    def gw(self) -> int:
        return int(self.d.shape[1])

    @property
    def gh(self) -> int:
        return int(self.d.shape[0])

    def max_abs(self) -> float:
        return float(np.abs(self.d).max()) if self.d.size else 0.0

    def scaled(self, s: float) -> tuple[np.ndarray, float, tuple[int, int]]:
        """The displacement in the form :mod:`warp` takes, for a canvas ``s`` times as wide."""
        return (self.d * np.float32(s), self.cell * s, (int(round(self.u0 * s)), int(round(self.v0 * s))))


@dataclass(slots=True)
class Constraints:
    """Alignment constraints in rigid-canvas coordinates, one row per (point, direction).

    Row ``k`` says ``n_k . (d_i(p_i) - d_j(p_j)) = n_k . (p_j - p_i)`` with weight ``w_k``,
    where ``d`` is a view's displacement field. Every array has the same length.
    """

    vi: np.ndarray
    vj: np.ndarray
    ui: np.ndarray
    vi_: np.ndarray
    uj: np.ndarray
    vj_: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    w: np.ndarray
    #: Groups rows that belong to one point so the robust loss sees one residual per point.
    group: np.ndarray

    @property
    def n(self) -> int:
        return int(self.vi.size)

    @classmethod
    def empty(cls) -> Constraints:
        z = np.zeros(0)
        return cls(z.astype(np.int64), z.astype(np.int64), z, z, z, z, z, z, z, z.astype(np.int64))

    @classmethod
    def concat(cls, parts: list[Constraints]) -> Constraints:
        parts = [p for p in parts if p.n > 0]
        if not parts:
            return cls.empty()
        groups: list[np.ndarray] = []
        base = 0
        for p in parts:
            _, dense = np.unique(p.group, return_inverse=True)
            groups.append(dense + base)
            base += int(dense.max()) + 1 if dense.size else 0
        return cls(
            *[np.concatenate([getattr(p, f) for p in parts]) for f in
              ("vi", "vj", "ui", "vi_", "uj", "vj_", "nx", "ny", "w")],
            np.concatenate(groups),
        )


def build_meshes(
    poses: list[CameraPose], canvas_w: int, canvas_h: int, *, cell: float
) -> list[ViewMesh]:
    """One zero mesh per view, over the same window :mod:`warp` will render into."""
    from .warp import footprint_window  # noqa: PLC0415 - avoid a circular import at load

    meshes: list[ViewMesh] = []
    for k, pose in enumerate(poses):
        u0, v0, w, h = footprint_window(pose, canvas_w, canvas_h)
        gw = int(math.ceil(w / cell)) + 1
        gh = int(math.ceil(h / cell)) + 1
        meshes.append(ViewMesh(view=k, u0=u0, v0=v0, w=w, h=h, cell=float(cell),
                               d=np.zeros((gh, gw, 2), dtype=np.float32)))
    return meshes


def canvas_coords(
    pose: CameraPose, pts: np.ndarray, canvas_w: int, canvas_h: int, u0: int
) -> tuple[np.ndarray, np.ndarray]:
    """Source pixels of ``pose``'s frame -> canvas index coordinates, unwrapped near ``u0``.

    ``u`` is returned in ``[u0 - W/2, u0 + W/2)`` so a footprint that straddles the cyclic
    seam still addresses one continuous window.
    """
    intr = pose.intrinsics
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    rays = geom.unproject_pixels(pts[:, 0], pts[:, 1], intr.K, intr.k1, intr.k2)
    world = rays @ np.asarray(pose.R, dtype=np.float64)  # R.T @ ray, row-vector form
    u, v = geom.equirect_from_rays(world, canvas_w, canvas_h)
    u = u.astype(np.float64)
    u = u0 + np.mod(u - u0 + canvas_w / 2.0, canvas_w) - canvas_w / 2.0
    return u, v.astype(np.float64)


def _bilinear(mesh: ViewMesh, u: np.ndarray, v: np.ndarray):
    """Vertex indices and weights of the four cell corners around each canvas point."""
    gx = np.clip((u - mesh.u0) / mesh.cell, 0.0, mesh.gw - 1.000001)
    gy = np.clip((v - mesh.v0) / mesh.cell, 0.0, mesh.gh - 1.000001)
    c0 = np.floor(gx).astype(np.int64)
    r0 = np.floor(gy).astype(np.int64)
    fx = gx - c0
    fy = gy - r0
    idx = np.stack([r0 * mesh.gw + c0, r0 * mesh.gw + c0 + 1,
                    (r0 + 1) * mesh.gw + c0, (r0 + 1) * mesh.gw + c0 + 1], axis=1)
    wts = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    return idx, wts


def _inside(mesh: ViewMesh, u: np.ndarray, v: np.ndarray, slack: float = 1.0) -> np.ndarray:
    return (
        (u >= mesh.u0 - slack) & (u <= mesh.u0 + mesh.w + slack)
        & (v >= mesh.v0 - slack) & (v <= mesh.v0 + mesh.h + slack)
    )


def _unwrap_to(u_ref: np.ndarray, u: np.ndarray, canvas_w: int) -> np.ndarray:
    """``u`` re-expressed on the same turn as ``u_ref``."""
    return u_ref + geom.wrap_pi((u - u_ref) * (2.0 * np.pi / canvas_w)) * (canvas_w / (2.0 * np.pi))


# ======================================================================================
# Constraint sources
# ======================================================================================


def sparse_constraints(
    meshes: list[ViewMesh],
    poses: list[CameraPose],
    pairs: list[MatchPair],
    canvas_w: int,
    canvas_h: int,
    *,
    view_of_frame: dict[int, int] | None = None,
    seed: int = 1729,
    weight: float = 1.0,
) -> Constraints:
    """The solver's verified feature matches, projected onto the canvas.

    ``pairs`` index *frames*; ``view_of_frame`` maps a frame index onto a position in
    ``poses`` (identity when omitted). Pairs touching a frame that is not rendered are
    ignored -- the render thins the solver's views and only the composited ones matter.
    """
    n_views = len(meshes)
    vof = view_of_frame if view_of_frame is not None else {k: k for k in range(n_views)}
    rng = np.random.default_rng(seed)
    parts: list[Constraints] = []
    for pair in pairs:
        vi, vj = vof.get(int(pair.i)), vof.get(int(pair.j))
        if vi is None or vj is None or vi == vj:
            continue
        pts_i, pts_j = pair.pts_i, pair.pts_j
        m = int(pts_i.shape[0])
        if m < 4:
            continue
        if m > _MAX_PER_PAIR:
            sel = np.sort(rng.choice(m, size=_MAX_PER_PAIR, replace=False))
            pts_i, pts_j = pts_i[sel], pts_j[sel]
        mi, mj = meshes[vi], meshes[vj]
        # Unwrap about the window's CENTRE, not its left edge. A view that straddles the
        # canvas edge gets a full-width window at u0 = 0, and unwrapping about u0 put the
        # half of its content that sits near u = W at negative u, where `_inside` dropped
        # it: the seam-straddling view was being aligned on one side only, and the wrap
        # column showed the difference.
        ui, vi_ = canvas_coords(poses[vi], pts_i, canvas_w, canvas_h, int(round(mi.u0 + mi.w / 2.0)))
        uj, vj_ = canvas_coords(poses[vj], pts_j, canvas_w, canvas_h, int(round(mj.u0 + mj.w / 2.0)))
        ok = _inside(mi, ui, vi_) & _inside(mj, uj, vj_)
        k = int(ok.sum())
        if k == 0:
            continue
        # Two isotropic rows per point: x and y.
        rep = np.repeat
        parts.append(Constraints(
            vi=np.full(2 * k, vi), vj=np.full(2 * k, vj),
            ui=rep(ui[ok], 2), vi_=rep(vi_[ok], 2), uj=rep(uj[ok], 2), vj_=rep(vj_[ok], 2),
            nx=np.tile([1.0, 0.0], k), ny=np.tile([0.0, 1.0], k),
            w=np.full(2 * k, float(weight)), group=rep(np.arange(k), 2),
        ))
    return Constraints.concat(parts)


def _block_shift(
    a: np.ndarray, b_search: np.ndarray, margin: int
) -> tuple[float, float, float]:
    """Sub-pixel shift that moves ``a``'s content onto ``b``'s, plus the match confidence.

    Normalised cross-correlation of ``a``'s centre against a search window of ``b`` that
    extends ``margin`` pixels beyond ``a``'s footprint on every side. Chosen over phase
    correlation because a block that contains one edge and nothing else -- the cornice, a
    door frame -- has a correlation *ridge* rather than a peak: phase correlation reports a
    near-zero response and throws the block away, whereas the ridge's maximum still gives
    the shift across the edge, which is the only component the caller uses for such blocks.
    """
    k = a.shape[0] // 4
    t = a[k : a.shape[0] - k, k : a.shape[1] - k]
    res = cv2.matchTemplate(b_search, t, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    px, py = int(loc[0]), int(loc[1])
    dx = px - (margin + k)
    dy = py - (margin + k)
    # Parabolic refinement, one axis at a time.
    fx = fy = 0.0
    if 0 < px < res.shape[1] - 1:
        l, c, r = res[py, px - 1], res[py, px], res[py, px + 1]
        den = l - 2 * c + r
        if abs(den) > 1e-9:
            fx = float(np.clip(0.5 * (l - r) / den, -1.0, 1.0))
    if 0 < py < res.shape[0] - 1:
        u, c, d = res[py - 1, px], res[py, px], res[py + 1, px]
        den = u - 2 * c + d
        if abs(den) > 1e-9:
            fy = float(np.clip(0.5 * (u - d) / den, -1.0, 1.0))
    return float(dx + fx), float(dy + fy), float(peak)


def dense_constraints(
    patches: list,
    meshes: list[ViewMesh],
    scale: float,
    *,
    block: int = 32,
    stride: int = 16,
    max_shift: float = 10.0,
    weight: float = 0.5,
    edge_weight: float = 1.0,
    wide_views: set[int] | None = None,
    wide_shift: float | None = None,
    only_view: int | None = None,
) -> Constraints:
    """Block-match every overlap of the reduced-scale warp and turn each block into a row.

    ``only_view`` restricts the matching to the overlaps that involve that one view. The
    rows it returns are exactly the rows the full call returns for that view, in the same
    order -- it only skips the pairs a caller interested in one view would throw away.

    ``patches`` are :class:`~biosphere.pano.warp.WarpedPatch` objects rendered at
    ``scale`` times the full canvas width *with* the meshes' current displacement applied;
    ``meshes`` are those displacements (full-canvas units). The returned constraints are in
    full-canvas rigid coordinates, i.e. what :func:`solve_meshes` expects.

    Per block the shift is measured by phase correlation; the block's structure tensor
    decides how much of it is believed: a corner-like block constrains both directions, an
    edge-like block (cornice, skirting, door frame) only the direction across the edge --
    the shift along an edge is the aperture problem and would only add noise.
    """
    from .warp import iter_overlaps  # noqa: PLC0415

    parts: list[Constraints] = []
    inv = 1.0 / float(scale)
    # A view placed by interpolation (see ``solve._bridge_track_gaps``) can sit a few
    # degrees from its neighbours, beyond ``max_shift``; its overlaps get ``wide_shift``.
    # Only its overlaps: measured on the reference capture, widening every pair to 24 px
    # let plain-wall noise in everywhere (mesh residual p90 3.2 -> 9.6 px, line
    # straightness 3.91 -> 4.23 px), while the bridged views' cornice did straighten.
    wide_views = wide_views or set()
    wide = float(wide_shift) if wide_shift is not None else float(max_shift)
    greys = [cv2.cvtColor(p.image, cv2.COLOR_BGR2GRAY).astype(np.float32) for p in patches]
    canvas_w_small = int(patches[0].canvas_w) if patches else 0
    n_blocks = n_kept = n_edge = 0
    for ia, ib, ov, both in iter_overlaps(patches, min_pixels=block * block, only=only_view):
        pair_shift = wide if (ia in wide_views or ib in wide_views) else float(max_shift)
        margin = int(math.ceil(pair_shift))
        ga = greys[ia][ov.si]
        gb = greys[ib][ov.sj]
        h, w = both.shape
        ma, mb = meshes[ia], meshes[ib]
        # Current (pass-1) displacement of each view at reduced scale, to convert block
        # centres back to rigid canvas coordinates.
        rows: list[tuple[float, float, float, float, float, float, float]] = []
        for y in range(0, h - block + 1, stride):
            for x in range(0, w - block + 1, stride):
                sub = both[y : y + block, x : x + block]
                if not sub.all():
                    continue
                n_blocks += 1
                a = ga[y : y + block, x : x + block]
                b = gb[y : y + block, x : x + block]
                gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
                jxx = float(np.mean(gx * gx)); jyy = float(np.mean(gy * gy)); jxy = float(np.mean(gx * gy))
                tr = jxx + jyy
                if tr < _MIN_TEXTURE:
                    continue
                det = jxx * jyy - jxy * jxy
                disc = math.sqrt(max(0.25 * tr * tr - det, 0.0))
                l1 = 0.5 * tr + disc
                l2 = max(0.5 * tr - disc, 0.0)
                # Search window of b around the block, clipped to the overlap rectangle.
                sy0, sy1 = max(0, y - margin), min(h, y + block + margin)
                sx0, sx1 = max(0, x - margin), min(w, x + block + margin)
                if sy1 - sy0 < block + 2 or sx1 - sx0 < block + 2:
                    continue
                bs = gb[sy0:sy1, sx0:sx1]
                dx, dy, resp = _block_shift(a, bs, margin)
                # `_block_shift` assumes a symmetric margin; correct for the clipping.
                dx += (x - margin) - sx0
                dy += (y - margin) - sy0
                if resp < _MIN_RESPONSE or math.hypot(dx, dy) > pair_shift:
                    continue
                # Dominant gradient direction (eigenvector of the larger eigenvalue).
                if abs(jxy) > 1e-9:
                    e1 = np.array([l1 - jyy, jxy])
                else:
                    e1 = np.array([1.0, 0.0]) if jxx >= jyy else np.array([0.0, 1.0])
                e1 /= np.linalg.norm(e1) + 1e-12
                edge_like = (l2 / max(l1, 1e-9)) < _EDGE_RATIO
                cx = ov.rect[0] + x + block / 2.0 - 0.5
                cy = ov.rect[1] + y + block / 2.0 - 0.5
                # Content of a at (cx, cy) matches content of b at (cx + dx, cy + dy).
                rows.append((cx, cy, cx + dx, cy + dy, e1[0], e1[1], 1.0 if not edge_like else 0.0))
                n_kept += 1
                n_edge += int(edge_like)
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        pa = arr[:, 0:2]; pb = arr[:, 2:4]; e1 = arr[:, 4:6]; iso = arr[:, 6] > 0.5
        e2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)
        # Undo the pass-1 displacement to get rigid coordinates (approximate: evaluate the
        # displacement at the displaced position, valid for smooth fields).
        def rigid(mesh: ViewMesh, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = p[:, 0] * inv
            v = p[:, 1] * inv
            u = _unwrap_to(np.full_like(u, mesh.u0 + mesh.w / 2.0), u, int(round(canvas_w_small * inv)))
            idx, wts = _bilinear(mesh, u, v)
            flat = mesh.d.reshape(-1, 2)
            du = np.sum(wts * flat[idx, 0], axis=1)
            dv = np.sum(wts * flat[idx, 1], axis=1)
            return u - du, v - dv

        ua, va = rigid(ma, pa)
        ub, vb = rigid(mb, pb)
        k = ua.size
        # Row 1: across the dominant gradient (always). Row 2: along it (corners only).
        vi = np.full(2 * k, ia); vj = np.full(2 * k, ib)
        rep = np.repeat
        nx = np.stack([e1[:, 0], e2[:, 0]], axis=1).ravel()
        ny = np.stack([e1[:, 1], e2[:, 1]], axis=1).ravel()
        # Edge-like blocks -- cornices, skirtings, door frames -- get `edge_weight` times
        # the weight of a textured block: they are the architecture whose misalignment the
        # eye reads first, and they are outnumbered by the texture of whatever near object
        # shares the overlap, whose parallax would otherwise out-vote them.
        w_across = np.where(iso, weight, weight * edge_weight)
        wt = np.stack([w_across, np.where(iso, weight, 0.0)], axis=1).ravel()
        keep = wt > 0
        parts.append(Constraints(
            vi=vi[keep], vj=vj[keep], ui=rep(ua, 2)[keep], vi_=rep(va, 2)[keep],
            uj=rep(ub, 2)[keep], vj_=rep(vb, 2)[keep], nx=nx[keep], ny=ny[keep], w=wt[keep],
            group=rep(np.arange(k), 2)[keep],
        ))
    log.info("mesh.dense", blocks=n_blocks, kept=n_kept, edge_like=n_edge)
    return Constraints.concat(parts)


# ======================================================================================
# The solve
# ======================================================================================


@dataclass(slots=True)
class RigidShift:
    """What the block matcher says about one whole view; see :func:`rigid_shift`."""

    du: float
    dv: float
    n: int
    n_du: int  # rows that can measure du (isotropic blocks and vertical edges)
    n_dv: int  # rows that can measure dv (isotropic blocks and horizontal edges)
    consensus_du: float  # share of those rows within 3 px of the median: agreement, not count
    consensus_dv: float
    residual: float  # median |normal residual| over every row, the thing a move must lower


def rigid_shift(
    k: int,
    patches: list,
    meshes: list[ViewMesh],
    scale: float,
    *,
    partners: list[int],
    block: int,
    stride: int,
    max_shift: float,
) -> RigidShift:
    """Median canvas shift ``(du, dv)`` of view ``k`` against ``partners``, in the patches'
    (reduced) pixels, with how many blocks stand behind each axis, how much they disagree,
    and the median residual of the whole overlap.

    The same block matcher as :func:`dense_constraints`, restricted to the overlaps of one
    view, with a search window wide enough to see a whole-view misplacement rather than the
    residual the mesh is meant to absorb. Used to re-aim a view that has no sparse
    constraints at all -- one placed by interpolation across a featureless gap (see
    ``solve._bridge_track_gaps``) -- whose orientation can be off by a few degrees, which
    is outside the dense pass's window and more than the stiffness terms will move a whole
    view. Measured on the reference capture: the two bridged views started 15-21 px (at
    2048) from their neighbours' cornice and three re-aim iterations brought them to 6-8,
    inside the dense window; the sign convention is the one those runs selected.
    """
    sub = [k] + [j for j in partners if j != k]
    sub_patches = [patches[j] for j in sub]
    sub_meshes = []
    for i, j in enumerate(sub):
        m = meshes[j]
        sub_meshes.append(ViewMesh(view=i, u0=m.u0, v0=m.v0, w=m.w, h=m.h, cell=m.cell, d=np.zeros_like(m.d)))
    # Only the overlaps that involve view k (index 0 here). The partners' overlaps with each
    # other used to be block-matched too and then discarded by the selection below -- 90 % of
    # the work. Measured on the reference capture: 7,634 blocks per verdict, of which 778
    # touch the view being judged; re-aiming its three bridged views took 171 s, now 19 s,
    # with the same verdicts to the last digit.
    dense = dense_constraints(
        sub_patches, sub_meshes, scale, block=block, stride=stride, max_shift=max_shift,
        weight=1.0, edge_weight=1.0, only_view=0,
    )
    sel = (dense.vi == 0) | (dense.vj == 0)
    if int(sel.sum()) == 0:
        return RigidShift(0.0, 0.0, 0, 0, 0, 0.0, 0.0, float("nan"))
    sign = np.where(dense.vi[sel] == 0, 1.0, -1.0)
    du = (dense.uj[sel] - dense.ui[sel]) * sign
    dv = (dense.vj_[sel] - dense.vi_[sel]) * sign
    b = dense.nx[sel] * (dense.uj[sel] - dense.ui[sel]) + dense.ny[sel] * (dense.vj_[sel] - dense.vi_[sel])
    # Across-edge rows only measure the component normal to the edge; use them for that
    # component alone so a horizontal cornice does not vote on yaw with noise.
    horiz = np.abs(dense.nx[sel]) > 0.9
    vert = np.abs(dense.ny[sel]) > 0.9
    iso = ~(horiz | vert)
    du_rows = du[iso | horiz]
    dv_rows = dv[iso | vert]

    def _consensus(x: np.ndarray) -> float:
        """Share of rows within 3 px of the median: a plain wall's blocks scatter across
        the whole window (about 10% land near the median by chance), a real whole-view
        shift stacks them up."""
        if x.size < 4:
            return 0.0
        return float(np.mean(np.abs(x - np.median(x)) <= 3.0))

    return RigidShift(
        du=float(np.median(du_rows)) if du_rows.size else 0.0,
        dv=float(np.median(dv_rows)) if dv_rows.size else 0.0,
        n=int(sel.sum()),
        n_du=int(du_rows.size),
        n_dv=int(dv_rows.size),
        consensus_du=_consensus(du_rows),
        consensus_dv=_consensus(dv_rows),
        residual=float(np.median(np.abs(b))),
    )


def _regulariser(meshes: list[ViewMesh], offsets: np.ndarray, canvas_w: int, config: PipelineConfig) -> sp.csc_matrix:
    """``A_reg^T A_reg`` for the prior + stiffness terms over interleaved (x, y) unknowns."""
    n_vert = int(offsets[-1])
    lam_prior = float(config.mesh_prior_weight)
    lam_first = float(config.mesh_stiffness_weight)
    lam_second = float(config.mesh_straightness_weight)
    blocks: list[sp.csr_matrix] = []
    for k, m in enumerate(meshes):
        gh, gw = m.gh, m.gw
        idx = np.arange(gh * gw).reshape(gh, gw) + offsets[k]
        blocks.append(sp.csr_matrix((np.full(gh * gw, lam_prior), (np.arange(gh * gw), idx.ravel())),
                                    shape=(gh * gw, n_vert)))
        for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
            a, b = a.ravel(), b.ravel()
            rr = np.repeat(np.arange(a.size), 2)
            cc = np.stack([a, b], axis=1).ravel()
            vv = np.tile([lam_first, -lam_first], a.size)
            blocks.append(sp.csr_matrix((vv, (rr, cc)), shape=(a.size, n_vert)))
        for a, b, c_ in ((idx[:, :-2], idx[:, 1:-1], idx[:, 2:]), (idx[:-2, :], idx[1:-1, :], idx[2:, :])):
            a, b, c_ = a.ravel(), b.ravel(), c_.ravel()
            if a.size == 0:
                continue
            rr = np.repeat(np.arange(a.size), 3)
            cc = np.stack([a, b, c_], axis=1).ravel()
            vv = np.tile([lam_second, -2.0 * lam_second, lam_second], a.size)
            blocks.append(sp.csr_matrix((vv, (rr, cc)), shape=(a.size, n_vert)))
        # A grid spanning the whole canvas width is a cylinder: its last column sits on the
        # same meridian as its first, and the two must agree or the wrap column would show
        # the displacement step.  Tied hard, plus the one second difference that straddles.
        if (gw - 1) * m.cell >= canvas_w - 1e-6 and gw >= 3:
            a, b = idx[:, gw - 1].ravel(), idx[:, 0].ravel()
            rr = np.repeat(np.arange(a.size), 2)
            cc = np.stack([a, b], axis=1).ravel()
            blocks.append(sp.csr_matrix((np.tile([10.0, -10.0], a.size), (rr, cc)), shape=(a.size, n_vert)))
            a, b, c_ = idx[:, gw - 2].ravel(), idx[:, 0].ravel(), idx[:, 1].ravel()
            rr = np.repeat(np.arange(a.size), 3)
            cc = np.stack([a, b, c_], axis=1).ravel()
            vv = np.tile([lam_second, -2.0 * lam_second, lam_second], a.size)
            blocks.append(sp.csr_matrix((vv, (rr, cc)), shape=(a.size, n_vert)))
    A = sp.vstack(blocks).tocsr()
    AtA = (A.T @ A).tocoo()
    # Interleave: unknown 2*v is x of vertex v, 2*v+1 is y; the regulariser is identical
    # for both components.
    r = np.concatenate([2 * AtA.row, 2 * AtA.row + 1])
    c = np.concatenate([2 * AtA.col, 2 * AtA.col + 1])
    d = np.concatenate([AtA.data, AtA.data])
    return sp.csc_matrix((d, (r, c)), shape=(2 * n_vert, 2 * n_vert))


def _solve_spd(A: sp.csr_matrix, b: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Conjugate gradients with a Jacobi preconditioner, warm-started from ``x0``.

    The normal matrix is symmetric positive definite and grid-structured; a direct sparse
    factorisation of it fills in badly (measured 446 s for 150 k unknowns against 22 s for
    the whole re-weighted solve this way, to the same answer).
    """
    d = A.diagonal()
    d = np.where(d > 1e-12, d, 1.0)
    M = spla.LinearOperator(A.shape, matvec=lambda v: v / d, dtype=np.float64)
    x, info = spla.cg(A, b, x0=x0, M=M, rtol=1e-7, maxiter=4000)
    if info != 0:
        log.warning("mesh.cg_not_converged", info=int(info))
    return x


def solve_meshes(
    meshes: list[ViewMesh],
    constraints: Constraints,
    canvas_w: int,
    config: PipelineConfig,
) -> dict[str, float]:
    """Fill every ``mesh.d`` in place from the constraints. Returns solve statistics."""
    n_views = len(meshes)
    if n_views < 2 or constraints.n < 8:
        return {"n_constraints": int(constraints.n), "max_disp_px": 0.0}
    offsets = np.cumsum([0] + [m.gh * m.gw for m in meshes])
    n_vert = int(offsets[-1])
    C = constraints
    n = C.n

    # Data matrix over interleaved unknowns: row k has 4 (+4) entries per view.
    rows_l, cols_l, vals_l = [], [], []
    for sign, views, us, vs in ((1.0, C.vi, C.ui, C.vi_), (-1.0, C.vj, C.uj, C.vj_)):
        for k_view in np.unique(views):
            sel = np.flatnonzero(views == k_view)
            m = meshes[int(k_view)]
            idx, wts = _bilinear(m, us[sel], vs[sel])
            base = 2 * (idx + offsets[int(k_view)])
            rows_l.append(np.repeat(sel, 8))
            cols_l.append(np.stack([base, base + 1], axis=2).reshape(sel.size, 8))
            vals_l.append(np.stack([wts * (sign * C.nx[sel])[:, None], wts * (sign * C.ny[sel])[:, None]], axis=2).reshape(sel.size, 8))
    rows = np.concatenate(rows_l)
    cols = np.concatenate([c.ravel() for c in cols_l])
    vals = np.concatenate([v.ravel() for v in vals_l])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, 2 * n_vert))
    b = C.nx * (C.uj - C.ui) + C.ny * (C.vj_ - C.vi_)
    AtA_reg = _regulariser(meshes, offsets, canvas_w, config)

    huber = float(config.mesh_huber_px) * (canvas_w / 4096.0)
    n_groups = int(C.group.max()) + 1
    w = C.w.copy()
    X = np.zeros(2 * n_vert)
    res_pt = np.zeros(n_groups)
    for _ in range(int(config.mesh_irls_iterations)):
        Aw = A.multiply(w[:, None]).tocsr()
        AtA = (A.T @ Aw).tocsr() + AtA_reg
        X = _solve_spd(AtA, A.T @ (w * b), X)
        r = A @ X - b
        # One robust residual per point: the norm over its rows (1 or 2 of them).
        sq = np.bincount(C.group, weights=r * r, minlength=n_groups)
        res_pt = np.sqrt(sq)
        hub = np.where(res_pt <= huber, 1.0, huber / np.maximum(res_pt, 1e-9))
        w = C.w * hub[C.group]
    before = np.sqrt(np.bincount(C.group, weights=b * b, minlength=n_groups))
    for k, m in enumerate(meshes):
        m.d = X[2 * offsets[k] : 2 * offsets[k + 1]].reshape(m.gh, m.gw, 2).astype(np.float32)
        # The regulariser ties a full-width grid's first and last columns with a weight,
        # which leaves them a fraction of a pixel apart; the wrap column is exactly where
        # that fraction shows. Make the cylinder exact.
        if (m.gw - 1) * m.cell >= canvas_w - 1e-6 and m.gw >= 3:
            edge = 0.5 * (m.d[:, 0] + m.d[:, m.gw - 1])
            m.d[:, 0] = edge
            m.d[:, m.gw - 1] = edge
    stats = {
        "n_constraints": int(n),
        "n_points": int(n_groups),
        "n_vertices": int(n_vert),
        "residual_before_median_px": float(np.median(before)),
        "residual_before_p90_px": float(np.percentile(before, 90)),
        "residual_after_median_px": float(np.median(res_pt)),
        "residual_after_p90_px": float(np.percentile(res_pt, 90)),
        "max_disp_px": float(max(m.max_abs() for m in meshes)),
        "median_disp_px": float(np.median(np.abs(X))),
    }
    log.info("mesh.solve", **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
    return stats
