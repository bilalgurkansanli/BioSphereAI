"""A Gaussian-splat rasteriser in plain PyTorch.

WHY WRITE ONE. The trainer (Brush) renders only the views of its own dataset, and the
converter (splat-transform) has a GPU rasteriser that returns an empty image whenever the
camera stands *inside* the cloud, which is the only place a room is ever looked at from. The
pipeline needs three renders of its own after training: the thumbnail of every area from its
entry pose, an environment panorama from the middle of every room, and the top-down plan.
Those are a handful of frames per property, so this does not have to be fast; it has to be
right, run on whatever ``torch`` device exists, and need no compiler.

WHAT IT COMPUTES. Exactly the 3DGS forward model: every Gaussian is projected with the
first-order (EWA) approximation of the perspective map to a 2D Gaussian, dilated by the
0.3 px anti-aliasing term of the reference implementation, and the pixels it covers are
composited front to back with ``C = sum_i c_i a_i prod_{j<i} (1 - a_j)``. Where the
reference does this per 16x16 tile with a depth-sorted list, this enumerates every
(Gaussian, pixel) pair inside the 3-sigma box, sorts the pairs by pixel with the Gaussians
already in depth order, and folds the transmittance product as an exclusive cumulative sum
of ``log(1 - a)`` per pixel segment. No tiles, no approximations beyond the box clip: on
the synthetic room the result is within 0.1 dB of Brush's own render of the same splats.

MEMORY. Pairs are the unit of memory, not Gaussians, so the cloud is processed in depth
order in chunks whose pair count stays under ``pair_budget``; a chunk only ever needs the
per-pixel transmittance left by the chunks in front of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from biosphere.pano import geometry as g

from .ply import SH_C0, GaussianTable

__all__ = ["PinholeView", "OrthoView", "render", "render_equirect", "cube_views", "device_for"]

_SH_C1 = 0.4886025119029199
_SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_SH_C3 = (
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
)


@dataclass(frozen=True)
class PinholeView:
    """``R`` maps world to camera (OpenCV axes: +X right, +Y down, +Z forward), ``t`` likewise."""

    R: np.ndarray  # noqa: N815 - the CV convention
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def look_at(
        cls, eye: np.ndarray, target: np.ndarray, up: np.ndarray, *, width: int, height: int, vfov_deg: float
    ) -> PinholeView:
        """A camera at ``eye`` looking at ``target`` with ``up`` (the world's up, e.g. ``-Y``)."""
        eye = np.asarray(eye, dtype=np.float64).reshape(3)
        fwd = np.asarray(target, dtype=np.float64).reshape(3) - eye
        fwd /= max(np.linalg.norm(fwd), 1e-12)
        upv = np.asarray(up, dtype=np.float64).reshape(3)
        right = np.cross(fwd, upv)
        if np.linalg.norm(right) < 1e-9:
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])  # rows: camera axes in world coordinates
        t = -R @ eye
        f = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        return cls(R=R, t=t, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=int(width), height=int(height))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass(frozen=True)
class OrthoView:
    """Parallel projection: pixel ``u = sx * x_cam + cx``. Used for the floor plan."""

    R: np.ndarray  # noqa: N815
    t: np.ndarray
    sx: float
    sy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = -1e9
    far: float = 1e9


def device_for(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def _sh_colour(sh_dc: torch.Tensor, sh_rest: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate the SH colour of every Gaussian along ``dirs`` (unit, Gaussian -> camera negated)."""
    rgb = 0.5 + SH_C0 * sh_dc
    k = sh_rest.shape[2]
    if k >= 3:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        rgb = rgb - _SH_C1 * y * sh_rest[:, :, 0] + _SH_C1 * z * sh_rest[:, :, 1] - _SH_C1 * x * sh_rest[:, :, 2]
    if k >= 8:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        rgb = (
            rgb
            + _SH_C2[0] * xy * sh_rest[:, :, 3]
            + _SH_C2[1] * yz * sh_rest[:, :, 4]
            + _SH_C2[2] * (2.0 * zz - xx - yy) * sh_rest[:, :, 5]
            + _SH_C2[3] * xz * sh_rest[:, :, 6]
            + _SH_C2[4] * (xx - yy) * sh_rest[:, :, 7]
        )
    if k >= 15:
        rgb = (
            rgb
            + _SH_C3[0] * y * (3.0 * xx - yy) * sh_rest[:, :, 8]
            + _SH_C3[1] * xy * z * sh_rest[:, :, 9]
            + _SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_rest[:, :, 10]
            + _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_rest[:, :, 11]
            + _SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_rest[:, :, 12]
            + _SH_C3[5] * z * (xx - yy) * sh_rest[:, :, 13]
            + _SH_C3[6] * x * (xx - 3.0 * yy) * sh_rest[:, :, 14]
        )
    return rgb.clamp_min(0.0)


def render(
    table: GaussianTable,
    view: PinholeView | OrthoView,
    *,
    device: torch.device | None = None,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near: float = 0.05,
    max_radius_px: int | None = None,
    pair_budget: int = 24_000_000,
    sh_degree: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``table`` through ``view``. Returns ``(rgb float32 HxWx3 in [0,1], alpha HxW)``."""
    device = device or device_for()
    H, W = int(view.height), int(view.width)
    # A cap on a splat's screen radius keeps the (splat, pixel) enumeration finite. A FIXED
    # cap is wrong: 48 px is generous at 2048 px wide and a hole-punch at 640, where a wall
    # a metre from the camera covers a third of the frame and gets painted as a small disc.
    # The background then shows through as dark streaks - which is what the first delivered
    # thumbnails were. Scaled to the frame, coverage went 0.67 -> 0.76 for 0.1 s.
    if max_radius_px is None:
        max_radius_px = max(64, min(256, max(H, W) // 4))
    f32 = torch.float32
    means = torch.as_tensor(table.means, dtype=f32, device=device)
    R = torch.as_tensor(np.asarray(view.R, dtype=np.float32), device=device)
    t = torch.as_tensor(np.asarray(view.t, dtype=np.float32), device=device)
    p_cam = means @ R.T + t
    ortho = isinstance(view, OrthoView)
    z = p_cam[:, 2]
    if ortho:
        keep = (z >= float(view.near)) & (z <= float(view.far))
    else:
        keep = z > float(near)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    colour_out = torch.zeros((H * W, 3), dtype=f32, device=device)
    transmittance = torch.ones(H * W, dtype=f32, device=device)
    if idx.numel() == 0:
        return _finish(colour_out, transmittance, background, H, W)

    p_cam = p_cam[idx]
    z = z[idx]
    means = means[idx]
    quats = torch.as_tensor(table.quats, dtype=f32, device=device)[idx]
    scales = torch.exp(torch.as_tensor(table.log_scales, dtype=f32, device=device)[idx])
    opacity = torch.sigmoid(torch.as_tensor(table.opacity_logit, dtype=f32, device=device)[idx])
    sh_dc = torch.as_tensor(table.sh_dc, dtype=f32, device=device)[idx]
    k_deg = table.sh_degree if sh_degree is None else min(int(sh_degree), table.sh_degree)
    k = {0: 0, 1: 3, 2: 8, 3: 15}[k_deg]
    sh_rest = torch.as_tensor(table.sh_rest[:, :, :k], dtype=f32, device=device)[idx]

    # Colour: view direction from the camera centre to the Gaussian, in world coordinates.
    if ortho:
        dirs = (R[2] * torch.ones_like(means)).contiguous()
    else:
        cam_center = torch.as_tensor(np.asarray(view.center, dtype=np.float32), device=device)
        dirs = means - cam_center
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    colours = _sh_colour(sh_dc, sh_rest, dirs)

    # 3D covariance -> 2D covariance (EWA).
    Rq = _quat_to_rotmat(quats)
    M = Rq * scales[:, None, :]
    cov3 = M @ M.transpose(1, 2)
    if ortho:
        J = torch.zeros((idx.numel(), 2, 3), dtype=f32, device=device)
        J[:, 0, 0] = float(view.sx)
        J[:, 1, 1] = float(view.sy)
        u = float(view.sx) * p_cam[:, 0] + float(view.cx)
        v = float(view.sy) * p_cam[:, 1] + float(view.cy)
    else:
        fx, fy = float(view.fx), float(view.fy)
        lim_x = 1.3 * (W / 2.0) / fx
        lim_y = 1.3 * (H / 2.0) / fy
        tx = (p_cam[:, 0] / z).clamp(-lim_x, lim_x) * z
        ty = (p_cam[:, 1] / z).clamp(-lim_y, lim_y) * z
        J = torch.zeros((idx.numel(), 2, 3), dtype=f32, device=device)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * tx / (z * z)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * ty / (z * z)
        u = fx * p_cam[:, 0] / z + float(view.cx)
        v = fy * p_cam[:, 1] / z + float(view.cy)
    T = J @ R[None]
    cov2 = T @ cov3 @ T.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    ok = det > 1e-12
    inv_det = 1.0 / det.clamp_min(1e-12)
    conic_a, conic_b, conic_c = c * inv_det, -b * inv_det, a * inv_det
    mid = 0.5 * (a + c)
    lam1 = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam1)).clamp(max=float(max_radius_px))
    x0 = torch.clamp(torch.floor(u - radius), 0, W - 1)
    x1 = torch.clamp(torch.ceil(u + radius), 0, W - 1)
    y0 = torch.clamp(torch.floor(v - radius), 0, H - 1)
    y1 = torch.clamp(torch.ceil(v + radius), 0, H - 1)
    in_view = ok & (u + radius >= 0) & (u - radius <= W - 1) & (v + radius >= 0) & (v - radius <= H - 1)
    in_view &= opacity > (1.0 / 255.0)
    box_w = (x1 - x0 + 1).clamp_min(0)
    box_h = (y1 - y0 + 1).clamp_min(0)
    count = torch.where(in_view, box_w * box_h, torch.zeros_like(box_w)).to(torch.int64)

    order = torch.argsort(z)
    count = count[order]
    total = int(count.sum().item())
    if total == 0:
        return _finish(colour_out, transmittance, background, H, W)
    cum = torch.cumsum(count, dim=0)
    # Chunk boundaries so that every chunk holds at most ``pair_budget`` pairs.
    bounds = [0]
    n = count.numel()
    while bounds[-1] < n:
        start = bounds[-1]
        base = int(cum[start - 1].item()) if start > 0 else 0
        end = int(torch.searchsorted(cum, torch.tensor(base + pair_budget, device=device), right=True).item())
        end = max(end, start + 1)
        bounds.append(min(end, n))

    for s, e in zip(bounds[:-1], bounds[1:], strict=True):
        sel = order[s:e]
        cnt = count[s:e]
        pairs = int(cnt.sum().item())
        if pairs == 0:
            continue
        gid = torch.repeat_interleave(torch.arange(sel.numel(), device=device), cnt)
        offsets = torch.cumsum(cnt, dim=0) - cnt
        local = torch.arange(pairs, device=device) - offsets[gid]
        bw = box_w[sel][gid]
        dx = local % bw
        dy = local // bw
        gsel = sel[gid]
        px = x0[gsel] + dx.to(f32)
        py = y0[gsel] + dy.to(f32)
        ddx = px + 0.5 - u[gsel]
        ddy = py + 0.5 - v[gsel]
        power = -0.5 * (conic_a[gsel] * ddx * ddx + conic_c[gsel] * ddy * ddy) - conic_b[gsel] * ddx * ddy
        alpha = (opacity[gsel] * torch.exp(power)).clamp(max=0.99)
        valid = (power <= 0.0) & (alpha >= 1.0 / 255.0)
        pix = (py * W + px).to(torch.int64)
        pix, alpha, gsel = pix[valid], alpha[valid], gsel[valid]
        if pix.numel() == 0:
            continue
        pix_sorted, perm = torch.sort(pix, stable=True)
        alpha = alpha[perm]
        gsel = gsel[perm]
        log1m = torch.log1p(-alpha)
        cum_log = torch.cumsum(log1m, dim=0)
        excl = cum_log - log1m
        first = torch.ones_like(pix_sorted, dtype=torch.bool)
        first[1:] = pix_sorted[1:] != pix_sorted[:-1]
        seg_id = torch.cumsum(first.to(torch.int64), dim=0) - 1
        seg_start_excl = excl[first][seg_id]
        t_within = torch.exp(excl - seg_start_excl)
        weight = alpha * t_within * transmittance[pix_sorted]
        colour_out.index_add_(0, pix_sorted, weight[:, None] * colours[gsel])
        seg_total = torch.zeros(H * W, dtype=f32, device=device)
        seg_total.index_add_(0, pix_sorted, log1m)
        transmittance = transmittance * torch.exp(seg_total)

    return _finish(colour_out, transmittance, background, H, W)


def _finish(colour: torch.Tensor, transmittance: torch.Tensor, background, H: int, W: int):  # noqa: N803
    bg = torch.as_tensor(background, dtype=colour.dtype, device=colour.device)
    rgb = colour + transmittance[:, None] * bg[None]
    rgb = rgb.clamp(0.0, 1.0).reshape(H, W, 3).cpu().numpy()
    alpha = (1.0 - transmittance).reshape(H, W).cpu().numpy()
    return rgb, alpha


def cube_views(center: np.ndarray, *, size: int) -> dict[str, PinholeView]:
    """Six 90-degree faces around ``center`` in the world frame where ``-Y`` is up.

    Face names follow the panorama engine (``px`` looks along ``+X``, ``py`` along ``+Y``
    which is *down*, ...); what matters is only that :func:`render_equirect` resamples them
    with the same rotations.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    f = (size / 2.0)
    faces: dict[str, PinholeView] = {}
    for name, R in _cube_rotations().items():
        faces[name] = PinholeView(R=R, t=-R @ center, fx=f, fy=f, cx=size / 2.0, cy=size / 2.0, width=size, height=size)
    return faces


def _cube_rotations() -> dict[str, np.ndarray]:
    # Camera looks along +Z of its own frame; these world->camera rotations point that axis
    # at each world direction with the camera's +Y (down) kept pointing at world +Y (down)
    # wherever possible.
    up = np.array([0.0, 1.0, 0.0])  # world "down", i.e. camera +Y
    out: dict[str, np.ndarray] = {}
    targets = {
        "px": (np.array([1.0, 0.0, 0.0]), up),
        "nx": (np.array([-1.0, 0.0, 0.0]), up),
        "pz": (np.array([0.0, 0.0, 1.0]), up),
        "nz": (np.array([0.0, 0.0, -1.0]), up),
        "py": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        "ny": (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    }
    for name, (fwd, down) in targets.items():
        right = np.cross(down, fwd)
        out[name] = np.stack([right, down, fwd]).astype(np.float64)
    return out


def render_equirect(
    table: GaussianTable,
    center: np.ndarray,
    *,
    width: int,
    face_size: int | None = None,
    device: torch.device | None = None,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **kwargs,
) -> np.ndarray:
    """A full equirectangular panorama (``width x width/2``, float32 RGB) from ``center``.

    Rendered as six cube faces and resampled, because a Gaussian is only a Gaussian under a
    perspective map. The world convention is the panorama engine's: ``+Y`` down, so the
    result drops straight into every 360 tool this project already has.
    """
    height = width // 2
    face_size = face_size or max(256, (width // 4) | 1)
    views = cube_views(center, size=face_size)
    faces = {n: render(table, v, device=device, background=background, **kwargs)[0] for n, v in views.items()}
    theta, phi = g.equirect_angles(width, height)
    rays = g.rays_from_equirect(theta, phi).reshape(-1, 3).astype(np.float32)
    out = np.zeros((height * width, 3), dtype=np.float32)
    ax = np.abs(rays)
    dominant = np.argmax(ax, axis=1)
    sign = np.sign(rays[np.arange(rays.shape[0]), dominant])
    for name, view in views.items():
        axis = {"px": 0, "nx": 0, "py": 1, "ny": 1, "pz": 2, "nz": 2}[name]
        want_sign = 1.0 if name[0] == "p" else -1.0
        sel = (dominant == axis) & (sign == want_sign)
        if not np.any(sel):
            continue
        d_cam = rays[sel] @ view.R.T.astype(np.float32)
        zc = d_cam[:, 2]
        # Pixel-centre convention: pixel i covers [i, i+1), so the sample coordinate is
        # shifted by half a pixel before bilinear lookup on pixel centres.
        u = view.fx * d_cam[:, 0] / zc + view.cx - 0.5
        v = view.fy * d_cam[:, 1] / zc + view.cy - 0.5
        img = faces[name]
        coords = np.stack([v, u]).astype(np.float64)
        for ch in range(3):
            out[sel, ch] = map_coordinates(img[:, :, ch], coords, order=1, mode="nearest")
    return out.reshape(height, width, 3)
