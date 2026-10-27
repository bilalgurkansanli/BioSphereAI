"""SOG compression and the per-area pictures: thumbnail, environment panorama, floor plan.

SOG is PlayCanvas's spatially-ordered, WebP-packed splat format: ~10 bytes per Gaussian
against ~250 for the PLY, decoded by the viewer's own loader. The writer is the
``@playcanvas/splat-transform`` CLI (Node), pinned in the tools directory; it is the same
tool SplatTour's assets were made with, and the meta.json it emits says so.

The pictures come from :mod:`.raster`, in the levelled frame (``+Y`` down, floor ``y=0``):

* the **thumbnail** is the room seen from its entry pose, levelled;
* the **environment panorama** is a small equirect from the room's centre, the viewer's
  background wherever the splat has holes (windows, the doorway you came in through);
* the **floor plan** is an orthographic top-down render of every area with the ceiling cut
  away, one image for the property, plus the pixel-per-metre mapping that puts a room's
  hotspot on it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biosphere.core.storage import write_jpeg

from .ply import GaussianTable
from .raster import OrthoView, PinholeView, device_for, render, render_equirect

__all__ = [
    "SogReport", "splat_transform_command", "to_sog", "render_thumbnail", "render_env_pano",
    "FloorPlanImage", "render_floor_plan",
]

_DEFAULT_TOOLS = Path(os.environ.get("BIOSPHERE_TOOLS_DIR", "C:/biosphere-data/tools"))


@dataclass
class SogReport:
    path: Path
    bytes: int
    count: int
    generator: str
    sh_bands: int

    def as_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "count": self.count, "generator": self.generator, "sh_bands": self.sh_bands}


def splat_transform_command() -> list[str]:
    """The CLI as an argv prefix: a pinned local install first, ``npx`` as the fallback."""
    override = os.environ.get("BIOSPHERE_SPLAT_TRANSFORM")
    if override:
        return override.split()
    cli = _DEFAULT_TOOLS / "node_modules" / "@playcanvas" / "splat-transform" / "bin" / "cli.mjs"
    node = shutil.which("node")
    if cli.exists() and node:
        return [node, str(cli)]
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx:
        return [npx, "-y", "@playcanvas/splat-transform@3.3.3"]
    raise RuntimeError("splat-transform not found: npm install @playcanvas/splat-transform in the tools directory")


def to_sog(ply_path: Path, sog_path: Path, *, sh_degree: int = 0, iterations: int = 10, use_gpu: bool = True) -> SogReport:
    """Compress a PLY to a bundled ``.sog``; higher SH bands are dropped to ``sh_degree``."""
    sog_path = Path(sog_path)
    sog_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = splat_transform_command() + ["-w", "-q", str(ply_path), "-N", "-H", str(int(sh_degree)), "-i", str(int(iterations))]
    if not use_gpu:
        cmd += ["-g", "cpu"]
    cmd += [str(sog_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not sog_path.exists():
        raise RuntimeError(f"splat-transform failed ({proc.returncode}): {proc.stderr[-1500:]}{proc.stdout[-500:]}")
    with zipfile.ZipFile(sog_path) as zf:
        meta = json.loads(zf.read("meta.json"))
    return SogReport(
        path=sog_path, bytes=sog_path.stat().st_size, count=int(meta.get("count", 0)),
        generator=str(meta.get("asset", {}).get("generator", "")), sh_bands=int(meta.get("shN", {}).get("bands", 0)),
    )


def _to_bgr8(rgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(rgb, 0.0, 1.0)
    return (srgb[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)


def render_thumbnail(
    table: GaussianTable, eye: np.ndarray, forward: np.ndarray, out_path: Path, *, size: tuple[int, int] = (640, 360), vfov_deg: float = 62.0, device=None
) -> Path:
    """The area from ``eye`` looking along ``forward`` (levelled: the pitch is dropped)."""
    fwd = np.asarray(forward, dtype=np.float64).copy()
    fwd[1] = 0.0
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([0.0, 0.0, 1.0])
    fwd /= np.linalg.norm(fwd)
    view = PinholeView.look_at(eye, np.asarray(eye) + fwd, np.array([0.0, -1.0, 0.0]), width=size[0], height=size[1], vfov_deg=vfov_deg)
    rgb, _ = render(table, view, device=device or device_for(), background=(0.02, 0.02, 0.03))
    return write_jpeg(Path(out_path), _to_bgr8(rgb), quality=86)


def render_env_pano(table: GaussianTable, centre: np.ndarray, out_path: Path, *, width: int = 1024, device=None) -> Path:
    """A small equirect from ``centre``; blurred a little, it is a background, not a view."""
    pano = render_equirect(table, centre, width=width, face_size=max(256, width // 3), device=device or device_for(), background=(0.02, 0.02, 0.03))
    bgr = _to_bgr8(pano)
    bgr = cv2.GaussianBlur(bgr, (0, 0), 1.2)
    return write_jpeg(Path(out_path), bgr, quality=80)


@dataclass
class FloorPlanImage:
    path: Path
    width: int
    height: int
    #: World -> pixel: ``u = (x - x0) * px_per_m``, ``v = (z0 - z) * px_per_m`` (``+Z`` is up the page).
    x0: float
    z0: float
    px_per_m: float

    def to_pixel(self, x: float, z: float) -> tuple[float, float]:
        return (x - self.x0) * self.px_per_m, (self.z0 - z) * self.px_per_m

    def to_fraction(self, x: float, z: float) -> tuple[float, float]:
        u, v = self.to_pixel(x, z)
        return u / self.width, v / self.height

    def as_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "x0": self.x0, "z0": self.z0, "px_per_m": self.px_per_m}


def render_floor_plan(
    tables: list[GaussianTable],
    out_path: Path,
    *,
    px_per_m: float = 48.0,
    margin_m: float = 0.6,
    ceiling_cut_m: float = 2.05,
    max_px: int = 2048,
    bounds: tuple[float, float, float, float] | None = None,
    #: Needles are widened before the top-down render, and only for it: from above they are
    #: shards, and a plan made of shards tells nobody which room is which.
    plan_anisotropy: float = 4.0,
    device=None,
) -> FloorPlanImage:
    """Top-down orthographic render of all areas with the ceiling removed.

    The camera hangs above the flat looking down ``+Y``; only splats lower than
    ``ceiling_cut_m`` above the floor are drawn, so the roof and the light fittings do not
    hide the rooms. The image's up is world ``+Z`` and its right is world ``+X``.
    """
    if not tables:
        raise ValueError("no splats to draw")
    merged = GaussianTable.concatenate(tables) if len(tables) > 1 else tables[0]
    # Extent from the well-supported splats only: opaque and below the ceiling cut.
    # A plan is a diagram, and it needs the cloud prepared for its own way of looking. Seen
    # from directly above, a long thin Gaussian that reads as a wall from inside the room
    # reads as a shard, and the street outside the window is simply not part of the flat.
    # The delivered cloud used to be cut down until it satisfied both this drawing and the
    # walkthrough, and the walkthrough paid 4.5 dB for it; the two are separated here.
    if bounds is not None:
        x_lo, x_hi, z_lo, z_hi = bounds
        inside = (
            (merged.means[:, 0] >= x_lo - margin_m) & (merged.means[:, 0] <= x_hi + margin_m)
            & (merged.means[:, 2] >= z_lo - margin_m) & (merged.means[:, 2] <= z_hi + margin_m)
        )
        if inside.sum() > 100:
            merged = merged.select(inside)
        scales = merged.scales
        widened = np.maximum(scales, scales.max(axis=1, keepdims=True) / plan_anisotropy)
        merged.log_scales = np.log(np.maximum(widened, 1e-9)).astype(merged.log_scales.dtype)
    else:
        solid = (merged.opacity > 0.3) & (merged.means[:, 1] > -ceiling_cut_m) & (merged.means[:, 1] < 0.6)
        pts = merged.means[solid] if solid.sum() > 100 else merged.means
        x_lo, x_hi = np.percentile(pts[:, 0], [0.5, 99.5])
        z_lo, z_hi = np.percentile(pts[:, 2], [0.5, 99.5])
    x_lo -= margin_m
    x_hi += margin_m
    z_lo -= margin_m
    z_hi += margin_m
    width_m, height_m = float(x_hi - x_lo), float(z_hi - z_lo)
    scale = float(px_per_m)
    if max(width_m, height_m) * scale > max_px:
        scale = max_px / max(width_m, height_m)
    W = max(int(round(width_m * scale)), 16)
    H = max(int(round(height_m * scale)), 16)
    # Camera axes in world: x_cam = +X (right), y_cam = -Z (down the page), z_cam = +Y (down).
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t = np.zeros(3)
    view = OrthoView(R=R, t=t, sx=scale, sy=scale, cx=-x_lo * scale, cy=z_hi * scale, width=W, height=H, near=-ceiling_cut_m, far=1.0)
    rgb, alpha = render(merged, view, device=device or device_for(), background=(0.0, 0.0, 0.0), max_radius_px=24)
    bgr = _to_bgr8(rgb)
    # Fade unsupported pixels to the page colour so the outline reads as a plan.
    page = np.array([10, 12, 16], dtype=np.float32)
    bgr = (bgr.astype(np.float32) * alpha[:, :, None] + page * (1.0 - alpha[:, :, None])).astype(np.uint8)
    write_jpeg(Path(out_path), bgr, quality=88)
    return FloorPlanImage(path=Path(out_path), width=W, height=H, x0=float(x_lo), z0=float(z_hi), px_per_m=scale)
