"""Which areas have something to show that the walkthrough does not already show.

A filmed walk breaks into pieces - a doorway, a turn, a moment the phone saw nothing to
match on - and each piece is trained and registered as its own area. The pieces overlap:
several of them reconstruct the same hallway from almost the same route. The viewer picks
the area whose camera path passes nearest and draws that one, which is right when the
pieces cover different places and wrong when they cover the same one, because then an
ordinary step across the middle of a room replaces every splat on screen with another
reconstruction of the same wall.

Measured on the first real capture, standing at the poses each area was itself filmed from
and asking what fraction of the frame each cloud actually puts anything on:

    area        its own cloud    the base's cloud
    base~3            68 %             29 %      the base does not reach here
    base~4            87 %             93 %      the base already covers this
    base~5            75 %             90 %      the base already covers this

The last two are the ones the walk kept flipping into, and switching to them made the
picture *worse*: rendered from a point on the base's own path, one of them came back almost
black - it has no geometry there at all, so the room went dark for a step and came back.

So an area earns the right to be chosen by position only by covering its own viewpoints
better than the base does. The others are still delivered and still open from the rail by
name; they simply never take the picture away from a walk that is already showing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ply import GaussianTable
from .raster import PinholeView, render

__all__ = ["CoverageReport", "frame_coverage", "redundant_areas"]

#: The property frame is COLMAP's: +Y points DOWN, so "up" is -Y.
_UP = np.array([0.0, -1.0, 0.0])


@dataclass
class CoverageReport:
    #: key -> fraction of the frame that area's own cloud covers at its own viewpoints.
    own: dict[str, float] = field(default_factory=dict)
    #: key -> what the base covers at those same viewpoints.
    base: dict[str, float] = field(default_factory=dict)
    #: Areas the base already covers, which the viewer must not switch into by position.
    redundant: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "own": {k: round(v, 4) for k, v in self.own.items()},
            "base": {k: round(v, 4) for k, v in self.base.items()},
            "redundant": list(self.redundant),
            "seconds": round(self.seconds, 1),
        }


def _basis(forward: np.ndarray) -> np.ndarray:
    """World-to-camera rotation looking along ``forward`` with the property frame's up."""
    z = forward / max(float(np.linalg.norm(forward)), 1e-9)
    x = np.cross(_UP, z)
    n = float(np.linalg.norm(x))
    if n < 1e-6:  # looking straight up or down: any horizontal right will do
        x = np.array([1.0, 0.0, 0.0])
        n = 1.0
    x /= n
    return np.stack([x, np.cross(z, x), z])


def _poses(cameras: list[list[float]], samples: int, fan: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stations along the walk, each looking ahead and to both sides."""
    cams = np.asarray(cameras, dtype=np.float64)
    if len(cams) < 2:
        return []
    step = max(len(cams) // samples, 1)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, len(cams), step):
        ahead = cams[min(i + 1, len(cams) - 1)] - cams[i]
        if float(np.linalg.norm(ahead)) < 1e-6:
            ahead = np.array([0.0, 0.0, 1.0])
        for turn in (-fan, 0.0, fan):
            c, s = np.cos(turn), np.sin(turn)
            out.append((cams[i], np.array([ahead[0] * c - ahead[2] * s, ahead[1], ahead[0] * s + ahead[2] * c])))
    return out[: samples * 3]


def frame_coverage(
    table: GaussianTable,
    poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    width: int = 384,
    fov_deg: float = 70.0,
    device: Any = None,
    threshold: float = 0.5,
) -> float:
    """Mean fraction of the frame ``table`` puts something solid on, over ``poses``.

    The rasteriser hands back coverage as well as colour, and coverage is the whole
    question: an empty cloud renders black, and a viewer that switches to it shows the
    visitor a dark room.
    """
    if not poses or table.count == 0:
        return 0.0
    height = int(round(width * 9 / 16))
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    seen = []
    for centre, forward in poses:
        R = _basis(forward)
        view = PinholeView(
            R=R, t=-R @ np.asarray(centre, dtype=np.float64), fx=f, fy=f,
            cx=width / 2, cy=height / 2, width=width, height=height,
        )
        alpha = np.asarray(render(table, view, device=device)[1])
        seen.append(float((alpha > threshold).mean()))
    return float(np.mean(seen))


def redundant_areas(
    tables: dict[str, GaussianTable],
    cameras: dict[str, list[list[float]]],
    base_key: str,
    *,
    margin: float = 0.05,
    samples: int = 8,
    fan: float = 1.0,
    device: Any = None,
) -> CoverageReport:
    """Areas the base already covers as well as they cover themselves.

    ``margin`` is the coverage an area must win by, as a fraction of the frame, before it is
    worth taking the picture over: the two redundant areas measured on the real capture lost
    by 6 and 15 points, and the one that earned its place won by 39.
    """
    import time

    t0 = time.perf_counter()
    rep = CoverageReport()
    base_table = tables.get(base_key)
    if base_table is None:
        return rep
    for key, table in tables.items():
        if key == base_key:
            continue
        poses = _poses(cameras.get(key, []), samples, fan)
        if not poses:
            continue
        rep.own[key] = frame_coverage(table, poses, device=device)
        rep.base[key] = frame_coverage(base_table, poses, device=device)
        if rep.own[key] <= rep.base[key] + margin:
            rep.redundant.append(key)
    rep.seconds = time.perf_counter() - t0
    return rep
