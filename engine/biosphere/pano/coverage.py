"""What the capture actually reached, measured honestly and reported so an agent can act.

The number that matters to a listing agent standing in an empty flat is not "94 % of
pixels are filled", it is "you are missing the corner behind you, go back and sweep it".
Two things follow.

*Everything is solid-angle weighted.*  An equirect canvas devotes as many pixels to the
half-degree around the zenith as to a 90-degree swathe of the equator.  Counting raw
pixels would report a capture that only grazed the ceiling as far better covered than one
that swept the whole horizon.  ``geometry.sphere_solid_angle_weights`` fixes that, and
every fraction here uses it.

*Gaps are reported inside the band the user attempted.*  A single-row sweep leaves the
whole sphere above ~25 degrees empty; saying so as a "gap" would bury the one actionable
fact -- the missing yaw sector -- under a ring that spans every direction.  Elevation
coverage is already reported by ``pitch_min_deg`` / ``pitch_max_deg`` and the zenith and
nadir flags, so the gap finder restricts itself to the elevation band the capture did
attempt and reports the holes *within* it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .types import CoverageReport

__all__ = ["build_coverage", "coverage_heatmap_png"]

_LOG = logging.getLogger(__name__)

_GRID_W, _GRID_H = 72, 36  # 5 degrees per cell
_MAX_GAPS = 8
#: Cell is "covered" once this fraction of it is filled.
_CELL_COVERED = 0.5
#: Elevation band considered "attempted", shrunk by one cell from the measured extremes so
#: the partially covered boundary rows do not masquerade as a ring-shaped gap.
_SCOPE_MARGIN_DEG = 5.0


def _union_mask(
    masks: list[np.ndarray], corners: list[tuple[int, int]], canvas_w: int, canvas_h: int
) -> np.ndarray:
    cov = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for m, (x0, y0) in zip(masks, corners, strict=True):
        h, w = m.shape[:2]
        x1, y1 = min(canvas_w, x0 + w), min(canvas_h, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        np.bitwise_or(cov[y0:y1, x0:x1], m[: y1 - y0, : x1 - x0], out=cov[y0:y1, x0:x1])
    cov[cov > 0] = 255
    return cov


def _cyclic_span(cols: np.ndarray) -> tuple[int, int]:
    """``(start_col, n_cols)`` of a possibly wrap-around set of grid columns."""
    n = cols.size
    idx = np.nonzero(cols)[0]
    if idx.size == 0:
        return 0, 0
    if idx.size == n:
        return 0, n
    missing = ~cols
    best_len, best_start = 0, 0
    run_len, run_start = 0, 0
    for k in range(2 * n):
        c = k % n
        if missing[c]:
            if run_len == 0:
                run_start = c
            run_len += 1
            if run_len > best_len and run_len <= n:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    start = (best_start + best_len) % n
    return int(start), int(n - best_len)


def _largest_uncovered_arc_deg(col_covered: np.ndarray) -> float:
    n = col_covered.size
    if col_covered.all():
        return 0.0
    if not col_covered.any():
        return 360.0
    best, run = 0, 0
    for k in range(2 * n):
        if not col_covered[k % n]:
            run += 1
            best = max(best, min(run, n))
        else:
            run = 0
    return 360.0 * best / n


def _find_gaps(grid_covered: np.ndarray, scope_rows: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Connected uncovered cells inside the attempted elevation band, largest first."""
    holes = (~grid_covered) & scope_rows[:, None]
    if not holes.any():
        return []
    n_labels, labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return []
    # The grid is cyclic in yaw: merge labels that touch across the column-0 boundary.
    parent = list(range(n_labels))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    left, right = labels[:, 0], labels[:, -1]
    for r in range(labels.shape[0]):
        if left[r] and right[r]:
            union(int(left[r]), int(right[r]))
        if r > 0 and left[r] and right[r - 1]:
            union(int(left[r]), int(right[r - 1]))
        if r + 1 < labels.shape[0] and left[r] and right[r + 1]:
            union(int(left[r]), int(right[r + 1]))
    roots = np.array([find(i) for i in range(n_labels)], dtype=np.int32)
    merged = roots[labels]

    weights = np.cos(np.radians(90.0 - (np.arange(_GRID_H) + 0.5) * (180.0 / _GRID_H)))
    out: list[tuple[float, float, float, float, float]] = []
    for lab in np.unique(merged):
        if lab == 0:
            continue
        sel = merged == lab
        rows = np.nonzero(sel.any(axis=1))[0]
        cols_any = sel.any(axis=0)
        start, span = _cyclic_span(cols_any)
        if span == 0:
            continue
        area = float((sel * weights[:, None]).sum())
        # Grid column 0 is canvas column 0, i.e. theta = -180 deg.  Gaps are reported in
        # the same yaw convention as CameraPose.yaw, lifted to [0, 360), so "the gap is at
        # 210-330" can be compared directly against the poses without a 180 deg shift.
        yaw0 = (start * (360.0 / _GRID_W) + 180.0) % 360.0
        yaw1 = yaw0 + span * (360.0 / _GRID_W)
        pitch_hi = 90.0 - int(rows.min()) * (180.0 / _GRID_H)
        pitch_lo = 90.0 - (int(rows.max()) + 1) * (180.0 / _GRID_H)
        out.append((area, yaw0, yaw1, pitch_lo, pitch_hi))
    out.sort(key=lambda t: -t[0])
    return [(y0, y1, p0, p1) for _a, y0, y1, p0, p1 in out[:_MAX_GAPS]]


def build_coverage(
    masks: list[np.ndarray],
    corners: list[tuple[int, int]],
    canvas_w: int,
    canvas_h: int,
) -> CoverageReport:
    """Measure the sphere coverage implied by a set of per-camera validity masks."""
    cov = _union_mask(masks, corners, canvas_w, canvas_h)
    covered = cov > 0
    weights = g.sphere_solid_angle_weights(canvas_h, canvas_w, dtype=np.float64)
    covered_fraction = float((weights * covered).sum())

    _, phi = g.equirect_angles(canvas_w, canvas_h, dtype=np.float64)
    elev = np.degrees(phi)
    band = np.abs(elev) <= 30.0
    band_w = weights[band]
    horizon_fraction = (
        float((band_w * covered[band]).sum() / band_w.sum()) if band_w.sum() > 0 else 0.0
    )
    del band_w

    col_covered = covered.any(axis=0)
    yaw_span_deg = 360.0 - _largest_uncovered_arc_deg(col_covered)

    row_covered = covered.any(axis=1)
    if row_covered.any():
        rows = np.nonzero(row_covered)[0]
        pitch_max = float(elev[int(rows.min())])
        pitch_min = float(elev[int(rows.max())])
    else:
        pitch_max = pitch_min = 0.0

    top = elev >= 85.0
    bottom = elev <= -85.0
    zenith_covered = bool(covered[top].mean() > 0.5) if top.any() else False
    nadir_covered = bool(covered[bottom].mean() > 0.5) if bottom.any() else False

    small = cv2.resize(cov, (_GRID_W, _GRID_H), interpolation=cv2.INTER_AREA)
    grid_covered = small >= int(255 * _CELL_COVERED)
    del small
    cell_elev = 90.0 - (np.arange(_GRID_H) + 0.5) * (180.0 / _GRID_H)
    scope_rows = (cell_elev <= pitch_max - _SCOPE_MARGIN_DEG) & (
        cell_elev >= pitch_min + _SCOPE_MARGIN_DEG
    )
    gaps = _find_gaps(grid_covered, scope_rows) if scope_rows.any() else []
    del grid_covered, weights, covered

    _LOG.info(
        "coverage: %.1f%% of the sphere, horizon %.1f%%, yaw span %.1f deg, %d gaps",
        100.0 * covered_fraction,
        100.0 * horizon_fraction,
        yaw_span_deg,
        len(gaps),
    )
    return CoverageReport(
        mask=cov,
        covered_fraction=covered_fraction,
        horizon_fraction=horizon_fraction,
        yaw_span_deg=float(yaw_span_deg),
        pitch_min_deg=pitch_min,
        pitch_max_deg=pitch_max,
        zenith_covered=zenith_covered,
        nadir_covered=nadir_covered,
        gaps=gaps,
    )


# ======================================================================================
# Visualisation
# ======================================================================================


def coverage_heatmap_png(report: CoverageReport, path: Path, *, width: int = 1024) -> Path:
    """Write the equirect coverage overlay the dashboard shows the agent.

    Teal is covered, amber is a reported gap, the dashed line is the horizon and the ticks
    are yaw in degrees, so "the amber block sits at 210-330" reads directly as "turn
    around and sweep that way again".
    """
    height = width // 2
    small = cv2.resize(report.mask, (width, height), interpolation=cv2.INTER_AREA)
    covered = small >= 128
    del small

    canvas = np.full((height, width, 3), (28, 26, 24), dtype=np.uint8)
    teal = np.array([150, 130, 30], dtype=np.float32)  # BGR
    canvas[covered] = (
        canvas[covered].astype(np.float32) * 0.25 + teal[None, :] * 0.75
    ).astype(np.uint8)

    amber = (40, 150, 235)
    overlay = canvas.copy()
    for yaw0, yaw1, p0, p1 in report.gaps:
        y_top = int(round((90.0 - p1) / 180.0 * height))
        y_bot = int(round((90.0 - p0) / 180.0 * height))
        # Undo the yaw convention: canvas column 0 is theta = -180 deg.
        x0 = int(round(((yaw0 - 180.0) % 360.0) / 360.0 * width))
        x1 = x0 + int(round((yaw1 - yaw0) / 360.0 * width))
        for shift in (0, -width):
            a = max(0, x0 + shift)
            b = min(width, x1 + shift)
            if b <= a:
                continue
            cv2.rectangle(overlay, (a, y_top), (b - 1, y_bot - 1), amber, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0, dst=canvas)
    del overlay

    mid = height // 2
    for x in range(0, width, 16):
        cv2.line(canvas, (x, mid), (min(width - 1, x + 8), mid), (220, 220, 220), 1)
    for deg in range(0, 360, 30):
        x = int(round(deg / 360.0 * width))
        cv2.line(canvas, (x, 0), (x, 10), (200, 200, 200), 1)
        cv2.line(canvas, (x, height - 11), (x, height - 1), (200, 200, 200), 1)
        cv2.putText(
            canvas, f"{deg}", (min(width - 28, x + 3), 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA,
        )
    label = (
        f"covered {100.0 * report.covered_fraction:.0f}%  "
        f"horizon {100.0 * report.horizon_fraction:.0f}%  "
        f"yaw {report.yaw_span_deg:.0f} deg"
    )
    cv2.putText(
        canvas, label, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write coverage heatmap to {path}")
    del canvas
    return path
