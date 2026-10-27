"""Single-view metrology on an equirectangular panorama.

This is exact geometry, not a learned model, and it works because one number is known: the
height of the optical centre above the floor. A 360 panorama fixes the *direction* of every
marked point perfectly; a direction plus a known plane gives a point.

    world:   +X right, +Y **down**, +Z forward   (see biosphere.pano.types)
    ray:     d(yaw, pitch) = (cos p · sin y,  -sin p,  cos p · cos y)
    floor:   the plane  y = +h   (h = camera height, positive because +Y is down)
    hit:     t = h / d_y  ->  P = t · d,  which only exists when d_y > 0, i.e. pitch < 0

Everything is computed at unit camera height first, so the result is a pure "in units of
camera height" measurement; the metric scale is applied once at the end from whichever
anchor the caller supplied. That ordering is what lets a reference segment rescale the whole
set consistently.

What this cannot do — and does not pretend to do: a point at or above the horizon
(``pitch >= 0``) never meets the floor. Its ray runs parallel to or away from the plane, and
"projecting" it anyway would produce a confidently wrong number. Those segments come back
``null`` with a hint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from biosphere.api.schemas import (
    BBoxCm,
    MeasurePoint,
    MeasureReference,
    MeasureResponse,
    MeasureSegment,
)

__all__ = ["measure_scene", "DEFAULT_CAMERA_HEIGHT_CM", "floor_point"]

#: Chest height for a hand-held phone sweep. Overridable per request.
DEFAULT_CAMERA_HEIGHT_CM = 155.0

#: Rays shallower than this never produce a usable measurement: at 0.5° of depression the
#: floor intersection is already 115 camera-heights away and moves ~4 m per tenth of a degree
#: of marking error.
MIN_DEPRESSION_DEG = 0.5

_HORIZON_HINT = (
    "This point is at or above the horizon, so its ray never reaches the floor. "
    "Mark the spot where the object meets the floor."
)
_GRAZING_HINT = (
    "This point is almost exactly on the horizon; the floor intersection is too far away to "
    "be meaningful. Mark a point closer to the camera."
)


@dataclass(slots=True)
class _FloorHit:
    """Intersection of one marked direction with the floor plane, at unit camera height."""

    index: int
    x: float | None
    z: float | None
    depression_deg: float
    hint: str | None

    @property
    def ok(self) -> bool:
        return self.x is not None and self.z is not None


def floor_point(yaw: float, pitch: float, height: float = 1.0) -> tuple[float, float] | None:
    """Intersect the ray ``(yaw, pitch)`` with the floor plane ``y = height``.

    Returns the horizontal ``(x, z)`` of the hit, or ``None`` when the ray cannot reach the
    floor. The radius reduces to the familiar ``h / tan(depression)``.
    """
    depression = -pitch
    if depression <= math.radians(MIN_DEPRESSION_DEG):
        return None
    radius = height / math.tan(depression)
    return radius * math.sin(yaw), radius * math.cos(yaw)


def _hit(index: int, point: MeasurePoint) -> _FloorHit:
    depression_deg = math.degrees(-point.pitch)
    if point.pitch >= 0.0:
        return _FloorHit(index, None, None, depression_deg, _HORIZON_HINT)
    hit = floor_point(point.yaw, point.pitch, 1.0)
    if hit is None:
        return _FloorHit(index, None, None, depression_deg, _GRAZING_HINT)
    return _FloorHit(index, hit[0], hit[1], depression_deg, None)


def _distance(a: _FloorHit, b: _FloorHit) -> float:
    """Euclidean distance between two floor hits (both lie on the same plane)."""
    return math.hypot(float(a.x) - float(b.x), float(a.z) - float(b.z))


def _point_confidence(depression_deg: float) -> float:
    """How trustworthy a floor hit at this depression angle is.

    Sensitivity of the floor intersection to marking error goes as ``1/sin²(depression)``:
    at 2° a one-pixel slip moves the point metres, at 20° it moves centimetres. The ramp
    below encodes that, saturating at 20°.
    """
    return float(min(max((depression_deg - MIN_DEPRESSION_DEG * 3.0) / 18.5, 0.0), 1.0))


def measure_scene(
    scene_id: str,
    points: Sequence[MeasurePoint],
    *,
    reference: MeasureReference | None = None,
    camera_height_cm: float | None = None,
) -> MeasureResponse:
    """Measure distances between floor points marked on a panorama.

    Scale is resolved in priority order — an explicit reference segment of known length wins,
    then the camera height, then nothing (relative units only) — and the winner is reported
    in ``scale_source`` so the UI can say where the numbers came from.
    """
    hits = [_hit(i, p) for i, p in enumerate(points)]

    # -- scale ------------------------------------------------------------------------
    scale_cm: float | None = None
    scale_source: str = "relative_only"

    if reference is not None:
        ref_a = _hit(-1, MeasurePoint(yaw=reference.yaw_a, pitch=reference.pitch_a))
        ref_b = _hit(-2, MeasurePoint(yaw=reference.yaw_b, pitch=reference.pitch_b))
        if ref_a.ok and ref_b.ok:
            ref_relative = _distance(ref_a, ref_b)
            if ref_relative > 1e-6:
                # Rescale everything so the reference segment measures exactly its known
                # length. The implied camera height falls out of the same factor.
                scale_cm = reference.length_cm / ref_relative
                scale_source = "reference"

    if scale_cm is None and any(h.ok for h in hits):
        scale_cm = float(camera_height_cm or DEFAULT_CAMERA_HEIGHT_CM)
        scale_source = "camera_height"

    # -- segments ----------------------------------------------------------------------
    segments: list[MeasureSegment] = []
    for a, b in zip(hits, hits[1:], strict=False):
        if a.ok and b.ok:
            relative = _distance(a, b)
            segments.append(
                MeasureSegment(
                    from_=a.index,
                    to=b.index,
                    length_cm=round(relative * scale_cm, 1) if scale_cm is not None else None,
                    length_relative=round(relative, 6),
                    hint=None,
                )
            )
        else:
            bad = a if not a.ok else b
            segments.append(
                MeasureSegment(
                    from_=a.index,
                    to=b.index,
                    length_cm=None,
                    length_relative=0.0,
                    hint=bad.hint,
                )
            )

    measurable = [s for s in segments if s.length_cm is not None]
    total_cm = (
        round(sum(float(s.length_cm) for s in segments), 1)
        if segments and len(measurable) == len(segments)
        else None
    )

    # -- footprint ---------------------------------------------------------------------
    bbox_cm: BBoxCm | None = None
    area_m2: float | None = None
    valid = [h for h in hits if h.ok]
    if len(valid) >= 3 and scale_cm is not None:
        pts = np.array(
            [[float(h.x) * scale_cm, float(h.z) * scale_cm] for h in valid], dtype=np.float32
        )
        hull = cv2.convexHull(pts)
        area_cm2 = float(cv2.contourArea(hull))
        area_m2 = round(area_cm2 / 10_000.0, 3)
        # A rotated rectangle beats an axis-aligned one here: rooms are rarely aligned with
        # the panorama's zero-yaw column, and an axis-aligned box over-reports every time.
        (_, _), (w, d), _ = cv2.minAreaRect(hull)
        width, depth = (float(w), float(d)) if w >= d else (float(d), float(w))
        bbox_cm = BBoxCm(
            width=round(width, 1),
            depth=round(depth, 1),
            # The optical-centre height is the only metric vertical a single viewpoint fixes.
            # Wall or ceiling extents would need a second anchor, so we report what we know.
            height=round(scale_cm, 1),
        )

    # -- confidence --------------------------------------------------------------------
    if valid:
        per_point = [_point_confidence(h.depression_deg) for h in valid]
        anchor_bonus = 0.2 if scale_source == "reference" else 0.1 if scale_source == "camera_height" else 0.0
        completeness = len(valid) / max(len(hits), 1)
        confidence = 0.45 * min(per_point) + 0.25 * (sum(per_point) / len(per_point))
        confidence += anchor_bonus + 0.1 * completeness
        confidence = float(min(max(confidence, 0.0), 0.98))
    else:
        confidence = 0.0

    return MeasureResponse(
        scene_id=scene_id,
        segments=segments,
        total_cm=total_cm,
        bbox_cm=bbox_cm,
        area_m2=area_m2,
        scale_source=scale_source,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        # Exact projective geometry — no network, no weights. Saying otherwise would be a lie
        # the UI would happily repeat to the user.
        model_backed=False,
    )
