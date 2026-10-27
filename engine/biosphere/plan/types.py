"""Contracts for the bird's-eye view: one room's floor, and the flat those rooms make.

Axes are the engine's own (see :mod:`biosphere.pano.types`): world is +X right, +Y **down**,
+Z forward, and a ray is ``(cos p * sin y, -sin p, cos p * cos y)``. A plan is that world
seen from above, so only ``x`` and ``z`` survive — ``y`` is the direction it is looked at
from. A plan angle of 0 therefore points along +Z and grows the way yaw does.

RADII ARE IN CAMERA HEIGHTS, NOT METRES
---------------------------------------
Everything a panorama can say about distance it says in units of the camera's height above
the floor, because that is the only length in the picture: a direction below the horizon
meets the floor at ``h / tan(depression)`` and the ``h`` divides straight back out again.
:mod:`biosphere.ai.measure` already works this way and for the same reason.

Keeping the stored profile dimensionless is not tidiness. The height is an *assumption* about
how somebody held a phone, and it is the one number in this feature most likely to be
corrected later — per property, or per capture once the client reports it. A cached profile in
metres would have to be recomputed (a segmentation pass per room) to apply that correction;
a dimensionless one is rescaled by a multiplication at read time.

:data:`CAMERA_HEIGHT_M` is where that assumption is applied, and it is deliberately the
player's ``EYE_HEIGHT`` rather than :data:`biosphere.ai.measure.DEFAULT_CAMERA_HEIGHT_CM`.
The two are the same physical guess rounded differently (1.60 m against 1.55 m), and this
side has to match the player: the plan draws a doorway where a floor ring stands, and a
three per cent disagreement would put the two a hand's breadth apart on a drawing whose
whole claim is that they are the same door.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "CAMERA_HEIGHT_M",
    "BINS",
    "MAX_RADIUS_H",
    "MIN_RADIUS_H",
    "PLAN_VERSION",
    "BinSource",
    "PlacementKind",
    "RadialProfile",
    "RoomOutline",
    "PlanDoor",
    "PlanRoom",
    "FloorPlan",
]

#: Height of the camera above the floor, in metres. See the module docstring: this is the
#: player's ``EYE_HEIGHT``, not ``measure``'s 1.55, so that the plan and the floor rings
#: agree about where a doorway is.
CAMERA_HEIGHT_M = 1.6

#: Azimuth bins in a stored profile. One degree: finer than the wall detail a semantic mask
#: can actually resolve, coarse enough that a whole flat's profiles are a few kilobytes.
BINS = 360

#: How far a wall may be, in camera heights, before the answer is discarded.
#:
#: Nine camera heights is about 14 m, which is the web side's ``portalGeometry.MAX_DISTANCE``
#: and the same argument: past it the boundary sits within a degree of the horizon, where a
#: one-pixel error in the mask moves the wall by metres. A room that big is also a room whose
#: far wall a hand-held phone panorama has not resolved.
MAX_RADIUS_H = 9.0

#: And how near. Closer than this the camera is inside the furniture, not the room.
MIN_RADIUS_H = 0.25

#: Bumped when the meaning of a cached ``layout.json`` changes. A file at another version is
#: recomputed rather than read, which is cheaper to reason about than a migration for a
#: derived artefact that can always be rebuilt from the panorama.
#:
#: 2: the polygon became a ROOM FITTED to the profile rather than the simplified profile
#:    itself. The profile in a version 1 file is still good, but the polygon beside it is a
#:    different kind of object, and a plan drawn half from each would be neither.
PLAN_VERSION = 2

#: Which boundary a bin's radius came from. ``none`` means neither was visible and the value
#: was interpolated from its neighbours — drawn, but never counted as measured.
BinSource = Literal["floor", "ceiling", "none"]

#: How a room got where it is on the plan.
#:
#: ``anchor``    the room the plan is built around; it defines the frame.
#: ``solved``    both sides of a doorway were marked, so position AND rotation are geometry.
#: ``estimated`` only one side was marked. The room is through that door, on that bearing, at
#:               a distance taken from its own outline — but nothing fixes which way it is
#:               turned, and the drawing has to say so.
#: ``placed``    somebody put it there. Outranks all three above, and is drawn as a solid
#:               line even when the shape is a placeholder: a person asserting where a room
#:               is is not the software guessing. See :mod:`biosphere.plan.placements`.
#: ``unlinked``  no doorway reaches it and nobody has placed it. It is listed, not drawn.
#:
#: Named ``PlacementKind`` rather than ``Placement`` because the noun is taken: a *placement*
#: is the thing an agent makes (``placements.Placement``), and this is the answer to how a
#: room ended up where it is.
PlacementKind = Literal["anchor", "solved", "estimated", "placed", "unlinked"]


def _round(value: float, places: int = 4) -> float:
    return float(round(float(value), places))


@dataclass(frozen=True, slots=True)
class RadialProfile:
    """Distance to the wall in every direction, in camera heights.

    This is the whole of what one panorama says about the shape of the room it was shot in.
    Everything else in this package is drawing.
    """

    radius: tuple[float, ...]
    source: tuple[BinSource, ...]
    #: Ceiling height above the FLOOR, in camera heights, when the two boundaries agreed
    #: often enough to fix it. ``None`` when they did not — in which case no bin can have
    #: come from the ceiling.
    ceiling_height: float | None

    def __post_init__(self) -> None:
        if len(self.radius) != len(self.source):
            raise ValueError("radius and source must be the same length")
        if not self.radius:
            raise ValueError("a profile needs at least one bin")

    @property
    def bins(self) -> int:
        return len(self.radius)

    @property
    def measured_fraction(self) -> float:
        """Share of directions where a boundary was actually found."""
        return sum(1 for s in self.source if s != "none") / float(self.bins)

    @property
    def ceiling_fraction(self) -> float:
        """Share of directions the ceiling line had to answer for, because the floor could not."""
        return sum(1 for s in self.source if s == "ceiling") / float(self.bins)

    def angles(self) -> list[float]:
        """Bin centres in radians, matching the engine's yaw."""
        step = 2.0 * math.pi / self.bins
        return [(-math.pi + (i + 0.5) * step) for i in range(self.bins)]

    def points(self, camera_height_m: float = CAMERA_HEIGHT_M) -> list[tuple[float, float]]:
        """The profile as ``(x, z)`` metres, camera at the origin."""
        return [
            (r * camera_height_m * math.sin(a), r * camera_height_m * math.cos(a))
            for r, a in zip(self.radius, self.angles(), strict=True)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius": [_round(r) for r in self.radius],
            "source": list(self.source),
            "ceiling_height": None if self.ceiling_height is None else _round(self.ceiling_height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RadialProfile:
        return cls(
            radius=tuple(float(v) for v in data["radius"]),
            source=tuple(str(v) for v in data["source"]),  # type: ignore[arg-type]
            ceiling_height=(
                None if data.get("ceiling_height") is None else float(data["ceiling_height"])
            ),
        )


@dataclass(frozen=True, slots=True)
class RoomOutline:
    """A room's floor as a closed polygon, in metres, with the camera at the origin.

    THE POLYGON IS A ROOM FITTED TO THE PROFILE, not the profile drawn. A profile is the
    distance to the nearest thing in every direction; a room is a rectangle, at most with a
    corner taken out of it. Both are kept, because they answer different questions and fail
    differently: every question about accuracy is a question about :attr:`profile`, and every
    question about what the plan shows is a question about the polygon. See
    :mod:`biosphere.plan.outline`.
    """

    #: Closed ring in (x, z), metres. The first point is not repeated at the end.
    polygon: tuple[tuple[float, float], ...]
    profile: RadialProfile
    camera_height_m: float
    #: ``model`` when a segmenter answered; ``none`` when nothing did, in which case there is
    #: no outline to draw and the room falls back to a placeholder disc.
    source: Literal["model", "none"]

    @property
    def area_m2(self) -> float:
        """Shoelace area of the polygon."""
        pts = self.polygon
        if len(pts) < 3:
            return 0.0
        total = 0.0
        for i, (x0, z0) in enumerate(pts):
            x1, z1 = pts[(i + 1) % len(pts)]
            total += x0 * z1 - x1 * z0
        return abs(total) * 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": PLAN_VERSION,
            "source": self.source,
            "camera_height_m": _round(self.camera_height_m, 3),
            "profile": self.profile.as_dict(),
            "polygon": [[_round(x, 3), _round(z, 3)] for x, z in self.polygon],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoomOutline:
        return cls(
            polygon=tuple((float(p[0]), float(p[1])) for p in data["polygon"]),
            profile=RadialProfile.from_dict(data["profile"]),
            camera_height_m=float(data.get("camera_height_m", CAMERA_HEIGHT_M)),
            source=str(data.get("source", "model")),  # type: ignore[arg-type]
        )


# ======================================================================================
# The assembled plan
# ======================================================================================


@dataclass(frozen=True, slots=True)
class PlanDoor:
    """A doorway in plan coordinates: the floor point a portal ring stands on.

    ``solved`` is true when the same door was marked from BOTH rooms, which is what fixes the
    second room's rotation. A one-way door places the room beyond it on a bearing and guesses
    the rest — see :func:`biosphere.plan.assemble.assemble`.
    """

    from_scene_id: str
    to_scene_id: str
    x: float
    z: float
    solved: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_scene_id": self.from_scene_id,
            "to_scene_id": self.to_scene_id,
            "x": _round(self.x, 3),
            "z": _round(self.z, 3),
            "solved": self.solved,
        }


@dataclass(frozen=True, slots=True)
class PlanRoom:
    """One room, placed."""

    scene_id: str
    name: str
    placement: PlacementKind
    #: Plan-frame position of the camera that shot this room, in metres.
    x: float
    z: float
    #: Rotation taking the room's own local frame to the plan frame, in radians.
    rotation: float
    #: The outline in PLAN coordinates, already rotated and translated. Empty when the room
    #: has no panorama to measure — a listed room rather than a drawn one.
    polygon: tuple[tuple[float, float], ...]
    #: Radius of the placeholder disc, in metres, for a room with no outline.
    radius: float
    area_m2: float
    #: True when this room has a finished panorama, i.e. somebody can actually stand in it.
    ready: bool
    measured_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "placement": self.placement,
            "x": _round(self.x, 3),
            "z": _round(self.z, 3),
            "rotation": _round(self.rotation, 5),
            "polygon": [[_round(x, 3), _round(z, 3)] for x, z in self.polygon],
            "radius": _round(self.radius, 3),
            "area_m2": _round(self.area_m2, 2),
            "ready": self.ready,
            "measured_fraction": _round(self.measured_fraction, 3),
        }


@dataclass(frozen=True, slots=True)
class FloorPlan:
    """A property seen from above."""

    property_id: str
    rooms: tuple[PlanRoom, ...] = ()
    doors: tuple[PlanDoor, ...] = ()
    camera_height_m: float = CAMERA_HEIGHT_M
    #: Rooms with a panorama but no cached outline yet — the work a build would do.
    pending_scene_ids: tuple[str, ...] = ()
    #: Axis-aligned bounds of everything drawn: ``(min_x, min_z, max_x, max_z)``.
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def drawn(self) -> tuple[PlanRoom, ...]:
        return tuple(r for r in self.rooms if r.placement != "unlinked")

    def as_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "rooms": [r.as_dict() for r in self.rooms],
            "doors": [d.as_dict() for d in self.doors],
            "camera_height_m": _round(self.camera_height_m, 3),
            "pending_scene_ids": list(self.pending_scene_ids),
            "bounds": [_round(v, 3) for v in self.bounds],
            "warnings": list(self.warnings),
        }
