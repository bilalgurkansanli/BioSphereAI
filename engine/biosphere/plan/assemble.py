"""Putting the rooms of a property into one frame.

A panorama fixes a room's shape but not its place: every capture has its own arbitrary yaw
origin — wherever the photographer happened to start turning — and no origin at all in the
horizontal plane. So a plan needs something that is the *same thing* seen from two rooms, and
the property already has one: a doorway.

    A portal ring is a point on the floor  (yaw, pitch, and an assumed camera height),
    and when both rooms have a ring for the same doorway, that is one point with two
    coordinates. Two coordinates for one point fix a relative pose completely.

This module is the arithmetic of that, and nothing else: no images, no model, no database.
Everything it consumes is a number some other module measured, which is why it can be tested
against rooms drawn on paper.

Why a doorway marked from BOTH sides is worth so much more than one
-------------------------------------------------------------------
One ring says where a door is *in the room you are standing in*, and that is two numbers
about a relative pose that has three (two of position, one of rotation). The missing one is
which way the room on the other side is turned, and nothing in a photograph taken in the
first room can supply it.

Two rings supply it, and they do it exactly: the door is the same floor point in both rooms,
and the two rooms are on opposite sides of it, so the second room's bearing to the door must
be the reverse of the first's. That is the whole solution — one subtraction of angles, no
optimisation.

The property cannot be required to have both, because the workflow that produces rings is an
agent walking through a flat dropping one at each doorway they pass, and the return ring gets
placed only when they shoot the far room. So the one-ring case is supported, marked
``estimated``, and drawn so that the difference is visible: the room is genuinely through
that door on that bearing, and its rotation is a guess this module makes explicitly rather
than a fact it inherited.

What is deliberately not done
-----------------------------
**Disconnected groups are not laid out side by side.** If two rooms are linked to each other
but nothing links either to the rest of the flat, their positions relative to the rest are
unknown, and placing them anyway — in a row, in a corner, anywhere — would be a drawing that
says something nobody measured. They are listed instead. The plan is a single connected
thing or it is a smaller plan and a list.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from biosphere.ai.measure import floor_point
from biosphere.plan.placements import Placement
from biosphere.plan.types import (
    CAMERA_HEIGHT_M,
    FloorPlan,
    PlanDoor,
    PlanRoom,
    RoomOutline,
)

__all__ = [
    "MIN_DEPRESSION_DEG",
    "MAX_DOOR_DISTANCE_M",
    "DEFAULT_ROOM_RADIUS_M",
    "RoomInput",
    "PortalInput",
    "assemble",
]

#: A portal any shallower than this has no usable floor point.
#:
#: The web player's ``portalGeometry.MIN_DEPRESSION`` in degrees, and the same number on
#: purpose: a door this module draws in a place the player refuses to draw a ring would be a
#: plan of a property nobody can walk.
MIN_DEPRESSION_DEG = 2.0

#: And the far limit, matching ``portalGeometry.MAX_DISTANCE`` for the same reason.
MAX_DOOR_DISTANCE_M = 14.0

#: Radius of the disc standing in for a room with no measured outline, in metres. A room, at
#: the size a room usually is — it is a placeholder and the drawing says so; the number only
#: has to be unsurprising.
DEFAULT_ROOM_RADIUS_M = 2.2

#: How far an ``estimated`` room may be pushed away from a doorway to stop it being drawn on
#: top of a room that was actually solved, in metres, and in how many steps.
_PUSH_STEP_M = 0.3
_PUSH_STEPS = 10


@dataclass(frozen=True, slots=True)
class RoomInput:
    """One room as this module needs it: a name, whether it can be entered, and its shape."""

    scene_id: str
    name: str
    ready: bool
    outline: RoomOutline | None = None


@dataclass(frozen=True, slots=True)
class PortalInput:
    """One floor ring: a direction in ``from_scene_id`` that leads to ``to_scene_id``."""

    from_scene_id: str
    to_scene_id: str
    yaw: float
    pitch: float


# ======================================================================================
# Small plane geometry
# ======================================================================================


def _rotate(point: tuple[float, float], angle: float) -> tuple[float, float]:
    """Rotate ``(x, z)`` by ``angle``, in the sense yaw grows."""
    x, z = point
    ca, sa = math.cos(angle), math.sin(angle)
    return (x * ca + z * sa, z * ca - x * sa)


def _bearing(point: tuple[float, float]) -> float:
    """The yaw of ``(x, z)``: 0 along +Z, growing the way yaw does."""
    return math.atan2(point[0], point[1])


def _door_point(portal: PortalInput, camera_height_m: float) -> tuple[float, float] | None:
    """Where a portal's ring stands, in its own room's frame, or ``None``.

    :func:`biosphere.ai.measure.floor_point` does the trigonometry — the same function the
    measuring tool uses, so a doorway on the plan and a distance the user measured to it
    cannot come out differently.
    """
    if math.degrees(-portal.pitch) < MIN_DEPRESSION_DEG:
        return None
    hit = floor_point(portal.yaw, portal.pitch, camera_height_m)
    if hit is None or math.hypot(hit[0], hit[1]) > MAX_DOOR_DISTANCE_M:
        return None
    return hit


def _polygon_of(room: RoomInput) -> tuple[tuple[float, float], ...]:
    return room.outline.polygon if room.outline is not None else ()


def _radius_of(room: RoomInput) -> float:
    """A single number for how big the room is, used for placeholders and for placement."""
    outline = room.outline
    if outline is None or not outline.polygon:
        return DEFAULT_ROOM_RADIUS_M
    radii = sorted(math.hypot(x, z) for x, z in outline.polygon)
    return float(radii[len(radii) // 2])


def _local_radius(room: RoomInput, bearing: float) -> float:
    """The room's own wall distance in one direction, from its profile."""
    outline = room.outline
    if outline is None:
        return DEFAULT_ROOM_RADIUS_M
    profile = outline.profile
    index = int(((bearing + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * profile.bins)
    index = min(max(index, 0), profile.bins - 1)
    return float(profile.radius[index] * outline.camera_height_m)


def _inside(polygon: tuple[tuple[float, float], ...], point: tuple[float, float]) -> bool:
    """Ray-cast point-in-polygon. Small, exact and with no OpenCV dependency."""
    if len(polygon) < 3:
        return False
    x, z = point
    inside = False
    for i in range(len(polygon)):
        x0, z0 = polygon[i]
        x1, z1 = polygon[(i + 1) % len(polygon)]
        if (z0 > z) != (z1 > z):
            cut = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
            if cut > x:
                inside = not inside
    return inside


def _overlap_score(
    candidate: tuple[tuple[float, float], ...],
    centre: tuple[float, float],
    placed: list[tuple[tuple[float, float], ...]],
) -> int:
    """How many of a room's own points land inside a room already on the plan.

    A count of sampled points rather than an intersection area: rooms here are small polygons
    that may be non-convex, the answer only has to rank four candidates against each other,
    and an exact area would be a clipping library for a tie-break.
    """
    probes = [centre, *candidate]
    return sum(1 for p in probes for other in placed if _inside(other, p))


def _place_polygon(
    room: RoomInput, position: tuple[float, float], rotation: float
) -> tuple[tuple[float, float], ...]:
    local = _polygon_of(room)
    return tuple(
        (position[0] + rx, position[1] + rz)
        for rx, rz in (_rotate(p, rotation) for p in local)
    )


def _dominant_room_angle(room: RoomInput) -> float | None:
    """The room's own wall grid, modulo 90 degrees, or ``None`` when it has no outline.

    Same trick as :func:`biosphere.plan.outline._dominant_angle` and for the same reason —
    walls 90 degrees apart should reinforce one another — but over the placed polygon, where
    the answer is what a guessed rotation is snapped to.
    """
    polygon = _polygon_of(room)
    if len(polygon) < 3:
        return None
    total = 0j
    for i in range(len(polygon)):
        x0, z0 = polygon[i]
        x1, z1 = polygon[(i + 1) % len(polygon)]
        length = math.hypot(x1 - x0, z1 - z0)
        if length < 1e-6:
            continue
        angle = 4.0 * math.atan2(z1 - z0, x1 - x0)
        total += length * complex(math.cos(angle), math.sin(angle))
    if abs(total) < 1e-9:
        return None
    return math.atan2(total.imag, total.real) / 4.0


# ======================================================================================
# Assembly
# ======================================================================================


@dataclass
class _Pose:
    x: float
    z: float
    rotation: float
    placement: str


def assemble(
    property_id: str,
    rooms: list[RoomInput],
    portals: list[PortalInput],
    *,
    camera_height_m: float = CAMERA_HEIGHT_M,
    pending_scene_ids: tuple[str, ...] = (),
    placements: dict[str, Placement] | None = None,
) -> FloorPlan:
    """Lay the property out. ``rooms`` is in the order the property presents them.

    The first room of the largest group of rooms that doorways connect becomes the anchor and
    defines the frame; everything else is reached from it. Rooms outside that group are
    returned as ``unlinked`` — present, named, and not drawn anywhere.

    ``placements`` are the rooms somebody has put somewhere by hand. They are not a bias on
    the solution, they ARE the solution for those rooms: each one is used as given and as a
    starting point for the walk, so correcting a hallway carries everything beyond it along
    instead of leaving its neighbours at their guessed positions. See
    :mod:`biosphere.plan.placements`.
    """
    placements = {sid: p for sid, p in (placements or {}).items()}
    by_id = {room.scene_id: room for room in rooms}
    order = {room.scene_id: i for i, room in enumerate(rooms)}

    # Door points, keyed by the room they were marked in. A portal to a room that is not in
    # this property, or one placed above the horizon, is simply not a doorway.
    doors: dict[tuple[str, str], tuple[float, float]] = {}
    for portal in portals:
        if portal.from_scene_id not in by_id or portal.to_scene_id not in by_id:
            continue
        if portal.from_scene_id == portal.to_scene_id:
            continue
        point = _door_point(portal, camera_height_m)
        if point is None:
            continue
        # Two rings in one room pointing at the same room: keep the nearer, which is the one
        # standing in the doorway rather than the one aimed across it.
        key = (portal.from_scene_id, portal.to_scene_id)
        current = doors.get(key)
        if current is None or math.hypot(*point) < math.hypot(*current):
            doors[key] = point

    neighbours: dict[str, set[str]] = defaultdict(set)
    for a, b in doors:
        neighbours[a].add(b)
        neighbours[b].add(a)

    # -- components ---------------------------------------------------------------------
    seen: set[str] = set()
    components: list[list[str]] = []
    for room in rooms:
        if room.scene_id in seen:
            continue
        stack, group = [room.scene_id], []
        seen.add(room.scene_id)
        while stack:
            node = stack.pop()
            group.append(node)
            for other in neighbours[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append(sorted(group, key=lambda sid: order[sid]))

    main = max(components, key=lambda g: (len(g), -order[g[0]]))
    warnings: list[str] = []
    stray = [g for g in components if g is not main and len(g) > 1]
    if stray:
        warnings.append("disconnected_groups")

    # -- walk out from the anchor -------------------------------------------------------
    # The anchor defines the frame, so it may as well define a frame worth drawing in: it is
    # turned until its OWN walls run square to the page. Every panorama's yaw origin is
    # wherever the photographer started turning, so leaving the anchor at zero draws the whole
    # flat at whatever angle one person happened to be facing — a plan tilted twenty degrees,
    # for no reason a reader could ever discover. There is no true north to use instead (see
    # `PlanView`), so the first room's walls are the best available reference.
    anchor = main[0]
    anchor_turn = _dominant_room_angle(by_id[anchor]) or 0.0

    # Hand-placed rooms are seeded FIRST and are walked out from in their own right. A property
    # whose rooms no doorway connects has no graph to solve and, until somebody drags a room
    # into place, no plan; after that it has one, and it grows from there.
    poses: dict[str, _Pose] = {}
    queue: deque[str] = deque()
    for scene_id, placement in placements.items():
        if scene_id in by_id:
            poses[scene_id] = _Pose(placement.x, placement.z, placement.rotation, "placed")
            queue.append(scene_id)
    if anchor not in poses:
        poses[anchor] = _Pose(0.0, 0.0, anchor_turn, "anchor")
        queue.appendleft(anchor)

    placed_polygons: list[tuple[tuple[float, float], ...]] = []
    for scene_id, pose in poses.items():
        polygon = _place_polygon(by_id[scene_id], (pose.x, pose.z), pose.rotation)
        if polygon:
            placed_polygons.append(polygon)

    while queue:
        here = queue.popleft()
        pose = poses[here]
        for there in sorted(neighbours[here], key=lambda sid: order[sid]):
            if there in poses:
                continue
            local = doors.get((here, there))
            if local is None:
                # Only the far room marked the doorway. Its ring is still a floor point in
                # ITS frame, but nothing in this room says where the door is, so the pair
                # cannot be walked in this direction; it will be reached from the other side.
                continue

            world_door = _door_world(pose, local)
            bearing = _bearing((world_door[0] - pose.x, world_door[1] - pose.z))
            back = doors.get((there, here))
            room = by_id[there]

            if back is not None:
                # Solved. The door is one point with two coordinates: the far room's bearing
                # to it must be the reverse of this one's, which fixes the rotation, and the
                # position then follows by subtraction.
                rotation = _wrap(bearing + math.pi - _bearing(back))
                offset = _rotate(back, rotation)
                poses[there] = _Pose(world_door[0] - offset[0], world_door[1] - offset[1], rotation, "solved")
            else:
                poses[there] = _estimate(room, world_door, bearing, placed_polygons)

            polygon = _place_polygon(room, (poses[there].x, poses[there].z), poses[there].rotation)
            if polygon:
                placed_polygons.append(polygon)
            queue.append(there)

    # -- results ------------------------------------------------------------------------
    plan_rooms: list[PlanRoom] = []
    for room in rooms:
        pose = poses.get(room.scene_id)
        outline = room.outline
        if pose is None:
            plan_rooms.append(
                PlanRoom(
                    scene_id=room.scene_id,
                    name=room.name,
                    placement="unlinked",
                    x=0.0,
                    z=0.0,
                    rotation=0.0,
                    polygon=(),
                    radius=_radius_of(room),
                    area_m2=outline.area_m2 if outline else 0.0,
                    ready=room.ready,
                    measured_fraction=outline.profile.measured_fraction if outline else 0.0,
                )
            )
            continue
        polygon = _place_polygon(room, (pose.x, pose.z), pose.rotation)
        plan_rooms.append(
            PlanRoom(
                scene_id=room.scene_id,
                name=room.name,
                placement=pose.placement,  # type: ignore[arg-type]
                x=pose.x,
                z=pose.z,
                rotation=pose.rotation,
                polygon=polygon,
                radius=_radius_of(room),
                area_m2=outline.area_m2 if outline else 0.0,
                ready=room.ready,
                measured_fraction=outline.profile.measured_fraction if outline else 0.0,
            )
        )

    plan_doors = _doors_in_plan(doors, poses)
    bounds = _bounds(plan_rooms, plan_doors)
    return FloorPlan(
        property_id=property_id,
        rooms=tuple(plan_rooms),
        doors=tuple(plan_doors),
        camera_height_m=camera_height_m,
        pending_scene_ids=tuple(pending_scene_ids),
        bounds=bounds,
        warnings=tuple(warnings),
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _door_world(pose: _Pose, local: tuple[float, float]) -> tuple[float, float]:
    rx, rz = _rotate(local, pose.rotation)
    return (pose.x + rx, pose.z + rz)


def _estimate(
    room: RoomInput,
    door: tuple[float, float],
    bearing: float,
    placed: list[tuple[tuple[float, float], ...]],
) -> _Pose:
    """Place a room that only the room before it marked a doorway to.

    Two of the three unknowns are honest: the room is beyond that door, on that bearing. The
    third — how it is turned — is not in any photograph taken on this side of the wall, so
    this picks:

    * the room's own wall grid, so its walls run parallel to the ones already drawn rather
      than at some angle nothing suggested; that leaves four candidates, a quarter turn apart;
    * of those four, the one that puts least of the new room on top of a room already placed.

    The distance from the door is then the room's own profile in whichever direction the door
    ends up lying — i.e. the door is put against the wall it would be in.

    None of that is a measurement, which is why the pose it returns is marked ``estimated``
    and drawn as a broken line. It is a defensible drawing of a real adjacency, not a claim
    about a rotation.
    """
    grid = _dominant_room_angle(room)
    base = 0.0 if grid is None else -grid
    candidates: list[tuple[int, float, _Pose]] = []
    for k in range(4 if grid is not None else 1):
        rotation = _wrap(base + k * math.pi / 2.0)
        # Where the door lies in the room's own frame, given that rotation.
        local_bearing = _wrap(bearing + math.pi - rotation)
        distance = max(_local_radius(room, local_bearing), 0.6)
        position = (door[0] + math.sin(bearing) * distance, door[1] + math.cos(bearing) * distance)
        polygon = _place_polygon(room, position, rotation)
        score = _overlap_score(polygon, position, placed)
        candidates.append((score, distance, _Pose(position[0], position[1], rotation, "estimated")))

    # Least overlap first; ties go to the candidate that puts the room FURTHEST out, which is
    # the one whose door lies in its longest wall — the more likely wall for a door, and the
    # placement least likely to need pushing below.
    candidates.sort(key=lambda c: (c[0], -c[1]))
    pose = candidates[0][2]

    # Still sitting on something? Push it further out along the same bearing — the one
    # direction the doorway actually licenses — rather than sliding it sideways.
    for _ in range(_PUSH_STEPS):
        polygon = _place_polygon(room, (pose.x, pose.z), pose.rotation)
        if _overlap_score(polygon, (pose.x, pose.z), placed) == 0:
            break
        pose = _Pose(
            pose.x + math.sin(_bearing((pose.x - door[0], pose.z - door[1]))) * _PUSH_STEP_M,
            pose.z + math.cos(_bearing((pose.x - door[0], pose.z - door[1]))) * _PUSH_STEP_M,
            pose.rotation,
            "estimated",
        )
    return pose


def _doors_in_plan(
    doors: dict[tuple[str, str], tuple[float, float]], poses: dict[str, _Pose]
) -> list[PlanDoor]:
    """Every doorway that reaches a placed room, in plan coordinates.

    A doorway marked from both sides is drawn once, at the midpoint of the two answers. They
    agree exactly for the pair the walk was built on; a second, redundant pair of rings can
    disagree, and the midpoint is both the honest summary and a visible measure of the
    disagreement.
    """
    out: dict[tuple[str, str], PlanDoor] = {}
    for (here, there), local in doors.items():
        pose = poses.get(here)
        if pose is None:
            continue
        point = _door_world(pose, local)
        key = (here, there) if here < there else (there, here)
        existing = out.get(key)
        if existing is None:
            out[key] = PlanDoor(here, there, point[0], point[1], solved=(there, here) in doors)
        else:
            out[key] = PlanDoor(
                existing.from_scene_id,
                existing.to_scene_id,
                (existing.x + point[0]) * 0.5,
                (existing.z + point[1]) * 0.5,
                solved=True,
            )
    return [out[key] for key in sorted(out)]


def _bounds(rooms: list[PlanRoom], doors: list[PlanDoor]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    zs: list[float] = []
    for room in rooms:
        if room.placement == "unlinked":
            continue
        if room.polygon:
            xs.extend(x for x, _ in room.polygon)
            zs.extend(z for _, z in room.polygon)
        else:
            xs.extend([room.x - room.radius, room.x + room.radius])
            zs.extend([room.z - room.radius, room.z + room.radius])
    for door in doors:
        xs.append(door.x)
        zs.append(door.z)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(zs), max(xs), max(zs))
