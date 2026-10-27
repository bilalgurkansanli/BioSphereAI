"""Tests for the bird's-eye view.

Both halves of this feature are checkable at a desk and neither needs a model, which is why
they were split apart in the first place:

* the estimator turns a *surface map* into distances, so a surface map ray-traced from a room
  of known dimensions — or drawn by hand, which is what these do — has a known answer;
* the assembler turns *floor points* into one drawing, so two rooms placed on paper have a
  known relative pose.

The occlusion tests are the ones worth reading. They are the reason the ceiling boundary
exists at all, and they encode the sign of the error: furniture can only ever make a wall
look nearer, never further, so an estimator that ignores the ceiling must come back short and
one that uses it must not.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from biosphere.plan import assemble as A
from biosphere.plan import outline as O
from biosphere.plan import placements as P
from biosphere.plan.placements import Placement
from biosphere.plan.types import RoomOutline

W, H = 1024, 512
EYE = 1.6


# ======================================================================================
# Hand-built rooms
# ======================================================================================


def box_radius(angles: np.ndarray, width: float, depth: float, eye: tuple[float, float]) -> np.ndarray:
    """Distance from ``eye`` to the walls of a ``width`` x ``depth`` box, per azimuth."""
    cx, cz = eye
    sx, sz = np.sin(angles), np.cos(angles)
    best = np.full(np.shape(angles), np.inf)
    for numerator, denominator in ((-cx, sx), (width - cx, sx), (-cz, sz), (depth - cz, sz)):
        safe = np.where(np.abs(denominator) > 1e-9, denominator, np.nan)
        t = numerator / safe
        best = np.minimum(best, np.where(np.isfinite(t) & (t > 1e-6), t, np.inf))
    return best


def box_surface(
    width: float = 5.2,
    depth: float = 3.9,
    ceiling: float = 2.7,
    eye: tuple[float, float] = (2.78, 1.71),
    camera_height: float = EYE,
    doorway: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A perfect surface map of an empty rectangular room, and its true radii in metres.

    ``doorway`` is ``(from_deg, to_deg, extra_metres)``: an opening in the wall through which
    the floor keeps going, which is what an open door actually looks like to this estimator.
    """
    theta = 2.0 * np.pi * (np.arange(W) + 0.5) / W - np.pi
    phi = 0.5 * np.pi - np.pi * (np.arange(H) + 0.5) / H
    radius = box_radius(theta, width, depth, eye)
    if doorway is not None:
        start, stop, extra = doorway
        through = (np.degrees(theta) >= start) & (np.degrees(theta) <= stop)
        radius = np.where(through, radius + extra, radius)

    floor_elevation = -np.arctan2(camera_height, radius)
    ceiling_elevation = np.arctan2(ceiling - camera_height, radius)

    surface = np.full((H, W), O.SURFACE_WALL, np.uint8)
    surface[phi[:, None] < floor_elevation[None, :]] = O.SURFACE_FLOOR
    surface[phi[:, None] > ceiling_elevation[None, :]] = O.SURFACE_CEILING
    return surface, radius


def columns_for(yaw_from_deg: float, yaw_to_deg: float) -> slice:
    """The column range covering an azimuth wedge, in the canvas's own indexing."""
    return slice(int((yaw_from_deg + 180.0) / 360.0 * W), int((yaw_to_deg + 180.0) / 360.0 * W))


def bins_for(yaw_from_deg: float, yaw_to_deg: float) -> slice:
    return slice(int(yaw_from_deg + 180.0), int(yaw_to_deg + 180.0))


def occlude(surface: np.ndarray, wedge: slice, hides_to_deg: float) -> np.ndarray:
    """Stand something in front of the wall: blank the floor above ``hides_to_deg``.

    Everything between the true wall foot and the near edge of the object stops being floor,
    exactly as a wardrobe does to the pixels behind it. The object itself is 255 — neither
    room nor a surface — which is what the segmenter leaves for a furnishing.
    """
    out = surface.copy()
    row = int((0.5 * math.pi - math.radians(hides_to_deg)) / math.pi * H)
    band = out[:row, wedge]
    band[band == O.SURFACE_FLOOR] = 255
    out[:row, wedge] = band
    return out


# ======================================================================================
# The estimator
# ======================================================================================


def test_profile_recovers_a_room_of_known_size():
    surface, truth = box_surface()
    profile = O.profile_from_surface(surface)

    angles = np.array(profile.angles())
    estimated = np.array(profile.radius) * EYE
    error = np.abs(estimated - box_radius(angles, 5.2, 3.9, (2.78, 1.71)))

    assert float(error.mean()) < 0.05
    assert float(np.percentile(error, 95)) < 0.10
    assert profile.measured_fraction == 1.0


def test_ceiling_height_is_measured_not_assumed():
    surface, _ = box_surface(ceiling=3.1)
    profile = O.profile_from_surface(surface)
    assert profile.ceiling_height is not None
    assert abs(profile.ceiling_height * EYE - 3.1) < 0.05


def test_without_a_ceiling_the_profile_says_so():
    surface, _ = box_surface()
    surface[surface == O.SURFACE_CEILING] = O.SURFACE_WALL

    profile = O.profile_from_surface(surface)
    assert profile.ceiling_height is None
    assert profile.ceiling_fraction == 0.0
    assert set(profile.source) <= {"floor", "none"}


def test_furniture_makes_the_floor_alone_report_a_nearer_wall():
    """The premise of the whole design: occlusion has a sign, and it is always inward."""
    surface, truth = box_surface()
    blocked = occlude(surface, columns_for(-30.0, 30.0), hides_to_deg=-40.0)

    floor_only = np.array(O.profile_from_surface(blocked, use_ceiling=False).radius) * EYE
    clean = np.array(O.profile_from_surface(surface, use_ceiling=False).radius) * EYE
    wedge = bins_for(-25.0, 25.0)

    assert (floor_only[wedge] < clean[wedge] - 0.2).all()


def test_the_ceiling_line_puts_the_occluded_wall_back():
    surface, _ = box_surface()
    blocked = occlude(surface, columns_for(-30.0, 30.0), hides_to_deg=-40.0)

    profile = O.profile_from_surface(blocked)
    angles = np.array(profile.angles())
    estimated = np.array(profile.radius) * EYE
    truth = box_radius(angles, 5.2, 3.9, (2.78, 1.71))
    wedge = bins_for(-25.0, 25.0)

    assert float(np.abs(estimated[wedge] - truth[wedge]).mean()) < 0.10
    assert all(source == "ceiling" for source in np.array(profile.source)[wedge])


def test_a_ceiling_label_near_the_horizon_is_not_the_ceiling():
    """A dark window mistaken for ceiling once asked for a wall twelve metres away.

    The rule that refuses it is structural rather than a threshold: a ceiling is the region
    that contains the zenith, so an island floating above the horizon is not one.
    """
    surface, _ = box_surface()
    clean = np.array(O.profile_from_surface(surface).radius)

    lied_to = surface.copy()
    lied_to[H // 2 - 22 : H // 2 - 6, columns_for(-20.0, 20.0)] = O.SURFACE_CEILING
    after = np.array(O.profile_from_surface(lied_to).radius)

    assert float(np.abs(after - clean).max()) < 0.05


def test_directions_nobody_saw_are_marked_rather_than_invented():
    surface, _ = box_surface()
    surface[:, columns_for(100.0, 140.0)] = 255

    profile = O.profile_from_surface(surface)
    wedge = bins_for(105.0, 135.0)
    assert all(source == "none" for source in np.array(profile.source)[wedge])
    # Still drawn — a closed outline has to cross the gap — but never counted as measured.
    assert all(r > 0.0 for r in np.array(profile.radius)[wedge])
    assert profile.measured_fraction < 0.9


def test_a_rectangular_room_comes_out_a_rectangle():
    surface, _ = box_surface()
    profile = O.profile_from_surface(surface)
    polygon = O.polygon_from_profile(profile, EYE)

    assert len(polygon) == 4
    xs = sorted(x for x, _ in polygon)
    zs = sorted(z for _, z in polygon)
    # The room is 5.2 by 3.9 with the camera at (2.78, 1.71) of it.
    assert abs((xs[-1] - xs[0]) - 5.2) < 0.2
    assert abs((zs[-1] - zs[0]) - 3.9) < 0.2
    assert abs(min(xs) + 2.78) < 0.15
    assert abs(max(zs) - 2.19) < 0.15


def side_lengths(polygon) -> list[float]:
    """The polygon's edge lengths, longest first — its shape without its orientation."""
    n = len(polygon)
    return sorted(
        (math.hypot(polygon[(i + 1) % n][0] - polygon[i][0], polygon[(i + 1) % n][1] - polygon[i][1])
         for i in range(n)),
        reverse=True,
    )


def test_a_room_shot_at_an_angle_still_comes_out_square():
    """The photographer's starting direction must not change the room's shape.

    Rolling the surface map sideways is exactly a different yaw origin, and it is the case a
    developer never sees by accident: the synthetic rooms are axis-aligned in the frame they
    are rendered from, so an estimator with no orientation search at all would pass every
    other test in this file.
    """
    surface, _ = box_surface()
    rolled = np.roll(surface, int(W * 23.0 / 360.0), axis=1)

    polygon = O.polygon_from_profile(O.profile_from_surface(rolled), EYE)
    assert len(polygon) == 4
    sides = side_lengths(polygon)
    assert sides[0] == pytest.approx(5.2, abs=0.2)
    assert sides[1] == pytest.approx(5.2, abs=0.2)
    assert sides[2] == pytest.approx(3.9, abs=0.2)
    assert sides[3] == pytest.approx(3.9, abs=0.2)


def test_an_open_doorway_does_not_stretch_the_room():
    """The failure the fit exists for: a wedge of the NEXT room's floor, seen through a door.

    Those directions are genuinely further away — the measurement is right — and they are not
    this room. Drawn as measured they put a two-metre spike through the wall.
    """
    surface, _ = box_surface(doorway=(20.0, 44.0, 2.5))
    polygon = O.polygon_from_profile(O.profile_from_surface(surface), EYE)

    assert len(polygon) == 4
    sides = side_lengths(polygon)
    assert sides[0] == pytest.approx(5.2, abs=0.25)
    assert sides[2] == pytest.approx(3.9, abs=0.25)


def test_a_wardrobe_does_not_dent_the_room():
    """And the opposite failure: something standing against a wall, hiding it."""
    surface, _ = box_surface()
    blocked = occlude(surface, columns_for(-30.0, 30.0), hides_to_deg=-40.0)
    polygon = O.polygon_from_profile(O.profile_from_surface(blocked), EYE)

    sides = side_lengths(polygon)
    assert len(polygon) == 4
    assert sides[0] == pytest.approx(5.2, abs=0.25)
    assert sides[2] == pytest.approx(3.9, abs=0.25)


def test_the_anchor_room_is_turned_until_its_walls_are_square():
    """A plan tilted by wherever one photographer happened to be standing is a worse drawing.

    There is no north to use instead, so the first room's own walls are the reference.
    """
    surface, _ = box_surface()
    rolled = np.roll(surface, int(W * 23.0 / 360.0), axis=1)
    outline = O.outline_from_surface(rolled)
    plan = A.assemble("t", [A.RoomInput("a", "Salon", ready=True, outline=outline)], [])

    placed = plan.rooms[0]
    assert placed.placement == "anchor"
    assert placed.rotation != 0.0
    for i in range(len(placed.polygon)):
        x0, z0 = placed.polygon[i]
        x1, z1 = placed.polygon[(i + 1) % len(placed.polygon)]
        # Every edge of a squared-up room runs along one axis or the other.
        assert min(abs(x1 - x0), abs(z1 - z0)) < 0.02


def test_an_outline_survives_the_cache():
    surface, _ = box_surface()
    outline = O.outline_from_surface(surface)
    restored = RoomOutline.from_dict(json.loads(json.dumps(outline.as_dict())))

    flat = [value for point in restored.polygon for value in point]
    assert flat == pytest.approx([value for point in outline.polygon for value in point], abs=1e-3)
    assert restored.profile.radius == pytest.approx(outline.profile.radius, abs=1e-3)
    assert restored.profile.ceiling_height == pytest.approx(outline.profile.ceiling_height, abs=1e-3)


# ======================================================================================
# The assembler
# ======================================================================================


def portal(from_id: str, to_id: str, x: float, z: float, *, height: float = EYE) -> A.PortalInput:
    """A ring standing at ``(x, z)`` in its own room's frame."""
    radius = math.hypot(x, z)
    return A.PortalInput(from_id, to_id, math.atan2(x, z), -math.atan2(height, radius))


def square_room(scene_id: str, name: str, half: float = 2.0) -> A.RoomInput:
    profile = O.profile_from_surface(box_surface(2 * half, 2 * half, eye=(half, half))[0])
    outline = RoomOutline(
        polygon=O.polygon_from_profile(profile, EYE),
        profile=profile,
        camera_height_m=EYE,
        source="model",
    )
    return A.RoomInput(scene_id, name, ready=True, outline=outline)


def test_a_doorway_marked_from_both_sides_fixes_the_far_room():
    """A is at the origin; the door is 2 m to its right; B is 2 m beyond it, facing back."""
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak")]
    portals = [portal("a", "b", 2.0, 0.0), portal("b", "a", -2.0, 0.0)]

    plan = A.assemble("t", rooms, portals)
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["a"].placement == "anchor"
    assert placed["b"].placement == "solved"
    assert placed["b"].x == pytest.approx(4.0, abs=0.01)
    assert placed["b"].z == pytest.approx(0.0, abs=0.01)
    assert placed["b"].rotation == pytest.approx(0.0, abs=1e-6)


def test_the_far_rooms_own_yaw_origin_does_not_matter():
    """The same doorway, with B's panorama started facing a different way."""
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak")]
    # B marked the same door straight ahead of itself instead of to its left.
    portals = [portal("a", "b", 2.0, 0.0), portal("b", "a", 0.0, 2.0)]

    plan = A.assemble("t", rooms, portals)
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["b"].x == pytest.approx(4.0, abs=0.01)
    assert placed["b"].z == pytest.approx(0.0, abs=0.01)
    assert placed["b"].rotation == pytest.approx(-math.pi / 2, abs=1e-6)


def test_both_rooms_end_up_agreeing_where_the_door_is():
    """The invariant the solution is built on, checked on the drawing rather than the maths."""
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak")]
    portals = [portal("a", "b", 1.6, 1.2), portal("b", "a", -0.4, -1.9)]

    plan = A.assemble("t", rooms, portals)
    placed = {room.scene_id: room for room in plan.rooms}
    assert len(plan.doors) == 1
    door = plan.doors[0]
    assert door.solved

    def from_room(scene_id: str, local: tuple[float, float]) -> tuple[float, float]:
        room = placed[scene_id]
        x, z = A._rotate(local, room.rotation)
        return (room.x + x, room.z + z)

    assert from_room("a", (1.6, 1.2)) == pytest.approx(from_room("b", (-0.4, -1.9)), abs=0.01)
    assert (door.x, door.z) == pytest.approx(from_room("a", (1.6, 1.2)), abs=0.01)


def test_a_doorway_marked_from_one_side_is_estimated_but_still_beyond_the_door():
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak")]
    plan = A.assemble("t", rooms, [portal("a", "b", 2.0, 0.0)])
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["b"].placement == "estimated"
    # Past the door, on the door's bearing: further out in +x and still on the axis.
    assert placed["b"].x > 2.0
    assert placed["b"].z == pytest.approx(0.0, abs=0.01)


def test_a_room_no_doorway_reaches_is_listed_rather_than_placed():
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak"), square_room("c", "Balkon")]
    plan = A.assemble("t", rooms, [portal("a", "b", 2.0, 0.0), portal("b", "a", -2.0, 0.0)])
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["c"].placement == "unlinked"
    assert placed["c"].polygon == ()
    assert len(plan.drawn) == 2


def test_a_portal_the_player_would_not_draw_is_not_a_doorway():
    """One degree below the horizon is not a floor point; the player refuses it too."""
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak")]
    shallow = A.PortalInput("a", "b", 0.0, math.radians(-1.0))

    plan = A.assemble("t", rooms, [shallow])
    placed = {room.scene_id: room for room in plan.rooms}
    assert placed["b"].placement == "unlinked"
    assert plan.doors == ()


def test_a_room_with_no_panorama_still_gets_a_place_on_the_plan():
    """The amber-ring case: the door is real, the room behind it has not been shot."""
    rooms = [square_room("a", "Salon"), A.RoomInput("b", "Çocuk odası", ready=False, outline=None)]
    plan = A.assemble("t", rooms, [portal("a", "b", 2.0, 0.0)])
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["b"].placement == "estimated"
    assert placed["b"].ready is False
    assert placed["b"].polygon == ()
    assert placed["b"].radius == pytest.approx(A.DEFAULT_ROOM_RADIUS_M)


def test_the_largest_group_of_linked_rooms_is_the_one_drawn():
    rooms = [square_room(sid, sid) for sid in ("a", "b", "c", "d")]
    portals = [
        portal("a", "b", 2.0, 0.0),
        portal("b", "a", -2.0, 0.0),
        portal("c", "d", 2.0, 0.0),
        portal("d", "c", -2.0, 0.0),
        portal("b", "c", 0.0, 2.0),  # one way only, but it joins the two pairs
        portal("c", "b", 0.0, -2.0),
    ]
    plan = A.assemble("t", rooms, portals)
    assert {room.placement for room in plan.rooms} == {"anchor", "solved"}
    assert plan.warnings == ()


# ======================================================================================
# What a person can override
# ======================================================================================


def test_a_placed_room_is_where_the_person_put_it():
    """Not a bias on the solution — the solution, for that room."""
    rooms = [square_room("a", "Salon"), square_room("b", "Mutfak")]
    portals = [portal("a", "b", 2.0, 0.0), portal("b", "a", -2.0, 0.0)]
    placements = {"b": Placement(x=-6.0, z=1.5, rotation=math.pi / 2)}

    plan = A.assemble("t", rooms, portals, placements=placements)
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["b"].placement == "placed"
    assert (placed["b"].x, placed["b"].z) == pytest.approx((-6.0, 1.5))
    assert placed["b"].rotation == pytest.approx(math.pi / 2)


def test_a_property_with_no_doorways_at_all_becomes_plannable_by_hand():
    """The case the solver cannot help with: no rings anywhere, so no graph to solve.

    One room still anchors itself, and every other room is unlinked until somebody puts it
    somewhere. That is the whole point of letting them.
    """
    rooms = [square_room(sid, sid) for sid in ("a", "b", "c")]

    before = {r.scene_id: r.placement for r in A.assemble("t", rooms, []).rooms}
    assert before == {"a": "anchor", "b": "unlinked", "c": "unlinked"}

    after = A.assemble(
        "t",
        rooms,
        [],
        placements={"b": Placement(5.0, 0.0, 0.0), "c": Placement(0.0, 5.0, 0.0)},
    )
    assert {r.scene_id: r.placement for r in after.rooms} == {
        "a": "anchor",
        "b": "placed",
        "c": "placed",
    }
    assert len(after.drawn) == 3


def test_correcting_a_room_carries_what_lies_beyond_it():
    """A placement is a new starting point, not a local nudge.

    Three rooms in a row. Moving the middle one has to take the far one with it, or the
    correction leaves a hole behind and the agent has to drag every room in the flat.
    """
    rooms = [square_room(sid, sid) for sid in ("a", "b", "c")]
    portals = [
        portal("a", "b", 2.0, 0.0),
        portal("b", "a", -2.0, 0.0),
        portal("b", "c", 2.0, 0.0),
        portal("c", "b", -2.0, 0.0),
    ]

    solved = {r.scene_id: (r.x, r.z) for r in A.assemble("t", rooms, portals).rooms}
    moved = A.assemble("t", rooms, portals, placements={"b": Placement(4.0, 9.0, 0.0)})
    placed = {r.scene_id: r for r in moved.rooms}

    assert placed["c"].placement == "solved"
    # `c` kept its relationship to `b` rather than staying where the solver first put it.
    carried = (placed["c"].x - placed["b"].x, placed["c"].z - placed["b"].z)
    original = (solved["c"][0] - solved["b"][0], solved["c"][1] - solved["b"][1])
    assert carried == pytest.approx(original, abs=0.01)


def test_a_placement_survives_a_room_that_cannot_be_measured():
    rooms = [square_room("a", "Salon"), A.RoomInput("b", "Balkon", ready=False, outline=None)]
    plan = A.assemble("t", rooms, [], placements={"b": Placement(3.0, -2.0, 0.0)})
    placed = {room.scene_id: room for room in plan.rooms}

    assert placed["b"].placement == "placed"
    assert placed["b"].polygon == ()
    assert placed["b"].radius == pytest.approx(A.DEFAULT_ROOM_RADIUS_M)


def test_a_placement_for_a_room_that_is_not_in_the_property_is_ignored():
    rooms = [square_room("a", "Salon")]
    plan = A.assemble("t", rooms, [], placements={"ghost": Placement(1.0, 1.0, 0.0)})
    assert [room.scene_id for room in plan.rooms] == ["a"]
    assert plan.rooms[0].placement == "anchor"


class _Storage:
    """The one thing `placements` asks of storage: where the property's directory is."""

    def __init__(self, root):
        self.root = root

    def property_dir(self, property_id: str):
        path = self.root / property_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def test_placements_round_trip_and_clear(tmp_path):
    storage = _Storage(tmp_path)
    assert P.read_placements("t", storage) == {}

    P.write_placements("t", storage, {"a": Placement(1.25, -2.5, 0.75)})
    back = P.read_placements("t", storage)
    assert back["a"].x == pytest.approx(1.25)
    assert back["a"].z == pytest.approx(-2.5)
    assert back["a"].rotation == pytest.approx(0.75)

    # "Nothing was corrected" and "the corrections were undone" are the same state, so the
    # file goes rather than being left behind as an empty decision.
    P.write_placements("t", storage, {})
    assert not P.placements_path("t", storage).exists()
    assert P.read_placements("t", storage) == {}


def test_a_corrupt_placements_file_is_not_an_outage(tmp_path):
    storage = _Storage(tmp_path)
    P.placements_path("t", storage).write_text("{ not json", encoding="utf-8")
    assert P.read_placements("t", storage) == {}


def test_updates_accumulate_and_release(tmp_path):
    storage = _Storage(tmp_path)
    P.update_placements("t", storage, changes={"a": Placement(1.0, 0.0, 0.0)})
    P.update_placements("t", storage, changes={"b": Placement(2.0, 0.0, 0.0)})
    assert sorted(P.read_placements("t", storage)) == ["a", "b"]

    P.update_placements("t", storage, changes={"a": None})
    assert sorted(P.read_placements("t", storage)) == ["b"]

    # One call can clear the flat and pin a room in it, in that order.
    P.update_placements("t", storage, changes={"c": Placement(3.0, 0.0, 0.0)}, reset=True)
    assert sorted(P.read_placements("t", storage)) == ["c"]


def test_two_drags_at_once_do_not_lose_one(tmp_path):
    """Measured, not imagined: every drag is its own request and they overlap.

    Without the lock both threads read the same file and the second write drops whatever the
    first had just added — which in the browser looks like a room springing back.
    """
    import threading

    storage = _Storage(tmp_path)
    ready = threading.Barrier(8)

    def put(index: int) -> None:
        ready.wait()
        P.update_placements(
            "t", storage, changes={f"room{index}": Placement(float(index), 0.0, 0.0)}
        )

    threads = [threading.Thread(target=put, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(P.read_placements("t", storage)) == sorted(f"room{i}" for i in range(8))
