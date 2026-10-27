"""The room's own planes: the geometry the empty-room fill is drawn on, and the fill itself.

Every test here renders its room with :func:`room_planes.trace` — the same function the fill
uses — so the ground truth is exact and cheap: a floor tiled in squares, walls painted with a
skirting band, a white ceiling, all as functions of where a ray lands. "Did the fill rebuild
the room" is then a comparison against the room with nothing standing in it, not an opinion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biosphere.ai import room_planes as rp

W, H = 1024, 512

#: A 4 x 3 room, camera off-centre, in camera heights; ceiling 1.65 camera heights up.
BOX = rp.RoomBox(
    polygon=rp._counter_clockwise(np.array([[-1.6, -1.1], [2.4, -1.1], [2.4, 1.9], [-1.6, 1.9]])),
    ceiling=1.65,
)

#: The colours the reference room is painted in (BGR).
WALL = np.array([196.0, 176.0, 186.0])
SKIRTING = np.array([235.0, 235.0, 235.0])
CEILING = np.array([238.0, 238.0, 240.0])
TILE_A = np.array([210.0, 214.0, 218.0])
TILE_B = np.array([150.0, 154.0, 158.0])
TILE = 0.30  # camera heights


def render(box: rp.RoomBox = BOX, width: int = W, height: int = H) -> np.ndarray:
    """The empty room, exactly: what every fill here is measured against."""
    camera = rp.EquirectCamera(width, height)
    ys, xs = np.mgrid[0:height, 0:width]
    hits = rp.trace(box, camera.rays(xs.astype(np.float64), ys.astype(np.float64)))
    out = np.zeros((height, width, 3), np.float32)
    floor = hits.plane == rp.FLOOR
    px, pz = hits.point[..., 0], hits.point[..., 2]
    checker = ((np.floor(px / TILE) + np.floor(pz / TILE)) % 2 == 0)
    out[floor & checker] = TILE_A
    out[floor & ~checker] = TILE_B
    out[hits.plane == rp.CEILING] = CEILING
    wall = hits.plane >= 0
    out[wall] = WALL
    # A skirting band: the last 0.06 camera heights of every wall above the floor.
    out[wall & (hits.point[..., 1] > 1.0 - 0.06)] = SKIRTING
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def surface_of(box: rp.RoomBox = BOX, width: int = W, height: int = H) -> np.ndarray:
    camera = rp.EquirectCamera(width, height)
    ys, xs = np.mgrid[0:height, 0:width]
    plane = rp.trace(box, camera.rays(xs.astype(np.float64), ys.astype(np.float64))).plane
    return np.where(plane == rp.FLOOR, 2, np.where(plane == rp.CEILING, 0, 1)).astype(np.uint8)


def furnished(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A dark block standing against a wall and on the floor — a wardrobe — and its mask."""
    out = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    rows = slice(int(H * 0.40), int(H * 0.72))
    cols = slice(int(W * 0.60), int(W * 0.72))
    out[rows, cols] = (40, 50, 70)
    mask[rows, cols] = 255
    return out, mask


# ======================================================================================
# Geometry
# ======================================================================================


def test_a_ray_straight_down_meets_the_floor_one_camera_height_below() -> None:
    hits = rp.trace(BOX, np.array([[0.0, 1.0, 0.0]]))
    assert hits.plane[0] == rp.FLOOR
    assert hits.point[0, 1] == pytest.approx(1.0)


def test_a_horizontal_ray_meets_the_wall_it_points_at() -> None:
    hits = rp.trace(BOX, np.array([[1.0, 0.0, 0.0]]))
    assert hits.plane[0] >= 0
    assert hits.point[0, 0] == pytest.approx(2.4)
    assert hits.point[0, 1] == pytest.approx(0.0)


def test_a_ray_straight_up_meets_the_ceiling_at_its_height() -> None:
    hits = rp.trace(BOX, np.array([[0.0, -1.0, 0.0]]))
    assert hits.plane[0] == rp.CEILING
    assert hits.point[0, 1] == pytest.approx(-(BOX.ceiling - 1.0))


def test_every_ray_of_a_closed_room_meets_something() -> None:
    camera = rp.EquirectCamera(256, 128)
    ys, xs = np.mgrid[0:128, 0:256]
    hits = rp.trace(BOX, camera.rays(xs.astype(np.float64), ys.astype(np.float64)))
    assert np.all((hits.plane >= 0) | (hits.plane == rp.FLOOR) | (hits.plane == rp.CEILING))


def test_the_box_knows_it_is_convex_and_the_camera_is_inside() -> None:
    assert BOX.convex_around_eye()
    outside = rp.RoomBox(polygon=BOX.polygon + np.array([5.0, 0.0]), ceiling=BOX.ceiling)
    assert not outside.convex_around_eye()


def test_an_equirect_camera_round_trips_its_own_rays() -> None:
    camera = rp.EquirectCamera(W, H)
    xs = np.array([0.0, 100.5, 511.0, 1023.0])
    ys = np.array([10.0, 255.5, 300.0, 500.0])
    u, v, inside = camera.project(camera.rays(xs, ys))
    assert inside.all()
    assert np.allclose(u, xs, atol=1e-3)
    assert np.allclose(v, ys, atol=1e-3)


def test_every_wall_raster_shares_one_pixel_size_so_rows_are_heights() -> None:
    rasters = rp.plane_rasters(BOX)
    walls = [r for pid, r in rasters.items() if pid >= 0]
    assert len({round(r.pixel, 9) for r in walls}) == 1
    assert len({r.height for r in walls}) == 1


def test_the_floor_raster_sees_a_regular_tiling() -> None:
    """Rectified top-down, a tiled floor is a lattice — which is what the fill relies on."""
    image = render()
    raster = rp.plane_rasters(BOX)[rp.FLOOR]
    rect = rp.rectify(
        image, rp.EquirectCamera(W, H), BOX, raster, hole=np.zeros((H, W), np.uint8), surface=surface_of()
    )
    assert rect is not None
    grey = rect.colour.mean(axis=2)
    # Brightness steps every TILE camera heights along the raster's own axes, on every row.
    spacings = []
    for index in range(raster.height // 4, 3 * raster.height // 4, 7):
        row, valid = grey[index], rect.valid[index]
        jumps = np.nonzero(valid[1:] & valid[:-1] & (np.abs(np.diff(row)) > 25))[0]
        step = np.diff(jumps) * raster.pixel
        spacings.extend(step[(step > 0.05) & (step < 2 * TILE)].tolist())
    assert len(spacings) >= 10
    assert np.median(spacings) == pytest.approx(TILE, rel=0.08)


# ======================================================================================
# Fitting the box
# ======================================================================================


def _junctions_of(box: rp.RoomBox, step_deg: float = 0.5) -> rp.Junctions:
    """The floor and ceiling lines of ``box``, as a camera would see them in every direction."""
    theta = np.radians(np.arange(-180.0, 180.0, step_deg))
    rays = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    hits = rp.trace(box, rays)
    radius = np.hypot(hits.wall_point[:, 0], hits.wall_point[:, 2])
    return rp.Junctions(
        floor_theta=theta,
        floor_depression=np.arctan2(1.0, radius),
        ceiling_theta=theta,
        ceiling_elevation=np.arctan2(box.ceiling - 1.0, radius),
    )


def test_the_fit_recovers_a_room_from_its_own_junctions() -> None:
    fitted, report = rp.fit_box(_junctions_of(BOX))
    assert fitted is not None, report
    assert fitted.ceiling == pytest.approx(BOX.ceiling, abs=0.03)
    ours = sorted(np.round(np.hypot(*(fitted.wall(i)[1] - fitted.wall(i)[0])), 2) for i in range(4))
    theirs = sorted(np.round(np.hypot(*(BOX.wall(i)[1] - BOX.wall(i)[0])), 2) for i in range(4))
    assert ours == pytest.approx(theirs, abs=0.06)


def test_furniture_tops_in_front_of_a_wall_do_not_pull_it_in() -> None:
    """A third of one wall's ceiling line read off a wardrobe's top, 15 % nearer than the wall."""
    junctions = _junctions_of(BOX)
    near = (junctions.ceiling_theta > math.radians(60)) & (junctions.ceiling_theta < math.radians(81))
    junctions.ceiling_elevation = np.where(
        near, np.arctan(np.tan(junctions.ceiling_elevation) / 0.85), junctions.ceiling_elevation
    )
    fitted, _ = rp.fit_box(junctions)
    assert fitted is not None
    clean, _ = rp.fit_box(_junctions_of(BOX))
    assert np.allclose(np.sort(fitted.polygon, axis=0), np.sort(clean.polygon, axis=0), atol=0.08)


def test_no_floor_readings_means_no_floor_line_correction() -> None:
    image = render()
    height, width = image.shape[:2]
    shift, report = rp.floor_line_shift(
        image,
        BOX,
        np.zeros((height, width), bool),
        np.zeros((height, width), bool),
        np.zeros((height, width), np.uint8),
    )
    assert np.all(shift == 0.0)
    assert report["columns_read"] == 0


# ======================================================================================
# Filling
# ======================================================================================


@pytest.fixture(scope="module")
def emptied() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = render()
    image, mask = furnished(truth)
    out, filled, _ = rp.fill_planes(image, rp.EquirectCamera(W, H), BOX, mask, surface_of())
    return truth, image, mask, out


def test_the_fill_only_writes_inside_its_hole(emptied) -> None:
    _, image, mask, out = emptied
    outside = mask == 0
    assert np.array_equal(out[outside], image[outside])


def test_the_fill_rebuilds_the_room_behind_a_wardrobe(emptied) -> None:
    truth, image, mask, out = emptied
    hole = mask > 0
    before = np.abs(image.astype(np.float32) - truth.astype(np.float32))[hole].mean()
    after = np.abs(out.astype(np.float32) - truth.astype(np.float32))[hole].mean()
    # The wardrobe was ~140 levels from the room; the rebuilt wall and floor are paint and tile.
    assert before > 100.0
    assert after < 18.0


def test_the_floor_line_is_drawn_where_the_room_puts_it(emptied) -> None:
    """Inside the hole, the rows either side of the true wall/floor line match the empty room.

    A line drawn even three rows off puts white skirting where tile belongs (or tile where the
    skirting is), which is fifty levels and more in this band; drawn in place it is the tile's
    and the skirting's own colours give or take the fill's grain.
    """
    truth, _, mask, out = emptied
    surface = surface_of()
    errors = []
    for col in np.nonzero(mask.any(axis=0))[0][3:-3]:
        true_row = int(np.argmax(surface[H // 2 :, col] == 2)) + H // 2
        band = slice(true_row - 3, true_row + 4)
        if not mask[band, col].all():
            continue
        difference = np.abs(out[band, col].astype(np.float32) - truth[band, col].astype(np.float32))
        errors.append(float(difference.max(axis=1).mean()))
    assert len(errors) > 20
    assert np.median(errors) < 25.0


def test_the_skirting_continues_through_the_hole(emptied) -> None:
    truth, _, mask, out = emptied
    surface = surface_of()
    hits = 0
    total = 0
    for col in np.nonzero(mask.any(axis=0))[0][2:-2]:
        rows = np.nonzero((np.abs(truth[:, col].astype(np.float32) - SKIRTING).max(axis=1) < 2) & (mask[:, col] > 0))[0]
        if rows.size == 0:
            continue
        total += 1
        values = out[rows, col].astype(np.float32)
        hits += int(np.median(np.abs(values - SKIRTING).max(axis=1)) < 20)
    assert total > 0
    assert hits / total > 0.8


def test_the_same_room_empties_the_same_way_twice() -> None:
    truth = render()
    image, mask = furnished(truth)
    first, _, _ = rp.fill_planes(image, rp.EquirectCamera(W, H), BOX, mask, surface_of())
    second, _, _ = rp.fill_planes(image, rp.EquirectCamera(W, H), BOX, mask, surface_of())
    assert np.array_equal(first, second)
