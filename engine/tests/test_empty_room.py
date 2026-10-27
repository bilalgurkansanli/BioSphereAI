"""What "the furniture is gone" means, as a number.

Two kinds of test here, and the split is deliberate.

**Property tests** use a panorama built by hand out of three flat bands and a box. They are
exact: the answer is known to the pixel, so they can assert that the seam is not an edge,
that the poles are untouched, that the layout is monotone and that the same input gives the
same output. Nothing about them is approximate, and a failure in one is a bug rather than a
regression in quality.

**The measurement** uses ``biosphere.synth`` — the same analytic ray tracer the stitch quality
gate runs on. It renders one room twice, once furnished and once with the furniture taken out
of the scene description, and the second render *is* the correct answer. That is what makes
"did it remove the furniture" an experiment rather than an opinion: the removal mask can be
scored against a ground truth nobody drew by hand.

The thresholds below are floors under measured behaviour, not targets. They are set clear of
the current numbers so that a real regression trips them and a tenth of a point of noise does
not. Where the algorithm is known to fail — furniture the colour of the floor it stands on —
that is asserted as a *limit*, so the day a learned ``SEGMENT_BACKEND`` lands, the test that
records the limitation is the one that has to be updated on purpose.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from biosphere.ai import empty_room as er
from biosphere.pano import geometry as geo

# ======================================================================================
# Fixtures
# ======================================================================================

#: Small enough that a test file runs in seconds, large enough that a 5-degree sector still
#: holds several columns at the decision width.
W, H = 512, 256


def phi_degrees(width: int = W, height: int = H) -> np.ndarray:
    _, phi = geo.equirect_angles(width, height, dtype=np.float32)
    return np.degrees(phi).astype(np.float32)


def flat_room(
    *,
    box: tuple[int, int, int, int] | None = None,
    box_colour: tuple[int, int, int] = (40, 40, 200),
    width: int = W,
    height: int = H,
) -> np.ndarray:
    """A three-band room: grey ceiling, beige walls, brown floor, plus one box.

    Flat by design. Every pixel's correct surface is known from its row alone, so anything
    the algorithm gets wrong here is wrong for a reason that has nothing to do with texture.
    ``box`` is ``(x0, y0, x1, y1)`` and may straddle the seam by passing ``x0 > x1``.
    """
    image = np.empty((height, width, 3), np.uint8)
    rows = np.arange(height)[:, None]
    ceiling_row, floor_row = int(height * 0.33), int(height * 0.62)
    image[...] = (150, 148, 145)                       # beige wall
    image[: ceiling_row, :] = (225, 224, 222)          # near-white ceiling
    image[floor_row:, :] = (60, 90, 130)               # brown floor
    # A little vertical shading, so nothing passes by being perfectly uniform.
    shade = (1.0 - 0.12 * (rows / float(height))).astype(np.float32)
    image = np.clip(image.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    if box is not None:
        x0, y0, x1, y1 = box
        if x0 <= x1:
            image[y0:y1, x0:x1] = box_colour
        else:  # straddles the cyclic seam
            image[y0:y1, x0:] = box_colour
            image[y0:y1, :x1] = box_colour
    return image


@pytest.fixture(scope="module")
def reference_rooms() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(name, furnished, truth)`` for a few procedural rooms.

    ``truth`` is where the furnished render and the same room with its furniture deleted
    actually differ — a ground truth that was rendered, not annotated.
    """
    from biosphere.synth.render import render_equirect
    from biosphere.synth.room import Room

    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for seed, style in ((7, "rich"), (3, "low_texture"), (11, "rich")):
        room = Room.random(seed=seed, style=style)
        bare = dataclasses.replace(room, boxes=[], rug=None)
        furnished = render_equirect(room, W, H)
        empty = render_equirect(bare, W, H)
        difference = np.abs(furnished.astype(np.int16) - empty.astype(np.int16)).max(axis=2)
        out.append((f"{seed}/{style}", furnished, difference > 18))
    return out


def run_classical(image: np.ndarray, strength: float) -> er.RoomPlan:
    """Always the colour path.

    These tests are about the classical segmentation and the classical fill, and on a machine
    where the models are installed `plan_empty_room` would quietly be testing neither. The
    model path has its own test below, which skips when the weights are absent.
    """
    return er.plan_empty_room(image, strength=strength, allow_model=False)


def scored(mask: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """``(recall, precision)`` of a removal mask against the rendered ground truth."""
    found = mask > 0
    hit = float((found & truth).sum())
    missed = float((~found & truth).sum())
    wrong = float((found & ~truth).sum())
    return hit / max(hit + missed, 1.0), hit / max(hit + wrong, 1.0)


# ======================================================================================
# Cyclic helpers — the canvas wraps, and every one of these has to know it
# ======================================================================================


def test_cyclic_smooth_crosses_the_seam() -> None:
    values = np.zeros(64, np.float32)
    values[0] = 10.0
    smoothed = er._cyclic_smooth(values, sigma=2.0)
    # The spike at column 0 has to bleed into the LAST column as well as the second.
    assert smoothed[-1] > 0.5
    assert smoothed[-1] == pytest.approx(smoothed[1], rel=1e-5)


def test_cyclic_fill_interpolates_through_the_seam() -> None:
    values = np.zeros(32, np.float32)
    trusted = np.ones(32, bool)
    values[:] = 5.0
    values[30:] = 0.0
    values[:2] = 0.0
    trusted[30:] = False
    trusted[:2] = False
    filled = er._cyclic_fill(values, trusted)
    # The untrusted run straddles column 0; every entry in it takes the surrounding value
    # rather than being clamped to an end.
    assert np.allclose(filled, 5.0)


def test_cyclic_percentile_is_a_median_over_a_window() -> None:
    values = np.array([1.0] * 10 + [9.0] + [1.0] * 10, np.float32)
    out = er._cyclic_percentile(values, window=5, q=50.0)
    assert out[10] == pytest.approx(1.0), "a single spike must not survive the window"


def test_wrap_pad_takes_the_opposite_edge() -> None:
    image = np.arange(12, dtype=np.uint8).reshape(1, 12, 1)
    padded = er._wrap_pad(image, 2)
    assert padded.shape[1] == 16
    assert padded[0, 0, 0] == 10 and padded[0, 1, 0] == 11
    assert padded[0, -2, 0] == 0 and padded[0, -1, 0] == 1


def test_pyramid_round_trip_keeps_the_canvas_size() -> None:
    image = np.random.default_rng(0).random((64, 128, 3)).astype(np.float32)
    down = er._pyr_down_cyclic(image)
    assert down.shape[:2] == (32, 64)
    up = er._pyr_up_cyclic(down, 64, 128)
    assert up.shape[:2] == (64, 128)


# ======================================================================================
# The layout solver
# ======================================================================================


def test_segmentation_is_monotone_down_every_column() -> None:
    """Ceiling, then wall, then floor. Never back up — the room has no other shape."""
    rng = np.random.default_rng(1)
    cost = rng.random((3, 40, 16)).astype(np.float32)
    labels = er._segment_columns(cost)
    assert np.all(np.diff(labels.astype(np.int16), axis=0) >= 0)
    assert labels.min() >= 0 and labels.max() <= 2


def test_layout_finds_the_bands_of_a_flat_room() -> None:
    image = flat_room()
    shell = er.estimate_shell(er._to_lab(image), phi_degrees())
    expected_ceiling, expected_floor = int(H * 0.33), int(H * 0.62)
    assert shell.ceiling_row.mean() == pytest.approx(expected_ceiling, abs=6)
    assert shell.floor_row.mean() == pytest.approx(expected_floor, abs=8)


# ======================================================================================
# Removal
# ======================================================================================


def test_removes_a_box_and_leaves_the_room_alone() -> None:
    box = (150, 120, 210, 175)
    plan = run_classical(flat_room(box=box), strength=0.6)

    x0, y0, x1, y1 = box
    inside = plan.mask[y0 + 8 : y1 - 8, x0 + 8 : x1 - 8]
    assert inside.mean() > 200, "the box itself was not selected for removal"

    # Two thirds of the canvas is plain room; almost none of it may be touched.
    assert plan.mask[: int(H * 0.30)].mean() < 4.0, "the ceiling was disturbed"
    assert plan.mask[:, 300:].mean() < 4.0, "a wall nowhere near the box was disturbed"

    assert plan.regions, "a removed box has to be reported"
    assert plan.removed_fraction > 0.0
    assert 0.0 < plan.removed_fraction < 0.5


def test_the_fill_replaces_the_box_with_the_room_behind_it() -> None:
    box = (150, 120, 210, 175)
    image = flat_room(box=box)
    plan = run_classical(image, strength=0.6)

    x0, y0, x1, y1 = box
    centre = plan.image[y0 + 12 : y1 - 12, x0 + 12 : x1 - 12].reshape(-1, 3).mean(axis=0)
    original = image[y0 + 12 : y1 - 12, x0 + 12 : x1 - 12].reshape(-1, 3).mean(axis=0)
    # The box was blue; whatever replaced it must not be.
    assert np.linalg.norm(centre - original) > 40.0, "the box is still there"

    # Rows either side of the wall-floor junction must come back as wall and as floor, not
    # as an average of the two: this is what the per-surface push-pull exists to guarantee.
    wall_reference = image[y0 + 12, 40].astype(np.float32)
    floor_reference = image[y1 - 12, 40].astype(np.float32)
    assert np.linalg.norm(plan.image[y0 + 12, (x0 + x1) // 2].astype(np.float32) - wall_reference) < 40.0
    assert np.linalg.norm(plan.image[y1 - 12, (x0 + x1) // 2].astype(np.float32) - floor_reference) < 40.0


def test_pixels_outside_the_mask_are_bit_identical() -> None:
    image = flat_room(box=(150, 120, 210, 175))
    plan = run_classical(image, strength=0.6)
    untouched = plan.mask == 0
    assert np.array_equal(plan.image[untouched], image[untouched])


def test_the_poles_are_never_touched() -> None:
    """The stitcher synthesised them and may have put branding there. Twice-invented pixels
    are worse than once-invented ones, so the caps are out of bounds by construction."""
    image = flat_room()
    # Something maximally foreign at both poles.
    image[:6, :] = (0, 0, 255)
    image[-6:, :] = (0, 255, 0)
    plan = run_classical(image, strength=1.0)
    assert plan.mask[:6, :].max() == 0
    assert plan.mask[-6:, :].max() == 0
    assert np.array_equal(plan.image[:6, :], image[:6, :])
    assert np.array_equal(plan.image[-6:, :], image[-6:, :])


def test_an_object_on_the_seam_is_one_object() -> None:
    """Column 0 and column W-1 are neighbours. A box across them is not two boxes."""
    plan = run_classical(flat_room(box=(W - 30, 120, 30, 175)), strength=0.6)
    assert plan.mask[140, 0] > 0 and plan.mask[140, W - 1] > 0

    # And the repair has to be continuous across the join, not two fills meeting at a wall.
    left = plan.image[125:170, W - 1].astype(np.float32)
    right = plan.image[125:170, 0].astype(np.float32)
    assert float(np.abs(left - right).mean()) < 12.0, "a seam appeared where the fill wrapped"


def test_running_it_twice_gives_the_same_answer() -> None:
    """A toggle that empties the room differently on the second press is a toggle nobody
    trusts, so every step from the palette to the grain transplant is deterministic."""
    image = flat_room(box=(150, 120, 210, 175))
    first = run_classical(image, strength=0.6)
    second = run_classical(image, strength=0.6)
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.image, second.image)


def test_strength_is_monotone() -> None:
    image = flat_room(box=(150, 120, 210, 175))
    areas = [float((run_classical(image, strength=s).mask > 0).mean()) for s in (0.1, 0.9)]
    assert areas[1] >= areas[0], "a higher strength must never remove less"


def test_nothing_to_remove_reports_no_regions() -> None:
    plan = run_classical(flat_room(), strength=0.5)
    assert plan.regions == []
    assert plan.removed_fraction == 0.0


# ======================================================================================
# The measurement
# ======================================================================================

#: Floors under measured behaviour on the reference rooms, with room for noise.
#:
#: Precision is the one that carries the product. A furnishing left standing is a job half
#: done and the user can see that it was; a smear where the cornice used to be is damage they
#: cannot undo, and it is the failure that makes the whole feature untrustworthy. So the bar
#: on not touching the room is set higher than the bar on finding everything.
MIN_MEAN_RECALL = 0.35
MIN_MEAN_PRECISION = 0.55
MIN_PER_ROOM_PRECISION = 0.45


def test_removal_scores_against_a_rendered_ground_truth(
    reference_rooms: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    recalls: list[float] = []
    precisions: list[float] = []
    for name, furnished, truth in reference_rooms:
        plan = run_classical(furnished, strength=0.6)
        recall, precision = scored(plan.mask, truth)
        assert precision >= MIN_PER_ROOM_PRECISION, f"{name}: erasing too much of the room"
        assert recall > 0.2, f"{name}: barely found anything"
        recalls.append(recall)
        precisions.append(precision)

    assert float(np.mean(recalls)) >= MIN_MEAN_RECALL
    assert float(np.mean(precisions)) >= MIN_MEAN_PRECISION


def test_the_known_limitation_is_still_the_known_limitation(
    reference_rooms: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Colour is the only signal, so furniture the colour of its own floor survives.

    This is asserted rather than merely written down, because it is the thing a learned
    ``SEGMENT_BACKEND`` is for. When one lands, recall goes well past this and the test
    fails — which is the point: the limitation should have to be retired deliberately.
    """
    recalls = [scored(run_classical(f, strength=0.85).mask, t)[0] for _, f, t in reference_rooms]
    assert float(np.mean(recalls)) < 0.95, (
        "recall is far above what colour-only segmentation can do — if a semantic backend "
        "is now in use, raise MIN_MEAN_RECALL and delete this test"
    )


# ======================================================================================
# The semantic path
# ======================================================================================


def test_room_shell_keeps_the_building_and_drops_the_furnishings() -> None:
    """The one table this feature's meaning rests on, asserted rather than trusted.

    ``_ROOM_SHELL`` is the definition of "the room" — change an entry and the product
    changes with it, silently. These are the memberships a user would notice: a wall that
    vanished, a toilet that vanished, a bed that stayed.
    """
    from biosphere.ai import furniture_seg as fseg

    # ADE20K ids. The room itself, and the fixtures an estate agent's empty room keeps.
    for name, class_id in (("wall", 0), ("floor", 3), ("ceiling", 5), ("windowpane", 8),
                           ("door", 14), ("column", 42), ("stairs", 53), ("bathtub", 37),
                           ("sink", 47), ("toilet", 65), ("radiator", 146), ("chandelier", 85)):
        assert class_id in fseg._ROOM_SHELL, f"{name} is part of the room and must survive"

    # Curtains and blinds STAY, and this is the one entry that changed its mind. They were
    # removed at first on the reading that a bare room has bare windows. Two things said
    # otherwise: the reference the feature is measured against (Matterport's own emptied
    # room) keeps its curtain, and a curtain covers a whole wall — taking it out replaces
    # real glass with invented glass and spends the context the inpainter needs elsewhere.
    for name, class_id in (("curtain", 18), ("blind", 63)):
        assert class_id in fseg._ROOM_SHELL, f"{name} belongs to the window, not the furniture"

    # Everything a room is furnished WITH, including the things a colour test never caught.
    for name, class_id in (("bed", 7), ("table", 15), ("chair", 19),
                           ("painting", 22), ("sofa", 23), ("shelf", 24), ("mirror", 27),
                           ("rug", 28), ("desk", 33), ("wardrobe", 35), ("lamp", 36),
                           ("cushion", 39), ("box", 41), ("chest of drawers", 44),
                           ("person", 12), ("apparel", 92), ("bag", 115), ("blanket", 131)):
        assert class_id not in fseg._ROOM_SHELL, f"{name} is a furnishing and must go"

    # A lamp stands on the floor and a sconce is screwed to the wall. The distinction is the
    # whole reason this list is hand-written rather than "everything that is not stuff".
    assert 36 not in fseg._ROOM_SHELL and 134 in fseg._ROOM_SHELL


def test_a_mask_drawn_in_a_view_lands_back_where_it_started() -> None:
    """The projection either way round, checked against itself.

    Every rotation convention in this engine is a chance to be off by a transpose or a sign,
    and the symptom here would not be an exception — it would be furniture masks landing on
    the wrong side of the room, which looks like a bad detector rather than bad geometry.
    """
    from biosphere.ai import furniture_seg as fseg

    K = fseg._intrinsics(fseg.VIEW_PIXELS, fseg.VIEW_FOV_DEG)
    centre = fseg.VIEW_PIXELS // 2
    for yaw, pitch in ((0.0, 0.0), (90.0, -30.0), (215.0, 12.0)):
        rotation = fseg._rotation(yaw, pitch)
        drawn = np.zeros((fseg.VIEW_PIXELS, fseg.VIEW_PIXELS), np.uint8)
        cv2.circle(drawn, (centre, centre), 60, 255, -1)

        sphere = np.zeros((512, 1024), np.uint8)
        fseg._accumulate(sphere, drawn, rotation, K)
        assert sphere.any(), f"the mask vanished on the way to the sphere at yaw {yaw}"

        returned = fseg._render_view(cv2.cvtColor(sphere, cv2.COLOR_GRAY2BGR), rotation, K, fseg.VIEW_PIXELS)
        ys, xs = np.nonzero(returned[:, :, 0] > 127)
        assert abs(float(xs.mean()) - centre) < 4.0
        assert abs(float(ys.mean()) - centre) < 4.0


@pytest.mark.slow
def test_the_model_path_empties_a_room_and_keeps_its_shell(
    reference_rooms: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """End to end with whatever segmenter this machine has. Skipped when it has none."""
    from biosphere.ai import furniture_seg as fseg

    if not fseg.available():
        pytest.skip("no scene-parsing backend installed on this machine")

    _, furnished, truth = reference_rooms[0]
    plan = er.plan_empty_room(furnished, strength=0.5, allow_model=True)
    recall, precision = scored(plan.mask, truth)

    assert plan.model_backed, "a model was available, so the run must say it used one"
    # A model that knows what a bed is should beat the colour path's measured recall by a
    # wide margin; the floor under it is that path's own number on the same rooms.
    assert recall > MIN_MEAN_RECALL, f"model recall {recall:.2f} is no better than colour"
    assert precision > 0.4, f"model precision {precision:.2f}: it is eating the room"
