"""The edge-aware seam cost: does it cost what it claims, and does it move the cut?

Every test here is about :mod:`biosphere.pano.seams`' second cost term -- the one that does
not depend on which pair of cameras is being compared.  The claims it rests on are
arithmetic (the penalty is exactly what the constant says, and the pairwise driver is the
same algorithm OpenCV runs) plus two behavioural ones (the cut leaves a coherent line alone,
and it stops ruling a straight line across a featureless surface).  All four are checked
here rather than trusted, because all four are silent when they break: a wrong penalty scale
just makes slightly different seams.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosphere.pano import seams as seams_mod


def _full(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, np.uint8)


def _boundary_rows(mask_a: np.ndarray, y_off: int, h: int, w: int) -> np.ndarray:
    """Row of the top patch's last owned pixel, per column, inside the overlap band."""
    band = mask_a[y_off : y_off + h] > 0
    rows = np.full(w, -1, np.int32)
    for x in range(w):
        owned = np.flatnonzero(band[:, x])
        if owned.size:
            rows[x] = int(owned.max())
    return rows


# --------------------------------------------------------------------------------------
# 1. The penalty costs exactly what its constant says
# --------------------------------------------------------------------------------------


def test_the_bump_adds_exactly_three_grey_levels_squared():
    """``4 a^2 = 3 g^2``: the added squared colour distance must be 3 * g**2, exactly.

    This is the whole reason the offset is built orthogonal to the pair's own difference.
    A fixed direction would leave a cross term of up to ``2 a |I_a - I_b|``, which on a
    strongly disagreeing pixel is larger than the penalty itself and can *cancel* it -- the
    cut would then be attracted to the very pixels the penalty exists to protect.
    """
    rng = np.random.default_rng(7)
    a = rng.uniform(0.0, 255.0, (64, 64, 3)).astype(np.float32)
    b = rng.uniform(0.0, 255.0, (64, 64, 3)).astype(np.float32)
    # Include exactly-agreeing pixels, where the orthogonal direction is undefined.
    b[:8] = a[:8]
    # ...and grey-only differences, which collapse a single fixed cross-product axis.
    b[8:16] = a[8:16] + rng.uniform(-30.0, 30.0, (8, 64, 1)).astype(np.float32)

    grey = rng.uniform(0.0, 25.0, (64, 64)).astype(np.float32)
    bump = seams_mod._orthogonal_bump(a, b, grey)

    before = ((a - b) ** 2).sum(axis=-1)
    after = (((a + bump) - (b - bump)) ** 2).sum(axis=-1)
    expected = 3.0 * grey * grey
    worst = float(np.abs(after - before - expected).max())
    print(f"\n[penalty] worst error over 4096 px = {worst:.4g} cost units")
    assert worst < 0.05, f"penalty is not 3*g^2: worst error {worst}"


# --------------------------------------------------------------------------------------
# 2. Driving OpenCV pair by pair is the same algorithm
# --------------------------------------------------------------------------------------


def test_pairwise_driver_reproduces_the_single_call_graph_cut():
    """With no penalty, the Python pair loop must equal ``GraphCutSeamFinder.find`` exactly.

    If this ever fails, every A/B measured with the edge-aware cost is confounded: a
    difference in the panorama could be the penalty or could be the driver.
    """
    rng = np.random.default_rng(11)
    world = rng.integers(0, 255, (90, 320, 3)).astype(np.float32)
    imgs, masks, corners = [], [], []
    for k, x0 in enumerate((0, 60, 120, 170)):
        imgs.append(world[:, x0 : x0 + 140].copy() + k * 2.0)
        masks.append(_full(90, 140))
        corners.append((x0, 0))

    ref = seams_mod._run_graphcut(imgs, corners, masks)
    ref = [seams_mod.as_numpy_mask(m) for m in ref]
    got = seams_mod._run_graphcut_edge_aware(
        imgs, corners, masks, np.zeros((90, 320), np.float32)
    )
    print(f"\n[parity] owned px single-call {[int((m > 0).sum()) for m in ref]}")
    print(f"[parity] owned px pairwise    {[int((m > 0).sum()) for m in got]}")
    assert all(np.array_equal(a, b) for a, b in zip(ref, got, strict=True))


def test_pairs_whose_masks_do_not_touch_are_skipped_without_changing_the_cut():
    """The canvas-wide window of a view straddling the wrap overlaps every rectangle and
    almost no mask. The driver skips those pairs; the partition must still equal OpenCV's.

    The fixture is that case in miniature: one patch spans the whole canvas but its mask holds
    only its two ends (the halves of a view cut by the canvas edge), and three ordinary patches
    sit in the middle -- one of them touching nothing but its neighbour.
    """
    rng = np.random.default_rng(17)
    world = rng.integers(0, 255, (80, 400, 3)).astype(np.float32)
    imgs, masks, corners = [], [], []
    wrap = _full(80, 400)
    wrap[:, 60:340] = 0
    imgs.append(world.copy())
    masks.append(wrap)
    corners.append((0, 0))
    for k, x0 in enumerate((40, 150, 260)):
        imgs.append(world[:, x0 : x0 + 110].copy() + 3.0 * (k + 1))
        masks.append(_full(80, 110))
        corners.append((x0, 0))

    ref = [seams_mod.as_numpy_mask(m) for m in seams_mod._run_graphcut(imgs, corners, masks)]
    got = seams_mod._run_graphcut_edge_aware(imgs, corners, masks, np.zeros((80, 400), np.float32))
    assert all(np.array_equal(a, b) for a, b in zip(ref, got, strict=True))


def test_the_driver_leaves_the_images_it_was_given_unchanged():
    """The bump is applied in place for memory; a leaked bump would poison later pairs."""
    rng = np.random.default_rng(13)
    imgs = [rng.uniform(0, 255, (60, 100, 3)).astype(np.float32) for _ in range(3)]
    before = [im.copy() for im in imgs]
    corners = [(0, 0), (40, 0), (80, 0)]
    masks = [_full(60, 100) for _ in imgs]
    pen = np.full((60, 180), 20.0, np.float32)
    seams_mod._run_graphcut_edge_aware(imgs, corners, masks, pen)
    worst = max(float(np.abs(a - b).max()) for a, b in zip(imgs, before, strict=True))
    print(f"\n[restore] worst residue after 3 pairs = {worst:.3g} grey levels")
    assert worst < 1e-3


# --------------------------------------------------------------------------------------
# 3. A coherent line is scored, a busy surface is not
# --------------------------------------------------------------------------------------


def test_a_moulding_scores_far_above_a_carpet():
    """The structure term must separate *coherent* contrast from merely *strong* contrast.

    A seam through a carpet is invisible and a seam through a cornice is not, so a penalty
    that fired on both would be worse than none: it would push the cut off the only places
    it can safely go.
    """
    # At the default seam scale, which is where the derivative and integration scales are
    # tuned: on a small canvas both collapse to their pixel floors and the noise rejection
    # is weaker than the delivered pipeline's.
    cw, ch = 2048, 400
    img = np.full((ch, cw, 3), 150.0, np.float32)
    img[100:104, :] = 215.0  # a moulding: high contrast, one orientation
    rng = np.random.default_rng(3)
    img[260:, :] += rng.normal(0.0, 26.0, (ch - 260, cw, 1)).astype(np.float32)  # carpet
    # The reach the delivered pipeline uses: the blender's coarsest band, 2**6 canvas px,
    # halved by the balanced mode's seam scale.
    reach = 32
    pen = seams_mod._line_penalty([img], [_full(ch, cw)], [(0, 0)], cw, ch, reach)

    # Sampled clear of the moulding's cone, which now reaches `reach` pixels either side.
    line = float(pen[98:107, 200:1848].mean())
    carpet = float(pen[300:360, 200:1848].mean())
    wall = float(pen[200:250, 200:1848].mean())
    print(f"\n[structure] moulding {line:.1f}, carpet {carpet:.1f}, blank wall {wall:.1f} DN")
    assert line > 10.0, "a moulding must be expensive to cut along"
    # A carpet is where a seam SHOULD go, so the structure term must be near zero on it. The
    # ruling term is allowed to fire there -- it is small by construction and only breaks
    # ties -- so the budget is its ceiling plus a little.
    assert carpet < seams_mod._RULING_PENALTY_GREY + 3.0, (
        "a carpet must stay cheap: it hides a seam, it does not show one"
    )
    # The blank wall carries only the ruling dither, which is bounded by its own constant.
    assert wall <= seams_mod._RULING_PENALTY_GREY + 1e-3


def test_the_cut_walks_around_a_line_the_two_views_agree_on():
    """The artefact this whole stage exists for, in miniature.

    Two views of a wall that is textured everywhere except for one clean bright bar.  The
    views disagree on the texture (different noise, as parallax and resampling guarantee)
    and agree exactly on the bar -- the aperture problem, because a bar slid along itself
    looks identical.  Colour cost alone therefore makes the bar the cheapest path in the
    whole overlap and the seam runs down it, handing over a structure the two sides place
    differently.  The penalty has to be worth more than the free ride.
    """
    h, w, overlap = 140, 220, 80
    cw, ch = w, h + (h - overlap)
    bar_row = h - overlap // 2  # dead centre of the overlap band, where the cut wants to be

    def view(seed: int) -> np.ndarray:
        img = np.full((ch, cw, 3), 130.0, np.float32)
        img += rng_of(seed).normal(0.0, 18.0, (ch, cw, 1)).astype(np.float32)
        img[bar_row - 2 : bar_row + 3, :] = 220.0  # identical in both views
        return img

    def rng_of(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    world_a, world_b = view(21), view(22)
    imgs = [world_a[:h].copy(), world_b[h - overlap :].copy()]
    corners = [(0, 0), (0, h - overlap)]
    masks = [_full(h, w), _full(imgs[1].shape[0], w)]

    plain = [seams_mod.as_numpy_mask(m) for m in seams_mod._run_graphcut(imgs, corners, masks)]
    # Scaled to this fixture: a quarter of the overlap band, so the cone has somewhere to
    # fall away to and the test measures steering rather than a blanket cost.
    pen = seams_mod._line_penalty(imgs, masks, corners, cw, ch, overlap // 4)
    aware = seams_mod._run_graphcut_edge_aware(imgs, corners, masks, pen)

    def on_the_bar(mask_a: np.ndarray) -> float:
        rows = _boundary_rows(mask_a, 0, h, w)
        seen = rows[rows >= 0]
        return float(np.mean(np.abs(seen - bar_row) <= 3))

    before, after = on_the_bar(plain[0]), on_the_bar(aware[0])
    print(f"\n[line] penalty on the bar {float(pen[bar_row, w // 2]):.1f} DN")
    print(f"[line] seam columns within 3 px of the bar: {before:.0%} -> {after:.0%}")
    assert before > 0.5, "fixture is not reproducing the artefact; nothing is being tested"
    assert after < 0.2 * before, f"the cut still follows the line: {before:.0%} -> {after:.0%}"


# --------------------------------------------------------------------------------------
# 4. A free cut across a featureless surface stops being a ruled line
# --------------------------------------------------------------------------------------


def test_a_free_cut_across_a_flat_surface_meanders():
    """Two views that agree exactly, so the cut is free and the colour term says nothing.

    A min-cut through a uniform field is a straight line, and that is what the finder draws:
    a ruled boundary with a sub-grey-level step, which is far more visible than the step
    deserves. The ruling term denies the field its uniformity.
    """
    h, w, overlap = 120, 1440, 60
    cw, ch = w, h + (h - overlap)
    world = np.full((ch, cw, 3), 140.0, np.float32)
    imgs = [world[:h].copy(), world[h - overlap :].copy()]
    corners = [(0, 0), (0, h - overlap)]
    masks = [_full(h, w), _full(imgs[1].shape[0], w)]

    plain = [seams_mod.as_numpy_mask(m) for m in seams_mod._run_graphcut(imgs, corners, masks)]
    pen = seams_mod._line_penalty(imgs, masks, corners, cw, ch, overlap // 4)
    aware = seams_mod._run_graphcut_edge_aware(imgs, corners, masks, pen)

    def spread(mask_a: np.ndarray) -> tuple[float, int]:
        rows = _boundary_rows(mask_a, 0, h, w)
        seen = rows[rows >= 0]
        # Longest run of columns at one row: the length of the ruled segment a viewer reads.
        best = run = 1
        for k in range(1, seen.size):
            run = run + 1 if seen[k] == seen[k - 1] else 1
            best = max(best, run)
        return float(seen.std()), int(best)

    sd_a, run_a = spread(plain[0])
    sd_b, run_b = spread(aware[0])
    print(f"\n[ruling] boundary row sd {sd_a:.2f} -> {sd_b:.2f} px")
    print(f"[ruling] longest ruled run {run_a} -> {run_b} columns of {w}")
    assert run_a > w // 3, "fixture is not producing a ruled cut; nothing is being tested"
    assert sd_b > 1.0, "the cut is still flat"
    assert run_b < run_a // 4, f"the cut is still ruled: {run_a} -> {run_b} columns"


# --------------------------------------------------------------------------------------
# 5. The switch really is a switch
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_the_master_switch_selects_the_backend(monkeypatch, enabled):
    """`_EDGE_AWARE_SEAM_COST` is what a bench script flips; it must actually reach the cut."""
    from biosphere.pano import types as types_mod
    from biosphere.pano import warp as warp_mod

    monkeypatch.setattr(seams_mod, "_EDGE_AWARE_SEAM_COST", enabled)
    cfg = types_mod.PipelineConfig(seam_finder="graphcut", seam_scale_width=256)
    rng = np.random.default_rng(17)
    canvas_w, canvas_h = 256, 128
    patches = []
    for k, x0 in enumerate((0, 60, 120)):
        img = rng.integers(0, 255, (80, 110, 3)).astype(np.uint8)
        patches.append(
            warp_mod.WarpedPatch(
                index=k,
                frame_index=k,
                image=img,
                mask=_full(80, 110),
                corner=(x0, 20),
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                pose=None,
            )
        )
    _hard, _feather, backend = seams_mod.find_seams_full(patches, cfg)
    print(f"\n[switch] enabled={enabled} -> backend {backend!r}")
    assert backend == ("graphcut+lines" if enabled else "graphcut")
