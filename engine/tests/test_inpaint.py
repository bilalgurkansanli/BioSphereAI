"""Which model fills the room, and what the filling machinery does with its answer.

Nothing here needs a GPU, and that is the point. The three things that can actually be wrong
in this half of the feature are wrong without one:

* **The wrong model is asked the wrong question.** A diffusion checkpoint handed the 2:1
  equirect canvas draws the same wall twice; LaMa handed a window of a furnished room has too
  little photograph left to continue. ``backend.local_inpainter`` is the one function that
  decides, so it is tested directly and against every combination of what a machine might
  have installed.
* **The window loop mishandles the backend's answer.** ``room_fill.fill_by_views`` takes the
  inpainter as a parameter precisely so it can be driven by a fake one — which is what lets a
  test assert that windows are filled easiest-first, that a refusal is not silently treated as
  a fill, and that the leftover it returns is genuinely smaller than what it was given.
* **The same room comes back different on the second press.** Diffusion is stochastic and the
  button is a toggle. ``sd_inpaint`` seeds from the content of the request, and that is a
  property of a pure function of two arrays.

The one test that does need the card is marked ``slow`` and skips itself.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from biosphere.ai import backend, room_fill, sd_inpaint, views

# ======================================================================================
# Fixtures
# ======================================================================================

W, H = 512, 256


def panorama() -> np.ndarray:
    """A three-band room, bright enough that "did anything change" is unambiguous."""
    image = np.zeros((H, W, 3), np.uint8)
    image[: H // 4] = (200, 200, 200)
    image[H // 4 : 3 * H // 4] = (170, 190, 210)
    image[3 * H // 4 :] = (60, 90, 130)
    return image


def hole(x0: int = 180, y0: int = 110, x1: int = 260, y1: int = 180) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


class FakeInpainter:
    """Paints every masked pixel one flat colour and remembers what it was asked.

    A real inpainter's *quality* is not testable without weights; its *contract* is, and the
    contract is all ``room_fill`` depends on — same size out as in, ``None`` for "I cannot",
    and the mask meaning "replace this".
    """

    def __init__(self, *, colour: tuple[int, int, int] = (0, 0, 255), fail_after: int | None = None):
        self.colour = colour
        self.fail_after = fail_after
        self.calls: list[dict[str, object]] = []

    def __call__(self, image, mask, *, cyclic=True):
        self.calls.append(
            {
                "shape": image.shape,
                "cyclic": cyclic,
                "load": float((mask > 127).mean()),
            }
        )
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            return None
        out = image.copy()
        out[mask > 127] = self.colour
        return out


class Stub:
    """Stands in for a whole inpainter module."""

    def __init__(self, *, present: bool, result=...):
        self.present = present
        self._result = result
        self.calls = 0

    def available(self) -> bool:
        return self.present

    def inpaint(self, image, mask, *, cyclic=True):
        self.calls += 1
        if self._result is ...:
            return (image.copy(), True) if self.present else None
        return self._result


@pytest.fixture
def models(monkeypatch):
    """Install stub ``sd_inpaint`` and ``lama`` modules and hand them back."""

    def install(*, sd: bool, lama: bool, preference: str = "auto", sd_result=...):
        sd_stub = Stub(present=sd, result=sd_result)
        lama_stub = Stub(present=lama)
        monkeypatch.setattr("biosphere.ai.sd_inpaint.available", sd_stub.available)
        monkeypatch.setattr("biosphere.ai.sd_inpaint.inpaint", sd_stub.inpaint)
        monkeypatch.setattr("biosphere.ai.lama.available", lama_stub.available)
        monkeypatch.setattr("biosphere.ai.lama.inpaint", lama_stub.inpaint)
        monkeypatch.setattr(backend, "_preference", lambda: preference)
        return sd_stub, lama_stub

    return install


# ======================================================================================
# Which model answers which question
# ======================================================================================


@pytest.mark.parametrize(
    ("sd", "lama", "window", "canvas"),
    [
        (True, True, "lama", "lama"),
        (True, False, None, None),
        (False, True, "lama", "lama"),
        (False, False, None, None),
    ],
)
def test_auto_never_reaches_for_the_diffusion_model(models, sd, lama, window, canvas):
    """The whole of the routing decision, in one table, and the point is the first two rows.

    A machine with BOTH models installed uses LaMa, and a machine with only the diffusion one
    uses nothing — which sends ``empty_room`` to the classical fill. That is not a ranking of
    architectures, it is what a real bedroom measured: the diffusion path drew a new bed into
    the bed-shaped hole at an 18 % mask and a stranger's flat at 74 %, through every guidance
    value and negative prompt tried. Softness is a defect; invention is a misrepresentation.

    This table is therefore the guard on that decision. If a later change makes ``auto`` pick
    `sd` again, it should be because someone measured a model that does not invent — and then
    this test is the one they have to edit on purpose.
    """
    models(sd=sd, lama=lama)
    assert backend.local_inpainter(cyclic=False) == window
    assert backend.local_inpainter(cyclic=True) == canvas


def test_asking_for_diffusion_by_name_still_gets_it(models):
    """Opt-in has to actually work, or the setting is a lie and the module is dead code."""
    sd_stub, lama_stub = models(sd=True, lama=True, preference="sd")
    assert backend.local_inpainter(cyclic=False) == "sd"
    backend.local_inpaint(panorama(), hole(), cyclic=False)
    assert sd_stub.calls == 1
    assert lama_stub.calls == 0


def test_pinning_to_lama_keeps_diffusion_out_of_the_windows(models):
    sd_stub, lama_stub = models(sd=True, lama=True, preference="lama")
    assert backend.local_inpainter(cyclic=False) == "lama"
    backend.local_inpaint(panorama(), hole(), cyclic=False)
    assert sd_stub.calls == 0
    assert lama_stub.calls == 1


def test_pinning_to_sd_still_will_not_put_it_on_the_canvas(models):
    """``sd`` is a choice about the window fill, not permission to duplicate a wall.

    An operator sets this to keep a run off big-lama's non-commercial weights. Honouring it on
    the canvas would hand a diffusion checkpoint a 4096x2048 image, which is not a stricter
    reading of the setting — it is a worse room.
    """
    sd_stub, _ = models(sd=True, lama=True, preference="sd")
    assert backend.local_inpainter(cyclic=True) is None
    assert backend.local_inpaint(panorama(), hole(), cyclic=True) is None
    assert sd_stub.calls == 0


def test_a_diffusion_failure_is_retried_through_lama(models):
    """Out of memory on a small card must cost softness, not a hole in the wall.

    Reachable only through the opt-in, since `auto` does not choose the diffusion path — but
    an operator who did choose it still gets a filled room rather than a cut-out one.
    """
    sd_stub, lama_stub = models(sd=True, lama=True, preference="sd", sd_result=None)
    out = backend.local_inpaint(panorama(), hole(), cyclic=False)
    assert out is not None, "the room came back with the furniture still cut out of it"
    assert sd_stub.calls == 1
    assert lama_stub.calls == 1


def test_nothing_installed_is_none_rather_than_an_exception(models):
    models(sd=False, lama=False)
    assert backend.local_inpaint(panorama(), hole(), cyclic=False) is None


def test_an_unknown_preference_falls_back_to_auto(monkeypatch):
    """A typo in an env var must not turn the feature off on a machine that can run it."""

    class Settings:
        ai_inpainter = "stable-diffusion"

    monkeypatch.setattr(backend, "_settings", lambda: Settings())
    assert backend._preference() == "auto"


# ======================================================================================
# The window loop
# ======================================================================================


def test_every_window_is_asked_as_a_photograph_not_as_a_canvas():
    """``cyclic=False`` on a window is not cosmetic: wrap-padding one glues opposite walls."""
    fake = FakeInpainter()
    room_fill.fill_by_views(panorama(), hole(), fake, passes=1)
    assert fake.calls, "no window was ever offered to the inpainter"
    assert all(call["cyclic"] is False for call in fake.calls)
    assert all(call["shape"][:2] == (room_fill.VIEW_SIZE, room_fill.VIEW_SIZE) for call in fake.calls)


def test_the_windows_are_planned_easiest_first():
    """A window with little to do has the most context to do it from, and its result is what
    the harder ones then see. Ordering is the whole reason the loop is sequential.

    Asserted on the PLAN and not on what each window is holding when it is finally rendered.
    Those are different numbers by design: the plan is made once per pass, and every window
    that runs strikes work off ``remaining``, so the eighth window renders a mask its
    neighbours have already eaten into. Asserting the second would be asserting that the loop
    does not work.
    """
    order = room_fill._view_order(hole())
    loads = [load for load, _, _ in order]
    assert loads == sorted(loads), f"the plan is out of order: {loads}"
    assert len(order) > 1, "one window is not an ordering"


def test_filling_shrinks_what_is_left_and_touches_only_the_hole():
    original = panorama()
    mask = hole()
    filled, leftover = room_fill.fill_by_views(original, mask, FakeInpainter(), passes=2)

    assert float((leftover > 0).mean()) < float((mask > 0).mean()), "nothing was struck off"
    outside = mask == 0
    assert np.array_equal(filled[outside], original[outside]), "a pixel outside the mask moved"
    assert (filled[mask > 0] != original[mask > 0]).any(), "the hole was never painted"


def test_a_refusal_stops_the_loop_instead_of_being_treated_as_a_fill():
    """``None`` means the model could not answer. Carrying on would composite the *previous*
    window's output into this one's directions and call the room finished."""
    fake = FakeInpainter(fail_after=1)
    _, leftover = room_fill.fill_by_views(panorama(), hole(), fake, passes=2)
    assert len(fake.calls) == 2, "the loop kept asking after a refusal"
    assert leftover.any(), "a refused fill was reported as finished work"


def test_an_empty_mask_asks_nobody():
    fake = FakeInpainter()
    filled, leftover = room_fill.fill_by_views(panorama(), np.zeros((H, W), np.uint8), fake)
    assert fake.calls == []
    assert not leftover.any()
    assert np.array_equal(filled, panorama())


def test_the_scout_finds_every_window_that_really_contains_the_mask():
    """The scouting pass must agree with the windows the loop then renders.

    IT DID NOT, AND THE DISAGREEMENT WAS INVISIBLE. ``_view_order`` took the caller's
    intrinsics — built for a 1024 px window — and rendered its scout at 128, so each probe
    pointed at a narrow cone near the corner of the window it was named after. On the fixture
    below nine windows genuinely contain the mask and the mismatched scout found one of them,
    which reaches the user as a room that comes back with the furniture still in it and nothing
    in the log except ``windows=0``.

    Asserted as agreement rather than as a count, because the count is a property of the plan
    and would change with it; the agreement is the invariant.
    """
    mask = np.zeros((H, W), np.uint8)
    mask[int(H * 0.45) : int(H * 0.62), int(W * 0.46) : int(W * 0.54)] = 255

    scouted = {(yaw, pitch) for _, yaw, pitch in room_fill._view_order(mask)}

    K = views.intrinsics(room_fill.VIEW_SIZE, views.VIEW_FOV_DEG)
    truly = set()
    for pitch, n_yaw in room_fill.FILL_VIEW_PLAN:
        for index in range(n_yaw):
            yaw = 360.0 * index / float(n_yaw)
            rendered = views.render_view(
                mask, views.rotation(yaw, pitch), K, room_fill.VIEW_SIZE,
                interpolation=cv2.INTER_NEAREST,
            )
            if rendered.ndim == 3:
                rendered = rendered[..., 0]
            if (rendered > 127).any():
                truly.add((yaw, pitch))

    assert truly, "the fixture mask should be visible from several windows"
    assert scouted == truly, (
        f"the scout disagrees with the renderer: missed {sorted(truly - scouted)}, "
        f"invented {sorted(scouted - truly)}"
    )


def test_the_scouted_load_matches_what_the_window_actually_holds():
    """Easiest-first is only easiest-first if the loads mean something.

    The sequential fill's whole design is that a window with little to do goes first, because
    it has the most context, and its output becomes context for the next. Loads that do not
    track the real occupancy do not order anything — they shuffle.
    """
    mask = np.zeros((H, W), np.uint8)
    mask[int(H * 0.45) : int(H * 0.62), int(W * 0.46) : int(W * 0.54)] = 255

    K = views.intrinsics(room_fill.VIEW_SIZE, views.VIEW_FOV_DEG)
    for load, yaw, pitch in room_fill._view_order(mask):
        rendered = views.render_view(
            mask, views.rotation(yaw, pitch), K, room_fill.VIEW_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )
        if rendered.ndim == 3:
            rendered = rendered[..., 0]
        actual = float((rendered > 127).mean())
        assert abs(load - actual) < 0.02, (
            f"window ({yaw:.0f}, {pitch:.0f}) scouted {load:.4f} against a real {actual:.4f}"
        )


def test_a_compact_mask_is_actually_filled():
    """The end of the chain the bug above broke: a single block of furniture, gone.

    ``removed_fraction`` said 4 % of the sphere, ``regions`` said one, and the delivered
    panorama came back byte-for-byte the room it started as — because no window was scouted,
    so no window was filled, and nothing anywhere reported a failure.
    """
    original = panorama()
    mask = np.zeros((H, W), np.uint8)
    mask[int(H * 0.45) : int(H * 0.62), int(W * 0.46) : int(W * 0.54)] = 255

    filled, leftover = room_fill.fill_by_views(original, mask, FakeInpainter(), passes=2)
    changed = float((filled[mask > 0] != original[mask > 0]).any(axis=-1).mean())
    assert changed > 0.9, f"only {changed:.0%} of the furnishing was rebuilt"
    assert float((leftover > 0).mean()) < 0.25 * float((mask > 0).mean())


def test_a_hole_on_the_seam_is_reached_by_a_window():
    """Column 0 is a wall like any other. The windows are placed by yaw, so one of them
    straddles the seam — and if ``views.equirect_window`` unwrapped it wrongly, this is the
    test that notices, because the seam pixels would never be struck off."""
    mask = np.zeros((H, W), np.uint8)
    mask[110:180, -30:] = 255
    mask[110:180, :30] = 255
    _, leftover = room_fill.fill_by_views(panorama(), mask, FakeInpainter(), passes=2)
    assert float((leftover > 0).mean()) < float((mask > 0).mean())


# ======================================================================================
# The diffusion backend itself
# ======================================================================================


def test_the_canvas_is_refused_without_loading_anything(monkeypatch):
    """``cyclic=True`` is declined by contract, and cheaply: a machine with no diffusion model
    must not pay a checkpoint load to be told the answer it was always going to get.

    Asserted by making the loader fatal rather than by reading ``is_loaded()`` afterwards.
    That state is per-process and the pipeline is cached for the life of it, so any earlier
    test that emptied a room leaves it loaded and the weaker assertion fails for a reason that
    has nothing to do with this code path — which is exactly what it did when the file was run
    after `test_api.py` instead of alone.
    """

    def explode() -> None:
        raise AssertionError("the canvas refusal reached the model loader")

    monkeypatch.setattr(sd_inpaint, "_load", explode)
    assert sd_inpaint.inpaint(panorama(), hole(), cyclic=True) is None


def test_the_seed_is_a_function_of_the_request():
    """Same window, same mask, same room — in this process and in the worker's."""
    image, mask = panorama(), hole()
    assert sd_inpaint._seed_for(image, mask) == sd_inpaint._seed_for(image.copy(), mask.copy())


def test_a_different_mask_is_a_different_question():
    image = panorama()
    assert sd_inpaint._seed_for(image, hole()) != sd_inpaint._seed_for(image, hole(x0=100, x1=140))


def test_the_seed_ignores_what_the_mask_says_below_threshold():
    """The mask is a threshold, not a number. Seeding off its raw bytes would make an
    anti-aliased edge one pixel different into an entirely different room."""
    image, mask_a = panorama(), hole()
    mask_b = mask_a.copy()
    mask_b[mask_b == 0] = 40  # still below 127, so still "keep"
    assert sd_inpaint._seed_for(image, mask_a) == sd_inpaint._seed_for(image, mask_b)


def test_the_window_size_is_stride_aligned():
    """Both sides handed to the VAE must divide by 8; a size that does not is a tensor-shape
    error deep inside the pipeline rather than anything readable."""
    assert sd_inpaint.WORK_SIZE % 8 == 0
    for value in (100, 511, 513, 1024):
        assert sd_inpaint._round_to_stride(value) % 8 == 0


# ======================================================================================
# With the real weights, if this machine has them
# ======================================================================================


@pytest.mark.slow
def test_diffusion_fills_a_window_and_leaves_the_rest_alone():
    """End to end on whatever card this machine has. Skipped when it has none."""
    if not sd_inpaint.available():
        pytest.skip("no diffusion inpainting checkpoint or no usable CUDA device")

    size = 512
    view = np.zeros((size, size, 3), np.uint8)
    view[: size // 2] = (170, 190, 210)
    view[size // 2 :] = (60, 90, 130)
    mask = np.zeros((size, size), np.uint8)
    mask[200:360, 180:340] = 255

    result = sd_inpaint.inpaint(view, mask, cyclic=False)
    assert result is not None, "the model is available but declined the window"
    filled, model_backed = result
    assert model_backed
    assert filled.shape == view.shape
    assert (filled[mask > 0] != view[mask > 0]).any(), "the hole came back untouched"

    # Deterministic, which is what makes the toggle trustworthy. Asserted on the hole only:
    # everything outside it is composited away by the caller.
    again = sd_inpaint.inpaint(view, mask, cyclic=False)
    assert again is not None
    assert np.array_equal(again[0][mask > 0], filled[mask > 0]), "the second press differed"
