"""Tests for the pre-solve capture validator.

The properties worth pinning here are not "does it return a verdict" but the three things
that make the verdict trustworthy:

* the **measurement** is right — a pure rotation must read as one at the recovered focal,
  and a wrong focal must be recognisably worse, because that failure is what made a naive
  version of this check useless (it read a real hand-held turn as a walk);
* the **thresholds separate what they claim to** — a nodal capture passes, a capture that
  translates through the scene does not, and neither answer depends on a single sample;
* the **cost** is a scan of a bounded subsample, with the pair baseline held fixed, since
  the residual is meaningless if the baseline drifts with the frame count.

Wherever the geometry is what is under test, correspondences are projected through the
same :mod:`biosphere.pano.geometry` primitives the validator uses. Where pixels are needed,
the pure-rotation case is rendered by resampling an equirect sphere (exactly nodal, and
free), and the translating case is ray-traced by :mod:`biosphere.synth`, which is the only
way to get real parallax from real depth.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from biosphere.pano import geometry as g
from biosphere.pano import validate as val
from biosphere.pano.extract import score_frames
from biosphere.pano.types import (
    CaptureKind,
    Frame,
    Intrinsics,
    PipelineConfig,
    QualityMode,
    StitchError,
)

IMG_W, IMG_H = 640, 360
HFOV = 62.0


def _config() -> PipelineConfig:
    return PipelineConfig.for_mode(QualityMode.BALANCED)


# ======================================================================================
# 1. Sampling: what gets measured, and over what baseline
# ======================================================================================


def _frames(n: int, fps: float = 8.0) -> list[Frame]:
    return [
        Frame(index=k, path=Path(f"f_{k:05d}.jpg"), timestamp=k / fps, width=IMG_W, height=IMG_H)
        for k in range(n)
    ]


def test_sample_pairs_hold_the_baseline_fixed_whatever_the_cap():
    """The residual scales with the baseline, so thinning must not stretch the pairs.

    If the cap were implemented by dropping the sample rate, a long capture would be
    measured over second-long baselines and a short one over quarter-second baselines, and
    the same motion would report numbers several times apart. Every threshold in the module
    would then mean something different depending on how long the video was.
    """
    for n in (40, 200, 900):
        frames = _frames(n, fps=8.0)
        pairs, spacing = val._sample_pairs(frames, sample_fps=4.0, max_samples=val.MAX_SAMPLES)
        assert spacing == pytest.approx(0.25, abs=1e-6), f"baseline drifted at n={n}"
        assert {j - i for i, j in pairs} == {2}, "pair stride must stay at 4 fps"
        assert len(pairs) <= val.MAX_SAMPLES
        assert all(0 <= i < j < n for i, j in pairs)


def test_sample_pairs_are_gathered_in_runs_so_detections_are_shared():
    """Contiguous runs are what make the scan affordable: L samples need L+1 frames.

    Isolated pairs would need 2L. Detection is ~95 % of the scan's cost, so this halves it.
    """
    frames = _frames(2000, fps=8.0)
    pairs, _ = val._sample_pairs(frames, sample_fps=4.0, max_samples=48)
    n_detections = len({k for pair in pairs for k in pair})
    assert len(pairs) <= 48
    assert n_detections < 1.5 * len(pairs), (
        f"{n_detections} detections for {len(pairs)} pairs — the runs are not sharing frames"
    )
    # ...and the runs must still be spread over the whole capture, or a room walked
    # through in the middle would never be looked at.
    times = np.array([frames[i].timestamp for i, _ in pairs])
    assert times.min() < 10.0 and times.max() > 0.9 * frames[-1].timestamp


def test_sample_pairs_uncapped_measures_every_moment():
    """The segmenter needs full temporal resolution; the validator does not."""
    frames = _frames(200, fps=8.0)
    pairs, _ = val._sample_pairs(frames, sample_fps=4.0, max_samples=None)
    assert len(pairs) == 99
    assert [i for i, _ in pairs] == list(range(0, 198, 2))


def test_sample_pairs_degrade_gracefully_on_a_tiny_capture():
    assert val._sample_pairs([], sample_fps=4.0, max_samples=8) == ([], 0.0)
    pairs, _ = val._sample_pairs(_frames(2, fps=8.0), sample_fps=4.0, max_samples=8)
    assert pairs == [(0, 1)]


# ======================================================================================
# 2. The measurement: rotation-only residual and the focal it is measured at
# ======================================================================================


def _rotation_fit(
    R: np.ndarray, K: np.ndarray, rng: np.random.Generator, n: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pure-rotation correspondence set plus its exact homography."""
    xs = rng.uniform(20.0, IMG_W - 20.0, 4 * n)
    ys = rng.uniform(20.0, IMG_H - 20.0, 4 * n)
    rays = np.asarray(g.unproject_pixels(xs, ys, K), dtype=np.float64)
    px, py, valid = g.project_rays(rays @ R.T, K)
    ok = valid & (px > 5) & (px < IMG_W - 5) & (py > 5) & (py < IMG_H - 5)
    idx = np.flatnonzero(ok)[:n]
    pa = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    pb = np.stack([np.asarray(px)[idx], np.asarray(py)[idx]], axis=1).astype(np.float32)
    return g.homography_from_rotation(R, K, K), pa, pb


def test_rotation_residual_is_zero_for_a_pure_rotation_at_the_true_focal():
    rng = np.random.default_rng(3)
    intr = Intrinsics.from_hfov(IMG_W, IMG_H, HFOV)
    fits = [
        _rotation_fit(g.from_yaw_pitch_roll(math.radians(d), math.radians(2.0), 0.0), intr.K, rng)
        for d in (5.0, 12.0, 20.0)
    ]
    res = val._rotation_residuals(fits, intr.focal, IMG_W, IMG_H)
    # A thousandth of a pixel: the float32 correspondence arrays quantise at ~1e-5 px on a
    # 640-wide frame, so anything below this is exactness rather than a small error.
    assert res.max() < 1e-3, f"a pure rotation must be reproduced exactly: {res}"


def test_a_wrong_focal_manufactures_a_residual_that_looks_like_translation():
    """The failure that makes a naive version of this check useless.

    Nothing about the motion changes here — the correspondences are an exact pure rotation
    throughout. Only the assumed field of view moves, and the residual it produces is large
    enough to be mistaken for a walk.
    """
    rng = np.random.default_rng(5)
    intr = Intrinsics.from_hfov(IMG_W, IMG_H, 40.0)
    fits = [
        _rotation_fit(g.from_yaw_pitch_roll(math.radians(d), math.radians(3.0), 0.0), intr.K, rng)
        for d in (8.0, 15.0, 22.0)
    ]
    wrong = Intrinsics.from_hfov(IMG_W, IMG_H, 67.0).focal
    assert float(np.median(val._rotation_residuals(fits, wrong, IMG_W, IMG_H))) > 3.0


def test_solve_focal_recovers_the_true_field_of_view():
    """Self-calibration by minimising the rotation-only residual, not by homography algebra."""
    rng = np.random.default_rng(11)
    for hfov in (38.0, 55.0, 78.0):
        intr = Intrinsics.from_hfov(IMG_W, IMG_H, hfov)
        fits = [
            _rotation_fit(
                g.from_yaw_pitch_roll(
                    math.radians(float(rng.uniform(6.0, 25.0))),
                    math.radians(float(rng.uniform(-6.0, 6.0))),
                    math.radians(float(rng.uniform(-3.0, 3.0))),
                ),
                intr.K,
                rng,
            )
            for _ in range(12)
        ]
        focal, clamped = val._solve_focal(fits, IMG_W, IMG_H)
        assert not clamped
        recovered = math.degrees(2.0 * math.atan((IMG_W / 2.0) / focal))
        assert recovered == pytest.approx(hfov, abs=2.0), f"wanted {hfov}, got {recovered:.1f}"


def test_solve_focal_stays_inside_the_field_of_view_envelope():
    """A degenerate set must be clamped and *flagged*, not quietly reported as a lens."""
    rng = np.random.default_rng(17)
    fits = [
        (np.eye(3), rng.uniform(0, IMG_W, (60, 2)).astype(np.float32),
         rng.uniform(0, IMG_W, (60, 2)).astype(np.float32))
        for _ in range(6)
    ]
    focal, _clamped = val._solve_focal(fits, IMG_W, IMG_H)
    hfov = math.degrees(2.0 * math.atan((IMG_W / 2.0) / focal))
    assert 29.0 <= hfov <= 111.0
    assert val._solve_focal([], IMG_W, IMG_H)[1] is True


# ======================================================================================
# 3. Statistics
# ======================================================================================


def _sample(t: float, rot: float, free: float = 1.0, *, yaw: float = 1.0, dt: float = 0.25,
            usable: bool = True, ratio: float = 0.6, turn: float | None = None) -> val.PuritySample:
    return val.PuritySample(
        i=int(t / dt), j=int(t / dt) + 1, time_s=t, dt_s=dt, n_matches=200, n_inliers=120,
        inlier_ratio=ratio, rot_residual_px=rot if usable else math.inf,
        free_residual_px=free if usable else math.inf,
        turn_deg=abs(yaw) if turn is None else turn, yaw_step_deg=yaw, mean_flow_px=30.0,
        usable=usable,
    )


def test_purity_stats_integrates_the_turn_instead_of_summing_samples():
    """The bug windowed sampling introduces, pinned.

    Two runs of samples an hour apart in a 100-second capture describe a capture that kept
    turning throughout. Summing the per-sample steps reports only the eight degrees inside
    the runs; integrating the rate over time reports the whole sweep. Getting this wrong
    made a full 360 turn read as 128 degrees.
    """
    runs = [_sample(t, 1.0, yaw=1.0) for t in np.arange(0.0, 2.0, 0.25)]
    runs += [_sample(t, 1.0, yaw=1.0) for t in np.arange(50.0, 52.0, 0.25)]
    stats = val.purity_stats(runs)
    naive_sum = sum(abs(s.yaw_step_deg) for s in runs)
    assert naive_sum == pytest.approx(16.0)
    # 1 deg per 0.25 s is 4 deg/s held across ~51 s of capture.
    assert stats["yaw_span_deg"] > 190.0, stats
    assert stats["total_turn_deg"] > 190.0


def test_purity_stats_excludes_unusable_samples_rather_than_zeroing_them():
    samples = [_sample(k * 0.25, 3.0) for k in range(8)]
    samples += [_sample((8 + k) * 0.25, 0.0, usable=False) for k in range(8)]
    stats = val.purity_stats(samples)
    assert stats["n_samples"] == 16.0
    assert stats["n_usable"] == 8.0
    assert stats["usable_fraction"] == pytest.approx(0.5)
    assert stats["median_rot_residual_px"] == pytest.approx(3.0)


def test_purity_stats_on_nothing_is_infinite_not_zero():
    """A capture that measured nothing must not look like a perfect one."""
    stats = val.purity_stats([])
    assert not math.isfinite(stats["median_rot_residual_px"])
    assert stats["n_usable"] == 0.0


# ======================================================================================
# 4. Classification
# ======================================================================================


def _stats(rot: float, free: float, n_usable: int = 40) -> dict[str, float]:
    return val.purity_stats([_sample(k * 0.25, rot, free) for k in range(n_usable)])


def test_classify_separates_the_two_real_captures():
    """The acceptance test, at the level of the numbers those captures actually produce.

    Video A measured 3.61 px against a 1.08 px floor and produced a usable panorama;
    Video B measured 21.17 px against 1.60 px and cannot produce one at all. The threshold
    has to sit between them, and it must be the *capture* that decides, not the noise.
    """
    assert val.classify_purity(_stats(3.61, 1.08))[0] is val.CaptureVerdict.DEGRADED
    assert val.classify_purity(_stats(21.17, 1.60))[0] is val.CaptureVerdict.WALKING
    # And with real headroom on both sides, not by a whisker.
    assert 3.61 * 2.5 < val.WALKING_ROT_RESIDUAL_PX < 21.17 / 2.0


def test_classify_calls_a_nodal_capture_good_and_a_hand_held_one_degraded():
    assert val.classify_purity(_stats(0.98, 0.59))[0] is val.CaptureVerdict.GOOD
    assert val.classify_purity(_stats(2.43, 0.81))[0] is val.CaptureVerdict.DEGRADED


def test_a_noisy_capture_is_not_blamed_for_translation_it_does_not_have():
    """The ratio guard. A dim, low-texture, heavily compressed capture matches badly, and
    its residual floor rises with it — but the *ratio* does not, because both halves rise
    together. Without this guard, bad conditions would be reported as a walk.
    """
    verdict, reason = val.classify_purity(_stats(12.0, 6.0))
    assert verdict is val.CaptureVerdict.DEGRADED, reason
    verdict, _ = val.classify_purity(_stats(12.0, 1.0))
    assert verdict is val.CaptureVerdict.WALKING


def test_a_capture_is_never_refused_on_a_handful_of_samples():
    verdict, reason = val.classify_purity(_stats(30.0, 1.0, n_usable=4))
    assert verdict is val.CaptureVerdict.DEGRADED
    assert reason == "translation_suspected_few_samples"
    assert val.classify_purity(_stats(30.0, 1.0, n_usable=20))[0] is val.CaptureVerdict.WALKING


def test_measuring_nothing_is_degraded_not_good():
    assert val.classify_purity(val.purity_stats([]))[0] is val.CaptureVerdict.DEGRADED


# ======================================================================================
# 5. Rendered captures, end to end
# ======================================================================================


PANO_W, PANO_H = 2048, 1024


def _sphere(rng: np.random.Generator) -> np.ndarray:
    coarse = cv2.resize(rng.integers(0, 256, (128, 256, 3), dtype=np.uint8),
                        (PANO_W, PANO_H), interpolation=cv2.INTER_LANCZOS4)
    mid = cv2.resize(rng.integers(0, 256, (48, 96, 3), dtype=np.uint8),
                     (PANO_W, PANO_H), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(coarse, 0.6, mid, 0.4, 0.0)


@pytest.fixture(scope="module")
def nodal_capture(tmp_path_factory) -> list[Frame]:
    """An exactly nodal turn: an equirect sphere resampled through a rotating camera.

    Sampling a sphere *is* a pure rotation by construction — the scene is at infinite
    depth, so there is no parallax to have — which makes this the cleanest possible
    positive control, and it costs no ray tracing.
    """
    out = tmp_path_factory.mktemp("nodal")
    rng = np.random.default_rng(20)
    pano = _sphere(rng)
    intr = Intrinsics.from_hfov(IMG_W, IMG_H, HFOV)
    rays = g.camera_rays(IMG_W, IMG_H, intr.K)
    frames = []
    n, fps = 60, 6.0
    for k in range(n):
        R = g.from_yaw_pitch_roll(
            math.radians(360.0 * k / n),
            math.radians(float(rng.uniform(-2.0, 2.0))),
            math.radians(float(rng.uniform(-1.5, 1.5))),
        )
        u, v = g.equirect_from_rays(rays @ R, PANO_W, PANO_H)
        path = out / f"f_{k:04d}.jpg"
        cv2.imwrite(str(path), g.sample_equirect(pano, u, v, interpolation=cv2.INTER_LANCZOS4),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frames.append(Frame(index=k, path=path, timestamp=k / fps, width=IMG_W, height=IMG_H))
    score_frames(frames)
    return frames


def test_a_nodal_capture_assesses_as_good(nodal_capture):
    """The validator must not start rejecting the captures the product is built for."""
    assessment = val.assess_capture(nodal_capture, _config(), kind=CaptureKind.VIDEO)
    assert assessment.verdict is val.CaptureVerdict.GOOD, assessment.detail()
    assert assessment.should_proceed
    assert assessment.median_rot_residual_px < val.GOOD_ROT_RESIDUAL_PX
    assert assessment.residual_ratio < val.WALKING_RESIDUAL_RATIO
    assert assessment.n_usable_samples >= 10
    # The focal search should land near the field of view the capture was rendered at.
    assert assessment.hfov_deg == pytest.approx(HFOV, abs=8.0)


def test_the_scan_is_a_small_fraction_of_a_solve(nodal_capture):
    """Assessment has to be cheap enough that running it always is obviously right."""
    assessment = val.assess_capture(nodal_capture, _config(), kind=CaptureKind.VIDEO)
    assert assessment.elapsed_s < 8.0, f"scan took {assessment.elapsed_s:.1f}s on 60 small frames"
    assert assessment.n_samples <= val.MAX_SAMPLES


@pytest.fixture(scope="module")
def walking_capture(tmp_path_factory) -> list[Frame]:
    """A camera translating through a ray-traced room while it pans.

    Real depth is the whole point: a sphere cannot produce parallax and a single plane
    cannot either (a translating camera viewing one plane is still exactly a homography).
    Only a scene with several depths breaks the rotation model, which is why this one is
    ray-traced even though it costs a few seconds.
    """
    from biosphere.synth import Room
    from biosphere.synth.render import render_view

    out = tmp_path_factory.mktemp("walk")
    room = Room.random(seed=7, style="rich")
    w, h = 224, 398
    intr = Intrinsics.from_hfov(w, h, 45.0)
    eye = np.asarray(room.eye, dtype=np.float64)
    reach = min(float(room.width_m), float(room.depth_m)) * 0.5 - 0.6
    frames = []
    n, fps = 40, 6.0
    for k in range(n):
        t = k / fps
        offset = reach * math.sin(0.28 * t)  # strolling back and forth, ~0.8 m/s
        origin = (eye + np.array([offset, 0.0, 0.35 * offset])).astype(np.float32)
        R = g.from_yaw_pitch_roll(math.radians(22.0 * t), math.radians(2.0 * math.sin(t)), 0.0)
        path = out / f"f_{k:04d}.png"
        cv2.imwrite(str(path), render_view(room, R, intr, origin=origin, subsamples=1))
        frames.append(Frame(index=k, path=path, timestamp=t, width=w, height=h))
    score_frames(frames)
    return frames


def test_a_walking_capture_is_refused(walking_capture):
    assessment = val.assess_capture(walking_capture, _config(), kind=CaptureKind.VIDEO)
    assert assessment.verdict is val.CaptureVerdict.WALKING, assessment.detail()
    assert not assessment.should_proceed
    assert assessment.median_rot_residual_px > val.WALKING_ROT_RESIDUAL_PX
    # The free-homography floor stays low — the features match fine. That is precisely why
    # no inlier-count health check in the pipeline can catch this.
    assert assessment.median_free_residual_px < 3.0
    with pytest.raises(val.WalkingCaptureError):
        assessment.raise_if_fatal()


def test_the_walking_error_tells_the_user_what_to_do(walking_capture):
    """The copy is the deliverable here, not the exception type."""
    assessment = val.assess_capture(walking_capture, _config(), kind=CaptureKind.VIDEO)
    with pytest.raises(val.WalkingCaptureError) as excinfo:
        assessment.raise_if_fatal(segments=[{"index": 0}])
    error = excinfo.value
    assert isinstance(error, StitchError)
    for fragment in ("odanın ortasında", "dikey", "yürümeden", "15 saniye"):
        assert fragment in error.user_message_tr, f"Turkish copy is missing {fragment!r}"
    for fragment in ("middle of the room", "upright", "without walking", "fifteen seconds"):
        assert fragment in error.user_message_en
    assert error.payload["segments"] == [{"index": 0}]
    assert error.payload["assessment"]["verdict"] == "walking"


def test_force_stitch_downgrades_a_hard_verdict_to_a_warning(walking_capture, monkeypatch):
    """A user who insists should be able to see what comes out."""
    monkeypatch.setenv("BIOSPHERE_FORCE_STITCH", "1")
    assert val.force_stitch_enabled()
    assessment = val.assess_capture(walking_capture, _config(), kind=CaptureKind.VIDEO)
    assert assessment.verdict is val.CaptureVerdict.WALKING, "the verdict must not be softened"
    assert assessment.forced and assessment.should_proceed
    assessment.raise_if_fatal()  # must not raise

    monkeypatch.delenv("BIOSPHERE_FORCE_STITCH")
    assert not val.force_stitch_enabled()


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("on", True),
                                            ("0", False), ("", False), ("no", False)])
def test_the_escape_hatches_read_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("BIOSPHERE_FORCE_STITCH", value)
    monkeypatch.setenv("BIOSPHERE_SEGMENT_ROOMS", value)
    assert val.force_stitch_enabled() is expected
    assert val.segment_rooms_enabled() is expected


def test_env_vars_default_to_off(monkeypatch):
    monkeypatch.delenv("BIOSPHERE_FORCE_STITCH", raising=False)
    monkeypatch.delenv("BIOSPHERE_SEGMENT_ROOMS", raising=False)
    assert not val.force_stitch_enabled() and not val.segment_rooms_enabled()


# ======================================================================================
# 6. The cheap rejects
# ======================================================================================


def test_too_short_is_decided_without_measuring_anything(nodal_capture):
    assessment = val.assess_capture(nodal_capture[:3], _config(), kind=CaptureKind.VIDEO)
    assert assessment.verdict is val.CaptureVerdict.TOO_SHORT
    assert assessment.n_samples == 0, "the scan must not have run"
    with pytest.raises(val.CaptureTooShortError):
        assessment.raise_if_fatal()


def test_a_short_photo_set_is_not_too_short(nodal_capture):
    """Photo timestamps are nominal, so the duration floor must apply to video only.

    A dozen stills of a room is a legitimate capture; measured against a six-second floor
    on one-second nominal stamps it would be refused for being brief when it is nothing of
    the kind.
    """
    # Six stills, nominally a second apart: five seconds "long" as a video, a perfectly
    # ordinary photo set otherwise.
    photos = [
        Frame(index=k, path=f.path, timestamp=float(k), width=f.width, height=f.height,
              sharpness=f.sharpness, brightness=f.brightness, quality=f.quality)
        for k, f in enumerate(nodal_capture[:6])
    ]
    assert len(photos) >= val.MIN_FRAMES
    assert photos[-1].timestamp - photos[0].timestamp < val.MIN_VIDEO_DURATION_S
    assert val.assess_capture(photos, _config(), kind=CaptureKind.PHOTOS).verdict is not (
        val.CaptureVerdict.TOO_SHORT
    )
    assert val.assess_capture(photos, _config(), kind=CaptureKind.VIDEO).verdict is (
        val.CaptureVerdict.TOO_SHORT
    )


def test_too_dark_is_decided_from_the_scores_extract_already_took(nodal_capture):
    dark = [
        Frame(index=f.index, path=f.path, timestamp=f.timestamp, width=f.width, height=f.height,
              sharpness=f.sharpness, brightness=6.0, quality=f.quality)
        for f in nodal_capture
    ]
    assessment = val.assess_capture(dark, _config(), kind=CaptureKind.VIDEO)
    assert assessment.verdict is val.CaptureVerdict.TOO_DARK
    assert assessment.n_samples == 0
    with pytest.raises(val.CaptureTooDarkError):
        assessment.raise_if_fatal()


def test_blur_needs_both_symptoms_before_it_is_called_blur(nodal_capture):
    """Featureless *or* unmatchable is not enough; a false reject here is expensive.

    A photo set with wide gaps fails the match test while being perfectly sharp; a flat
    white wall is textureless while matching fine. Only both together mean the frames
    cannot be used.
    """
    sharp_but_unmatchable = val.purity_stats([])  # nothing matched
    assert sharp_but_unmatchable["usable_fraction"] == 0.0
    blurry = [
        Frame(index=f.index, path=f.path, timestamp=f.timestamp, width=f.width, height=f.height,
              sharpness=1.0, brightness=f.brightness, quality=f.quality)
        for f in nodal_capture
    ]
    # These frames are labelled textureless but actually match perfectly, so the verdict
    # must come from the geometry, not from the Laplacian.
    assert val.assess_capture(blurry, _config(), kind=CaptureKind.VIDEO).verdict is (
        val.CaptureVerdict.GOOD
    )


def test_unscored_frames_are_not_mistaken_for_a_pitch_black_capture(nodal_capture):
    """``Frame`` defaults every score to zero, and only ``score_frames`` fills them in.

    The pipeline always scores first, but a caller using this module directly need not
    have — and an unscored capture would otherwise be refused as ``too_dark`` on the
    strength of a field nobody ever wrote to.
    """
    unscored = [
        Frame(index=f.index, path=f.path, timestamp=f.timestamp, width=f.width, height=f.height)
        for f in nodal_capture
    ]
    assert all(f.brightness == 0.0 for f in unscored)
    assert val.assess_capture(unscored, _config(), kind=CaptureKind.VIDEO).verdict is (
        val.CaptureVerdict.GOOD
    )


def test_an_implausible_fit_is_marked_unusable_rather_than_averaged_in():
    """One real sample in Video B claimed a 165 deg turn and a 3259 px residual.

    Left in, a single fit like that shifts the median and wrecks the trapezoid integration
    of the turn. It is not a fast pan — 165 deg in a quarter second is not a thing a camera
    does — it is a degenerate homography, and the honest record is "nothing was measured".
    """
    assert val.MAX_SAMPLE_TURN_DEG < 90.0
    good = [_sample(k * 0.25, 3.0, yaw=2.0) for k in range(20)]
    stats = val.purity_stats(good + [_sample(5.0, 3259.0, yaw=-72.0, turn=164.9, usable=False)])
    assert stats["median_rot_residual_px"] == pytest.approx(3.0)
    assert stats["n_usable"] == 20.0


# ======================================================================================
# 7. Reporting
# ======================================================================================


def test_the_assessment_serialises_for_the_api(nodal_capture):
    payload = val.assess_capture(nodal_capture, _config(), kind=CaptureKind.VIDEO).as_dict()
    for key in ("verdict", "reason", "message_tr", "message_en", "median_rot_residual_px",
                "median_free_residual_px", "median_rot_residual_deg", "residual_ratio",
                "frac_below_2px", "frac_below_4px", "yaw_span_deg", "total_turn_deg",
                "elapsed_s", "focal_px", "hfov_deg", "forced"):
        assert key in payload, f"as_dict is missing {key}"
    assert payload["message_tr"] and payload["message_en"]
    assert payload["message_tr"] != payload["message_en"], "the copy must actually be bilingual"
    import json

    json.dumps(payload)  # must survive the API boundary, infinities and all


def test_the_angular_residual_is_resolution_independent(nodal_capture):
    """Pixels depend on the frame size; degrees do not, and both are reported."""
    assessment = val.assess_capture(nodal_capture, _config(), kind=CaptureKind.VIDEO)
    expected = math.degrees(
        math.atan(
            assessment.median_rot_residual_px
            / (assessment.focal_px * val.REFERENCE_WIDTH / assessment.analysis_width)
        )
    )
    assert assessment.median_rot_residual_deg == pytest.approx(expected, rel=1e-9)


def test_every_verdict_carries_bilingual_copy_and_an_error_where_it_should():
    for verdict in val.CaptureVerdict:
        tr, en = val._MESSAGES[verdict]
        assert tr and en and tr != en
        if verdict.is_fatal:
            assert verdict in val._ERROR_FOR_VERDICT
            error = val._ERROR_FOR_VERDICT[verdict]
            assert issubclass(error, StitchError)
            assert error.user_message_tr and error.user_message_en
