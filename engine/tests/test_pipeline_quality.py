"""End-to-end acceptance gate: synthetic capture in, measured panorama out.

This is the test that decides whether the product claim holds. It builds a virtual room,
films it with a simulated phone (motion blur, rolling shutter, auto-exposure drift, hand
jitter and real parallax from the phone orbiting the user's spine), encodes an mp4, runs
the *actual* shipping pipeline over it, and scores the result against the panorama the
room was rendered from.

Two traps are avoided deliberately, and both live inside ``synth.metrics``:

* **Gauge freedom.** A stitched panorama is only defined up to a global rotation, so raw
  PSNR is meaningless; ``compare`` aligns before scoring and ``pose_errors`` factors out
  the global rotation before measuring orientation accuracy.
* **Pole weighting.** An unweighted mean over an equirect canvas lets the poles — which
  occupy almost none of the sphere — dominate. Every average is solid-angle weighted.

Note on ``seam_score``: a *correct* panorama of a real room scores well above 1.0,
because a wall corner is a genuine vertical discontinuity and no single-image statistic
tells it apart from a bad join. The meaningful quantity is the **ratio** against the
ground truth, which is ~1.0 when the stitch added nothing of its own.

Run the fast subset with::

    pytest tests/test_pipeline_quality.py -q -m "not slow"

Set ``BIOSPHERE_TEST_REPORT=<dir>`` to also drop the ground truth, the result and a
difference map into that directory for eyeballing.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from biosphere.pano.types import Intrinsics, PipelineConfig, QualityMode

# A portrait phone capture. Held upright, a 26 mm-equivalent main camera gives roughly
# 43 deg across the short edge and 69 deg down the long one — the tall field of view is
# exactly why the capture guide tells users to hold the phone vertically.
#
# The cost of these tests is dominated by *ray-tracing the synthetic capture*, not by the
# pipeline: roughly 2 s per 720x1280 frame, so a 20 s / 24 fps capture spends around
# fifteen minutes before the stitcher even starts. The quick tier therefore films at a
# smaller sensor size — same aspect, same field of view, every code path exercised — so
# the regression gate is something a person will actually run. The slow tier keeps full
# size for the numbers we quote.
QUICK_FRAME = (480, 854)
FULL_FRAME = (720, 1280)
CAPTURE_HFOV_DEG = 45.0


@dataclass(frozen=True)
class Gate:
    """Acceptance thresholds for one scenario."""

    psnr_horizon: float
    ssim_horizon: float
    pose_mean_deg: float
    pose_max_deg: float
    wrap_seam: float
    seam_ratio: float
    covered_fraction: float
    hf_retention: float


# Two tiers. The nodal case isolates the algorithm; the realistic case is what a real
# agent's phone actually produces and is the number that matters commercially.
# Calibrated from a measured baseline on 2026-08-16 (nodal quick tier: psnr_horizon
# 24.15, ssim_horizon 0.844, pose 0.403/0.694 deg, seam_ratio 0.857, wrap 1.00, hf 1.02,
# coverage 0.586 against a 0.593 geometric ceiling), then loosened by a regression margin.
# The photometric numbers are scene-specific: this room's procedural texture is dense
# high-frequency detail, and PSNR punishes a two-pixel registration error there far
# harder than any viewer would. `hf_retention` is the honest sharpness check — it
# separates "shifted slightly" from "blurred", which PSNR cannot.
GATES: dict[str, Gate] = {
    "nodal": Gate(
        psnr_horizon=22.0,
        ssim_horizon=0.80,
        pose_mean_deg=0.55,
        pose_max_deg=1.00,
        hf_retention=0.92,
        wrap_seam=2.00,
        seam_ratio=1.15,
        # A *level single row* can only ever reach sin(vfov/2) of the sphere — about 0.59
        # at this field of view. Demanding more would be demanding that synthesised poles
        # be counted as photographed, which is exactly the dishonesty this suite exists to
        # prevent. `test_nodal_single_row_is_near_perfect` pins the geometric limit too.
        covered_fraction=0.55,
    ),
    "realistic": Gate(
        psnr_horizon=20.0,
        ssim_horizon=0.74,
        pose_mean_deg=1.20,
        pose_max_deg=3.50,
        hf_retention=0.85,
        wrap_seam=2.20,
        seam_ratio=1.35,
        # Three rows genuinely do reach almost the whole sphere.
        covered_fraction=0.85,
    ),
}


def _synth():
    """Import the harness lazily so a missing piece fails with a readable message."""
    from biosphere.synth import capture, encode, metrics, room

    return room, capture, encode, metrics


def _report_dir() -> Path | None:
    raw = os.environ.get("BIOSPHERE_TEST_REPORT")
    if not raw:
        return None
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_report(name: str, truth: np.ndarray, aligned: np.ndarray, scores: dict) -> None:
    out = _report_dir()
    if out is None:
        return
    h, w = truth.shape[:2]
    res = cv2.resize(aligned, (w, h), interpolation=cv2.INTER_AREA)
    diff = cv2.applyColorMap(
        np.clip(cv2.absdiff(truth, res).mean(axis=2) * 4.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    q = [cv2.IMWRITE_JPEG_QUALITY, 92]
    cv2.imwrite(str(out / f"{name}_truth.jpg"), truth, q)
    cv2.imwrite(str(out / f"{name}_result.jpg"), res, q)
    cv2.imwrite(str(out / f"{name}_diff.jpg"), diff, q)
    (out / f"{name}_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")


def _coverage_mask_like(result, shape: tuple[int, int]) -> np.ndarray | None:
    """Resample the pipeline's own coverage mask onto the ground truth's grid.

    Scoring against the auto-detected mask would let JPEG ringing around an uncovered
    region read as covered and tank the score; the pipeline knows exactly which pixels it
    actually filled, so use that.
    """
    mask = getattr(result.coverage, "mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8) * 255


def _hf_retention(candidate: np.ndarray, reference: np.ndarray, band=(0.32, 0.68)) -> float:
    """Ratio of high-frequency energy to the ground truth's, over the captured band.

    1.0 means the stitch is exactly as sharp as the original. Below ~0.9 means it blurred
    — the failure mode that PSNR conflates with harmless sub-pixel registration error.
    """
    h = reference.shape[0]
    lo, hi = int(h * band[0]), int(h * band[1])
    a = cv2.cvtColor(reference[lo:hi], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(
        cv2.resize(candidate, (reference.shape[1], h), interpolation=cv2.INTER_AREA)[lo:hi],
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)
    ea = float(np.abs(cv2.Laplacian(a, cv2.CV_32F)).mean())
    eb = float(np.abs(cv2.Laplacian(b, cv2.CV_32F)).mean())
    return eb / ea if ea > 1e-6 else 0.0


def _wrap_step_ratio(pano: np.ndarray, band: tuple[float, float] = (0.30, 0.62)) -> float:
    """How far the wrap column stands out from ordinary scene edges, in the horizon band.

    ``synth.metrics.wrap_seam_score`` measures the whole canvas, which on a single-row
    capture includes the synthesised poles. Fill is almost perfectly smooth, so the local
    baseline there collapses toward zero and any trace of a difference explodes the ratio
    — it reported 2.2 on a panorama whose wrap column measured *exactly* the median
    adjacent-column difference. Restricting to the band a viewer actually looks at gives a
    number that means what it says: 1.0 is indistinguishable from the scene's own edges.

    The wrap is the *last* difference (column W-1 back to column 0). This used to read
    ``diffs[0]`` -- the step from column 0 to column 1, an ordinary interior boundary -- so
    the gate had never looked at the wrap at all; the calibrated "wrap 1.00" was that.
    """
    lo, hi = int(pano.shape[0] * band[0]), int(pano.shape[0] * band[1])
    strip = pano[lo:hi].astype(np.float32)
    cyclic = np.concatenate([strip, strip[:, :1]], axis=1)
    diffs = np.abs(np.diff(cyclic, axis=1)).mean(axis=(0, 2))
    median = float(np.median(diffs))
    return float(diffs[-1] / median) if median > 1e-6 else 0.0


def _truth_poses_for(result, synthetic) -> list:
    """Pick the ground-truth pose contemporaneous with each solved keyframe."""
    stamps = list(getattr(result, "keyframe_timestamps", []) or [])
    times = np.asarray(getattr(synthetic, "times", []), dtype=float)
    assert stamps, "StitchResult.keyframe_timestamps is empty; the pipeline must populate it"
    assert len(stamps) == len(result.poses), (
        f"{len(stamps)} timestamps for {len(result.poses)} poses — the two must stay aligned"
    )
    assert times.size == len(synthetic.poses), "synthetic times/poses length mismatch"
    idx = [int(np.argmin(np.abs(times - t))) for t in stamps]
    return [synthetic.poses[i] for i in idx]


def _expected_single_row_coverage(frame_size: tuple[int, int], hfov_deg: float) -> float:
    """Solid-angle fraction a *level* single sweep can physically reach.

    A level row spans elevation +/- vfov/2, and the sphere fraction between two parallels
    at +/- a is sin(a). Nothing about the algorithm can beat this — which is why the
    capture guide asks for three rows, and why a single-row property honestly reports that its
    ceiling and floor were synthesised rather than photographed.
    """
    intr = Intrinsics.from_hfov(frame_size[0], frame_size[1], hfov_deg)
    return float(math.sin(math.radians(intr.vfov_deg / 2.0)))


def _run_scenario(
    tmp_path: Path,
    *,
    name: str,
    style: str,
    trajectory: str,
    effects: str,
    arm_radius_m: float,
    mode: QualityMode,
    output_width: int,
    duration_s: float,
    fps: int,
    seed: int = 7,
    frame_size: tuple[int, int] = QUICK_FRAME,
    revolutions: float | None = None,
    overrides: dict | None = None,
) -> dict:
    """Simulate a capture, stitch it with the real pipeline, and score the result.

    ``revolutions`` is how much of a turn the sweep completes, and only ``single_row``
    accepts it. It is what makes an *incomplete* capture incomplete: duration alone does
    not, because the angle programme is a function of progress rather than of time, so a
    shorter capture sweeps the same arc faster rather than stopping short of it.
    """
    sroom, scap, sencode, smetrics = _synth()
    from biosphere.pano.pipeline import stitch_capture

    room = sroom.Room.random(seed=seed, style=style)
    traj_kwargs = {} if revolutions is None else {"revolutions": float(revolutions)}
    traj = getattr(scap.Trajectory, trajectory)(duration_s=duration_s, fps=fps, **traj_kwargs)

    intr = Intrinsics.from_hfov(frame_size[0], frame_size[1], CAPTURE_HFOV_DEG)
    synthetic = scap.simulate_capture(
        room,
        traj,
        intr,
        out_dir=tmp_path / f"{name}_frames",
        effects=effects,
        seed=seed,
        arm_radius_m=arm_radius_m,
        ground_truth_width=output_width,
    )
    truth = synthetic.ground_truth

    video = sencode.encode_video(synthetic.frame_paths, tmp_path / f"{name}.mp4", fps=fps, crf=20)

    config = PipelineConfig.for_mode(mode, output_width=output_width, **(overrides or {}))
    result = stitch_capture(
        video,
        config,
        work_dir=tmp_path / f"{name}_work",
        out_dir=tmp_path / f"{name}_out",
        cleanup=False,
    )

    mask = _coverage_mask_like(result, truth.shape[:2])
    aligned, yaw_offset = smetrics.align_panoramas(result.panorama, truth)
    quality = smetrics.compare(result.panorama, truth, mask=mask)

    # The solver reports one pose per *keyframe*; the simulator holds one per *source
    # frame*, and which frames became keyframes is data-dependent. `keyframe_timestamps`
    # is the only thing that lines the two up, so refuse to score rather than guess if it
    # is missing — a silently mismatched comparison is worse than no comparison.
    truth_poses = _truth_poses_for(result, synthetic)
    poses = smetrics.pose_errors(result.poses, truth_poses)

    truth_seam = smetrics.seam_score(truth)
    scores = {
        "scenario": name,
        "psnr": float(quality.psnr),
        "ssim": float(quality.ssim),
        "psnr_horizon": float(quality.psnr_horizon),
        "ssim_horizon": float(quality.ssim_horizon),
        "yaw_offset_deg": float(yaw_offset),
        "pose_n": int(poses.n),
        "pose_mean_deg": float(poses.mean_deg),
        "pose_max_deg": float(poses.max_deg),
        "pose_yaw_max_deg": float(poses.yaw_max_deg),
        "pose_pitch_max_deg": float(poses.pitch_max_deg),
        "wrap_seam": _wrap_step_ratio(result.panorama),
        "wrap_seam_global": float(smetrics.wrap_seam_score(result.panorama)),
        # Sharpness retention: if the stitcher softened the image, high-frequency energy
        # would fall below the ground truth's. PSNR alone cannot tell blurring apart from
        # a sub-pixel shift, and only one of those two is a defect a viewer would notice.
        "hf_retention": _hf_retention(aligned, truth),
        # Ratio, not absolute: a real room's own corners score well above 1.0 on their own.
        "seam_ratio": float(smetrics.seam_score(result.panorama) / max(truth_seam, 1e-6)),
        "covered_fraction": float(result.coverage.covered_fraction),
        "horizon_fraction": float(result.coverage.horizon_fraction),
        "zenith_covered": bool(result.coverage.zenith_covered),
        "nadir_covered": bool(result.coverage.nadir_covered),
        "n_gaps": len(result.coverage.gaps),
        "n_keyframes": int(result.metrics.n_keyframes),
        "mean_reprojection_error_px": float(result.metrics.mean_reprojection_error_px),
        "loop_closure_residual_deg": float(result.metrics.loop_closure_residual_deg),
        "hfov_deg": float(result.metrics.hfov_deg),
        "hfov_error_deg": abs(float(result.metrics.hfov_deg) - CAPTURE_HFOV_DEG),
        "stage_seconds": dict(result.metrics.stage_seconds),
        "warnings": list(result.metrics.warnings),
    }
    _write_report(name, truth, aligned, scores)
    return scores


def _assert_gate(scores: dict, gate: Gate) -> None:
    # `BIOSPHERE_GATE_CALIBRATE=1` reports instead of failing. Photometric thresholds are
    # scene-dependent — PSNR against procedurally-textured ground truth punishes a
    # sub-pixel misregistration far harder than a viewer ever would — so they are
    # *regression* guards calibrated from a measured baseline, not absolute quality
    # claims. This switch is how that baseline gets re-measured after a deliberate change.
    if os.environ.get("BIOSPHERE_GATE_CALIBRATE") == "1":
        print("\nCALIBRATION " + json.dumps(scores, indent=2))
        return
    checks = [
        ("psnr_horizon", scores["psnr_horizon"], gate.psnr_horizon, "min"),
        ("ssim_horizon", scores["ssim_horizon"], gate.ssim_horizon, "min"),
        ("pose_mean_deg", scores["pose_mean_deg"], gate.pose_mean_deg, "max"),
        ("pose_max_deg", scores["pose_max_deg"], gate.pose_max_deg, "max"),
        ("wrap_seam", scores["wrap_seam"], gate.wrap_seam, "max"),
        ("seam_ratio", scores["seam_ratio"], gate.seam_ratio, "max"),
        ("covered_fraction", scores["covered_fraction"], gate.covered_fraction, "min"),
        ("hf_retention", scores["hf_retention"], gate.hf_retention, "min"),
    ]
    failures = [
        f"{label} {value:.4f} {'<' if kind == 'min' else '>'} {limit}"
        for label, value, limit, kind in checks
        if (value < limit if kind == "min" else value > limit)
    ]
    assert not failures, (
        "quality gate failed:\n  " + "\n  ".join(failures) + "\nfull scores: " + json.dumps(scores, indent=2)
    )


# ======================================================================================
# Scenarios
# ======================================================================================


def test_nodal_single_row_is_near_perfect(tmp_path: Path):
    """No parallax, no artefacts: this isolates the geometry and compositing algorithms.

    If this one degrades, the regression is in the engine itself rather than in its
    robustness to real-world capture noise.
    """
    scores = _run_scenario(
        tmp_path,
        name="nodal_single",
        style="rich",
        trajectory="single_row",
        effects="clean",
        arm_radius_m=0.0,
        mode=QualityMode.BALANCED,
        output_width=2048,
        duration_s=18.0,
        fps=16,
    )
    _assert_gate(scores, GATES["nodal"])
    assert scores["hfov_error_deg"] < 2.0, f"focal recovery drifted: {scores}"

    # Coverage must land close to the physical ceiling of a level row — neither well
    # under it (frames were lost) nor over it (synthesised poles counted as captured).
    ceiling = _expected_single_row_coverage(QUICK_FRAME, CAPTURE_HFOV_DEG)
    assert scores["covered_fraction"] <= ceiling + 0.04, (
        f"reported coverage {scores['covered_fraction']:.3f} exceeds the {ceiling:.3f} a level "
        "single row can physically reach — synthesised poles are being counted as captured"
    )
    assert scores["covered_fraction"] > ceiling - 0.08, (
        f"coverage {scores['covered_fraction']:.3f} well under the {ceiling:.3f} ceiling — "
        "keyframes were probably dropped"
    )
    assert not scores["zenith_covered"] and not scores["nadir_covered"], (
        "a level single row cannot see the ceiling or the floor; saying otherwise is the "
        "one dishonesty this product must never commit"
    )


@pytest.mark.slow
def test_realistic_three_row_meets_product_bar(tmp_path: Path):
    """The commercial case: a real phone, a real hand, a real 0.28 m arm radius."""
    scores = _run_scenario(
        tmp_path,
        name="realistic_three_row",
        style="rich",
        trajectory="three_row",
        effects="realistic",
        arm_radius_m=0.28,
        mode=QualityMode.BALANCED,
        output_width=2048,
        duration_s=45.0,
        fps=24,
        frame_size=FULL_FRAME,
    )
    _assert_gate(scores, GATES["realistic"])
    assert scores["zenith_covered"], "three rows should actually reach the ceiling"


@pytest.mark.slow
def test_low_texture_room_still_solves(tmp_path: Path):
    """Plain white walls are the case that breaks naive stitchers.

    The bar is deliberately lower on photometric similarity but *not* on geometry: the
    solver still has to close the loop, because a drifting loop produces a visible tear.
    """
    scores = _run_scenario(
        tmp_path,
        name="low_texture",
        style="low_texture",
        trajectory="three_row",
        effects="realistic",
        arm_radius_m=0.28,
        mode=QualityMode.BALANCED,
        output_width=2048,
        duration_s=45.0,
        fps=24,
        frame_size=FULL_FRAME,
    )
    assert scores["loop_closure_residual_deg"] < 3.0, f"loop did not close: {scores}"
    assert scores["pose_mean_deg"] < 2.5, f"poses drifted on low texture: {scores}"
    assert scores["wrap_seam"] < 1.8, f"visible wrap seam: {scores}"
    assert scores["psnr_horizon"] > 19.0, f"too degraded: {scores}"


@pytest.mark.slow
def test_studio_mode_beats_balanced_on_parallax(tmp_path: Path):
    """Studio mode exists to buy back parallax ghosting; prove that it actually does."""
    common = dict(
        style="rich",
        trajectory="single_row",
        effects="harsh",
        arm_radius_m=0.35,
        output_width=2048,
        duration_s=24.0,
        fps=20,
        frame_size=FULL_FRAME,
    )
    balanced = _run_scenario(tmp_path, name="parallax_balanced", mode=QualityMode.BALANCED, **common)
    studio = _run_scenario(tmp_path, name="parallax_studio", mode=QualityMode.STUDIO, **common)
    assert studio["ssim_horizon"] >= balanced["ssim_horizon"], (
        "studio mode must not be worse than balanced; "
        f"studio={studio['ssim_horizon']:.4f} balanced={balanced['ssim_horizon']:.4f}"
    )
    assert studio["seam_ratio"] <= balanced["seam_ratio"] * 1.05


def test_partial_sweep_is_reported_not_faked(tmp_path: Path):
    """A short sweep must come back honestly labelled, with the gap located.

    Silently inventing the missing arc would be the single most damaging thing this
    product could do, so the contract is: produce what you have, and say what you do not.

    ``revolutions`` is the whole point of the scenario and has to be set explicitly: the
    trajectory defaults to a *full* turn, and a full level row leaves nothing uncovered
    except the poles. Poles are not gaps — ``coverage._find_gaps`` searches only the
    elevation band the capture actually attempted, precisely so that the unreachable
    ceiling and floor of every single-row property do not get reported as a ring-shaped hole
    (their absence is reported by ``zenith_covered`` / ``nadir_covered`` instead). So a
    full turn yields zero gaps by design, and asking for one back would be asking the
    coverage report to invent a defect. 0.7 of a turn leaves a real ~110 deg hole in the
    horizon band, which is the thing this test exists to see located, and still clears the
    200 deg of yaw ``solve.MIN_YAW_SPAN_DEG`` requires of a panorama.
    """
    scores = _run_scenario(
        tmp_path,
        name="partial",
        style="rich",
        trajectory="single_row",
        effects="clean",
        arm_radius_m=0.0,
        mode=QualityMode.FAST,
        output_width=1024,
        duration_s=12.0,
        fps=14,
        seed=11,
        revolutions=0.7,
    )
    assert scores["covered_fraction"] < 0.99
    assert scores["n_gaps"] >= 1, "an incomplete sphere must come back with its gaps located"


def test_fast_mode_is_actually_fast(tmp_path: Path):
    """Fast mode is a promise about latency, but not a licence to be wrong.

    This test used to assert the clock and nothing else, which let the mode's geometry rot
    unnoticed: a run at 17 s came back with 2.2 deg mean pose error, a 5.6 deg loop
    residual — past our own 2.0 deg "expect misaligned detail" warning — and coverage a
    quarter under the ceiling a level row can physically reach. The dashboard advertises
    fast mode as "4K, ~40 sn", not as "may not close the loop", so the bar below is what
    the mode has to clear to keep making that offer.

    The thresholds are deliberately looser than balanced mode's and still far from the
    measured failure: fast trades resolution, seam search and refinement for time, but it
    must not trade away whether the panorama closes.
    """
    scores = _run_scenario(
        tmp_path,
        name="fast_timing",
        style="rich",
        trajectory="single_row",
        effects="realistic",
        arm_radius_m=0.28,
        mode=QualityMode.FAST,
        output_width=2048,
        duration_s=16.0,
        fps=14,
    )
    ceiling = _expected_single_row_coverage(QUICK_FRAME, CAPTURE_HFOV_DEG)
    failures = []
    if scores["loop_closure_residual_deg"] > 2.0:
        failures.append(
            f"loop {scores['loop_closure_residual_deg']:.2f} deg > 2.0 — past the threshold "
            "at which we warn the user to expect misaligned detail"
        )
    # This scenario holds the phone 0.28 m from the pivot, and that arm is what sets the
    # bar here, not the solver. Tangential translation adds image flow f*r/d in the same
    # direction as the rotation flow, so a rotation-only model closes the 360-degree loop
    # by inflating the focal by about (1 + r/d) -- 13% in this room -- and by spreading
    # the depth-dependent remainder over the yaw spacing. Measured on this fixture: every
    # start (520, 579.4, 640, 700 px) converges to the same 655.4 px against a true 579.4,
    # the fixed-focal cost profile is within 8% from 600 to 670 px, and a per-pair
    # translation nuisance term moves the answer only a third of the way back (629.7 px)
    # because it is the rotations plus focal, not the nuisance, that absorb a uniform
    # parallax. The release-time solver on the same capture: 652.2 px, 1.71 deg mean pose
    # error, in 154 s. So `hfov_error < 2.0` and `pose_mean < 1.20` never held here; the
    # nodal scenario is where the focal is a claim about the solver, and there it is
    # asserted at 2 degrees. What this scenario CAN promise is that the bias stays the
    # explained one: an effective field between the true 45 and the fully parallax-scaled
    # one (r/d up to ~0.2 at the nearest wall => 38 deg), and pose errors of the size the
    # depth spread implies rather than the 2.2 deg of a solve that did not close.
    if not (38.0 <= scores["hfov_deg"] <= 46.0):
        failures.append(
            f"hfov {scores['hfov_deg']:.2f} deg outside [38, 46] — beyond what a 0.28 m arm "
            "in this room can explain; the focal is off for another reason"
        )
    if scores["pose_mean_deg"] > 2.0:
        failures.append(f"pose_mean {scores['pose_mean_deg']:.3f} deg > 2.0")
    if scores["covered_fraction"] < ceiling - 0.10:
        failures.append(
            f"coverage {scores['covered_fraction']:.3f} more than 0.10 under the {ceiling:.3f} "
            "ceiling — keyframes are being lost"
        )
    if scores["wrap_seam"] > 2.0:
        failures.append(f"wrap_seam {scores['wrap_seam']:.2f} > 2.0")
    assert not failures, (
        "fast mode met its latency budget but not its quality floor:\n  "
        + "\n  ".join(failures)
        + "\nfull scores: "
        + json.dumps(scores, indent=2)
    )

    total = scores["stage_seconds"].get("total", 1e9)
    assert total < 120.0, f"fast mode took {total:.1f}s on a 20s capture: {scores['stage_seconds']}"
