"""End-to-end orchestration: a phone capture goes in, an equirectangular panorama comes out.

The engine is deliberately split in two halves that meet at :class:`SolveResult`:

``solve``   — figures out *where the camera was looking* for every usable frame.
              Frame extraction, keyframe selection, feature matching, rotation-only
              bundle adjustment, loop closure, horizon levelling.

``render``  — figures out *what the sphere looks like* given those orientations.
              Exposure compensation, equirectangular warping, seam selection,
              multi-band blending, parallax refinement, pole filling, output levels.

Keeping the halves apart is what makes the pipeline debuggable: a ghosted panorama is
either a geometry failure (solve) or a compositing failure (render), and the metrics
each half reports tell you which. This module owns only the seam between them —
progress accounting, timing, work-directory hygiene and error translation.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import structlog

from biosphere.pano.types import (
    NullProgress,
    PipelineConfig,
    ProgressReporter,
    QualityMode,
    SolveResult,
    StitchError,
    StitchMetrics,
    StitchResult,
)

log = structlog.get_logger(__name__)

__all__ = ["stitch_capture", "solve_only", "STAGE_WEIGHTS", "StageWeightedProgress"]


# --------------------------------------------------------------------------------------
# Progress accounting
# --------------------------------------------------------------------------------------

# Relative cost of each stage, measured on the reference machine (Ryzen 5 3600, 12
# threads) for a 60-second 1080p capture at 8K output. These do not have to be exact,
# but they do have to be *monotonic and believable*: a progress bar that sits at 12 %
# for ninety seconds and then jumps to 100 % reads as a hang, and users cancel hangs.
STAGE_WEIGHTS: dict[str, float] = {
    "probe": 0.01,
    "extract": 0.10,
    "select": 0.03,
    "features": 0.10,
    "match": 0.14,
    "bundle": 0.07,
    "exposure": 0.05,
    "warp": 0.16,
    "seam": 0.08,
    "blend": 0.14,
    "refine": 0.06,
    "poles": 0.03,
    "levels": 0.03,
}

SOLVE_STAGES = ("probe", "extract", "select", "features", "match", "bundle")
RENDER_STAGES = ("exposure", "warp", "seam", "blend", "refine", "poles", "levels")


def _active_weights(config: PipelineConfig) -> dict[str, float]:
    """Drop stages this configuration will not run, then renormalise to sum to 1.

    Without this a fast-mode run would top out at 94 % because ``refine`` never fires.
    """
    weights = dict(STAGE_WEIGHTS)
    if not config.optical_flow_refine:
        weights.pop("refine", None)
    if not config.fill_poles or config.pole_method == "none":
        weights.pop("poles", None)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


class StageWeightedProgress:
    """Adapts per-stage fractions from the two halves onto one global 0..1 track.

    Each half reports ``stage(name, fraction)`` where ``fraction`` is progress *within*
    that stage. This wrapper turns that into overall progress by summing the weights of
    every stage already finished plus the partial weight of the current one, and it
    clamps the result to be non-decreasing — a bar that goes backwards looks broken even
    when the underlying estimate legitimately improved.
    """

    def __init__(self, inner: ProgressReporter, config: PipelineConfig) -> None:
        self._inner = inner
        self._weights = _active_weights(config)
        self._order = [s for s in STAGE_WEIGHTS if s in self._weights]
        self._completed = 0.0
        self._current: str | None = None
        self._high_water = 0.0

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        weight = self._weights.get(name)
        if weight is None:
            # An unexpected stage name should never silently vanish from the UI.
            self._inner.stage(name, self._high_water, message)
            return

        if name != self._current:
            if self._current is not None:
                self._completed += self._weights.get(self._current, 0.0)
            self._current = name

        fraction = min(max(float(fraction), 0.0), 1.0)
        overall = self._completed + weight * fraction
        self._high_water = max(self._high_water, min(overall, 0.999))
        self._inner.stage(name, self._high_water, message)

    def finish(self) -> None:
        self._inner.stage("done", 1.0, "")

    def log(self, message: str, **fields: Any) -> None:
        self._inner.log(message, **fields)


_PERFORMANCE_REQUESTED = False


def _prefer_performance_cores() -> None:
    """Ask Windows not to run this process as background (EcoQoS) work. Once per process.

    THE ENGINE WAS RUNNING ON A QUARTER OF THE LAPTOP. On a hybrid CPU (the reference
    machine is an i5-13450HX: 6 performance cores, 4 efficiency cores) Windows 11 gives a
    process with no foreground window "EcoQoS", and Thread Director places its threads on
    the efficiency cores at reduced clocks. A stitch server is exactly such a process.
    Measured on that machine with a fixed single-threaded loop launched the same way the
    engine is: 1487 iterations in 8 s as-is, 4194 after this call -- 2.8x; and a live
    render's graph-cut seam stage went from 0.27 to 1.0 cores the moment it was pinned off
    the efficiency cores, while those four cores sat at 100 % and the performance cores
    idled.

    `SetProcessInformation(ProcessPowerThrottling)` with EXECUTION_SPEED in the control
    mask and not in the state mask is the documented per-process opt-out: no privilege, no
    system setting, nothing outside this process. It only matters while the engine is
    computing something somebody asked for -- an idle server burns nothing either way.
    ``BIOSPHERE_ECO_QOS=1`` keeps Windows' default. Anywhere else this is a no-op.
    """
    global _PERFORMANCE_REQUESTED
    if _PERFORMANCE_REQUESTED:
        return
    _PERFORMANCE_REQUESTED = True
    import os
    import sys

    if sys.platform != "win32" or os.environ.get("BIOSPHERE_ECO_QOS", "").strip() in ("1", "true", "yes", "on"):
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        version, execution_speed, process_power_throttling = 1, 0x1, 4
        state = _PowerThrottlingState(version, execution_speed, 0)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), process_power_throttling,
            ctypes.byref(state), ctypes.sizeof(state),
        )
        log.info("stitch.power_throttling_opt_out", ok=bool(ok), error=ctypes.get_last_error())
    except Exception as exc:  # noqa: BLE001 - an optimisation, never a reason to fail
        log.info("stitch.power_throttling_opt_out_unavailable", error=str(exc))


class _Timer:
    """Records wall-clock per stage so the metrics can explain where the time went."""

    def __init__(self) -> None:
        self.spans: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Span:
            def __enter__(self) -> None:
                self._t0 = time.perf_counter()

            def __exit__(self, *exc: object) -> None:
                timer.spans[name] = timer.spans.get(name, 0.0) + (time.perf_counter() - self._t0)

        return _Span()


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def _normalise_source(source: Path | str | Sequence[Path | str]) -> Path | list[Path]:
    """Accept a video path, a directory of images, or an explicit image list."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            images = sorted(
                q
                for q in p.iterdir()
                if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
            )
            if not images:
                raise StitchError(f"No images found in {p}")
            return images
        if not p.exists():
            raise StitchError(f"Capture source does not exist: {p}")
        return p
    if isinstance(source, Iterable):
        paths = [Path(s) for s in source]
        if not paths:
            raise StitchError("Empty capture source")
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise StitchError(f"Missing input images: {missing[:3]}")
        return paths[0] if len(paths) == 1 and paths[0].suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
        } else paths
    raise StitchError(f"Unsupported capture source type: {type(source)!r}")


def solve_only(
    source: Path | str | Sequence[Path | str],
    config: PipelineConfig,
    *,
    work_dir: Path,
    progress: ProgressReporter | None = None,
) -> SolveResult:
    """Run just the geometry half. Useful for diagnosing alignment without paying for 8K."""
    from biosphere.pano.solve import solve_capture  # noqa: PLC0415 - heavy import, keep lazy

    work_dir.mkdir(parents=True, exist_ok=True)
    _prefer_performance_cores()
    reporter = StageWeightedProgress(progress or NullProgress(), config)
    return solve_capture(_normalise_source(source), config, work_dir=work_dir, progress=reporter)


def stitch_capture(
    source: Path | str | Sequence[Path | str],
    config: PipelineConfig | QualityMode | str | None = None,
    *,
    work_dir: Path,
    out_dir: Path | None = None,
    progress: ProgressReporter | None = None,
    cleanup: bool = True,
) -> StitchResult:
    """Turn a capture into a finished panorama.

    ``source`` may be a video file, a directory of stills, or a list of image paths.
    ``work_dir`` holds extracted frames and intermediates; it is removed afterwards
    unless ``cleanup`` is False. ``out_dir`` receives the delivered levels and defaults
    to ``work_dir / "out"``.

    Raises :class:`StitchError` (or one of its subclasses, which carry bilingual
    user-facing copy) for anything the user could plausibly fix by re-shooting.
    """
    # Imported lazily: pulling in OpenCV, SciPy and possibly Torch costs seconds and
    # hundreds of megabytes, which the API process should not pay at import time.
    from biosphere.pano.render import render_panorama  # noqa: PLC0415
    from biosphere.pano.solve import solve_capture  # noqa: PLC0415

    if config is None:
        config = PipelineConfig.for_mode(QualityMode.BALANCED)
    elif not isinstance(config, PipelineConfig):
        config = PipelineConfig.for_mode(config)

    normalised = _normalise_source(source)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir) if out_dir is not None else work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _prefer_performance_cores()

    reporter = StageWeightedProgress(progress or NullProgress(), config)
    timer = _Timer()
    started = time.perf_counter()

    log.info(
        "stitch.start",
        source=str(normalised if isinstance(normalised, Path) else f"{len(normalised)} images"),
        mode=config.mode.value,
        output_width=config.output_width,
        work_dir=str(work_dir),
    )

    try:
        with timer("solve"):
            solved: SolveResult = solve_capture(
                normalised, config, work_dir=work_dir, progress=reporter
            )

        log.info(
            "stitch.solved",
            keyframes=len(solved.frames),
            pairs=solved.metrics.n_pairs,
            mean_reproj_px=round(solved.metrics.mean_reprojection_error_px, 3),
            loop_residual_deg=round(solved.metrics.loop_closure_residual_deg, 3),
            hfov_deg=round(solved.metrics.hfov_deg, 2),
        )

        # Optical-flow refinement must not run on a capture that actually translated.
        #
        # It exists to remove parallax ghosting, but it removes it by *warping* one image
        # toward another, and under real translation the disagreement it is warping away
        # is a genuine depth difference rather than a small alignment error. Measured by
        # ablation on a 0.28 m-arm capture: with flow, straightness 2.53 px and detail
        # 3.70; without, 0.97 px and 4.91 — flow costs a factor of 2.6 in geometry, a
        # fifth of the sharpness, and 22 % more time. The same effect on the real
        # bedroom video was 0.98 px -> 5.27 px.
        #
        # Its own `flow_max_shift_px` gate cannot catch this: 24 px at an 8192 canvas is
        # 1.05 deg, while the parallax in a hand-held sweep runs 0.22-0.32 deg — a fifth
        # of the threshold. So the decision has to be made here, from the capture
        # assessment, which is the only thing that knows the camera moved.
        if config.optical_flow_refine and _capture_translated(solved.metrics):
            config = replace(config, optical_flow_refine=False)
            reporter.log(
                "refine.skipped",
                reason="capture_geometry_degraded",
                detail="the camera translated; local warping would bend geometry rather than fix it",
            )
            solved.metrics.warnings.append(
                "parallax_refinement_skipped: the phone moved as well as turned, so local "
                "warping was disabled — it would have distorted the panorama instead of "
                "sharpening it"
            )

        with timer("render"):
            result: StitchResult = render_panorama(
                solved.frames,
                solved.poses,
                config,
                out_dir=out_dir,
                progress=reporter,
                metrics=solved.metrics,
                pairs=solved.pairs,
            )

        # Only the pipeline sees both halves, so it is the only place that can record
        # which moment of the capture each pose came from. Keyed by ``frame_index`` rather
        # than by position: the renderer composites from a THINNED subset of the solved
        # views (`render.select_render_views`), so ``result.poses`` is shorter than
        # ``solved.frames`` and a positional copy would line pose k up with the wrong
        # moment of the capture — silently, and every downstream pose comparison with it.
        if not result.keyframe_timestamps:
            result.keyframe_timestamps = [
                float(solved.frames[int(p.frame_index)].timestamp) for p in result.poses
            ]

        _merge_metrics(result.metrics, solved.metrics, timer.spans, time.perf_counter() - started)
        reporter.finish()

        log.info(
            "stitch.done",
            seconds=round(time.perf_counter() - started, 2),
            covered=round(result.coverage.covered_fraction, 4),
            seam_energy=round(result.metrics.seam_gradient_energy, 4),
            levels=[lvl.name for lvl in result.levels],
        )
        return result

    except StitchError:
        raise
    except MemoryError as exc:
        raise StitchError(
            f"Ran out of memory rendering a {config.output_width}px panorama. "
            "Try the 'balanced' or 'fast' quality mode."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - deliberately broad; translated for the user
        log.exception("stitch.failed", error=str(exc))
        raise StitchError(f"Panorama generation failed: {exc}") from exc
    finally:
        if cleanup and not _keep_work_dirs():
            _safe_rmtree(work_dir, keep=out_dir)


#: Emitted by the capture validator when rotation stops explaining the motion.
_TRANSLATION_WARNINGS = ("capture_geometry_degraded", "high_reprojection_error")


def _capture_translated(metrics: StitchMetrics) -> bool:
    """Did the phone move through space as well as turn?

    Reads the solver's own verdict rather than re-deriving it: the validator already
    measured rotational purity before any of the expensive stages ran, and the solver
    already knows its reprojection error. Duplicating that judgement here would give us
    two definitions of "translated" that could disagree.
    """
    if any(w.startswith(_TRANSLATION_WARNINGS) for w in metrics.warnings):
        return True
    # Belt and braces for a capture that skipped assessment: above ~1.5 px of mean
    # reprojection on a rotation-only model, the residual is structural rather than noise.
    return metrics.mean_reprojection_error_px > 1.5


def _merge_metrics(
    target: StitchMetrics,
    solved: StitchMetrics,
    spans: dict[str, float],
    total_seconds: float,
) -> None:
    """Carry the solver's numbers onto the final metrics without clobbering render's."""
    for field in (
        "n_frames_extracted",
        "n_keyframes",
        "n_pairs",
        "n_loop_closures",
        "mean_reprojection_error_px",
        "max_reprojection_error_px",
        "loop_closure_residual_deg",
        "loop_closure_max_deg",
        "loop_closure_span_pairs",
        "parallax_median_px",
        "parallax_p90_px",
        "parallax_fraction_over_6px",
        "focal_px",
        "hfov_deg",
        "sweep_pattern",
    ):
        value = getattr(solved, field, None)
        if value:
            setattr(target, field, value)

    merged = dict(solved.stage_seconds)
    merged.update(target.stage_seconds)
    merged.update({k: round(v, 3) for k, v in spans.items()})
    merged["total"] = round(total_seconds, 3)
    target.stage_seconds = merged

    seen = set(target.warnings)
    target.warnings.extend(w for w in solved.warnings if w not in seen)


def _keep_work_dirs() -> bool:
    try:
        from biosphere.core.config import settings  # noqa: PLC0415

        return bool(settings.keep_work_dirs)
    except Exception:  # noqa: BLE001 - the engine must run without the API package
        return False


def _safe_rmtree(path: Path, *, keep: Path | None = None) -> None:
    """Remove intermediates, never the outputs, and never raise out of a finally block."""
    try:
        if keep is not None and keep.resolve().is_relative_to(path.resolve()):
            for child in path.iterdir():
                if child.resolve() == keep.resolve():
                    continue
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(
                    missing_ok=True
                )
        else:
            shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("stitch.cleanup_failed", path=str(path), error=str(exc))
