"""Stitch a real phone capture and print an honest report about what came out.

Usage:
    python scripts/stitch_real.py <video> <out_dir> [--mode balanced] [--width 4096]

Unlike the test suite there is no ground truth here, so the report leans on the
self-measurable signals: how much of the sphere was actually reached, how well the
rotations agreed with each other, and whether the seams and the wrap column stand out
from the scene's own edges.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.pano.pipeline import stitch_capture  # noqa: E402
from biosphere.pano.types import PipelineConfig, QualityMode  # noqa: E402


class ConsoleProgress:
    """One line per stage, with the elapsed clock — enough to see where time goes."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.stage_name = ""

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        if name != self.stage_name:
            self.stage_name = name
            print(f"  [{time.perf_counter() - self.t0:6.1f}s] {int(fraction * 100):3d}%  {name}"
                  f"{('  ' + message) if message else ''}", flush=True)

    def log(self, message: str, **fields: object) -> None:
        print(f"        {message} {fields if fields else ''}", flush=True)


def wrap_step_ratio(pano: np.ndarray, band: tuple[float, float] = (0.30, 0.62)) -> float:
    lo, hi = int(pano.shape[0] * band[0]), int(pano.shape[0] * band[1])
    strip = pano[lo:hi].astype(np.float32)
    cyclic = np.concatenate([strip, strip[:, :1]], axis=1)
    diffs = np.abs(np.diff(cyclic, axis=1)).mean(axis=(0, 2))
    med = float(np.median(diffs))
    # The wrap is the last difference (column W-1 back to 0); diffs[0] is columns 0 -> 1.
    return float(diffs[-1] / med) if med > 1e-6 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--mode", default="balanced", choices=[m.value for m in QualityMode])
    ap.add_argument("--width", type=int, default=4096)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig.for_mode(args.mode, output_width=args.width)

    print(f"stitching {Path(args.video).name}  mode={args.mode}  width={args.width}")
    t0 = time.perf_counter()
    result = stitch_capture(
        args.video,
        cfg,
        work_dir=out / "work",
        out_dir=out / "levels",
        progress=ConsoleProgress(),
        cleanup=False,
    )
    elapsed = time.perf_counter() - t0

    cv2.imwrite(str(out / "panorama.jpg"), result.panorama, [cv2.IMWRITE_JPEG_QUALITY, 94])
    m, c = result.metrics, result.coverage

    report = {
        "seconds": round(elapsed, 1),
        "output": f"{result.width}x{result.height}",
        "frames_extracted": m.n_frames_extracted,
        "keyframes": m.n_keyframes,
        "pairs": m.n_pairs,
        "loop_closures": m.n_loop_closures,
        "sweep_pattern": m.sweep_pattern.value,
        "hfov_deg": round(m.hfov_deg, 2),
        "focal_px": round(m.focal_px, 1),
        "mean_reprojection_px": round(m.mean_reprojection_error_px, 3),
        "loop_residual_deg": round(m.loop_closure_residual_deg, 3),
        "seam_gradient_energy": round(m.seam_gradient_energy, 4),
        "wrap_step_ratio": round(wrap_step_ratio(result.panorama), 3),
        "covered_fraction": round(c.covered_fraction, 4),
        "horizon_fraction": round(c.horizon_fraction, 4),
        "yaw_span_deg": round(c.yaw_span_deg, 1),
        "pitch_range_deg": [round(c.pitch_min_deg, 1), round(c.pitch_max_deg, 1)],
        "zenith_covered": c.zenith_covered,
        "nadir_covered": c.nadir_covered,
        "gaps": [[round(v, 1) for v in g] for g in c.gaps[:6]],
        "levels": [f"{lvl.name}:{lvl.width}x{lvl.height}" for lvl in result.levels],
        "stage_seconds": {k: round(v, 2) for k, v in m.stage_seconds.items()},
        "warnings": m.warnings,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))
    print(f"\npanorama -> {out / 'panorama.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
