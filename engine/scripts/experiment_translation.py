"""How much can the pipeline claw back from a capture where the phone moved?

The synthetic control says the stitcher is geometrically exact on a proper pivot capture
— its straightness matches the ground-truth panorama to about a tenth of a pixel. So the
bent architecture in a real capture comes from translation, not from the algorithm.

That is true but not useful on its own: real users walk. This measures which of the
levers we already have actually recovers geometry when the camera moved, so the default
for a `capture_geometry_degraded` verdict can be chosen from evidence rather than taste.

Variants:
  balanced        - the current default
  studio          - adds RAFT/DIS optical-flow local warping over each overlap
  sparse          - balanced, but far fewer keyframes

`sparse` is in here because of a specific hypothesis: under translation every pair of
frames disagrees, so each extra frame contributes another inconsistent join. Fewer, more
widely spaced frames mean fewer joins to be wrong at.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.pano.pipeline import stitch_capture  # noqa: E402
from biosphere.pano.types import PipelineConfig, QualityMode  # noqa: E402
from inspect_views import render_view, straightness  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit("usage: experiment_translation.py <capture-video> [out-dir]")
VIDEO = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiment-out")

VARIANTS = {
    "balanced": lambda: PipelineConfig.for_mode(QualityMode.BALANCED, output_width=4096),
    "studio": lambda: PipelineConfig.for_mode(QualityMode.STUDIO, output_width=4096),
    "sparse": lambda: PipelineConfig.for_mode(
        QualityMode.BALANCED, output_width=4096, max_keyframes=26, target_overlap=0.45
    ),
}

VIEWS = [(y, p) for p in (20.0, 0.0, -20.0) for y in (-135.0, -45.0, 45.0, 135.0)]


def score(pano: np.ndarray) -> dict:
    devs, edges, hfs = [], 0, []
    for yaw, pitch in VIEWS:
        v = render_view(pano, yaw, pitch, 75.0)
        d, n = straightness(v)
        if n > 0:
            devs.append(d)
            edges += n
        hfs.append(float(np.abs(cv2.Laplacian(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY),
                                              cv2.CV_32F)).mean()))
    return {
        "straightness_median_px": round(float(np.median(devs)) if devs else 0.0, 3),
        "straightness_worst_px": round(float(np.max(devs)) if devs else 0.0, 3),
        "n_edge_sets": edges,
        "hf_mean": round(float(np.mean(hfs)), 3),
        "hf_min": round(float(np.min(hfs)), 3),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, make in VARIANTS.items():
        print(f"\n===== {name} =====", flush=True)
        t0 = time.perf_counter()
        try:
            r = stitch_capture(VIDEO, make(), work_dir=OUT / f"{name}_work",
                               out_dir=OUT / f"{name}_out", cleanup=False)
        except Exception as exc:  # noqa: BLE001 - a variant failing is a result, not a crash
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[name] = {"failed": f"{type(exc).__name__}: {exc}"}
            continue
        secs = time.perf_counter() - t0
        cv2.imwrite(str(OUT / f"{name}.jpg"), r.panorama, [cv2.IMWRITE_JPEG_QUALITY, 93])
        s = score(r.panorama)
        s.update({
            "seconds": round(secs, 1),
            "keyframes": r.metrics.n_keyframes,
            "mean_reprojection_px": round(r.metrics.mean_reprojection_error_px, 3),
            "loop_residual_deg": round(r.metrics.loop_closure_residual_deg, 3),
            "seam_energy": round(r.metrics.seam_gradient_energy, 3),
            "covered_fraction": round(r.coverage.covered_fraction, 4),
        })
        results[name] = s
        print("  " + json.dumps(s), flush=True)

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(f"{'variant':10s} {'straight_med':>12s} {'straight_worst':>14s} {'hf_mean':>8s} "
          f"{'keyfr':>6s} {'reproj':>7s} {'loop':>6s} {'cover':>6s} {'secs':>6s}")
    for name, r in results.items():
        if "failed" in r:
            print(f"{name:10s} {r['failed']}")
            continue
        print(f"{name:10s} {r['straightness_median_px']:12.2f} {r['straightness_worst_px']:14.2f} "
              f"{r['hf_mean']:8.2f} {r['keyframes']:6d} {r['mean_reprojection_px']:7.2f} "
              f"{r['loop_residual_deg']:6.2f} {r['covered_fraction']:6.3f} {r['seconds']:6.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
