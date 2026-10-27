"""Run the solve half once on a real capture and cache every intermediate to disk.

The solve costs ~10 minutes on the reference 210 s three-row capture, and almost all of
that is spent BEFORE bundle adjustment (extract, select, features, match). Anything we want
to try in the bundle stage or the renderer should not pay that again, so this script runs
`solve_capture` with two hooks that pickle its state:

  <work>/cache/pre_bundle.pkl   poses0, pairs, intr, keyframes, prior_yaw, config
  <work>/cache/solved.pkl       the SolveResult (frames, poses, pairs, metrics)

Frames keep pointing at <work>/frames/frame_*.jpg, so the work dir must be kept.

Usage:
    python scripts/bench_solve_cache.py <work_dir> <video> [<video> ...] [--mode balanced]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.pano import solve as solve_mod  # noqa: E402
from biosphere.pano.pipeline import solve_only  # noqa: E402
from biosphere.pano.types import PipelineConfig, QualityMode  # noqa: E402


class ConsoleProgress:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.stage_name = ""

    def stage(self, name: str, fraction: float, message: str = "") -> None:
        if name != self.stage_name or fraction >= 1.0:
            self.stage_name = name
            print(
                f"  [{time.perf_counter() - self.t0:6.1f}s] {int(fraction * 100):3d}%  {name}"
                f"{('  ' + message) if message else ''}",
                flush=True,
            )

    def log(self, message: str, **fields: object) -> None:
        print(f"        {message} {fields if fields else ''}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--mode", default="balanced", choices=[m.value for m in QualityMode])
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(exist_ok=True)
    config = PipelineConfig.for_mode(args.mode)

    state: dict = {}
    real_keep = solve_mod._keep_largest_component
    real_ba = solve_mod.bundle_adjust

    def keep_hook(keyframes, pairs, prior_yaw, metrics):
        out = real_keep(keyframes, pairs, prior_yaw, metrics)
        state["keyframes"] = out[0]
        state["prior_yaw"] = out[2]
        state["dropped"] = out[3]
        return out

    def ba_hook(poses0, pairs, intr, cfg, *, progress=None, span=(0.35, 0.9)):
        with (cache / "pre_bundle.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "poses0": poses0,
                    "pairs": pairs,
                    "intr": intr,
                    "keyframes": state.get("keyframes"),
                    "prior_yaw": state.get("prior_yaw"),
                    "dropped": state.get("dropped"),
                    "config": cfg,
                },
                fh,
            )
        print(f"  cached pre_bundle.pkl ({len(pairs)} pairs, {len(poses0)} cameras)", flush=True)
        return real_ba(poses0, pairs, intr, cfg, progress=progress, span=span)

    solve_mod._keep_largest_component = keep_hook
    solve_mod.bundle_adjust = ba_hook

    t0 = time.perf_counter()
    result = solve_only(
        [Path(v) for v in args.videos], config, work_dir=work, progress=ConsoleProgress()
    )
    with (cache / "solved.pkl").open("wb") as fh:
        pickle.dump(result, fh)
    print(f"solved in {time.perf_counter() - t0:.1f}s; cached solved.pkl", flush=True)
    m = result.metrics.as_dict()
    for k in (
        "n_keyframes",
        "n_pairs",
        "mean_reprojection_error_px",
        "max_reprojection_error_px",
        "loop_closure_residual_deg",
        "loop_closure_max_deg",
        "focal_px",
        "hfov_deg",
    ):
        print(f"  {k}: {m[k]}")
    print("  stage_seconds:", m["stage_seconds"])
    for w in m["warnings"]:
        print("  warning:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
