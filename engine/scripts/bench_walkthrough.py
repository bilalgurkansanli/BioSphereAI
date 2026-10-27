"""Bench the walkthrough pipeline on a real flat and write the numbers out to be looked at.

    uv run python scripts/bench_walkthrough.py --mode fast
    uv run python scripts/bench_walkthrough.py --mode balanced --out C:/biosphere-data/work/bench_walk

The dataset is the ZipNeRF "berlin" flat (1,525 photographs walked through an apartment,
with COLMAP ground-truth poses), re-projected by ``prepare_berlin_bench.py`` from its 160°
fisheye into the portrait rectilinear frame a phone's ultra-wide records (the raw fisheye
measures the camera model, not the pipeline - see that script). Three videos are cut from
it the way a user would film: a general walkthrough (the 480-photograph stretch that covers the
most floor) and two rooms (contiguous stretches, see ``SPLITS``). The pipeline gets the videos only; the ground
truth is used afterwards to score what it recovered:

* poses: how far each area's cameras land from where COLMAP put them (position in metres
  after the property frame's own scale, rotation in degrees), through one Sim3 fitted on the
  base - so a room's error includes its registration into the property;
* training: Brush's held-out PSNR per area, splat counts, SOG bytes, wall clock per stage.

Everything lands in ``<out>/report.json`` and ``<out>/report.md`` next to the property itself,
so a regression is a diff and a claim has a number behind it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.splat.pipeline import build_tour  # noqa: E402
from biosphere.splat.types import AreaInput, SplatConfig, SplatQuality  # noqa: E402

BERLIN = Path("C:/biosphere-data/datasets/berlin_rect")  # from scripts/prepare_berlin_bench.py
VIDEOS = Path("C:/biosphere-data/datasets/berlin_rect_tour2")
# Photo ranges (start, stop, step) of the three videos. The berlin capture is ~500 bursts
# of three photographs 0.5 m apart, walked through the flat several times over; a real
# general walkthrough is ONE pass, so the base is the 480-photograph stretch that covers the
# most floor, and the rooms are two later/earlier stretches whose floor it covers.
SPLITS = {"base": (700, 1180, 1), "room1": (540, 700, 1), "room2": (1200, 1360, 1)}


def make_videos() -> dict[str, Path]:
    src = BERLIN / "images"
    names = sorted(p.name for p in src.glob("*.jpg"))
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, (a, b, step) in SPLITS.items():
        target = VIDEOS / f"{key}.mp4"
        out[key] = target
        if target.exists():
            continue
        sel = names[a:b:step]
        lst = target.with_suffix(".txt")
        lst.write_text("".join(f"file '{(src / n).as_posix()}'\nduration 0.5\n" for n in sel) + f"file '{(src / sel[-1]).as_posix()}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "fps=2", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(target)],
            check=True,
        )
    return out


def sim3_fit(src: np.ndarray, tgt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: ``tgt = s R src + t``."""
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    a, b = src - mu_s, tgt - mu_t
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a**2).sum(1).mean())
    return s, R, mu_t - s * (R @ mu_s)


def score_against_truth(work: Path, keys: list[str]) -> dict:
    """Compare every area's property-frame cameras with the ground truth, through ONE Sim3 fitted on the base."""
    import pycolmap

    gt = pycolmap.Reconstruction(str(BERLIN / "sparse" / "0"))
    truth = {img.name: np.asarray(img.projection_center()) for img in gt.images.values() if img.has_pose}
    truth_R = {img.name: np.asarray(img.cam_from_world().rotation.matrix()) for img in gt.images.values() if img.has_pose}
    names_all = sorted(truth)

    # The bench videos show one photograph per 0.5 s at 2 fps, so a kept frame's timestamp
    # (recorded in frames.json by prepare_frames) names the photograph it came from.
    report: dict = {}
    recs: dict[str, dict[str, np.ndarray]] = {}
    rots: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        d = work / "sfm" / key / "property_frame"
        if not d.exists():
            continue
        rec = pycolmap.Reconstruction(str(d))
        stamps = json.loads((work / "frames" / key / "frames.json").read_text())["timestamps"] if (work / "frames" / key / "frames.json").exists() else None
        a, b, step = SPLITS[key]
        mapping: dict[str, np.ndarray] = {}
        rmap: dict[str, np.ndarray] = {}
        for img in rec.images.values():
            if not img.has_pose or not img.name.startswith(f"{key}/"):
                continue
            idx = int(img.name.split("f_")[-1].split(".")[0])
            # Photo j is on screen during [j/2, (j+1)/2) s, so a frame's stamp floors to it.
            if stamps is not None and idx < len(stamps):
                photo = a + int(np.floor(stamps[idx] * 2.0 + 1e-6)) * step
            else:
                photo = a + idx * step
            if 0 <= photo < len(names_all) and names_all[photo] in truth:
                mapping[names_all[photo]] = np.asarray(img.projection_center())
                rmap[names_all[photo]] = np.asarray(img.cam_from_world().rotation.matrix())
        recs[key] = mapping
        rots[key] = rmap
    if "base" not in recs or len(recs["base"]) < 10:
        return {"error": "no base reconstruction to score"}
    src = np.stack(list(recs["base"].values()))
    tgt = np.stack([truth[n] for n in recs["base"]])
    s, R, t = sim3_fit(src, tgt)  # property frame -> GT frame
    gt_scale_per_m = 1.0 / s  # GT units per metre (property frame is metric)
    for key, mapping in recs.items():
        est = np.stack(list(mapping.values()))
        ref = np.stack([truth[n] for n in mapping])
        err_m = np.linalg.norm((s * (est @ R.T) + t) - ref, axis=1) / max(gt_scale_per_m, 1e-9)
        # cam_from_world in the GT frame is R_cam @ R^T; the residual to the truth is the rotation error.
        rot_deg = np.array([
            np.degrees(np.arccos(np.clip((np.trace(truth_R[n] @ (rots[key][n] @ R.T).T) - 1.0) / 2.0, -1.0, 1.0)))
            for n in mapping
        ])
        report[key] = {
            "n": int(len(est)),
            "pos_err_median_m": round(float(np.median(err_m)), 4),
            "pos_err_p90_m": round(float(np.percentile(err_m, 90)), 4),
            "pos_err_max_m": round(float(err_m.max()), 4),
            "rot_err_median_deg": round(float(np.median(rot_deg)), 3),
            "rot_err_p90_deg": round(float(np.percentile(rot_deg, 90)), 3),
        }
    report["gt_units_per_metre"] = round(gt_scale_per_m, 4)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fast", choices=["fast", "balanced", "studio"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="keep frames/train work dirs")
    ap.add_argument("--base-only", action="store_true", help="solve and register every area, train only the base")
    ap.add_argument("--no-dense-init", action="store_true", help="start the trainer from the SfM points (the old init)")
    args = ap.parse_args()
    videos = make_videos()
    out = Path(args.out or f"C:/biosphere-data/work/bench_walk_{args.mode}")
    if out.exists():
        shutil.rmtree(out)
    work = out / "work"
    deliver = out / "property"

    class P:
        t0 = time.perf_counter()
        marks: dict[str, float] = {}

        def stage(self, name, frac, message=""):
            self.marks.setdefault(name, time.perf_counter() - self.t0)
            if frac >= 1.0 or message:
                print(f"[{time.perf_counter() - self.t0:7.1f}s] {name:9s} {frac:5.2f} {message}", flush=True)

        def log(self, message, **fields):
            print("   ", message, fields, flush=True)

    prog = P()
    quality = SplatQuality.for_mode(args.mode)
    if args.no_dense_init:
        quality = replace(quality, dense_init=replace(quality.dense_init, enabled=False))
    t = time.perf_counter()
    res = build_tour(
        AreaInput("base", "Genel gezinti", (videos["base"],)),
        [AreaInput("room1", "Oda 1", (videos["room1"],)), AreaInput("room2", "Oda 2", (videos["room2"],))],
        config=SplatConfig(quality=quality, keep_work=True, base_only=args.base_only),
        work_dir=work, out_dir=deliver, progress=prog,
    )
    total = time.perf_counter() - t
    report = {
        "mode": args.mode, "seconds_total": round(total, 1), "stage_started_at": {k: round(v, 1) for k, v in prog.marks.items()},
        "warnings": res.warnings, "metrics": res.metrics, "areas": {a.key: a.as_dict() for a in [res.base] + res.rooms},
        "truth": score_against_truth(work, ["base", "room1", "room2"]),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = [f"# Walkthrough bench · {args.mode} · {total / 60:.1f} min", ""]
    lines.append(
        "| area | frames | registered | folded pairs | GT pos err med / p90 m | GT rot err med / p90 deg | "
        "reg. residual m / tilt deg | placed | dense init pts | train PSNR | splats | SOG MB | train s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    truth = report["truth"]
    for a in [res.base] + res.rooms:
        m = a.metrics
        fr = m.get("frames", {}).get("n_frames", "-")
        sf = m.get("sfm", {})
        reg = m.get("registration", {})
        tr = m.get("train", {})
        gt = truth.get(a.key, {}) if isinstance(truth, dict) else {}
        lines.append(
            f"| {a.name} | {fr} | {sf.get('n_registered', '-')}/{sf.get('n_images', '-')} ({sf.get('mapper', '-')}) | "
            f"{sf.get('strong_pair_violation_fraction', '-')} | "
            f"{gt.get('pos_err_median_m', '-')} / {gt.get('pos_err_p90_m', '-')} | "
            f"{gt.get('rot_err_median_deg', '-')} / {gt.get('rot_err_p90_deg', '-')} | "
            f"{reg.get('median_residual_m', '-')} / {reg.get('tilt_deg', '-')} | {a.placed} | "
            f"{(m.get('dense_init') or {}).get('n_points', '-')} | {tr.get('psnr', '-')} | {a.splat_count:,} | {a.sog_bytes / 1e6:.1f} | {tr.get('seconds', '-')} |"
        )
    lines += ["", "## Against COLMAP ground truth", "```", json.dumps(report["truth"], indent=1), "```", "", "## Warnings", *[f"- {w}" for w in res.warnings]]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if not args.keep:
        shutil.rmtree(work / "train", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
