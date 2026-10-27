"""Decide what a walking capture can actually yield, before promising anything.

A walkthrough breaks the pure-rotation assumption the stitcher is built on: the camera
translates continuously, so consecutive frames are related by a homography only when the
scene is far away or the motion happens to be a pivot. This script measures, per moment
of the video:

* **rotational purity** — how well a rotation-only homography explains the matched
  points. Low residual means "the camera pivoted here"; high residual means "the camera
  moved through space", and no amount of bundle adjustment will fix that.
* **scene continuity** — the inlier ratio between neighbouring samples. It collapses when
  the operator walks through a doorway, which is what marks a room boundary.
* **angular rate** — how fast the view is turning, from the fitted rotation.

From those it proposes segments: contiguous stretches inside one room, and within each,
the sub-interval that is most nearly a pivot — the part a panorama could be built from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.pano import geometry as g  # noqa: E402
from biosphere.pano.types import Intrinsics  # noqa: E402


@dataclass
class Sample:
    index: int
    time_s: float
    inlier_ratio: float
    n_inliers: int
    rot_residual_px: float   # residual after fitting rotation-only; high => translation
    full_residual_px: float  # residual of the unconstrained homography, as a floor
    turn_deg: float
    mean_flow_px: float


def extract(video: Path, out: Path, fps: float, width: int) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("f_*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps},scale='min({width},iw)':-2:flags=lanczos",
         "-q:v", "3", str(out / "f_%05d.jpg")],
        check=True,
    )
    return sorted(out.glob("f_*.jpg"))


def analyse(paths: list[Path], fps: float, hfov_deg: float) -> list[Sample]:
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.015, edgeThreshold=12)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    matcher = cv2.BFMatcher()

    img0 = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    h, w = img0.shape[:2]
    intr = Intrinsics.from_hfov(w, h, hfov_deg)
    K, Kinv = intr.K, np.linalg.inv(intr.K)

    prev = sift.detectAndCompute(clahe.apply(img0), None)
    samples: list[Sample] = []

    for i, p in enumerate(paths[1:], start=1):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cur = sift.detectAndCompute(clahe.apply(img), None)
        ka, da = prev
        kb, db = cur
        prev = cur
        if da is None or db is None or len(ka) < 12 or len(kb) < 12:
            samples.append(Sample(i, i / fps, 0.0, 0, 99.0, 99.0, 0.0, 0.0))
            continue

        good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.8 * n.distance]
        if len(good) < 12:
            samples.append(Sample(i, i / fps, 0.0, len(good), 99.0, 99.0, 0.0, 0.0))
            continue

        pa = np.float32([ka[m.queryIdx].pt for m in good])
        pb = np.float32([kb[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 3.0, maxIters=8000, confidence=0.999)
        if H is None or mask is None:
            samples.append(Sample(i, i / fps, 0.0, 0, 99.0, 99.0, 0.0, 0.0))
            continue

        inl = mask.ravel().astype(bool)
        ratio = float(inl.mean())
        ia, ib = pa[inl], pb[inl]
        flow = float(np.linalg.norm(ib - ia, axis=1).mean())

        # Unconstrained homography residual: the best any 8-dof model can do here.
        proj = cv2.perspectiveTransform(ia.reshape(-1, 1, 2), H).reshape(-1, 2)
        full_res = float(np.linalg.norm(proj - ib, axis=1).mean())

        # Rotation-only residual: force H onto the K R K^-1 manifold and re-measure. The
        # gap between this and full_res is the part of the motion that is *not* a rotation.
        R = g.nearest_rotation(Kinv @ H @ K)
        Hrot = K @ R @ Kinv
        projr = cv2.perspectiveTransform(ia.reshape(-1, 1, 2), Hrot).reshape(-1, 2)
        rot_res = float(np.linalg.norm(projr - ib, axis=1).mean())
        turn = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

        samples.append(Sample(i, i / fps, ratio, int(inl.sum()), rot_res, full_res, turn, flow))
    return samples


def segment(samples: list[Sample], fps: float) -> list[dict]:
    """Cut at sustained collapses in scene continuity — i.e. doorways."""
    ratios = np.array([s.inlier_ratio for s in samples])
    smooth = cv2.GaussianBlur(ratios.reshape(-1, 1).astype(np.float32), (1, 9), 0).ravel()
    cut = smooth < max(0.18, float(np.percentile(smooth, 20)) * 0.6)

    bounds, run = [], 0
    for i, c in enumerate(cut):
        run = run + 1 if c else 0
        if run == 2:
            bounds.append(i - 1)
    edges = [0, *bounds, len(samples) - 1]

    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        if b - a < int(fps * 2.5):
            continue
        seg = samples[a:b]
        rot = np.array([s.rot_residual_px for s in seg])
        turn = np.array([s.turn_deg for s in seg])
        # Best pivot window: where rotation explains the motion best over >=3 seconds.
        win = max(4, int(fps * 3))
        best, best_score = None, 1e9
        for i in range(0, max(1, len(seg) - win)):
            score = float(np.median(rot[i:i + win]))
            if score < best_score:
                best_score, best = score, i
        out.append({
            "start_s": round(seg[0].time_s, 2),
            "end_s": round(seg[-1].time_s, 2),
            "duration_s": round(seg[-1].time_s - seg[0].time_s, 2),
            "median_rot_residual_px": round(float(np.median(rot)), 2),
            "median_full_residual_px": round(float(np.median([s.full_residual_px for s in seg])), 2),
            "total_turn_deg": round(float(turn.sum()), 1),
            "best_pivot_start_s": round(seg[best].time_s, 2) if best is not None else None,
            "best_pivot_residual_px": round(best_score, 2) if best is not None else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=62.0)
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else video.parent / (video.stem + "_analysis")
    paths = extract(video, out / "frames", args.fps, args.width)
    print(f"{len(paths)} frames at {args.fps} fps")

    samples = analyse(paths, args.fps, args.hfov)
    segs = segment(samples, args.fps)

    rot = np.array([s.rot_residual_px for s in samples if s.n_inliers > 20])
    full = np.array([s.full_residual_px for s in samples if s.n_inliers > 20])
    report = {
        "n_samples": len(samples),
        "rotational_purity": {
            "median_rot_residual_px": round(float(np.median(rot)), 2),
            "median_full_residual_px": round(float(np.median(full)), 2),
            "frac_below_2px": round(float((rot < 2.0).mean()), 3),
            "frac_below_4px": round(float((rot < 4.0).mean()), 3),
        },
        "segments": segs,
    }
    (out / "analysis.json").write_text(
        json.dumps({"report": report, "samples": [asdict(s) for s in samples]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
