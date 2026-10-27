"""SfM points vs Depth Anything 3 as the trainer's starting cloud, on held-out frames.

    uv run python scripts/bench_dense_init.py <posed model dir> <image root> <out dir> [--steps 12000] [--max-splats 800000]

The model dir is any solved COLMAP model of one area (for example a part's
``sfm/<key>/property_frame`` from a build run with ``keep_work``); the image root is the dir
its image names are relative to. Every 8th frame (by name) is held out of training for both
runs; both are trained with the shipped fast settings, cleaned the way the delivery cleans,
and scored on the held-out frames with the repo's own rasteriser: PSNR, SSIM, local contrast
(std of the Laplacian; the photographs' own value is printed for scale) and splat count.
Writes ``<out>/report.json`` and side-by-side held-out renders under ``<out>/<variant>/held``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pycolmap  # noqa: E402

from biosphere.splat import pipeline as pl  # noqa: E402
from biosphere.splat.clean import clean_table  # noqa: E402
from biosphere.splat.densify import DenseInitConfig, dense_init  # noqa: E402
from biosphere.splat.exposure import apply_gains, estimate_gains  # noqa: E402
from biosphere.splat.ply import read_ply  # noqa: E402
from biosphere.splat.raster import PinholeView, device_for, render  # noqa: E402
from biosphere.splat.sfm import undistort_for_training  # noqa: E402
from biosphere.splat.train import TrainConfig, train_splat  # noqa: E402


def _sparse(d: Path) -> Path:
    return d / "sparse" / "0" if (d / "sparse" / "0" / "cameras.bin").exists() else d / "sparse"


def _split(src: Path, out: Path, keep) -> Path:
    """A copy of dataset ``src`` holding only the images ``keep(name)`` accepts."""
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(_sparse(src)))
    for img in list(rec.images.values()):
        if img.has_pose and not keep(img.name):
            rec.deregister_frame(img.frame_id)
    for pid in [pid for pid, pt in rec.points3D.items() if pt.track.length() == 0]:
        rec.delete_point3D(pid)
    rec.write_binary(str(out / "sparse" / "0"))
    for img in rec.images.values():
        if img.has_pose:
            dst = out / "images" / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "images" / img.name, dst)
    return out


def _contrast(rgb: np.ndarray) -> float:
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(cv2.Laplacian(g, cv2.CV_32F)))


def _score(table, held: Path, save: Path) -> dict:
    from skimage.metrics import structural_similarity

    rec = pycolmap.Reconstruction(str(_sparse(held)))
    device = device_for()
    ps, ss, cr, cp = [], [], [], []
    save.mkdir(parents=True, exist_ok=True)
    for img in sorted((i for i in rec.images.values() if i.has_pose), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(v) for v in cam.params[:4])
        pose = img.cam_from_world()
        view = PinholeView(R=np.asarray(pose.rotation.matrix()), t=np.asarray(pose.translation), fx=fx, fy=fy, cx=cx, cy=cy,
                           width=int(cam.width), height=int(cam.height))
        rgb, _ = render(table, view, device=device)
        gt = cv2.cvtColor(cv2.imread(str(held / "images" / img.name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ps.append(10 * np.log10(1.0 / max(float(np.mean((rgb - gt) ** 2)), 1e-10)))
        ss.append(float(structural_similarity(rgb, gt, channel_axis=2, data_range=1.0)))
        cr.append(_contrast(rgb))
        cp.append(_contrast(gt))
        both = np.concatenate([gt, rgb], axis=1)
        cv2.imwrite(str(save / (Path(img.name).stem + ".jpg")), cv2.cvtColor((both * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return {"n": len(ps), "psnr": round(float(np.mean(ps)), 3), "ssim": round(float(np.mean(ss)), 4),
            "contrast": round(float(np.mean(cr)), 3), "contrast_photos": round(float(np.mean(cp)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-splats", type=int, default=None)
    ap.add_argument("--every", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rec = pycolmap.Reconstruction(args.model)
    names = sorted(img.name for img in rec.images.values() if img.has_pose)
    held = set(names[args.every // 2 :: args.every])
    allset = undistort_for_training(Path(args.model), Path(args.images), out / "all", max_image_size=1280, image_names=names)
    gains, _ = estimate_gains(pycolmap.Reconstruction(str(_sparse(allset))), allset / "images")
    apply_gains(allset / "images", gains)
    held_dir = _split(allset, out / "held", lambda n: n in held)
    cfg = TrainConfig.for_mode("fast")
    cfg = replace(cfg, eval_split_every=0, total_steps=args.steps or cfg.total_steps, max_splats=args.max_splats or cfg.max_splats)
    report: dict = {"model": args.model, "n_train": len(names) - len(held), "n_held": len(held), "train": {}}
    for variant in ("sfm", "da3"):
        ds = _split(allset, out / variant / "dataset", lambda n: n not in held)
        t0 = time.perf_counter()
        info: dict = {}
        if variant == "da3":
            info["dense_init"] = dense_init(ds, config=DenseInitConfig()).as_dict()
        trep = train_splat(ds, out / variant / "train", config=cfg)
        trec = pycolmap.Reconstruction(str(_sparse(ds)))
        own = {i.name for i in trec.images.values() if i.has_pose}
        _, cams, _ = pl._rotations_and_centres(trec, own)
        raw = read_ply(trep.ply_path)
        clean, _ = clean_table(raw, cams, views=pl._views_of(trec, own))
        info.update({"seconds": round(time.perf_counter() - t0, 1), "splats": raw.count,
                     "raw": _score(raw, held_dir, out / variant / "held_raw"), "delivered": _score(clean, held_dir, out / variant / "held")})
        report["train"][variant] = info
        print(variant, json.dumps(info), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
