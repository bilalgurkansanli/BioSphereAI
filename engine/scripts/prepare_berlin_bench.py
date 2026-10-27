"""Turn the Zip-NeRF *berlin* fisheye photos into phone-like footage with ground-truth poses.

The berlin capture is a 160° fisheye (OPENCV_FISHEYE, k1 = 0.037) and a phone's ultra-wide
video is not: the ISP hands the browser a rectilinear frame of roughly 75° x 107° in
portrait. A pinhole-plus-radial camera model cannot fit a 160° fisheye, so feeding the raw
photos to the walkthrough pipeline measured the camera model, not the pipeline. This script
re-projects every photo through its ground-truth intrinsics into exactly the frame a phone
would record - portrait, 1080 x 1920, f = 800 px - and writes a COLMAP model with the same
poses and the new PINHOLE camera next to it, so ``bench_walkthrough.py`` can score against
the truth.

Usage: uv run python scripts/prepare_berlin_bench.py [--src C:/biosphere-data/datasets/berlin_extract/berlin]
                                                     [--out C:/biosphere-data/datasets/berlin_rect]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 1080, 1920
#: 800 px on a 1080-wide portrait frame: 68° horizontal, 100° vertical - close to an iPhone 0.5x in
#: portrait, and the widest crop the 2336-px-tall fisheye covers without black corners.
OUT_F = 800.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="C:/biosphere-data/datasets/berlin_extract/berlin")
    ap.add_argument("--out", default="C:/biosphere-data/datasets/berlin_rect")
    ap.add_argument("--images", default="images_2", help="which downsample of the photos to read (images_2 = 3504 x 2336)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import pycolmap

    src = Path(args.src)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    gt = pycolmap.Reconstruction(str(src / "sparse" / "0"))
    cam = next(iter(gt.cameras.values()))
    if cam.model_name != "OPENCV_FISHEYE":
        print(f"expected an OPENCV_FISHEYE ground truth, got {cam.model_name}", file=sys.stderr)
        return 2
    sample = cv2.imread(str(next((src / args.images).glob("*.JPG"))))
    scale = sample.shape[1] / cam.width
    fx, fy, cx, cy, k1, k2, k3, k4 = cam.params
    K = np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    K_new = np.array([[OUT_F, 0, OUT_W / 2], [0, OUT_F, OUT_H / 2], [0, 0, 1]], dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (OUT_W, OUT_H), cv2.CV_16SC2)
    # The centre of the fisheye is downsampled ~1.7x on the way out; a light blur first
    # keeps the remap from aliasing (remap has no area filter).
    sigma = 0.5 * (fx * scale) / OUT_F

    names = sorted(p.name for p in (src / args.images).glob("*.JPG"))

    def convert(name: str) -> str:
        dst = out / "images" / (Path(name).stem + ".jpg")
        if dst.exists():
            return name
        im = cv2.imread(str(src / args.images / name), cv2.IMREAD_COLOR)
        im = cv2.GaussianBlur(im, (0, 0), sigma)
        rect = cv2.remap(im, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(dst), rect, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return name

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(convert, names)):
            if i % 100 == 0:
                print(f"{i}/{len(names)}", flush=True)

    # Ground truth for the rectified frames: same poses, a PINHOLE camera.
    rec = pycolmap.Reconstruction(str(src / "sparse" / "0"))
    for c in rec.cameras.values():
        c.model = pycolmap.CameraModelId.PINHOLE
        c.width, c.height = OUT_W, OUT_H
        c.params = np.array([OUT_F, OUT_F, OUT_W / 2, OUT_H / 2], dtype=np.float64)
    for img in rec.images.values():
        img.name = Path(img.name).stem + ".jpg"
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    rec.write(str(out / "sparse" / "0"))
    print(f"wrote {len(names)} rectified frames and the pinhole ground truth to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
