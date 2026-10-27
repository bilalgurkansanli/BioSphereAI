"""Render rectilinear views out of a panorama, exactly as the player does.

An equirectangular image flatters itself: the eye forgives its built-in distortion, so a
panorama can look plausible laid out flat and still be wrong. The moment it is projected
back to a perspective view — which is all a 360 viewer ever shows — any error in the
angular mapping turns into bent architecture.

This renders the same views the player would, plus a straightness check: it finds long
edges and measures how far they deviate from straight. In a correct panorama of a room,
door frames and wall junctions must come back straight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.pano import geometry as g  # noqa: E402
from biosphere.pano.types import Intrinsics  # noqa: E402


def render_view(pano: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float,
                size: tuple[int, int] = (720, 480)) -> np.ndarray:
    """One perspective frame, using the same conventions the stitcher warped with."""
    w, h = size
    intr = Intrinsics.from_hfov(w, h, fov_deg)
    rays_cam = g.camera_rays(w, h, intr.K)
    R = g.from_yaw_pitch_roll(np.radians(yaw_deg), np.radians(pitch_deg), 0.0)
    # camera -> world is R.T; for row vectors that is rays @ R
    rays_world = rays_cam.reshape(-1, 3) @ R
    u, v = g.equirect_from_rays(rays_world.reshape(h, w, 3), pano.shape[1], pano.shape[0])
    return g.sample_equirect(pano, u, v, interpolation=cv2.INTER_CUBIC)


def straightness(view: np.ndarray) -> tuple[float, int]:
    """Median deviation from straight, in pixels, over long detected edges.

    Uses the LSD-style trick of comparing a probabilistic Hough segment against the actual
    edge pixels beneath it. A correct rectilinear view of a room scores near zero.
    """
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                            minLineLength=min(view.shape[:2]) // 3, maxLineGap=6)
    if lines is None:
        return 0.0, 0
    devs = []
    ys, xs = np.nonzero(edges)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    for seg in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (float(t) for t in seg)
        d = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        n = np.linalg.norm(d)
        if n < 1e-3:
            continue
        d /= n
        nrm = np.array([-d[1], d[0]], dtype=np.float32)
        rel = pts - np.array([x1, y1], dtype=np.float32)
        along = rel @ d
        near = (along >= 0) & (along <= n)
        if near.sum() < 30:
            continue
        off = np.abs(rel[near] @ nrm)
        close = off[off < 6.0]
        if close.size >= 30:
            devs.append(float(np.percentile(close, 90)))
    return (float(np.median(devs)) if devs else 0.0), len(devs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("panorama")
    ap.add_argument("out")
    ap.add_argument("--fov", type=float, default=75.0)
    args = ap.parse_args()

    pano = cv2.imread(args.panorama)
    if pano is None:
        print(f"cannot read {args.panorama}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"panorama {pano.shape[1]}x{pano.shape[0]}")

    tiles = []
    print("\n yaw  pitch   straightness(px)  edges   coverage(non-fill proxy)")
    for pitch in (20.0, 0.0, -20.0):
        row = []
        for yaw in (-135.0, -45.0, 45.0, 135.0):
            v = render_view(pano, yaw, pitch, args.fov)
            dev, n = straightness(v)
            # high-frequency energy as a proxy for "real content vs smooth fill"
            hf = float(np.abs(cv2.Laplacian(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY),
                                            cv2.CV_32F)).mean())
            print(f" {yaw:+6.0f} {pitch:+5.0f}      {dev:6.2f}        {n:3d}     hf {hf:6.2f}")
            label = v.copy()
            cv2.rectangle(label, (0, 0), (label.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(label, f"yaw {yaw:+.0f} pitch {pitch:+.0f}  hf {hf:.1f}",
                        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(label)
            cv2.imwrite(str(out / f"view_y{int(yaw)}_p{int(pitch)}.jpg"), v,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(np.hstack(row))
    cv2.imwrite(str(out / "_grid.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nwrote {out / '_grid.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
