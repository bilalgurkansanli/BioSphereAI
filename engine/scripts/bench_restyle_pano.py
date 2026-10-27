"""Restyle a whole sphere and give back something that can actually be judged.

An equirectangular canvas is the wrong thing to look at — the bit that matters is whether the
seams between windows agree, and on an equirect a seam is a curve you cannot see. So this
writes THREE things:

* the restyled canvas, for the record;
* a strip of rectilinear windows around the horizon, at yaws deliberately OFFSET from the ones
  the pipeline used, so every picture in the strip is centred on a seam rather than on a
  window. If two neighbours disagreed about what the room contains, that is where it shows;
* the same yaws from the original, above them, so "is it still this room" stays answerable.

    python scripts/bench_restyle_pano.py --pano pano_sd.jpg --style rustik --out C:\\temp\\p
    python scripts/bench_restyle_pano.py --pano pano_hd.jpg --style modern --view-size 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.ai import restyle, restyle_pano, views  # noqa: E402

_LABEL_H = 30


def _label(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((_LABEL_H, image.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pano", required=True, type=Path)
    parser.add_argument("--style", default="modern")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--view-size", type=int, default=restyle_pano.VIEW_SIZE)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--free-curtains", action="store_true")
    args = parser.parse_args()

    pano = cv2.imread(str(args.pano))
    if pano is None:
        print(f"cannot read {args.pano}", file=sys.stderr)
        return 2
    print(f"pano       {args.pano}  {pano.shape[1]}x{pano.shape[0]}")

    if not restyle.available():
        print("NO BACKEND: needs a CUDA device, diffusers, both ControlNets and a depth model.")
        return 1

    extra: dict = {}
    if args.strength is not None:
        extra["strength"] = args.strength
    if args.steps is not None:
        extra["steps"] = args.steps

    started = time.perf_counter()
    result = restyle_pano.restyle_equirect(
        pano,
        args.style,
        window_dressing=not args.free_curtains,
        view_size=args.view_size,
        progress=lambda f, m: print(f"  {f * 100:5.1f}%  {m}", flush=True),
        **extra,
    )
    if result is None:
        print("failed", file=sys.stderr)
        return 1

    print()
    print(f"windows    {result.windows}")
    print(f"seconds    {(time.perf_counter() - started):.1f}   ({result.elapsed_ms / 1000:.1f} in engine)")
    print(f"restyled   {result.restyled_fraction * 100:.1f}% of the sphere (solid-angle weighted)")
    print(f"kept       {result.protected_fraction * 100:.1f}% of the sphere is still the photograph")

    if args.out is None:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "00_original_equirect.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(
        str(args.out / f"01_{result.style_id}_equirect.jpg"), result.panorama, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    # Centred on the SEAMS: the pipeline's yaws are multiples of 45, so these sit between them.
    seam_yaws = [22.5 + 45.0 * i for i in range(8)]
    size = 480
    K = views.intrinsics(size, views.VIEW_FOV_DEG)
    before_row, after_row = [], []
    for yaw in seam_yaws:
        rot = views.rotation(yaw, -8.0)
        before_row.append(_label(views.render_view(pano, rot, K, size), f"orijinal {yaw:.0f}"))
        after_row.append(
            _label(views.render_view(result.panorama, rot, K, size), f"{result.style_id} {yaw:.0f}")
        )
    for name, row in (("02_seams_before.jpg", before_row), (f"03_seams_{result.style_id}.jpg", after_row)):
        half = len(row) // 2
        sheet = np.vstack([np.hstack(row[:half]), np.hstack(row[half:])])
        cv2.imwrite(str(args.out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
