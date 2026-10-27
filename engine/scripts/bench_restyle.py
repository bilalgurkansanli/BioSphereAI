"""What the five styles cost on this machine, and — the part that matters — what they look like.

``biosphere.ai.restyle`` has one quality question and no metric can answer it: *does this look
like a room somebody could live in, and is it still THIS room?* So this script's real output is
the pictures. It writes every style beside the original, writes the two conditioning images the
model was actually given, and lays the lot out as a contact sheet.

The numbers are there to size the feature rather than to score it, with one exception.
``drift`` is the mean absolute difference, over the PROTECTED pixels only, between the original
photograph and the styled one. It should be zero. It is a regression test for the compositing
rather than a judgement about the model — if protection is working, the window in the styled
image is the same window, pixel for pixel, and anything above about one grey level means a mask
or a resize is off somewhere.

    python scripts/bench_restyle.py --image C:\\path\\room.jpg --out C:\\temp\\restyle
    python scripts/bench_restyle.py --image room.jpg --styles modern rustik --steps 20 28
    python scripts/bench_restyle.py --image room.jpg --control 0.75,0.55 1.0,0.8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.ai import restyle  # noqa: E402

_LABEL_H = 34


def _pair(text: str) -> tuple[float, float]:
    depth, _, canny = text.partition(",")
    if not canny:
        raise argparse.ArgumentTypeError("control scales are a pair, e.g. 0.75,0.55")
    return float(depth), float(canny)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """A caption bar under a tile, so a contact sheet is readable without counting columns."""
    bar = np.full((_LABEL_H, image.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


def _sheet(tiles: list[tuple[str, np.ndarray]], columns: int = 3, width: int = 560) -> np.ndarray:
    scaled = []
    for name, image in tiles:
        rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height = int(round(rgb.shape[0] * width / rgb.shape[1]))
        scaled.append(_label(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA), name))
    cell_h = max(t.shape[0] for t in scaled)
    rows = []
    for start in range(0, len(scaled), columns):
        chunk = scaled[start : start + columns]
        padded = [
            np.vstack([t, np.full((cell_h - t.shape[0], t.shape[1], 3), 24, np.uint8)]) for t in chunk
        ]
        while len(padded) < columns:
            padded.append(np.full((cell_h, width, 3), 24, np.uint8))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def _drift(original: np.ndarray, styled: np.ndarray, alpha: np.ndarray) -> float:
    """Mean absolute difference over the fully protected pixels. Should be ~0."""
    solid = alpha > 0.99
    if not solid.any():
        return 0.0
    diff = np.abs(original.astype(np.float32) - styled.astype(np.float32)).mean(axis=2)
    return float(diff[solid].mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True, type=Path, help="room photograph to restyle")
    parser.add_argument("--out", type=Path, default=None, help="directory to write results to")
    parser.add_argument(
        "--styles", nargs="*", default=None, help=f"subset of {', '.join(restyle.STYLE_IDS)}"
    )
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--guidance", nargs="*", type=float, default=None)
    parser.add_argument(
        "--control", nargs="*", type=_pair, default=None, help="depth,canny pairs e.g. 0.75,0.55"
    )
    parser.add_argument("--strength", nargs="*", type=float, default=None)
    parser.add_argument("--pixels", type=int, default=restyle.WORK_PIXELS, help="latent budget in pixels")
    parser.add_argument(
        "--free-curtains",
        action="store_true",
        help="let curtains and blinds be restyled (they are held by default)",
    )
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2

    out_dir = args.out
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"image      {args.image}  {image.shape[1]}x{image.shape[0]}")
    if not restyle.available():
        print("NO BACKEND: needs a CUDA device, diffusers, both ControlNets and a depth model.")
        return 1
    print(f"checkpoint {restyle.describe()['checkpoint']}")

    t0 = time.perf_counter()
    room = restyle.analyse_room(image, window_dressing=not args.free_curtains)
    analyse_ms = (time.perf_counter() - t0) * 1000
    if room is None:
        print("analysis failed: no depth backend", file=sys.stderr)
        return 1
    print(
        f"analyse    {analyse_ms:7.0f} ms   mode {room.mode:<7s} "
        f"editable {(1 - room.protected_fraction) * 100:5.1f}%   "
        f"edges {float((room.edges > 0).mean()) * 100:.2f}%   segmented {room.segmented}"
    )
    if room.objects:
        print(f"in the room: {', '.join(room.objects)}")

    if out_dir is not None:
        cv2.imwrite(str(out_dir / "00_original.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / "01_depth.png"), room.depth)
        cv2.imwrite(str(out_dir / "02_edges.png"), room.edges)
        cv2.imwrite(str(out_dir / "03_editable.png"), room.editable)
        overlay = image.copy()
        held = room.editable < 128
        if held.any():
            overlay[held] = (0.45 * overlay[held] + 0.55 * np.array([40, 220, 40])).astype(np.uint8)
        cv2.imwrite(str(out_dir / "04_protected.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    wanted = args.styles or list(restyle.STYLE_IDS)
    steps_list = args.steps or [None]
    guidance_list = args.guidance or [None]
    control_list = args.control or [None]
    strength_list = args.strength or [None]
    sweeping = len(steps_list) * len(guidance_list) * len(control_list) * len(strength_list) > 1

    tiles: list[tuple[str, np.ndarray]] = [("orijinal", image)]
    if not sweeping:
        tiles.append(("depth + mimari cizgiler", cv2.addWeighted(
            cv2.cvtColor(room.depth, cv2.COLOR_GRAY2BGR), 1.0,
            cv2.cvtColor(room.edges, cv2.COLOR_GRAY2BGR), 1.0, 0.0)))

    print()
    print(
        f"{'style':<12} {'steps':>5} {'cfg':>5} {'str':>5} {'depth':>6} {'canny':>6} "
        f"{'seconds':>8} {'drift':>7}"
    )
    print("-" * 68)

    for style_id in wanted:
        style = restyle.style_by_id(style_id)
        if style is None:
            print(f"unknown style {style_id!r}", file=sys.stderr)
            continue
        for steps in steps_list:
            for guidance in guidance_list:
              for strength in strength_list:
                for control in control_list:
                    kwargs: dict = {"work_pixels": args.pixels, "analysis": room}
                    if strength is not None:
                        kwargs["strength"] = strength
                    if steps is not None:
                        kwargs["steps"] = steps
                    if guidance is not None:
                        kwargs["guidance"] = guidance
                    if control is not None:
                        kwargs["control_scales"] = control

                    result = restyle.restyle_photo(image, style, **kwargs)
                    if result is None:
                        print(f"{style_id:<12} FAILED")
                        continue

                    drift = _drift(image, result.image, room.protect_alpha)
                    d, c = result.control_scales
                    print(
                        f"{style_id:<12} {result.steps:>5} {result.guidance:>5.1f} "
                        f"{result.strength:>5.2f} {d:>6.2f} {c:>6.2f} "
                        f"{result.elapsed_ms / 1000:>8.1f} {drift:>7.2f}"
                    )

                    tag = (
                        style_id
                        if not sweeping
                        else f"{style_id}_str{result.strength:g}_d{d:g}_c{c:g}_g{result.guidance:g}"
                    )
                    tiles.append((tag, result.image))
                    if out_dir is not None:
                        cv2.imwrite(
                            str(out_dir / f"{tag}.jpg"), result.image, [cv2.IMWRITE_JPEG_QUALITY, 94]
                        )

    if out_dir is not None and len(tiles) > 1:
        sheet = _sheet(tiles, columns=args.columns)
        cv2.imwrite(str(out_dir / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\nwrote {out_dir}\\sheet.jpg  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
