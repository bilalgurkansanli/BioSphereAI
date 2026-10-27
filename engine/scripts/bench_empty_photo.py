"""What emptying an ordinary photograph costs, and — the part that matters — what it leaves.

``biosphere.ai.empty_photo`` is two steps with two very different failure modes, and lumping
them into one "before / after" picture hides which one went wrong. So this writes the MASK
beside the result: a bad mask and a bad fill look identical in the output and are fixed in
completely different places.

    python scripts/bench_empty_photo.py --image room.jpg --out C:\\temp\\empty
    python scripts/bench_empty_photo.py --image a.jpg b.jpg --out C:\\temp\\empty
    python scripts/bench_empty_photo.py --image room.jpg --free-curtains
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.ai import empty_photo, restyle  # noqa: E402

_LABEL_H = 30


def _label(image: np.ndarray, text: str) -> np.ndarray:
    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bar = np.full((_LABEL_H, rgb.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([rgb, bar])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--free-curtains",
        action="store_true",
        help="let curtains and blinds be removed (they are held by default)",
    )
    parser.add_argument("--width", type=int, default=520, help="tile width in the contact sheet")
    parser.add_argument(
        "--no-complete",
        action="store_true",
        help="the segmenter's mask alone, without empty_photo.complete_mask (the 'before')",
    )
    args = parser.parse_args()
    empty_photo.COMPLETE_MASK = not args.no_complete

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    print(f"{'image':<22} {'hole':>6} {'seconds':>8}  contents")
    print("-" * 72)

    rows: list[np.ndarray] = []
    for path in args.image:
        image = cv2.imread(str(path))
        if image is None:
            print(f"cannot read {path}", file=sys.stderr)
            continue

        started = time.perf_counter()
        result = empty_photo.empty_photo(image, window_dressing=not args.free_curtains)
        seconds = time.perf_counter() - started
        if result is None:
            print(f"{path.name:<22} NO BACKEND: needs a segmenter and an inpainter.")
            continue

        print(
            f"{path.name:<22} {result.removed_fraction * 100:5.1f}% {seconds:8.1f}  "
            f"{', '.join(result.objects) or '(nothing found)'}"
        )

        # The mask over the photograph, because a bad mask and a bad fill are different bugs.
        overlay = image.copy()
        selected = result.mask > 127
        if selected.any():
            overlay[selected] = (
                0.35 * overlay[selected] + 0.65 * np.array([40, 60, 230])
            ).astype(np.uint8)

        tiles = [("orijinal", image), ("maske", overlay), ("bosaltilmis", result.image)]
        scaled = []
        for name, tile in tiles:
            height = int(round(tile.shape[0] * args.width / tile.shape[1]))
            scaled.append(
                _label(cv2.resize(tile, (args.width, height), interpolation=cv2.INTER_AREA), name)
            )
        rows.append(np.hstack(scaled))

        if args.out is not None:
            stem = path.stem
            cv2.imwrite(str(args.out / f"{stem}_empty.jpg"), result.image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            cv2.imwrite(str(args.out / f"{stem}_mask.png"), result.mask)

    if args.out is not None and rows:
        widest = max(r.shape[1] for r in rows)
        padded = [
            np.hstack([r, np.full((r.shape[0], widest - r.shape[1], 3), 24, np.uint8)]) for r in rows
        ]
        cv2.imwrite(str(args.out / "sheet.jpg"), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\nwrote {args.out}\\sheet.jpg")

    # A reminder rather than a check: the mask comes from the same place the restyle path's does,
    # so a surprise here is a surprise there too.
    print(f"\nmask source: restyle.read_room (furnished threshold {restyle._FURNISHED_FRACTION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
