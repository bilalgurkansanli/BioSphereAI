"""What each inpainter costs on this machine, and what it produces.

The numbers in ``biosphere/ai/sd_inpaint.py`` are from this script, and they are card-specific
enough that copying them onto another machine would be a lie: the measurement that stalled the
feature in the first place was 185 seconds per window on a 4 GB GTX 1650, against a couple of
seconds on a card one generation newer. Re-run this when the hardware changes.

It does two things a bare timer does not:

* **Sweeps the hole size.** The whole disagreement between the two models lives here. Over a
  small hole LaMa is quicker and at least as good; over the half-a-window hole a furnished
  living room actually presents, it has almost no photograph left to continue and collapses to
  a smooth field. ``flatness`` below is an attempt at that collapse expressed as a number — the
  mean gradient magnitude inside the filled region, against the same measure on the surrounding
  room.

  READ IT AS A TREND, NOT AS A SCORE. The reference window is drawn with two-pixel-hard
  floorboard lines and grain on top, so its gradient energy is higher than any real room's and
  every fill lands well under 1.0 against it — SD 1.5 measured 0.24 to 0.36 here while producing
  an unbroken skirting board and correctly converging boards. What the number is good for is one
  model against itself as the hole grows, and one model against another on the same window.

* **Writes the windows out.** The metric is a proxy and it knows it. ``--out`` saves every
  filled window so the actual question — does this look like a room — stays answerable by
  looking.

    python scripts/bench_inpaint.py --out C:\\temp\\inpaint-bench
    python scripts/bench_inpaint.py --models sd --steps 20 30 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.ai import lama, sd_inpaint  # noqa: E402

#: Share of the window the mask covers, from "a chair" to "a furnished living room".
_LOADS = (0.05, 0.15, 0.30, 0.45)


def room_window(size: int = 1024) -> np.ndarray:
    """A synthetic window with the two things a fill has to continue: a horizon and a line.

    Deliberately not a photograph. A real capture makes the timings honest and the quality
    metric meaningless, because every window is a different room; a constructed one has a known
    answer — the wall meets the floor at one straight line and the skirting runs along it — so
    "did the structure survive" is something the eye can check in a second.
    """
    view = np.zeros((size, size, 3), np.uint8)
    horizon = int(size * 0.62)
    view[:horizon] = (168, 186, 205)          # wall
    view[horizon:] = (58, 88, 128)            # floor
    view[horizon - int(size * 0.02) : horizon] = (232, 236, 240)   # skirting

    # Floorboards, converging: the structure a smooth field cannot fake.
    for offset in range(-size, size * 2, size // 12):
        cv2.line(view, (offset, size), (size // 2 + offset // 6, horizon), (48, 74, 110), 2)

    # Grain, so "the fill is smoother than the room" is measurable rather than trivially true.
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 4.0, view.shape).astype(np.float32)
    return np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def mask_covering(size: int, load: float) -> np.ndarray:
    """A block sitting on the floor and rising up the wall, like the furniture it stands for."""
    mask = np.zeros((size, size), np.uint8)
    side = int(round(np.sqrt(load) * size))
    x0 = (size - side) // 2
    y0 = min(size - side, int(size * 0.62) - side // 2)
    mask[max(0, y0) : y0 + side, x0 : x0 + side] = 255
    return mask


def flatness(original: np.ndarray, filled: np.ndarray, mask: np.ndarray) -> float:
    """Detail inside the fill as a fraction of the detail in the room around it.

    Near 1.0 means the fill carries the same amount of structure and grain as the photograph it
    sits in. Well under it is the failure this whole exercise is about: a wash of wall colour
    where the furniture used to be.
    """
    grey_filled = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey_room = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges_filled = np.abs(cv2.Laplacian(grey_filled, cv2.CV_32F))
    edges_room = np.abs(cv2.Laplacian(grey_room, cv2.CV_32F))

    inside = mask > 127
    # The reference is the room OUTSIDE the hole, in the original: comparing against the same
    # image's masked block would be comparing a fill to the thing it replaced.
    outside = ~cv2.dilate(mask, np.ones((31, 31), np.uint8)).astype(bool)
    if not inside.any() or not outside.any():
        return float("nan")
    return float(edges_filled[inside].mean() / max(edges_room[outside].mean(), 1e-6))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["sd", "lama"], choices=["sd", "lama"])
    parser.add_argument("--size", type=int, default=1024, help="window size, as room_fill uses")
    parser.add_argument("--loads", nargs="+", type=float, default=list(_LOADS))
    parser.add_argument("--steps", nargs="+", type=int, default=None, help="sd only; sweeps _STEPS")
    parser.add_argument("--out", type=Path, default=None, help="write the filled windows here")
    args = parser.parse_args()

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    view = room_window(args.size)
    if args.out:
        cv2.imwrite(str(args.out / "00-source.png"), view)

    modules = {"sd": sd_inpaint, "lama": lama}
    print(f"window {args.size}px\n")
    print(f"{'model':>6} {'steps':>6} {'hole':>6} {'seconds':>8} {'flatness':>9}")
    print("-" * 40)

    for key in args.models:
        module = modules[key]
        if not module.available():
            print(f"{key:>6}  not available on this machine")
            continue

        step_sweep = args.steps if (key == "sd" and args.steps) else [None]
        for steps in step_sweep:
            original_steps = None
            if steps is not None:
                original_steps = sd_inpaint._STEPS
                sd_inpaint._STEPS = steps
            try:
                for load in args.loads:
                    mask = mask_covering(args.size, load)
                    started = time.perf_counter()
                    result = module.inpaint(view, mask, cyclic=False)
                    elapsed = time.perf_counter() - started
                    if result is None:
                        print(f"{key:>6} {str(steps or '-'):>6} {load:>6.2f}  declined")
                        continue
                    filled = result[0]
                    score = flatness(view, filled, mask)
                    print(
                        f"{key:>6} {str(steps or '-'):>6} {load:>6.2f} "
                        f"{elapsed:>8.2f} {score:>9.2f}"
                    )
                    if args.out:
                        tag = f"{key}-{steps or 'default'}-{int(load * 100):02d}"
                        cv2.imwrite(str(args.out / f"{tag}.png"), filled)
            finally:
                if original_steps is not None:
                    sd_inpaint._STEPS = original_steps

    if args.out:
        print(f"\nwindows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
