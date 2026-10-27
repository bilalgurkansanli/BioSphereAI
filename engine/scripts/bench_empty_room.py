"""Empty a REAL panorama with each fill, and give back numbers and pictures to judge them by.

``bench_inpaint.py`` measures one window of a synthetic room, and its own header says why that
flattered the diffusion model: the synthetic furniture is untextured blocks. This one runs the
whole of ``empty_room.plan_empty_room`` on real captures — the reference bedroom is three
captures of one very cluttered room — and reports, per fill:

``seconds``     the plan, segmentation excluded when ``--cache`` supplies it.
``changed``     share of pixels OUTSIDE the hole that moved by more than 3 levels. Must be ~0:
                the fill may only ever write inside its mask.
``dE``          mean CIE76 distance between the filled pixels (lightly smoothed, so grain and
                grout do not count) and the median colour of the SAME PLANE where it is visible
                in the same direction. A wall the colour of the wall scores a few units; the grey
                and black smears this bench was written against score thirty and more.
``dark``        share of the hole more than 15 L* darker than its plane's own colour there — the
                "dark smear" a learned fill drags in from the remains around a hole.
``edges``       gradient energy inside the hole over the same plane's elsewhere. Near 1 is a
                surface with the room's own texture; well under is plastic; well over is clutter.
``furniture``   (``--resegment``, GPU) share of the hole the segmenter STILL calls furniture
                when it is run again on the result: what is left behind plus what was invented.
``furniture_was`` the same, over what the segmenter called furniture in the ORIGINAL — one
                region for every fill, where ``furniture`` is over each fill's own hole.

and writes, per panorama: the emptied equirect, the mask, and a contact sheet of the six
directions with the most hole in them, before above and after below. The numbers rank the
fills; the sheet is what decides whether either is a room.

    python scripts/bench_empty_room.py --out C:\\temp\\empty-bench
    python scripts/bench_empty_room.py --cache C:\\biosphere-data\\work\\claude\\ai\\cache --fill both
    python scripts/bench_empty_room.py --pano name=C:\\path\\pano_uhd.jpg --resegment

``--cache`` reads ``<name>_furniture1024.npy``, ``<name>_surface1024.npy`` and
``<name>_classes2048.npy`` written by an earlier run with ``--write-cache``, so the fill can be
iterated on without the GPU: the segmentation is most of the wall clock and does not change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.ai import empty_room as er  # noqa: E402
from biosphere.ai import furniture_seg, room_planes, views  # noqa: E402

#: No reference captures ship with the repository: the measurements in docs/ARCHITECTURE.md
#: were taken on private rooms. Pass your own with ``--pano name=path/to/pano_uhd.jpg``.
DEFAULT_PANOS: dict[str, str] = {}

_WIDTH = 4096

#: The segmenter itself, kept before ``--cache`` swaps the module's function for a disk answer:
#: ``--resegment`` must look at the RESULT, and through the swapped function it would only have
#: been handed the original's cached segmentation back.
_SEGMENT_LABELS = furniture_seg.segment_equirect_labels


def _lab(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _plane_ids(box: room_planes.RoomBox, width: int, height: int) -> np.ndarray:
    camera = room_planes.EquirectCamera(width, height)
    ys, xs = np.mgrid[0:height, 0:width]
    return room_planes.trace(box, camera.rays(xs.astype(np.float64), ys.astype(np.float64))).plane


def metrics(original: np.ndarray, result: np.ndarray, hole: np.ndarray, room: er.RoomModel | None) -> dict[str, float]:
    """The numbers in the module docstring, for one result."""
    height, width = hole.shape
    inside = hole > 127
    outside = ~(cv2.dilate(hole, np.ones((7, 7), np.uint8)) > 127)
    moved = np.abs(original.astype(np.int16) - result.astype(np.int16)).max(axis=2) > 3
    out: dict[str, float] = {
        "hole": round(float(inside.mean()), 4),
        "changed": round(float(moved[outside].mean()), 5),
    }
    if room is None or not inside.any():
        return out

    planes = _plane_ids(room.box, width, height)
    sample = (~inside) & (room.surface != 255)
    lab_in = _lab(cv2.blur(result, (7, 7)))
    lab_ref = _lab(original)
    grey_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_r = cv2.magnitude(cv2.Sobel(grey_r, cv2.CV_32F, 1, 0), cv2.Sobel(grey_r, cv2.CV_32F, 0, 1))
    grad_o = cv2.magnitude(cv2.Sobel(grey_o, cv2.CV_32F, 1, 0), cv2.Sobel(grey_o, cv2.CV_32F, 0, 1))

    sector = (np.arange(width) * 18 // width)[None, :].repeat(height, axis=0)
    de_all, dark_all, edge_in, edge_ref = [], [], [], []
    for plane in np.unique(planes[inside]):
        on = planes == plane
        for s in range(18):
            here = on & (sector == s)
            hole_here = here & inside
            if not hole_here.any():
                continue
            # The plane's colour in this direction, from where it is visible; the neighbouring
            # sectors when this one has too little of it.
            ref_mask = here & sample
            if ref_mask.sum() < 200:
                ref_mask = on & sample & (np.abs(((sector - s + 9) % 18) - 9) <= 1)
            if ref_mask.sum() < 200:
                continue
            ref = np.median(lab_ref[ref_mask], axis=0)
            diff = lab_in[hole_here] - ref
            de_all.append(np.sqrt((diff ** 2).sum(axis=1)))
            dark_all.append(diff[:, 0] < -15.0)
        if (on & inside).sum() > 500 and (on & sample).sum() > 500:
            edge_in.append(float(grad_r[on & inside].mean()))
            edge_ref.append(float(grad_o[on & sample].mean()))
    if de_all:
        de = np.concatenate(de_all)
        dark = np.concatenate(dark_all)
        out["dE"] = round(float(de.mean()), 2)
        out["dE_p90"] = round(float(np.percentile(de, 90)), 2)
        out["dark"] = round(float(dark.mean()), 4)
    if edge_in:
        out["edges"] = round(float(np.sum(edge_in) / max(np.sum(edge_ref), 1e-6)), 3)
    return out


def contact_sheet(original: np.ndarray, results: list[tuple[str, np.ndarray]], hole: np.ndarray, size: int = 520) -> np.ndarray:
    """Six views with the most hole in them: the original on top, each fill below it."""
    K = views.intrinsics(size, 90.0)
    small_hole = cv2.resize(hole, (512, 256), interpolation=cv2.INTER_AREA)
    K_s = views.intrinsics(64, 90.0)
    scored = []
    for yaw in range(0, 360, 30):
        for pitch in (-25.0, 5.0):
            rot = views.rotation(float(yaw), pitch)
            load = float((views.render_view(small_hole, rot, K_s, 64) > 127).mean())
            scored.append((load, yaw, pitch))
    scored.sort(reverse=True)
    chosen: list[tuple[float, float]] = []
    for _, yaw, pitch in scored:
        if all(min(abs(yaw - y), 360 - abs(yaw - y)) >= 50 for y, _ in chosen):
            chosen.append((float(yaw), pitch))
        if len(chosen) == 6:
            break

    def label(tile: np.ndarray, text: str) -> np.ndarray:
        tile = tile.copy()
        cv2.putText(tile, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return tile

    columns = []
    for yaw, pitch in chosen:
        rot = views.rotation(yaw, pitch)
        cells = [label(views.render_view(original, rot, K, size), f"orijinal {yaw:.0f}/{pitch:.0f}")]
        for name, image in results:
            cells.append(label(views.render_view(image, rot, K, size), name))
        columns.append(np.vstack(cells))
    return np.hstack(columns)


def _install_cache(cache: Path, name: str) -> None:
    """Answer the segmentation from disk, exactly as the engine would have computed it."""
    furniture = np.load(cache / f"{name}_furniture1024.npy")
    surface = np.load(cache / f"{name}_surface1024.npy")
    classes = cv2.resize(
        np.load(cache / f"{name}_classes2048.npy"), (furniture.shape[1], furniture.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    furniture_seg.segment_equirect_labels = lambda _pano: (furniture, surface, classes)  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pano", nargs="*", default=None, help="name=path pairs (required)")
    parser.add_argument("--cache", type=Path, default=None, help="read the segmentation from here")
    parser.add_argument("--write-cache", type=Path, default=None, help="write the segmentation here")
    parser.add_argument("--fill", choices=["planes", "legacy", "both"], default="planes")
    parser.add_argument("--cpu", action="store_true", help="never touch the GPU for the fill (classical leftovers)")
    parser.add_argument("--resegment", action="store_true", help="re-run the segmenter on each result (GPU)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panos = dict(DEFAULT_PANOS)
    if args.pano:
        panos = dict(item.split("=", 1) for item in args.pano)
    if not panos:
        parser.error("give at least one --pano name=path/to/pano_uhd.jpg")

    if args.cpu:
        from biosphere.ai import backend

        backend.ready = lambda: False  # type: ignore[assignment]

    fills = ["planes", "legacy"] if args.fill == "both" else [args.fill]
    report: dict[str, dict[str, dict[str, float]]] = {}
    for name, path in panos.items():
        image = er._imread(Path(path))
        if image.shape[1] > _WIDTH:
            image = er.resize_cyclic(image, _WIDTH, _WIDTH // 2, cv2.INTER_AREA)
        if args.cache is not None:
            _install_cache(args.cache, name)
        elif args.write_cache is not None:
            answered = furniture_seg.segment_equirect_labels(image)
            if answered is not None:
                args.write_cache.mkdir(parents=True, exist_ok=True)
                np.save(args.write_cache / f"{name}_furniture1024.npy", answered[0])
                np.save(args.write_cache / f"{name}_surface1024.npy", answered[1])
                np.save(
                    args.write_cache / f"{name}_classes2048.npy",
                    cv2.resize(answered[2], (2048, 1024), interpolation=cv2.INTER_NEAREST),
                )
                furniture_seg.segment_equirect_labels = lambda _pano, a=answered: a  # type: ignore[assignment]

        results: list[tuple[str, np.ndarray]] = []
        report[name] = {}
        room = None
        for fill in fills:
            er.PLANE_FILL = fill == "planes"
            started = time.perf_counter()
            plan = er.plan_empty_room(image, er.DEFAULT_STRENGTH)
            seconds = time.perf_counter() - started
            # The SAME room for every fill's numbers: the planes' own when they ran, so a legacy
            # fill is scored against the surfaces the plane fill would have drawn.
            if plan.room is not None:
                room = plan.room
            elif room is None:
                semantic = er._semantic_mask(image, 512, 1024)
                if semantic is not None:
                    room = er._room_model(image, semantic[1], semantic[2], plan.mask)
            row = metrics(image, plan.image, plan.mask, room)
            if plan.room is not None:
                print(name, fill, "box", json.dumps(plan.room.fit, default=str), flush=True)
                print(name, fill, "planes", [(p.plane, p.method) for p in plan.planes], flush=True)
            row["seconds"] = round(seconds, 1)
            row["removed"] = round(plan.removed_fraction, 4)
            if args.resegment:
                before = furniture_seg.segment_equirect_labels(image)
                answered = _SEGMENT_LABELS(plan.image)
                if answered is not None and before is not None:
                    size = (image.shape[1], image.shape[0])
                    again = cv2.resize(answered[0], size, interpolation=cv2.INTER_NEAREST) > 127
                    was = cv2.resize(before[0], size, interpolation=cv2.INTER_NEAREST) > 127
                    row["furniture"] = round(float(again[plan.mask > 127].mean()), 4)
                    # The same region for every fill: what the segmenter called furniture BEFORE.
                    row["furniture_was"] = round(float(again[was].mean()), 4)
            report[name][fill] = row
            print(name, fill, json.dumps(row), flush=True)
            cv2.imwrite(str(args.out / f"{name}_{fill}.jpg"), plan.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(args.out / f"{name}_{fill}_mask.png"), plan.mask)
            results.append((fill, plan.image))
        sheet = contact_sheet(image, results, plan.mask)
        cv2.imwrite(str(args.out / f"{name}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
