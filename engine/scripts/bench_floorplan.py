"""What the plan view actually measures, against a room whose size is known exactly.

The bird's-eye view makes a numeric claim — this room is 4.1 m by 3.4 m — so the question
"how wrong is it" has to have an answer before anybody puts it in front of a buyer. The
synthetic bench in ``biosphere.synth`` already builds procedural rooms of known dimensions
and can ray-trace them, so the answer is available exactly rather than by eye.

Two runs, and the gap between them is the point:

``truth``   the estimator is handed the surface map the ray tracer knows to be correct —
            every pixel labelled ceiling, wall, floor or furniture with no error at all.
            What comes out is the error of the GEOMETRY: the binning, the filtering, the
            wall fitting and the corner crossing.
``model``   the estimator is handed what the real segmenter makes of a render of the same
            room. The difference between the two rows is the error of the PERCEPTION, and
            it is the half that a better checkpoint would move.

``--pano`` skips all of that and runs on a real panorama, where there is no ground truth and
the output is therefore a picture rather than a number: the detected floor and ceiling
boundaries drawn on the equirect, beside the plan they produce. A boundary in the wrong place
and a room of the wrong shape look identical in the plan alone and are fixed in different
places.

    python scripts/bench_floorplan.py
    python scripts/bench_floorplan.py --seeds 1 2 3 --out C:\\temp\\plan
    python scripts/bench_floorplan.py --pano pano_hd.jpg --out C:\\temp\\plan
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.plan import outline as O  # noqa: E402
from biosphere.plan.types import RoomOutline  # noqa: E402
from biosphere.synth import room as synth_room  # noqa: E402

_CANVAS = (1024, 512)


# ======================================================================================
# Ground truth
# ======================================================================================


def ground_truth_surface(room: synth_room.Room, width: int, height: int) -> np.ndarray:
    """The surface map a perfect segmenter would produce for ``room``.

    Furniture becomes 255 rather than a surface, which is exactly what
    ``furniture_seg.segment_equirect`` does with it: it paints only the classes that ARE the
    room's shell and leaves everything else unpainted. Handing the estimator a nicer input
    than the real one would make the ``truth`` row meaningless.
    """
    from biosphere.pano import geometry as geo

    theta, phi = geo.equirect_angles(width, height)
    rays = geo.rays_from_equirect(theta, phi).reshape(-1, 3)
    _, _, face = room.trace(room.eye, rays)
    face = face.reshape(height, width)

    surface = np.full((height, width), 255, np.uint8)
    surface[np.isin(face, synth_room.WALL_FACES)] = O.SURFACE_WALL
    surface[face == synth_room.FACE_CEILING] = O.SURFACE_CEILING
    surface[face == synth_room.FACE_FLOOR] = O.SURFACE_FLOOR
    return surface


def true_radius(room: synth_room.Room, angles: np.ndarray) -> np.ndarray:
    """Distance from the eye to the room's own walls, in metres, per azimuth."""
    eye = np.asarray(room.eye, np.float64)
    cx, cz = float(eye[0]), float(eye[2])
    sx, sz = np.sin(angles), np.cos(angles)
    best = np.full(angles.shape, np.inf)
    for numerator, denominator in (
        (0.0 - cx, sx),
        (room.width_m - cx, sx),
        (0.0 - cz, sz),
        (room.depth_m - cz, sz),
    ):
        safe = np.where(np.abs(denominator) > 1e-9, denominator, np.nan)
        t = numerator / safe
        t = np.where(np.isfinite(t) & (t > 1e-6), t, np.inf)
        best = np.minimum(best, t)
    return best


def true_polygon(room: synth_room.Room) -> np.ndarray:
    eye = np.asarray(room.eye, np.float64)
    x0, z0 = -float(eye[0]), -float(eye[2])
    x1, z1 = room.width_m - float(eye[0]), room.depth_m - float(eye[2])
    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], np.float64)


def iou(a: np.ndarray, b: np.ndarray, *, resolution: float = 0.02) -> float:
    """Intersection over union of two plan polygons, by rasterising both."""
    points = np.vstack([a, b])
    lo = points.min(axis=0) - 0.2
    hi = points.max(axis=0) + 0.2
    size = np.maximum(np.ceil((hi - lo) / resolution).astype(int), 1)

    def raster(polygon: np.ndarray) -> np.ndarray:
        canvas = np.zeros((size[1], size[0]), np.uint8)
        pixels = np.round((polygon - lo) / resolution).astype(np.int32)
        cv2.fillPoly(canvas, [pixels], 1)
        return canvas

    ra, rb = raster(a), raster(b)
    union = int((ra | rb).sum())
    return float((ra & rb).sum()) / union if union else 0.0


# ======================================================================================
# One measurement
# ======================================================================================


def evaluate(room: synth_room.Room, surface: np.ndarray, *, use_ceiling: bool) -> dict[str, float]:
    height = float(room.eye_height_m)
    profile = O.profile_from_surface(surface, use_ceiling=use_ceiling)
    angles = np.array(profile.angles())
    estimated = np.array(profile.radius) * height
    truth = true_radius(room, angles)

    error = estimated - truth
    polygon = np.array(O.polygon_from_profile(profile, height, rectify=True), np.float64)
    outline = RoomOutline(
        polygon=tuple(map(tuple, polygon)), profile=profile, camera_height_m=height, source="model"
    )
    return {
        "bias_cm": float(np.mean(error) * 100.0),
        "mean_cm": float(np.mean(np.abs(error)) * 100.0),
        "p95_cm": float(np.percentile(np.abs(error), 95) * 100.0),
        "iou": iou(polygon, true_polygon(room)),
        "area_err": (outline.area_m2 - room.width_m * room.depth_m) / (room.width_m * room.depth_m),
        "ceiling": -1.0 if profile.ceiling_height is None else profile.ceiling_height * height,
        "from_ceiling": profile.ceiling_fraction,
        "vertices": float(len(polygon)),
    }


def draw_plan(
    size: int,
    layers: list[tuple[np.ndarray, tuple[int, int, int], int, bool]],
    *,
    span: float | None = None,
    marks: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    canvas = np.full((size, size, 3), 20, np.uint8)
    everything = np.vstack([layer[0] for layer in layers])
    extent = float(np.abs(everything).max()) * 1.1 if span is None else span

    def to_pixels(points: np.ndarray) -> np.ndarray:
        xs = size / 2 + points[:, 0] / extent * size / 2
        ys = size / 2 - points[:, 1] / extent * size / 2
        return np.stack([xs, ys], 1).astype(np.int32)

    for points, colour, thickness, closed in layers:
        cv2.polylines(canvas, [to_pixels(points)], closed, colour, thickness, cv2.LINE_AA)
    for point, colour in marks or []:
        cv2.circle(canvas, tuple(to_pixels(point.reshape(1, 2))[0]), 6, colour, 2, cv2.LINE_AA)
    cv2.circle(canvas, (size // 2, size // 2), 4, (90, 90, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas, f"{2 * extent:.1f} m across", (10, size - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA,
    )
    return canvas


def boundary_picture(pano: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """The panorama with the surface tint and both detected boundaries drawn on it."""
    height, width = surface.shape
    base = cv2.resize(pano, (width, height))
    tint = np.zeros_like(base)
    tint[surface == O.SURFACE_CEILING] = (200, 120, 40)
    tint[surface == O.SURFACE_WALL] = (60, 60, 60)
    tint[surface == O.SURFACE_FLOOR] = (40, 180, 90)
    out = cv2.addWeighted(base, 0.62, tint, 0.38, 0)

    floor_elev, ceiling_elev = O._column_boundaries(surface)
    for u in range(width):
        for elevations, colour in ((floor_elev, (60, 255, 255)), (ceiling_elev, (255, 120, 255))):
            if np.isfinite(elevations[u]):
                v = int((0.5 * math.pi - elevations[u]) / math.pi * height)
                cv2.circle(out, (u, min(max(v, 0), height - 1)), 1, colour, -1)
    return out


# ======================================================================================
# Runs
# ======================================================================================


def run_synthetic(seeds: list[int], style: str, out: Path | None, with_model: bool) -> int:
    from biosphere.synth.render import render_equirect

    print(f"{'room':<16} {'source':<7} {'bias':>7} {'mean':>7} {'p95':>7} {'IoU':>6} {'area':>7} {'ceil':>6} {'from ceil':>10} {'verts':>6}")
    print("-" * 92)

    for seed in seeds:
        room = synth_room.Room.random(seed=seed, style=style)
        label = f"{seed} {room.width_m:.1f}x{room.depth_m:.1f}"
        surface = ground_truth_surface(room, *_CANVAS)

        rows: list[tuple[str, dict[str, float]]] = [
            ("floor", evaluate(room, surface, use_ceiling=False)),
            ("truth", evaluate(room, surface, use_ceiling=True)),
        ]

        pano = None
        if with_model:
            from biosphere.ai import furniture_seg

            pano = render_equirect(room, *_CANVAS)
            started = time.time()
            answered = furniture_seg.segment_equirect(pano)
            if answered is None:
                print(f"{label:<16} model   (no segmenter on this machine)")
            else:
                rows.append(("model", evaluate(room, answered[1], use_ceiling=True)))
                rows[-1][1]["seconds"] = time.time() - started

        for name, m in rows:
            print(
                f"{label:<16} {name:<7} {m['bias_cm']:>6.1f}c {m['mean_cm']:>6.1f}c {m['p95_cm']:>6.1f}c "
                f"{m['iou']:>6.3f} {m['area_err']*100:>6.1f}% {m['ceiling']:>6.2f} "
                f"{m['from_ceiling']*100:>9.0f}% {int(m['vertices']):>6d}"
            )
            label = ""

        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            profile = O.profile_from_surface(surface)
            polygon = np.array(O.polygon_from_profile(profile, room.eye_height_m), np.float64)
            picture = draw_plan(
                560,
                [
                    (true_polygon(room), (110, 110, 110), 1, True),
                    (np.array(profile.points(room.eye_height_m)), (150, 150, 90), 1, True),
                    (polygon, (120, 255, 160), 2, True),
                ],
            )
            cv2.imwrite(str(out / f"synthetic_{seed}_plan.png"), picture)
            if pano is not None:
                cv2.imwrite(str(out / f"synthetic_{seed}_boundary.jpg"), boundary_picture(pano, surface))
    return 0


def run_pano(path: Path, out: Path | None, camera_height: float) -> int:
    from biosphere.ai import furniture_seg

    pano = cv2.imread(str(path))
    if pano is None:
        print(f"cannot read {path}", file=sys.stderr)
        return 2
    started = time.time()
    answered = furniture_seg.segment_equirect(pano)
    if answered is None:
        print("no segmenter on this machine; nothing to measure", file=sys.stderr)
        return 3
    _, surface = answered
    elapsed = time.time() - started

    for name, use_ceiling in (("floor only", False), ("floor+ceiling", True)):
        profile = O.profile_from_surface(surface, use_ceiling=use_ceiling)
        polygon = O.polygon_from_profile(profile, camera_height)
        outline = RoomOutline(
            polygon=polygon, profile=profile, camera_height_m=camera_height, source="model"
        )
        radii = np.array(profile.radius) * camera_height
        print(
            f"{name:<14} area {outline.area_m2:6.1f} m2  radius {radii.min():.2f}-{radii.max():.2f} m "
            f"(median {np.median(radii):.2f})  from ceiling {profile.ceiling_fraction*100:3.0f}%  "
            f"measured {profile.measured_fraction*100:3.0f}%  {len(polygon)} vertices"
        )

    profile = O.profile_from_surface(surface)
    print(f"segmentation {elapsed:.1f} s, ceiling height "
          f"{'unknown' if profile.ceiling_height is None else f'{profile.ceiling_height * camera_height:.2f} m'}")

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{path.stem}_boundary.jpg"), boundary_picture(pano, surface))
        polygon = np.array(O.polygon_from_profile(profile, camera_height), np.float64)
        picture = draw_plan(
            700,
            [
                (np.array(profile.points(camera_height)), (150, 150, 90), 1, True),
                (polygon, (120, 255, 160), 2, True),
            ],
        )
        cv2.imwrite(str(out / f"{path.stem}_plan.png"), picture)
        print(f"wrote {out / (path.stem + '_boundary.jpg')} and {out / (path.stem + '_plan.png')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pano", type=Path, default=None, help="measure a real panorama instead")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 11])
    parser.add_argument("--style", default="rich", choices=sorted(synth_room.STYLES))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--camera-height", type=float, default=1.6)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="skip the segmenter and measure the geometry alone",
    )
    args = parser.parse_args()

    if args.pano is not None:
        return run_pano(args.pano, args.out, args.camera_height)
    return run_synthetic(args.seeds, args.style, args.out, not args.no_model)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
