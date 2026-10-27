"""Register an already-stitched panorama as a property scene, without re-running the pipeline.

Useful when a capture has been stitched offline (`scripts/stitch_real.py`) and you want to
walk through the result in the real player rather than looking at a flat JPEG. Everything
it writes goes through the same storage layout and the same `Scene` shape the job worker
produces, so the property is indistinguishable from one created by an upload.

Usage:
    python scripts/import_panorama.py <levels_dir> --title "Yatak Odası" --scene "Oda 1"

`<levels_dir>` is the directory `stitch_real.py` wrote (`uhd.jpg`, `sd.jpg`,
`preview.jpg`, `coverage.png`, ...). A sibling `report.json` is picked up automatically
for the metrics panel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosphere.core.config import get_settings  # noqa: E402
from biosphere.core.db import init_db, session_scope  # noqa: E402
from biosphere.core.models import Scene, Property  # noqa: E402
from biosphere.core.storage import get_storage  # noqa: E402

LEVEL_FILES = {
    "preview": ("preview.jpg", "preview_url"),
    "sd": ("sd.jpg", "sd_url"),
    "hd": ("hd.jpg", "hd_url"),
    "uhd": ("uhd.jpg", "uhd_url"),
}


#: report.json key -> StitchMetrics field name (see web/src/lib/types.ts)
_METRIC_ALIASES = {
    "frames_extracted": "n_frames_extracted",
    "keyframes": "n_keyframes",
    "pairs": "n_pairs",
    "loop_closures": "n_loop_closures",
    "mean_reprojection_px": "mean_reprojection_error_px",
    "loop_residual_deg": "loop_closure_residual_deg",
}

_METRIC_DEFAULTS: dict[str, object] = {
    "n_frames_extracted": 0,
    "n_keyframes": 0,
    "n_pairs": 0,
    "n_loop_closures": 0,
    "mean_reprojection_error_px": 0.0,
    "max_reprojection_error_px": 0.0,
    "loop_closure_residual_deg": 0.0,
    "focal_px": 0.0,
    "hfov_deg": 0.0,
    "seam_gradient_energy": 0.0,
    "exposure_spread": 0.0,
    "sweep_pattern": "unknown",
    "stage_seconds": {},
    "warnings": [],
}


def _metrics_from_report(report: dict) -> dict:
    out = dict(_METRIC_DEFAULTS)
    out["stage_seconds"] = {}
    out["warnings"] = []
    for key, value in report.items():
        field = _METRIC_ALIASES.get(key, key)
        if field in out:
            out[field] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("levels_dir")
    ap.add_argument("--title", default="İçe Aktarılan Tur")
    ap.add_argument("--scene", default="Oda 1")
    ap.add_argument("--address", default=None)
    ap.add_argument("--property-id", default=None, help="add to an existing property instead of creating one")
    args = ap.parse_args()

    src = Path(args.levels_dir).resolve()
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 2

    settings = get_settings()
    storage = get_storage()
    init_db()

    report: dict = {}
    for candidate in (src / "report.json", src.parent / "report.json"):
        if candidate.exists():
            report = json.loads(candidate.read_text(encoding="utf-8"))
            break

    with session_scope() as session:
        if args.property_id:
            property = session.get(Property, args.property_id)
            if property is None:
                print(f"no such property: {args.property_id}", file=sys.stderr)
                return 2
        else:
            property = Property(title=args.title, address=args.address)
            session.add(property)
            session.flush()

        existing = len(getattr(property, "scenes", []) or [])
        scene = Scene(
            property_id=property.id,
            name=args.scene,
            status="ready",
            capture_kind="video",
            quality_mode=report.get("mode", "balanced"),
            order=existing,
        )
        session.add(scene)
        session.flush()

        dest = storage.scene_dir(property.id, scene.id)
        dest.mkdir(parents=True, exist_ok=True)

        panorama: dict = {"cube_urls": None}
        width = height = 0
        for _level, (filename, url_key) in LEVEL_FILES.items():
            source = src / filename
            if not source.exists():
                panorama[url_key] = None
                continue
            target = dest / f"pano_{filename}"
            shutil.copy2(source, target)
            panorama[url_key] = storage.url_for(target)
            if url_key == "uhd_url" or width == 0:
                img = cv2.imread(str(target))
                if img is not None:
                    height, width = img.shape[:2]
        panorama["width"], panorama["height"] = width, height

        # A preview is what the player shows before anything else loads; without one the
        # first paint is a black sphere, so fall back to the smallest level we do have.
        if not panorama.get("preview_url"):
            for key in ("sd_url", "hd_url", "uhd_url"):
                if panorama.get(key):
                    panorama["preview_url"] = panorama[key]
                    break

        for extra, target_name in (("coverage.png", "coverage.png"), ("thumb.jpg", "thumb.jpg")):
            if (src / extra).exists():
                shutil.copy2(src / extra, dest / target_name)

        coverage = {
            "covered_fraction": report.get("covered_fraction", 0.0),
            "horizon_fraction": report.get("horizon_fraction", 0.0),
            "yaw_span_deg": report.get("yaw_span_deg", 0.0),
            "pitch_min_deg": (report.get("pitch_range_deg") or [0.0, 0.0])[0],
            "pitch_max_deg": (report.get("pitch_range_deg") or [0.0, 0.0])[1],
            "zenith_covered": bool(report.get("zenith_covered", False)),
            "nadir_covered": bool(report.get("nadir_covered", False)),
            "gaps": [
                {
                    "yaw_start_deg": g[0],
                    "yaw_end_deg": g[1],
                    "pitch_start_deg": g[2],
                    "pitch_end_deg": g[3],
                }
                for g in (report.get("gaps") or [])
            ],
            "map_url": storage.url_for(dest / "coverage.png") if (dest / "coverage.png").exists() else None,
        }

        scene.panorama = panorama
        scene.coverage = coverage
        # `report.json` uses the CLI's short names; the API schema uses the contract names
        # from web/src/lib/types.ts. Translate explicitly — the schema forbids extras, so a
        # stray key fails the whole response with a 500 rather than being ignored.
        scene.metrics = _metrics_from_report(report)
        scene.initial_view = {"yaw": 0.0, "pitch": 0.0, "fov": 75.0}
        if not property.cover_scene_id:
            property.cover_scene_id = scene.id
        session.add(scene)
        session.add(property)
        session.flush()

        print(json.dumps({
            "property_id": property.id,
            "scene_id": scene.id,
            "public_token": property.public_token,
            "panorama": f"{width}x{height}",
            "dashboard": f"{settings.web_base_url}/properties/{property.id}",
            "embed": f"{settings.web_base_url}/embed/{property.public_token}",
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
