"""The property manifest the viewer consumes.

One JSON per property, modelled on the streaming config SplatTour serves: a *base* area (the
general walkthrough, loaded first and shown wherever no room is loaded) and *rooms*, every
one with its asset, its bounding box, a thinned camera path, where to stand on entry, a
thumbnail and an environment panorama, all in the one property frame. Plus the floor plan and
where each room sits on it.

Paths in the manifest written to disk are relative to the property's output directory; the API
layer swaps them for media URLs when it serves the document, so the file on disk never
knows a hostname.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import AreaResult, PropertySplatResult

__all__ = ["MANIFEST_VERSION", "area_dict", "manifest_dict", "write_manifest"]

MANIFEST_VERSION = 1


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def area_dict(area: AreaResult, root: Path) -> dict[str, Any]:
    return {
        "key": area.key,
        "name": area.name,
        "floor_index": area.floor_index,
        "url": _rel(area.sog_path, root),
        "bytes": area.sog_bytes,
        "splat_count": area.splat_count,
        "thumbnail_url": _rel(area.thumbnail_path, root),
        "env_pano_url": _rel(area.env_pano_path, root),
        # Each camera is [x, y, z] or, when known, [x, y, z, lx, ly, lz, t]: also where it
        # pointed and when (seconds into the area's video). The API's manifest schema is closed
        # (extra keys are refused), and a camera that says more is still a camera to every
        # reader that takes the first three numbers.
        "cameras": [list(c) + list(lk) for c, lk in zip(area.cameras, area.looks, strict=True)]
        if area.looks and len(area.looks) == len(area.cameras)
        else area.cameras,
        "centroid": area.centroid,
        "entry": {"position": area.entry_position, "forward": area.entry_forward},
        "bbox": {"min": area.bbox_min, "max": area.bbox_max},
        "placed": area.placed and area.failed is None and bool(area.sog_path),
        "redundant": area.redundant,
        "warnings": list(area.warnings),
        "failed": area.failed,
    }


def manifest_dict(result: PropertySplatResult) -> dict[str, Any]:
    root = result.out_dir
    return {
        "version": MANIFEST_VERSION,
        "frame": {
            "convention": "colmap",
            "up": [0.0, -1.0, 0.0],
            "floor_y": 0.0,
            "units": "m",
            "note": "Right-handed, +Y down. Rotate 180 degrees about X for a Y-up renderer.",
        },
        "base": area_dict(result.base, root),
        "rooms": [area_dict(r, root) for r in result.rooms],
        "floor_plan": {
            "url": _rel(result.floor_plan_path, root),
            **result.floor_plan,
        },
        "metrics": result.metrics,
        "warnings": list(result.warnings),
    }


def write_manifest(result: PropertySplatResult) -> Path:
    doc = manifest_dict(result)
    path = result.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
