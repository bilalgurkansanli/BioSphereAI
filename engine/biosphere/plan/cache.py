"""Where a room's measured outline lives between requests.

Measuring one room is sixteen passes of a scene-parsing model — about four seconds on this
laptop's card and rather more without one — so it cannot happen while somebody is opening a
plan. It also cannot sensibly happen on every request, because the answer does not change:
the panorama it was read from is written once.

So the outline is a **derived artefact cached beside the panorama it came from**, exactly
like the emptied and restyled ladders next to it, and it is invalidated the same way those
are — by the source file's modification time rather than by its identity. A scene id is
stable across a reprocess; the file it points at is not.

Why a file and not a column on ``scene``
----------------------------------------
Every other rich sub-object on a scene (``panorama``, ``coverage``, ``metrics``) is a JSON
column, and this one deliberately is not. Those are written by the stitch worker as part of
the run that creates the scene; an outline is written by a *different* pass that may never
run at all — a deployment with no segmenter never produces one — and adding a nullable column
for it would mean a schema change to ship a feature that a room can perfectly well live
without. The bytes it describes are in the scene's own directory, so deleting a property already
takes it with them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from biosphere.core.logging import get_logger
from biosphere.core.models import Scene
from biosphere.core.storage import LocalStorage
from biosphere.plan.outline import outline_from_pano
from biosphere.plan.types import CAMERA_HEIGHT_M, PLAN_VERSION, RoomOutline

__all__ = ["LAYOUT_FILENAME", "layout_path", "read_outline", "build_outline", "forget_outline"]

log = get_logger("biosphere.plan.cache")

LAYOUT_FILENAME = "layout.json"


def layout_path(scene: Scene, storage: LocalStorage) -> Path:
    return storage.scene_dir(scene.property_id, scene.id) / LAYOUT_FILENAME


def _source(scene: Scene, storage: LocalStorage) -> Path | None:
    """The panorama the outline is read from, or ``None`` when the scene has none.

    :func:`biosphere.ai.empty_room.source_level` picks it, so the plan measures the same rung
    the other tools work from. Which rung hardly matters to this feature — the segmenter looks
    through 640-pixel windows either way — but "which file is this scene's panorama" having
    one answer does.
    """
    from biosphere.ai.empty_room import source_level  # noqa: PLC0415 - avoids a cycle at import

    if not scene.panorama:
        return None
    try:
        path, _ = source_level(scene, storage)
    except Exception:  # noqa: BLE001 - a missing rung is "no outline", not a failure
        return None
    return path


def read_outline(scene: Scene, storage: LocalStorage) -> RoomOutline | None:
    """The cached outline, or ``None`` when there is none, it is stale, or it is older.

    Never runs a model and never raises: this is the read path the player and every embed go
    through, and a plan is better drawn without a room than not drawn at all.
    """
    path = layout_path(scene, storage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if int(payload.get("version", 0)) != PLAN_VERSION:
        return None
    source = _source(scene, storage)
    if source is None or int(payload.get("source_mtime", -1)) != int(source.stat().st_mtime):
        return None
    try:
        return RoomOutline.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def forget_outline(scene: Scene, storage: LocalStorage) -> None:
    layout_path(scene, storage).unlink(missing_ok=True)


def build_outline(
    scene: Scene,
    storage: LocalStorage,
    *,
    camera_height_m: float = CAMERA_HEIGHT_M,
    refresh: bool = False,
) -> RoomOutline | None:
    """Measure the room if it has not been measured, and return what is on disk either way.

    ``None`` means this room has no outline and will not get one from this call: it has no
    panorama, or this deployment has no segmenter. Neither is an error — the plan draws a
    placeholder and says which it is.
    """
    if not refresh:
        cached = read_outline(scene, storage)
        if cached is not None:
            return cached

    source = _source(scene, storage)
    if source is None:
        return None
    pano = cv2.imread(str(source))
    if pano is None:
        log.info("plan.unreadable_panorama", scene_id=scene.id, path=str(source))
        return None

    outline = outline_from_pano(pano, camera_height_m=camera_height_m)
    if outline is None:
        return None

    payload = outline.as_dict()
    payload["source_mtime"] = int(source.stat().st_mtime)
    path = layout_path(scene, storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info("plan.outline_cached", scene_id=scene.id, area_m2=round(outline.area_m2, 2))
    return outline
