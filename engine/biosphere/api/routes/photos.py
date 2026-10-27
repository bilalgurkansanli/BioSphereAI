"""A room's photographs: editing them by id, and the two AI tools that work on them.

The first half mirrors ``hotspots.py`` — creating a photograph is a fact about a SCENE and
lives under it, while changing or removing one is a fact about the photograph and does not need
the room's id to say so.

The second half is the point of the module. The player already offers two tools on a 360:
decorate it, or take the furniture out. A room's ordinary photographs deserve the same two, and
there is no new algorithm behind them — ``ai.restyle`` already works on any photograph, and
``ai.empty_photo`` is the perspective-canvas sibling of ``ai.empty_room``. What this file owns
is where the results live and how they are cached.

CACHED AGAINST THE ORIGINAL'S MTIME, not merely against its id. A photograph is immutable in
practice — the id is a row, the file is written once — but "in practice" is how stale
derivatives come to be served, and the check costs one ``stat``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, Response
from fastapi.concurrency import run_in_threadpool

from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, SessionDep, StorageDep, gpu_lock, require_scene_photo
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException, Property, Scene, ScenePhoto, User, utcnow
from biosphere.core.storage import LocalStorage, write_jpeg

router = APIRouter(prefix="/scene-photos", tags=["photos"])

log = get_logger("biosphere.api.photos")

#: Derived versions live beside the originals, one directory per photograph.
_DERIVED = "derived"
_META = "meta.json"

#: The variant name for "the furniture is gone". Never a style id, so the two share one
#: namespace on disk without colliding.
_EMPTY = "empty"

#: What produced a variant, versioned. An entry written by an older algorithm is stale: it is
#: still listed (the sidecar contract), but a request for it renders it again instead of
#: serving yesterday's picture as today's. 2 (empty): the hole grows over what the segmenter called room but is
#: not — see ``ai.empty_photo.complete_mask``. 2 (style): curtains held by default and the loft
#: palette's glowing orange removed — see ``ai.restyle``.
_ALGORITHM = {"empty": 2, "style": 2}


def _kind(variant: str) -> str:
    return "empty" if variant == _EMPTY else "style"


def _current(variant: str, entry: dict[str, Any]) -> bool:
    return int(entry.get("algorithm", 1)) == _ALGORITHM[_kind(variant)]


def _photos_dir(storage: LocalStorage, session: Any, photo: ScenePhoto) -> Path:
    scene = session.get(Scene, photo.scene_id)
    if scene is None:  # pragma: no cover - the foreign key forbids it
        raise ApiException(
            "scene_not_found", "Scene was not found.", "Sahne bulunamadi.", status_code=404
        )
    return storage.scene_photos_dir(scene.property_id, photo.scene_id)


def _photo_url(storage: LocalStorage, session: Any, photo: ScenePhoto) -> str:
    return storage.url_for(_photos_dir(storage, session, photo) / photo.filename)


# --------------------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------------------


@router.patch("/{photo_id}", response_model=sc.ScenePhoto, summary="Caption or reorder a photo")
def update_photo(
    photo_id: str,
    payload: sc.UpdateScenePhotoRequest,
    session: SessionDep,
    storage: StorageDep,
    user: CurrentUser,
) -> sc.ScenePhoto:
    photo = require_scene_photo(session, photo_id, user)
    data = payload.model_dump(exclude_unset=True)
    if "caption" in data:
        caption = data["caption"]
        photo.caption = (str(caption).strip() or None) if caption is not None else None
    if "order" in data and data["order"] is not None:
        photo.order = int(data["order"])
    session.add(photo)
    session.commit()
    session.refresh(photo)
    return sc.scene_photo_to_schema(photo, _photo_url(storage, session, photo))


@router.delete("/{photo_id}", status_code=204, summary="Delete a photo")
def delete_photo(photo_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> Response:
    photo = require_scene_photo(session, photo_id, user)
    scene = session.get(Scene, photo.scene_id)
    # The files first, then the row. The other order can leave a row pointing at nothing, which
    # is a broken image in the UI; this order can at worst leak a file, which is invisible and
    # is swept with the scene directory when the room goes.
    if scene is not None:
        directory = storage.scene_photos_dir(scene.property_id, photo.scene_id)
        (directory / photo.filename).unlink(missing_ok=True)
        storage.delete_tree(directory / _DERIVED / photo.id)
        property = session.get(Property, scene.property_id)
        if property is not None:
            property.updated_at = utcnow()
            session.add(property)
    session.delete(photo)
    session.commit()
    return Response(status_code=204)


# --------------------------------------------------------------------------------------
# Decorating and emptying
# --------------------------------------------------------------------------------------


def _read_meta(directory: Path) -> dict[str, Any]:
    try:
        return json.loads((directory / _META).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing or corrupt sidecar is not a failure
        return {}


def _variant_schema(
    photo_id: str,
    variant: str,
    entry: dict[str, Any],
    url: str,
    original_url: str,
    cached: bool,
) -> sc.ScenePhotoVariant:
    return sc.ScenePhotoVariant(
        photo_id=photo_id,
        kind="empty" if variant == _EMPTY else "style",
        variant=variant,
        url=url,
        original_url=original_url,
        protected_fraction=entry.get("protected_fraction"),
        removed_fraction=entry.get("removed_fraction"),
        objects=list(entry.get("objects", [])),
        elapsed_ms=int(entry.get("elapsed_ms", 0)),
        cached=cached,
        model_backed=bool(entry.get("model_backed", True)),
        variation=int(entry.get("variation", 0)),
    )


@router.get(
    "/{photo_id}/variants",
    response_model=list[sc.ScenePhotoVariant],
    summary="Styles and the bare room already rendered for a photo",
)
def list_variants(
    photo_id: str, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> list[sc.ScenePhotoVariant]:
    """What has already been made from this photograph.

    Asked once when a photograph is opened, so a reload does not spend a GPU pass on something
    already sitting on disk. An entry whose file has gone is skipped rather than reported: a
    sidecar can outlive what it points at, and offering a broken image is worse than offering
    nothing.
    """
    photo = require_scene_photo(session, photo_id, user)
    directory = _photos_dir(storage, session, photo)
    original = directory / photo.filename
    if not original.is_file():
        return []

    derived = directory / _DERIVED / photo.id
    meta = _read_meta(derived)
    if int(meta.get("source_mtime", -1)) != int(original.stat().st_mtime):
        return []

    original_url = storage.url_for(original)
    out: list[sc.ScenePhotoVariant] = []
    for variant, entry in dict(meta.get("variants", {})).items():
        path = derived / f"{variant}.jpg"
        if path.is_file():
            out.append(
                _variant_schema(photo.id, variant, entry, storage.url_for(path), original_url, True)
            )
    return out


async def _render_variant(
    photo_id: str,
    variant: str,
    session: SessionDep,
    storage: StorageDep,
    refresh: bool,
    work: Any,
    user: User,
    variation: int = 0,
) -> sc.ScenePhotoVariant:
    """Cache lookup, the lock, the pass, the write, the sidecar.

    ``work`` is the only part that differs between decorating and emptying: a callable taking
    the decoded photograph and returning ``(image, entry)`` — or ``None`` when this machine
    cannot answer.
    """
    photo = require_scene_photo(session, photo_id, user)
    directory = _photos_dir(storage, session, photo)
    original = directory / photo.filename
    if not original.is_file():
        raise ApiException(
            "photo_not_found",
            "That photograph is no longer on the server.",
            "Bu fotograf sunucuda bulunamadi.",
            status_code=404,
        )
    session.commit()

    derived = directory / _DERIVED / photo.id
    target = derived / f"{variant}.jpg"
    original_url = storage.url_for(original)
    mtime = int(original.stat().st_mtime)

    def cached() -> sc.ScenePhotoVariant | None:
        meta = _read_meta(derived)
        entry = dict(meta.get("variants", {})).get(variant)
        if refresh or entry is None or int(meta.get("source_mtime", -1)) != mtime:
            return None
        if not _current(variant, entry) or int(entry.get("variation", 0)) != variation:
            return None
        if not target.is_file():
            return None
        return _variant_schema(photo.id, variant, entry, storage.url_for(target), original_url, True)

    hit = cached()
    if hit is not None:
        return hit

    image = await run_in_threadpool(cv2.imread, str(original))
    if image is None:  # pragma: no cover - the file was stat'ed a moment ago
        raise ApiException(
            "photo_unreadable",
            "The stored photograph could not be read.",
            "Kayitli fotograf okunamadi.",
            status_code=500,
        )

    async with gpu_lock:
        # Re-checked inside the lock: while this request queued, the caller ahead of it may
        # have asked for exactly this variant, and a second pass would spend a GPU minute to
        # produce a byte-identical file.
        hit = cached()
        if hit is not None:
            return hit
        produced = await run_in_threadpool(work, image)

    if produced is None:
        raise ApiException(
            "ai_unavailable",
            "This tool is not available on this server.",
            "Bu arac bu sunucuda kullanilamiyor.",
            hint="It needs a CUDA GPU with the segmentation and inpainting weights present.",
            status_code=503,
        )
    result_image, entry = produced

    derived.mkdir(parents=True, exist_ok=True)
    await run_in_threadpool(write_jpeg, target, result_image, 92)

    meta = _read_meta(derived)
    if int(meta.get("source_mtime", -1)) != mtime:
        # The original changed under an older sidecar, so every other variant in it is stale.
        meta = {"variants": {}}
    meta["source_mtime"] = mtime
    entry = {**entry, "algorithm": _ALGORITHM[_kind(variant)], "variation": variation}
    meta.setdefault("variants", {})[variant] = {**entry, "rendered_at": time.time()}
    await run_in_threadpool((derived / _META).write_text, json.dumps(meta, indent=2), "utf-8")

    log.info("photos.variant_done", photo_id=photo.id, variant=variant, ms=entry.get("elapsed_ms"))
    return _variant_schema(photo.id, variant, entry, storage.url_for(target), original_url, False)


@router.post(
    "/{photo_id}/restyle",
    response_model=sc.ScenePhotoVariant,
    summary="Furnish a room photo in one decorating style",
)
async def restyle_scene_photo(
    photo_id: str,
    payload: sc.ScenePhotoRestyleRequest,
    session: SessionDep,
    storage: StorageDep,
    user: CurrentUser,
) -> sc.ScenePhotoVariant:
    """The same pass ``/ai/restyle`` runs, on a photograph that belongs to a room.

    Deliberately not routed through that endpoint: it addresses photographs by the hash of
    their own bytes and knows nothing about scenes, and teaching it would give one endpoint two
    notions of identity.
    """
    from biosphere.ai import restyle as engine

    style = engine.style_by_id(payload.style)
    if style is None:
        raise ApiException(
            "unknown_style",
            f"'{payload.style}' is not a style. Use one of: {', '.join(engine.STYLE_IDS)}.",
            f"'{payload.style}' gecerli bir stil degil.",
            status_code=422,
        )

    def work(image: Any) -> Any:
        room = engine.analyse_room(image, window_dressing=payload.window_dressing)
        if room is None:
            return None
        result = engine.restyle_photo(image, style, analysis=room, variation=payload.variation)
        if result is None:
            return None
        return result.image, {
            "protected_fraction": round(result.protected_fraction, 5),
            "objects": list(room.objects),
            "elapsed_ms": result.elapsed_ms,
            "model_backed": result.model_backed,
            "seed": result.seed,
        }

    return await _render_variant(
        photo_id, style.id, session, storage, payload.refresh, work, user, variation=payload.variation
    )


@router.post(
    "/{photo_id}/empty",
    response_model=sc.ScenePhotoVariant,
    summary="Take the furniture out of a room photo",
)
async def empty_scene_photo(
    photo_id: str,
    payload: sc.ScenePhotoEmptyRequest,
    session: SessionDep,
    storage: StorageDep,
    user: CurrentUser,
) -> sc.ScenePhotoVariant:
    """Erase the furnishings and leave the room.

    An already-empty room comes back as a RESULT with ``removed_fraction: 0`` rather than as an
    error: nothing went wrong, there was simply nothing standing in it, and the caller is told
    so plainly. See ``ai.empty_photo`` for what the fill can and cannot do.
    """
    from biosphere.ai import empty_photo as engine

    def work(image: Any) -> Any:
        result = engine.empty_photo(image, window_dressing=payload.window_dressing)
        if result is None:
            return None
        return result.image, {
            "removed_fraction": round(result.removed_fraction, 5),
            "objects": list(result.objects),
            "elapsed_ms": result.elapsed_ms,
            "model_backed": result.model_backed,
        }

    return await _render_variant(photo_id, _EMPTY, session, storage, payload.refresh, work, user)
