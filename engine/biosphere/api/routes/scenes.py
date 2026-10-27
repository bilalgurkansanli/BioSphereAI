"""Scene read/update/delete, reprocessing, and the two things attached to a room:
hotspots and ordinary photographs."""

from __future__ import annotations

from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func
from sqlmodel import select

from biosphere.api import schemas as sc
from biosphere.api.deps import (
    CurrentUser,
    JobsDep,
    SessionDep,
    StorageDep,
    hotspots_of_scene,
    photos_of_scene,
    require_scene,
)
from biosphere.core.models import ApiException, Hotspot, Property, Scene, ScenePhoto, utcnow
from biosphere.core.storage import write_jpeg

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.get("/{scene_id}", response_model=sc.Scene, summary="Get a scene")
def get_scene(scene_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> sc.Scene:
    scene = require_scene(session, scene_id, user)
    return sc.scene_to_schema(scene, hotspots_of_scene(session, scene_id), _photos(storage, scene))


@router.patch("/{scene_id}", response_model=sc.Scene, summary="Rename, reorder or re-aim a scene")
def update_scene(
    scene_id: str, payload: sc.UpdateSceneRequest, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> sc.Scene:
    scene = require_scene(session, scene_id, user)
    data = payload.model_dump(exclude_unset=True)
    if "name" in data and data["name"] is not None:
        scene.name = str(data["name"]).strip()
    if "order" in data and data["order"] is not None:
        scene.order = int(data["order"])
    if "initial_view" in data and data["initial_view"] is not None:
        # Stored as a whole document: SQLAlchemy does not track mutation inside JSON columns.
        scene.initial_view = dict(data["initial_view"])
    session.add(scene)

    property = session.get(Property, scene.property_id)
    if property is not None:
        property.updated_at = utcnow()
        session.add(property)
    session.commit()
    session.refresh(scene)
    return sc.scene_to_schema(scene, hotspots_of_scene(session, scene_id), _photos(storage, scene))


@router.delete("/{scene_id}", status_code=204, summary="Delete a scene and its media")
def delete_scene(scene_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> Response:
    scene = require_scene(session, scene_id, user)
    property_id = scene.property_id
    property = session.get(Property, property_id)
    if property is not None and property.cover_scene_id == scene.id:
        property.cover_scene_id = None
        property.updated_at = utcnow()
        session.add(property)
    session.delete(scene)
    session.commit()
    storage.delete_tree(storage.settings.uploads_dir / scene_id)
    storage.delete_tree(storage.settings.properties_dir / property_id / "scenes" / scene_id)
    return Response(status_code=204)


@router.post("/{scene_id}/reprocess", response_model=sc.Job, status_code=202, summary="Re-run the stitch")
def reprocess_scene(scene_id: str, session: SessionDep, jobs: JobsDep, user: CurrentUser) -> sc.Job:
    """Queue a fresh stitch from the original upload, which is why we keep it around."""
    scene = require_scene(session, scene_id, user)
    if scene.active_job_id is not None and scene.status == "processing":
        raise ApiException(
            "already_processing",
            "This scene is already being processed.",
            "Bu sahne zaten işleniyor.",
            hint="Cancel the running job first.",
            status_code=409,
        )
    if not scene.source_paths:
        raise ApiException(
            "no_source_media",
            "The original capture for this scene is gone, so it cannot be reprocessed.",
            "Bu sahnenin orijinal çekimi bulunmadığı için yeniden işlenemez.",
            hint="Upload the capture again as a new scene.",
            status_code=409,
        )
    session.commit()  # release the read transaction before the worker opens its own
    return sc.job_to_schema(jobs.submit(scene_id))


@router.post(
    "/{scene_id}/hotspots", response_model=sc.Hotspot, status_code=201, summary="Add a hotspot"
)
def create_hotspot(scene_id: str, payload: sc.CreateHotspotRequest, session: SessionDep, user: CurrentUser) -> sc.Hotspot:
    scene = require_scene(session, scene_id, user)
    target_id = payload.target_scene_id

    if payload.type in sc.NAVIGATION_TYPES:
        if not target_id:
            raise ApiException(
                "missing_link_target",
                "A navigation hotspot needs a target scene.",
                "Geçiş noktası için hedef sahne gerekli.",
                status_code=422,
            )
        target = session.get(Scene, target_id)
        if target is None:
            raise ApiException(
                "invalid_link_target",
                "The target scene does not exist.",
                "Hedef sahne mevcut değil.",
                hint="Pick a scene from this property.",
                status_code=422,
            )
        if target.property_id != scene.property_id:
            raise ApiException(
                "invalid_link_target",
                "A hotspot can only link to a scene in the same property.",
                "Bir etkileşim noktası yalnızca aynı evdeki bir sahneye bağlanabilir.",
                status_code=422,
            )

    hotspot = Hotspot(
        scene_id=scene.id,
        type=payload.type,
        yaw=payload.yaw,
        pitch=payload.pitch,
        title=payload.title.strip(),
        body=payload.body,
        icon=payload.icon,
        target_scene_id=target_id,
        media_url=payload.media_url,
        accent=payload.accent,
    )
    session.add(hotspot)
    session.commit()
    session.refresh(hotspot)
    return sc.hotspot_to_schema(hotspot)


# --------------------------------------------------------------------------------------
# Photographs
# --------------------------------------------------------------------------------------

#: Longest side a stored photograph keeps.
#:
#: Generous on purpose, and for a different reason than the restyle feature's own cap: these
#: are DELIVERABLES. The point of putting photographs on a room is to hand them to a listing —
#: today by download, later by whatever uploads to a portal — and a portal re-encodes what it
#: is given, so arriving already small means arriving small twice.
_PHOTO_MAX_SIDE = 2560
_PHOTO_QUALITY = 88

#: Refused before anything is decoded.
_PHOTO_MAX_BYTES = 32 * 1024 * 1024

#: One request cannot carry more than this. A room is a handful of pictures; a hundred is a
#: mistake or a probe, and either way it should fail fast rather than half way through.
MAX_PHOTOS_PER_REQUEST = 24


def _photos(storage: object, scene: Scene) -> list[sc.ScenePhoto]:
    """The room's photographs as schemas, resolved to URLs."""
    from biosphere.core.db import session_scope

    with session_scope() as session:
        rows = photos_of_scene(session, scene.id)
        directory = storage.scene_photos_dir(scene.property_id, scene.id)  # type: ignore[attr-defined]
        return [
            sc.scene_photo_to_schema(row, storage.url_for(directory / row.filename))  # type: ignore[attr-defined]
            for row in rows
        ]


def _fit(image: np.ndarray, longest: int = _PHOTO_MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = longest / float(max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )


@router.post(
    "/{scene_id}/photos",
    response_model=list[sc.ScenePhoto],
    status_code=201,
    summary="Add photographs to a room",
)
async def add_photos(
    scene_id: str,
    session: SessionDep,
    storage: StorageDep,
    files: Annotated[list[UploadFile], File()],
    user: CurrentUser,
) -> list[sc.ScenePhoto]:
    """Attach ordinary photographs to a room.

    Independent of whether the room has a 360: a listing usually wants both, and an agent
    photographing a flat takes the panorama and the stills on the same visit. Each file is
    DECODED rather than sniffed — if OpenCV can turn the bytes into an image then it is an
    image, which is a stronger statement than any extension check.

    Partial success is deliberate: with four good photographs and one corrupt file, refusing
    all five would cost the user four uploads to punish one. The good ones are stored and the
    response says which arrived; a request where NOTHING was an image is a 415.
    """
    scene = require_scene(session, scene_id, user)
    if not files:
        raise ApiException(
            "no_file", "No file was uploaded.", "Hiçbir dosya yüklenmedi.", status_code=422
        )
    if len(files) > MAX_PHOTOS_PER_REQUEST:
        raise ApiException(
            "too_many_files",
            f"At most {MAX_PHOTOS_PER_REQUEST} photographs per request.",
            f"Bir seferde en fazla {MAX_PHOTOS_PER_REQUEST} fotoğraf yüklenebilir.",
            status_code=422,
        )

    directory = storage.scene_photos_dir(scene.property_id, scene.id)
    next_order = int(
        session.exec(
            select(func.coalesce(func.max(ScenePhoto.order), -1)).where(
                ScenePhoto.scene_id == scene.id
            )
        ).one()
    ) + 1

    created: list[ScenePhoto] = []
    for upload in files:
        payload = await upload.read(_PHOTO_MAX_BYTES + 1)
        if not payload or len(payload) > _PHOTO_MAX_BYTES:
            continue
        image = await run_in_threadpool(
            cv2.imdecode, np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            continue
        fitted = _fit(image)
        row = ScenePhoto(
            scene_id=scene.id,
            filename="",
            width=int(fitted.shape[1]),
            height=int(fitted.shape[0]),
            order=next_order,
        )
        row.filename = f"{row.id}.jpg"
        await run_in_threadpool(write_jpeg, directory / row.filename, fitted, _PHOTO_QUALITY)
        session.add(row)
        created.append(row)
        next_order += 1

    if not created:
        raise ApiException(
            "unsupported_media",
            "None of those files is an image this server can read.",
            "Bu dosyaların hiçbiri okunabilir bir görsel değil.",
            hint="Use JPEG, PNG or WebP photographs.",
            status_code=415,
        )

    property = session.get(Property, scene.property_id)
    if property is not None:
        property.updated_at = utcnow()
        session.add(property)
    session.commit()
    for row in created:
        session.refresh(row)

    return [
        sc.scene_photo_to_schema(row, storage.url_for(directory / row.filename)) for row in created
    ]
