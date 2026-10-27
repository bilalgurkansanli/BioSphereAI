"""The property's 3D walkthrough: upload the videos, build, read, rebuild, remove.

One general walkthrough video plus one video per room arrive in a single upload and are
built together, because the rooms are placed in the general video's frame (see
:mod:`biosphere.splat.pipeline`). Two entry points share one implementation: a multipart
POST for small uploads and tests, and the staged-parts finaliser the web client uses for
anything phone-sized, exactly as the scene upload does.

Rooms become :class:`Scene` rows (``capture_kind="walk"``) so the 360 and the 3D area of a
room live on the same record; a room that already exists can be named by ``scene_id`` and
keeps its panorama.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from sqlmodel import func, select

from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, JobsDep, SessionDep, SettingsDep, StorageDep, require_property
from biosphere.api.routes.properties import (
    MIN_FREE_BYTES,
    QUALITY_MODES,
    _PartStream,
    _staging_dir,
)
from biosphere.core.models import ApiException, Job, Property, Scene, User, Walkthrough, new_id, utcnow
from biosphere.core.storage import MediaProbe, validate_capture_duration, validate_media

router = APIRouter(prefix="/properties", tags=["walkthrough"])

#: At most this many videos per area; the general property of a large flat may be several segments.
MAX_SEGMENTS_PER_AREA = 16


def walkthrough_of_property(session: SessionDep, property_id: str) -> Walkthrough | None:
    return session.exec(select(Walkthrough).where(Walkthrough.property_id == property_id)).first()


def _require_walkthrough(session: SessionDep, property_id: str) -> Walkthrough:
    walk = walkthrough_of_property(session, property_id)
    if walk is None:
        raise ApiException(
            "walkthrough_not_found",
            "This property has no 3D walkthrough yet.",
            "Bu evin henüz bir 3B gezintisi yok.",
            status_code=404,
        )
    return walk


def _quality(quality_mode: str, settings) -> str:
    quality = (quality_mode or settings.default_quality_mode).strip().lower()
    if quality not in QUALITY_MODES:
        raise ApiException(
            "invalid_quality_mode",
            f"'{quality}' is not a valid quality mode. Use one of: {', '.join(sorted(QUALITY_MODES))}.",
            f"'{quality}' geçerli bir kalite modu değil. Şunlardan birini kullanın: {', '.join(sorted(QUALITY_MODES))}.",
            status_code=422,
        )
    return quality


async def _save_area(
    storage, settings, uploads: list[UploadFile], *, scene_id: str, prefix: str
) -> list[Path]:
    """Stream one area's segments to disk and validate them as one capture."""
    saved: list[Path] = []
    probes: list[MediaProbe] = []
    for index, upload in enumerate(uploads):
        path = await run_in_threadpool(
            storage.save_upload, upload.file, scene_id=scene_id, filename=upload.filename or f"walk_{index}.mp4",
            index=index, prefix=prefix,
        )
        saved.append(path)
        probes.append(await run_in_threadpool(validate_media, path, expect="video", settings=settings))
    await run_in_threadpool(validate_capture_duration, probes, settings)
    return saved


async def _create_walkthrough(
    property_id: str,
    *,
    session,
    storage,
    jobs,
    settings,
    spec: sc.CreateWalkthroughSpec,
    files: list[UploadFile],
    user: User,
) -> sc.WalkthroughJobResponse:
    property = require_property(session, property_id, user)
    quality = _quality(spec.quality_mode, settings)
    by_name: dict[str, UploadFile] = {}
    for upload in files:
        name = (upload.filename or "").strip()
        if not name:
            continue
        if name in by_name:
            raise ApiException(
                "duplicate_filename",
                f"Two uploaded files are both called '{name}'.",
                f"İki yüklenen dosyanın adı da '{name}'.",
                hint="Give every video a distinct file name.",
                status_code=422,
            )
        by_name[name] = upload

    def take(names: list[str], area: str) -> list[UploadFile]:
        out = []
        for n in names:
            if n not in by_name:
                raise ApiException(
                    "missing_file",
                    f"The spec names '{n}' for {area} but no such file was uploaded.",
                    f"Belirtimde {area} için '{n}' adı geçiyor ama böyle bir dosya yüklenmedi.",
                    status_code=422,
                )
            out.append(by_name[n])
        if len(out) > MAX_SEGMENTS_PER_AREA:
            raise ApiException(
                "too_many_files",
                f"{area}: at most {MAX_SEGMENTS_PER_AREA} video segments per area.",
                f"{area}: alan başına en fazla {MAX_SEGMENTS_PER_AREA} video parçası.",
                status_code=422,
            )
        return out

    base_files = take(spec.base_filenames, "the general walkthrough")
    room_files = [take(r.filenames, r.name) for r in spec.rooms]
    if not base_files and not spec.rooms:
        raise ApiException(
            "no_file",
            "Upload a general walkthrough video and/or room videos.",
            "Bir genel gezinti videosu ve/veya oda videoları yükleyin.",
            status_code=422,
        )
    if storage.free_bytes() < MIN_FREE_BYTES:
        raise ApiException(
            "insufficient_storage",
            "The server is out of disk space for new captures.",
            "Sunucuda yeni çekimler için disk alanı kalmadı.",
            status_code=507,
        )

    existing = walkthrough_of_property(session, property_id)
    if existing is not None and existing.status == "processing" and existing.active_job_id:
        raise ApiException(
            "already_processing",
            "This property's 3D walkthrough is already being built.",
            "Bu evin 3B gezintisi zaten oluşturuluyor.",
            hint="Cancel the running job first.",
            status_code=409,
        )
    walk = existing or Walkthrough(property_id=property.id)
    walk.quality_mode = quality
    has_base = bool(base_files) or bool(walk.base_source_paths)
    existing_rooms = [session.get(Scene, sid) for sid in list(walk.room_scene_ids or [])]
    existing_rooms = [s for s in existing_rooms if s is not None and s.walk_source_paths]
    n_rooms_after = len(existing_rooms) + sum(
        1 for r in spec.rooms if not any((r.scene_id and s.id == r.scene_id) or s.name.strip().lower() == r.name.strip().lower() for s in existing_rooms)
    )
    if n_rooms_after > 1 and not has_base and not spec.defer_build:
        raise ApiException(
            "walkthrough_required",
            "A general walkthrough video is required when there is more than one room: it is what places the rooms in the flat.",
            "Birden fazla oda varsa genel gezinti videosu gerekli: odaları daireye yerleştiren odur.",
            status_code=422,
        )

    # Rooms: existing scenes named by id keep their 360; the rest are created.
    next_order = int(
        session.exec(select(func.coalesce(func.max(Scene.order), -1)).where(Scene.property_id == property_id)).one()
    ) + 1
    scenes: list[Scene] = []
    for spec_room in spec.rooms:
        scene = session.get(Scene, spec_room.scene_id) if spec_room.scene_id else None
        if spec_room.scene_id and (scene is None or scene.property_id != property_id):
            raise ApiException(
                "scene_not_found",
                f"Room '{spec_room.scene_id}' does not belong to this property.",
                f"'{spec_room.scene_id}' odası bu tura ait değil.",
                status_code=404,
            )
        if scene is None:
            # A re-recorded room replaces the one with the same name rather than doubling it.
            scene = next((s for s in existing_rooms if s.name.strip().lower() == spec_room.name.strip().lower()), None)
        if scene is None:
            scene = Scene(
                id=new_id(), property_id=property.id, name=spec_room.name.strip(), status="pending",
                capture_kind="walk", quality_mode=quality, order=next_order,
            )
            next_order += 1
        scenes.append(scene)

    saved_dirs: list[Path] = []
    try:
        if base_files:
            base_dir_id = f"walk_{walk.id}"
            storage.delete_tree(settings.uploads_dir / base_dir_id)
            paths = await _save_area(storage, settings, base_files, scene_id=base_dir_id, prefix="source")
            saved_dirs.append(settings.uploads_dir / base_dir_id)
            walk.base_source_paths = [str(p) for p in paths]
        for scene, uploads in zip(scenes, room_files, strict=True):
            for old in scene.walk_source_paths or []:
                Path(old).unlink(missing_ok=True)
            paths = await _save_area(storage, settings, uploads, scene_id=scene.id, prefix="walk")
            scene.walk_source_paths = [str(p) for p in paths]
            scene.walkthrough = {"status": "pending", "url": None, "thumbnail_url": None, "splat_count": 0, "bytes": 0, "placed": False, "error": None}
    except Exception:
        for d in saved_dirs:
            storage.delete_tree(d)
        raise

    # Existing rooms stay, in their order; new ones follow.
    kept = [s for s in existing_rooms if s.id not in {x.id for x in scenes}]
    walk.room_scene_ids = [s.id for s in kept] + [s.id for s in scenes]
    walk.status = "pending"
    walk.error = None
    walk.updated_at = utcnow()
    session.add(walk)
    for scene in scenes:
        session.add(scene)
    property.updated_at = utcnow()
    session.add(property)
    session.commit()
    session.refresh(walk)

    if spec.defer_build:
        # Stored, not built: the caller will POST .../rebuild when every area is in.
        placeholder = Job(
            property_id=property.id, scene_id=None, kind="walkthrough", status="queued", stage="upload", progress=0.0,
            stage_progress=0.0, message="Waiting for the remaining videos", message_tr="Kalan videolar bekleniyor",
            quality_mode=quality, capture_kind="walk",
        )
        return sc.WalkthroughJobResponse(walkthrough=sc.walkthrough_to_schema(walk), job=sc.job_to_schema(placeholder))

    job: Job = jobs.submit_walkthrough(walk.id)
    session.expire_all()
    session.refresh(walk)
    return sc.WalkthroughJobResponse(walkthrough=sc.walkthrough_to_schema(walk), job=sc.job_to_schema(job))


@router.post(
    "/{property_id}/walkthrough",
    response_model=sc.WalkthroughJobResponse,
    status_code=201,
    summary="Upload the walkthrough videos and start the 3D build",
)
async def create_walkthrough(
    property_id: str,
    session: SessionDep,
    storage: StorageDep,
    jobs: JobsDep,
    settings: SettingsDep,
    user: CurrentUser,
    spec: Annotated[str, Form(min_length=2)],
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> sc.WalkthroughJobResponse:
    """``spec`` is JSON (:class:`CreateWalkthroughSpec`); every video travels in ``files[]``
    and is assigned to an area by its file name."""
    try:
        parsed = sc.CreateWalkthroughSpec.model_validate(json.loads(spec))
    except (ValueError, ValidationError) as exc:
        raise ApiException(
            "invalid_spec",
            "The walkthrough spec is not valid JSON of the expected shape.",
            "Gezinti belirtimi beklenen biçimde geçerli bir JSON değil.",
            hint=str(exc)[:300],
            status_code=422,
        ) from exc
    return await _create_walkthrough(
        property_id, session=session, storage=storage, jobs=jobs, settings=settings, spec=parsed, files=list(files or []), user=user
    )


@router.post(
    "/{property_id}/walkthrough/from-parts",
    response_model=sc.WalkthroughJobResponse,
    status_code=201,
    summary="Assemble staged walkthrough videos and start the 3D build",
)
async def create_walkthrough_from_parts(
    property_id: str,
    payload: sc.CreateWalkthroughFromPartsRequest,
    session: SessionDep,
    storage: StorageDep,
    jobs: JobsDep,
    settings: SettingsDep,
    user: CurrentUser,
) -> sc.WalkthroughJobResponse:
    directory = _staging_dir(settings, payload.upload_id)
    if not directory.is_dir():
        raise ApiException(
            "upload_not_found",
            "That staged upload does not exist, or has already been assembled.",
            "Bu yükleme bulunamadı ya da zaten birleştirildi.",
            hint="Upload the parts again.",
            status_code=404,
        )
    streams: list[_PartStream] = []
    files: list[UploadFile] = []

    def assemble(entry: sc.ScenePartFile) -> str:
        paths: list[Path] = []
        for index in entry.parts:
            part = directory / f"{index:06d}.part"
            if not part.is_file():
                raise ApiException(
                    "missing_part",
                    f"Part {index} of '{entry.filename}' never arrived.",
                    f"'{entry.filename}' dosyasının {index}. parçası gelmemiş.",
                    hint="Upload the missing part and finalise again.",
                    status_code=422,
                )
            paths.append(part)
        stream = _PartStream(paths)
        streams.append(stream)
        files.append(UploadFile(file=stream, filename=entry.filename))
        return entry.filename

    try:
        spec = sc.CreateWalkthroughSpec(
            quality_mode=payload.quality_mode,
            defer_build=payload.defer_build,
            base_filenames=[assemble(e) for e in payload.base],
            rooms=[
                sc.WalkthroughRoomSpec(
                    name=r.name, scene_id=r.scene_id, floor_index=r.floor_index, filenames=[assemble(e) for e in r.files]
                )
                for r in payload.rooms
            ],
        )
        response = await _create_walkthrough(
            property_id, session=session, storage=storage, jobs=jobs, settings=settings, spec=spec, files=files, user=user
        )
    finally:
        for stream in streams:
            stream.close()
    storage.delete_tree(directory)
    return response


@router.get("/{property_id}/walkthrough", response_model=sc.Walkthrough, summary="The property's 3D walkthrough")
def get_walkthrough(property_id: str, session: SessionDep, user: CurrentUser) -> sc.Walkthrough:
    require_property(session, property_id, user)
    return sc.walkthrough_to_schema(_require_walkthrough(session, property_id))


@router.post(
    "/{property_id}/walkthrough/rebuild",
    response_model=sc.Job,
    status_code=202,
    summary="Rebuild the 3D walkthrough from the uploaded videos",
)
def rebuild_walkthrough(
    property_id: str, payload: sc.RebuildWalkthroughRequest, session: SessionDep, jobs: JobsDep, settings: SettingsDep
,
    user: CurrentUser,
) -> sc.Job:
    require_property(session, property_id, user)
    walk = _require_walkthrough(session, property_id)
    if payload.quality_mode:
        walk.quality_mode = _quality(payload.quality_mode, settings)
        session.add(walk)
    rooms_with_video = [
        s for s in (session.get(Scene, sid) for sid in list(walk.room_scene_ids or [])) if s is not None and s.walk_source_paths
    ]
    if len(rooms_with_video) > 1 and not walk.base_source_paths:
        raise ApiException(
            "walkthrough_required",
            "A general walkthrough video is required when there is more than one room: it is what places the rooms in the flat.",
            "Birden fazla oda varsa genel gezinti videosu gerekli: odaları daireye yerleştiren odur.",
            status_code=422,
        )
    session.commit()
    return sc.job_to_schema(jobs.submit_walkthrough(walk.id))


@router.delete("/{property_id}/walkthrough", status_code=204, summary="Remove the 3D walkthrough and its videos")
def delete_walkthrough(property_id: str, session: SessionDep, storage: StorageDep, settings: SettingsDep, user: CurrentUser) -> Response:
    require_property(session, property_id, user)
    walk = _require_walkthrough(session, property_id)
    if walk.status == "processing" and walk.active_job_id:
        raise ApiException(
            "already_processing",
            "Cancel the running build before removing the walkthrough.",
            "Gezintiyi kaldırmadan önce çalışan işi iptal edin.",
            status_code=409,
        )
    walk_id = walk.id
    room_ids = list(walk.room_scene_ids or [])
    for scene_id in room_ids:
        scene = session.get(Scene, scene_id)
        if scene is None:
            continue
        for p in scene.walk_source_paths or []:
            Path(p).unlink(missing_ok=True)
        scene.walk_source_paths = []
        scene.walkthrough = None
        if scene.capture_kind == "walk" and not scene.panorama:
            scene.status = "ready"  # a room without any capture, like a photo-only room
        session.add(scene)
    session.delete(walk)
    property = session.get(Property, property_id)
    if property is not None:
        property.updated_at = utcnow()
        session.add(property)
    session.commit()
    storage.delete_tree(settings.uploads_dir / f"walk_{walk_id}")
    storage.delete_tree(settings.properties_dir / property_id / "walk")
    return Response(status_code=204)
