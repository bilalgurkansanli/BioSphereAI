"""Property CRUD and the scene-upload endpoint that starts a stitch job."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session, func, select

from biosphere.api import schemas as sc
from biosphere.api.deps import (
    CurrentUser,
    JobsDep,
    SessionDep,
    SettingsDep,
    StorageDep,
    hotspots_by_scene,
    photos_by_scene,
    require_property,
    scenes_of_property,
)
from biosphere.core.config import Settings
from biosphere.core.models import ApiException, Job, Property, Scene, new_id, utcnow
from biosphere.core.storage import LocalStorage, MediaProbe, validate_capture_duration, validate_media

router = APIRouter(prefix="/properties", tags=["properties"])

QUALITY_MODES = {"fast", "balanced", "studio"}
CAPTURE_KINDS = {"video", "photos"}

#: Refuse an upload when the volume is nearly full: running out mid-render corrupts the
#: output, and the user has already filmed the room by then.
#:
#: It has to clear `max_upload_bytes` with room to spare or the check means nothing — a floor
#: below the largest upload the service accepts would pass a request it cannot then store.
#: 4 GB is the 1.5 GB ceiling plus the extracted frames and the delivered levels. The frames
#: are NOT proportional to the video's length: `_sampling_fps` divides the decimation rate by
#: the duration so a ten-minute capture yields the same `max_extracted_frames` as a one-minute
#: one, which is why this number does not have to grow with the capture ceiling.
MIN_FREE_BYTES = 4 * 1024**3

#: Extensions in ``files[]`` that are capture TELEMETRY rather than footage.
#:
#: The guided capture on the phone posts its sensor log — the orientation track, the
#: warnings it raised and the coverage it measured — in the same ``files[]`` array as the
#: video. It does so because an upload over a phone tether should cost one request with one
#: progress bar; a second round trip is a second chance to fail at the worst possible
#: moment, after the user has already filmed the room.
#:
#: The sidecar is kept beside the footage for diagnosis and is deliberately NOT a source
#: path: the stitch is driven by the video alone, and a solver that started depending on
#: the phone's own numbers would stop working for the uploads that never had them.
SIDECAR_SUFFIXES = frozenset({".json"})

#: Where sidecar filenames start, far enough above the footage that the two can never
#: collide no matter how many segments a capture was split into.
SIDECAR_INDEX_BASE = 90

#: A video capture legitimately arrives in SEVERAL PIECES. The recorder seals a segment at
#: its own 120 s length ceiling, an incoming call ends the camera track outright, and the
#: silent-encoder watchdog restarts onto a different codec mid-sweep. They are consecutive
#: footage of one room from one spot, so they are stitched as one capture.
#:
#: A full-length capture is 600 s / 120 s = five segments before anything goes wrong, so the
#: cap has to sit well above five or an interrupted capture is refused for having been
#: interrupted. It exists only to keep a malformed request from queueing unbounded work.
MAX_VIDEO_SEGMENTS = 16


def _is_sidecar(upload: UploadFile) -> bool:
    """True for a telemetry file riding along with the footage.

    Decided on the extension rather than the field name because the client sends everything
    through one ``files[]`` array; ``.json`` is not in either media allow-list, so a file
    that reaches here with that suffix could never have been a capture in the first place.
    """
    return Path(upload.filename or "").suffix.lower() in SIDECAR_SUFFIXES


@router.post("", response_model=sc.Property, status_code=201, summary="Create a property")
def create_property(payload: sc.CreatePropertyRequest, session: SessionDep, user: CurrentUser) -> sc.Property:
    property = Property(
        owner_id=user.id,
        title=payload.title.strip(),
        description=payload.description,
        address=payload.address,
    )
    session.add(property)
    session.commit()
    session.refresh(property)
    return sc.property_to_schema(property, [])


@router.get("", response_model=sc.Paginated[sc.Property], summary="List properties")
def list_properties(
    session: SessionDep,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> sc.Paginated[sc.Property]:
    # Scoped in the QUERY, not filtered afterwards. A list endpoint that fetches everything
    # and then drops what the caller may not see gets the pagination wrong the moment there
    # is a second account — page one would be short and page two would repeat.
    owned = select(Property).where(Property.owner_id == user.id)
    total = int(session.exec(select(func.count()).select_from(Property).where(Property.owner_id == user.id)).one())
    properties = list(
        session.exec(
            owned.order_by(Property.created_at.desc()).offset(offset).limit(limit)  # type: ignore[attr-defined]
        ).all()
    )
    # One query for every listed property's scenes rather than N+1 round trips.
    property_ids = [t.id for t in properties]
    scenes_by_property: dict[str, list[Scene]] = {tid: [] for tid in property_ids}
    if property_ids:
        for scene in session.exec(select(Scene).where(Scene.property_id.in_(property_ids))).all():  # type: ignore[attr-defined]
            scenes_by_property.setdefault(scene.property_id, []).append(scene)
    return sc.Paginated[sc.Property](
        items=[sc.property_to_schema(t, scenes_by_property.get(t.id, [])) for t in properties],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{property_id}", response_model=sc.PropertyDetail, summary="Property with its scenes")
def get_property(property_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> sc.PropertyDetail:
    property = require_property(session, property_id, user)
    scenes = scenes_of_property(session, property_id)
    from biosphere.api.routes.walkthrough import walkthrough_of_property

    return sc.property_detail_to_schema(
        property,
        scenes,
        hotspots_by_scene(session, [s.id for s in scenes]),
        _photos_by_scene(storage, session, scenes),
        walkthrough=walkthrough_of_property(session, property_id),
    )


@router.patch("/{property_id}", response_model=sc.Property, summary="Update a property")
def update_property(property_id: str, payload: sc.UpdatePropertyRequest, session: SessionDep, user: CurrentUser) -> sc.Property:
    property = require_property(session, property_id, user)
    data = payload.model_dump(exclude_unset=True)

    if "cover_scene_id" in data and data["cover_scene_id"] is not None:
        scene = session.get(Scene, data["cover_scene_id"])
        if scene is None or scene.property_id != property.id:
            raise ApiException(
                "invalid_cover_scene",
                "The chosen cover scene does not belong to this property.",
                "Seçilen kapak sahnesi bu eve ait değil.",
                status_code=422,
            )
    for key, value in data.items():
        setattr(property, key, value)
    property.updated_at = utcnow()
    session.add(property)
    session.commit()
    session.refresh(property)
    return sc.property_to_schema(property, scenes_of_property(session, property_id))


def _photos_by_scene(
    storage: LocalStorage, session: Session, scenes: list[Scene]
) -> dict[str, list[sc.ScenePhoto]]:
    """Every room's photographs in one query, resolved to URLs."""
    grouped = photos_by_scene(session, [s.id for s in scenes])
    out: dict[str, list[sc.ScenePhoto]] = {}
    for scene in scenes:
        directory = storage.scene_photos_dir(scene.property_id, scene.id)
        out[scene.id] = [
            sc.scene_photo_to_schema(row, storage.url_for(directory / row.filename))
            for row in grouped.get(scene.id, [])
        ]
    return out


@router.post(
    "/{property_id}/rooms",
    response_model=sc.Scene,
    status_code=201,
    summary="Create a room with no capture",
)
def create_room(property_id: str, payload: sc.CreateRoomRequest, session: SessionDep, user: CurrentUser) -> sc.Scene:
    """A room that exists before anything has been shot in it.

    The property's unit is a ROOM, and not every room starts with a panorama: an agent may have
    four photographs of the kitchen and no 360 of it, or may want the room in the property now and
    the capture later. Such a room has ``capture_kind = "none"``, no job, no panorama, and is
    ``ready`` the moment it exists — there is nothing to wait for.

    Deliberately NOT folded into ``create_scene``: that endpoint is a multipart upload whose
    every branch is about validating footage, and a JSON route that uploads nothing has no
    business sharing it.
    """
    property = require_property(session, property_id, user)
    next_order = int(
        session.exec(select(func.coalesce(func.max(Scene.order), -1)).where(Scene.property_id == property_id)).one()
    ) + 1
    scene = Scene(
        property_id=property.id,
        name=payload.name.strip(),
        status="ready",
        capture_kind="none",
        order=next_order,
        source_paths=[],
    )
    session.add(scene)
    property.updated_at = utcnow()
    session.add(property)
    session.commit()
    session.refresh(scene)
    return sc.scene_to_schema(scene, [], [])


@router.delete("/{property_id}", status_code=204, summary="Delete a property and its media")
def delete_property(property_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> Response:
    property = require_property(session, property_id, user)
    # Read the ids *before* the delete: the cascade removes the scene rows, after which the
    # ORM can no longer refresh those expired instances.
    scene_ids = [scene.id for scene in scenes_of_property(session, property_id)]
    # Rows go first: if the media delete fails half way we would rather leak bytes on disk
    # than leave the dashboard pointing at scenes whose images have vanished.
    session.delete(property)
    session.commit()
    for scene_id in scene_ids:
        storage.delete_tree(storage.settings.uploads_dir / scene_id)
    storage.delete_tree(storage.settings.properties_dir / property_id)
    return Response(status_code=204)


# --------------------------------------------------------------------------------------
# Scene upload
# --------------------------------------------------------------------------------------



#: A capture is staged in numbered parts under this directory, then assembled.
_STAGING_DIRNAME = "_staging"
#: Nothing about a part is trusted from the URL: both are matched against this first.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
#: Enough parts for a 4 GB capture at the client's 8 MB chunk, and no more.
_MAX_PARTS = 512


#: How long an unfinished upload's parts are kept. Long enough that a user who lost the link
#: mid-transfer can finish from the same page after a break; short enough that abandoned
#: captures do not accumulate. At half a gigabyte each, they accumulate fast.
STAGING_TTL_HOURS = 24


def sweep_staging(settings: "Settings", *, ttl_hours: float = STAGING_TTL_HOURS) -> int:
    """Delete staged parts nobody came back for. Returns how many uploads were removed.

    A finalise that fails leaves the parts deliberately — a validation error should not cost
    the user a re-upload of a quarter of a gigabyte, and retrying the finalise is free. The
    cost of that kindness is that nothing ever cleans up after an upload that is simply
    abandoned, and one abandoned 500 MB capture a day fills a disk in a year. So the parts
    have an expiry rather than a promise.
    """
    root = settings.uploads_dir / _STAGING_DIRNAME
    if not root.is_dir():
        return 0
    cutoff = time.time() - ttl_hours * 3600.0
    removed = 0
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            newest = max((p.stat().st_mtime for p in directory.iterdir()), default=directory.stat().st_mtime)
        except OSError:
            continue
        if newest < cutoff:
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    return removed


def _staging_dir(settings: "Settings", upload_id: str) -> Path:
    if not _SAFE_ID.match(upload_id):
        raise ApiException(
            "invalid_upload_id",
            "That upload id is not valid.",
            "Bu yükleme kimliği geçersiz.",
            status_code=422,
        )
    return settings.uploads_dir / _STAGING_DIRNAME / upload_id


class _PartStream:
    """The staged parts of one file, read as if they were the file.

    `save_upload` wants a binary stream and enforces the size limit while it copies, so
    handing it one of these means the assembled path is byte-for-byte the path a single POST
    would have taken — same naming, same limit, same validation, no second copy and no second
    code path to keep honest.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._index = 0
        self._handle: BinaryIO | None = None

    def read(self, size: int = -1) -> bytes:
        while True:
            if self._handle is None:
                if self._index >= len(self._paths):
                    return b""
                self._handle = self._paths[self._index].open("rb")
                self._index += 1
            chunk = self._handle.read(size)
            if chunk:
                return chunk
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@router.put(
    "/uploads/{upload_id}/parts/{index}",
    status_code=200,
    summary="Stage one part of a large capture",
)
async def put_upload_part(
    upload_id: str,
    index: int,
    request: Request,
    settings: SettingsDep,
    storage: StorageDep,
    user: CurrentUser,
) -> dict[str, int]:
    """Take one slice of a capture and write it to disk, streamed.

    WHY THIS EXISTS. The single multipart POST works and stays the path for ordinary
    captures, but it asks the browser to build the whole body before sending it. On iOS a
    270 MB capture never left the phone at all: no request reached the proxy, no error
    reached the page, and the upload sat at 0% under "keep this page open" until the user
    gave up — the capture lost, twice over, because the recording only ever existed in that
    tab. Safari builds the body in memory and simply gives up when it cannot.

    A part is a `Blob.slice`, which costs the browser nothing to produce, and arrives here as
    a raw body rather than multipart so nothing has to be parsed or buffered on this side
    either. Ordering is the client's business: parts are numbered and assembled by number.
    """
    if not (0 <= index < _MAX_PARTS):
        raise ApiException(
            "invalid_part_index",
            f"Part index must be between 0 and {_MAX_PARTS - 1}.",
            f"Parça numarası 0 ile {_MAX_PARTS - 1} arasında olmalı.",
            status_code=422,
        )
    if storage.free_bytes() < MIN_FREE_BYTES:
        raise ApiException(
            "insufficient_storage",
            "The server is out of disk space for new captures.",
            "Sunucuda yeni çekimler için disk alanı kalmadı.",
            status_code=507,
        )
    directory = _staging_dir(settings, upload_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{index:06d}.part"

    written = 0
    with target.open("wb") as handle:
        async for chunk in request.stream():
            if not chunk:
                continue
            await run_in_threadpool(handle.write, chunk)
            written += len(chunk)
    return {"received": written}


@router.post(
    "/{property_id}/scenes/from-parts",
    response_model=sc.SceneJobResponse,
    status_code=201,
    summary="Assemble a staged capture and queue the stitch",
)
async def create_scene_from_parts(
    property_id: str,
    payload: sc.CreateSceneFromPartsRequest,
    session: SessionDep,
    storage: StorageDep,
    jobs: JobsDep,
    settings: SettingsDep,
) -> sc.SceneJobResponse:
    """Turn staged parts into the same upload a single POST would have produced.

    Deliberately thin: it assembles streams and hands them to :func:`create_scene`, so the
    size limit, the media sniff, the capture-duration gate, the sidecar handling and the job
    submission are all the ones already in use. A second implementation of any of that would
    be a second thing to keep correct.
    """
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
    try:
        for entry in payload.files:
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

        response = await create_scene(
            property_id=property_id,
            session=session,
            storage=storage,
            jobs=jobs,
            settings=settings,
            name=payload.name,
            quality_mode=payload.quality_mode,
            capture_kind=payload.capture_kind,
            file=None,
            files=files,
        )
    finally:
        for stream in streams:
            stream.close()

    # Only once the scene owns a copy. A failure leaves the parts where they are, so the
    # client can finalise again without re-uploading a quarter of a gigabyte.
    storage.delete_tree(directory)
    return response



@router.post(
    "/{property_id}/scenes",
    response_model=sc.SceneJobResponse,
    status_code=201,
    summary="Upload a capture and start processing",
)
async def create_scene(
    property_id: str,
    session: SessionDep,
    storage: StorageDep,
    jobs: JobsDep,
    settings: SettingsDep,
    name: Annotated[str, Form(min_length=1, max_length=200)],
    user: CurrentUser,
    quality_mode: Annotated[str, Form()] = "",
    capture_kind: Annotated[str, Form()] = "video",
    file: Annotated[UploadFile | None, File()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> sc.SceneJobResponse:
    """Accept ``file`` (a video) or ``files[]`` (photos), then queue the stitch.

    Everything that touches the disk — the streamed copy and the ffprobe sniff — is pushed to
    a worker thread; a 500 MB upload must not stall the event loop that is streaming progress
    to every other client.
    """
    property = require_property(session, property_id, user)

    quality = (quality_mode or settings.default_quality_mode).strip().lower()
    kind = (capture_kind or "video").strip().lower()
    if quality not in QUALITY_MODES:
        raise ApiException(
            "invalid_quality_mode",
            f"'{quality}' is not a valid quality mode. Use one of: {', '.join(sorted(QUALITY_MODES))}.",
            f"'{quality}' geçerli bir kalite modu değil. Şunlardan birini kullanın: {', '.join(sorted(QUALITY_MODES))}.",
            status_code=422,
        )
    if kind not in CAPTURE_KINDS:
        raise ApiException(
            "invalid_capture_kind",
            f"'{kind}' is not a valid capture kind. Use 'video' or 'photos'.",
            f"'{kind}' geçerli bir çekim türü değil. 'video' veya 'photos' kullanın.",
            status_code=422,
        )

    uploads = [u for u in ([file] if file is not None else []) + list(files or []) if u is not None]
    uploads = [u for u in uploads if (u.filename or "").strip()]
    media = [u for u in uploads if not _is_sidecar(u)]
    sidecars = [u for u in uploads if _is_sidecar(u)]
    if not media:
        raise ApiException(
            "no_file",
            "No file was uploaded.",
            "Hiçbir dosya yüklenmedi.",
            hint="Send the capture as 'file' (video) or 'files' (photos).",
            status_code=422,
        )
    if kind == "video" and len(media) > MAX_VIDEO_SEGMENTS:
        raise ApiException(
            "too_many_files",
            f"A video scene takes at most {MAX_VIDEO_SEGMENTS} segments of one capture.",
            f"Bir video sahnesi en fazla {MAX_VIDEO_SEGMENTS} parçadan oluşabilir.",
            hint="Use capture_kind='photos' to upload a set of stills.",
            status_code=422,
        )
    if storage.free_bytes() < MIN_FREE_BYTES:
        raise ApiException(
            "insufficient_storage",
            "The server is out of disk space for new captures.",
            "Sunucuda yeni çekimler için disk alanı kalmadı.",
            status_code=507,
        )

    scene_id = new_id()
    saved: list[Path] = []
    probes: list[MediaProbe] = []
    try:
        for index, upload in enumerate(media):
            path = await run_in_threadpool(
                storage.save_upload,
                upload.file,
                scene_id=scene_id,
                filename=upload.filename or f"capture_{index}",
                index=index,
            )
            saved.append(path)
            probes.append(
                await run_in_threadpool(validate_media, path, expect=kind, settings=settings)
            )
        # The length gate belongs to the CAPTURE, not to any one file, and is therefore
        # checked here rather than inside `validate_media`. A guided capture legitimately
        # arrives in several segments — the recorder seals one at its ceiling, a call ends
        # the track, a stall restarts the encoder — and a two-minute capture was being
        # refused because its opening segment was 4.7 s long.
        if kind == "video":
            await run_in_threadpool(validate_capture_duration, probes, settings)
        # Written only once the footage has passed, so a rejected capture never leaves a
        # telemetry file behind with no video to explain it.
        for offset, upload in enumerate(sidecars):
            await run_in_threadpool(
                storage.save_upload,
                upload.file,
                scene_id=scene_id,
                filename=upload.filename or "motion.json",
                index=SIDECAR_INDEX_BASE + offset,
            )
    except Exception:
        # Never leave a rejected upload lying around; it is the largest kind of garbage
        # this service can accumulate.
        storage.delete_tree(settings.uploads_dir / scene_id)
        raise

    next_order = int(
        session.exec(select(func.coalesce(func.max(Scene.order), -1)).where(Scene.property_id == property_id)).one()
    ) + 1

    scene = Scene(
        id=scene_id,
        property_id=property.id,
        name=name.strip(),
        status="pending",
        capture_kind=kind,
        quality_mode=quality,
        order=next_order,
        source_paths=[str(p) for p in saved],
    )
    session.add(scene)
    property.updated_at = utcnow()
    session.add(property)
    session.commit()

    # submit() runs in its own transaction (the worker thread needs its own session), so the
    # scene must be committed before we hand the id over.
    job: Job = jobs.submit(scene.id)

    session.expire_all()
    session.refresh(scene)
    return sc.SceneJobResponse(scene=sc.scene_to_schema(scene, []), job=sc.job_to_schema(job))
