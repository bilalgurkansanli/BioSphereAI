"""Shared FastAPI dependencies and row lookups.

The ``require_*`` helpers exist so that every route raises the *same* bilingual 404 instead
of each handler inventing its own copy — the dashboard shows these strings directly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, Request
from sqlmodel import Session, select

from biosphere.core import auth as auth_lib
from biosphere.core.config import Settings, get_settings
from biosphere.core.db import get_session
from biosphere.core.jobs import JobManager, get_job_manager
from biosphere.core.models import (
    ApiException,
    AuthSession,
    CaptureGrant,
    Hotspot,
    Job,
    Property,
    Scene,
    ScenePhoto,
    User,
    ensure_utc,
)
from biosphere.core.storage import LocalStorage, get_storage

__all__ = [
    "SessionDep",
    "SettingsDep",
    "StorageDep",
    "JobsDep",
    "CurrentUser",
    "current_user",
    "SESSION_COOKIE",
    "require_property",
    "require_scene",
    "require_hotspot",
    "require_scene_photo",
    "require_job",
    "scenes_of_property",
    "hotspots_of_scene",
    "gpu_lock",
    "hotspots_by_scene",
    "photos_of_scene",
    "photos_by_scene",
]

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StorageDep = Annotated[LocalStorage, Depends(get_storage)]
JobsDep = Annotated[JobManager, Depends(get_job_manager)]

#: One model pass at a time in this process.
#:
#: NOT PER RESOURCE. Two diffusion passes on one card is not twice the throughput, it is an
#: out-of-memory error on an eight-gigabyte laptop GPU and a failed request for both callers.
#: It lives here rather than in a router because there are now several routes that reach for
#: the same card — restyling a photograph, restyling a sphere, emptying a room, emptying a
#: photograph — and a lock owned by one of them is a lock the others cannot see.
gpu_lock = asyncio.Lock()


def _not_found(kind_en: str, kind_tr: str, code: str) -> ApiException:
    return ApiException(
        code,
        f"{kind_en} was not found.",
        f"{kind_tr} bulunamadı.",
        status_code=404,
    )


#: Name of the cookie the login endpoints set. Read as a FALLBACK only — see
#: :func:`current_user` for why the header comes first.
SESSION_COOKIE = "bs_session"

#: ``last_seen_at`` is refreshed at most this often. Writing it on every request would put a
#: database write in front of every read, including the job-progress poll that runs twice a
#: second while a panorama is stitching.
_LAST_SEEN_THROTTLE = timedelta(hours=1)


def _unauthorised() -> ApiException:
    return ApiException(
        "unauthorized",
        "Sign in to continue.",
        "Devam etmek için giriş yapın.",
        status_code=401,
    )


def _bearer_token(request: Request) -> str | None:
    """The credential, from the header first and the cookie second.

    THE ORDER IS THE SECURITY PROPERTY, not a preference. A cookie is attached by the
    browser to any request that reaches this origin, including one a third-party page
    caused; an ``Authorization`` header is only ever set by our own code. Preferring the
    header means that in the deployment where both exist, a cross-site request forgery has
    nothing to forge with, because the header it would need cannot be set cross-origin
    without a CORS preflight this API does not grant.

    The cookie is still read, because it is what makes a same-origin deployment able to keep
    the token out of JavaScript's reach entirely.
    """
    header = request.headers.get("authorization")
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    cookie = request.cookies.get(SESSION_COOKIE)
    return cookie.strip() if cookie else None


#: Everything a capture link may do, as ``(method, route template below /api/v1)``.
#:
#: AN ALLOW-LIST OF ROUTES, NOT A DENY-LIST, and kept beside the one place credentials are
#: resolved. The private routers are mounted behind a single ``current_user`` dependency; a
#: grant that resolved to the owner and was then trusted by every handler would quietly be a
#: full sign-in in a QR code. So a grant is refused on every route not named here, including
#: the ones that do not exist yet — a new endpoint is unreachable from a phone until somebody
#: decides it should be.
#:
#: The list is what ``/cek`` actually calls: confirm the property, create a room for the 3D
#: walk, stage and assemble an upload (the chunked path and the single multipart POST), then
#: follow the job it started. Cancelling is included because the progress card offers it and a
#: phone that started a wrong capture should be able to stop it.
_CAPTURE_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/properties/{property_id}"),
        ("POST", "/properties/{property_id}/rooms"),
        ("POST", "/properties/{property_id}/scenes"),
        ("PUT", "/properties/uploads/{upload_id}/parts/{index}"),
        ("POST", "/properties/{property_id}/scenes/from-parts"),
        ("GET", "/properties/{property_id}/walkthrough"),
        ("POST", "/properties/{property_id}/walkthrough"),
        ("POST", "/properties/{property_id}/walkthrough/from-parts"),
        ("GET", "/scenes/{scene_id}"),
        ("GET", "/jobs/{job_id}"),
        ("GET", "/jobs/{job_id}/events"),
        ("POST", "/jobs/{job_id}/cancel"),
    }
)

#: The prefix ``api.main`` mounts every router under. Repeated rather than imported because
#: ``api.main`` imports this module.
_API_PREFIX = "/api/v1"


def _capture_scope_allows(request: Request, session: Session, grant: CaptureGrant) -> bool:
    """Whether this request is one a capture link for ``grant.property_id`` may make.

    Two checks, both required. The ROUTE must be on :data:`_CAPTURE_ROUTES` (matched on the
    template FastAPI routed to, so ``/scenes/x`` and ``/scenes/y`` are the same entry). And
    every id in the path must belong to the grant's property: a grant resolves to the owner,
    so without the second check it would read the owner's OTHER properties' scenes and jobs.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", "") or ""
    if template.startswith(_API_PREFIX):
        template = template[len(_API_PREFIX) :]
    if (request.method.upper(), template) not in _CAPTURE_ROUTES:
        return False

    params = request.path_params
    if "property_id" in params and params["property_id"] != grant.property_id:
        return False
    if "scene_id" in params:
        scene = session.get(Scene, params["scene_id"])
        if scene is None or scene.property_id != grant.property_id:
            return False
    if "job_id" in params:
        job = session.get(Job, params["job_id"])
        if job is None or job.property_id != grant.property_id:
            return False
    return True


def _user_from_capture_grant(request: Request, session: Session, token: str) -> User:
    """The owner a capture link acts for — or 401, for every way it can be wrong.

    A grant is not trusted for being in the table. Its creator must still exist, still be
    enabled and still OWN the property (a listing handed to a colleague takes its old QR codes
    with it), and the request must be in scope. Every failure is the same 401 the phone already
    turns into "this link has expired, scan a new one": telling a QR-code holder WHICH check
    failed tells them nothing they can act on and something about the account.
    """
    grant = session.exec(
        select(CaptureGrant).where(CaptureGrant.token_hash == auth_lib.token_digest(token))
    ).first()
    if grant is None:
        raise _unauthorised()

    now = datetime.now(UTC)
    if (ensure_utc(grant.expires_at) or now) <= now:
        session.delete(grant)
        session.commit()
        raise _unauthorised()

    user = session.get(User, grant.created_by)
    prop = session.get(Property, grant.property_id)
    if user is None or user.disabled or prop is None or prop.owner_id != user.id:
        raise _unauthorised()
    if not _capture_scope_allows(request, session, grant):
        raise _unauthorised()

    request.state.capture_property_id = grant.property_id
    return user


def _local_mode() -> bool:
    return str(get_settings().auth_mode).strip().lower() != "accounts"


def _local_owner(session: Session) -> User:
    """The installation's owner, for local mode: its oldest account, made on first use.

    The OLDEST account rather than a dedicated "local" one, so an installation that had
    accounts before keeps showing its owner's properties; an empty installation gets one
    account, which also adopts any property that predates accounts.
    """
    owner = session.exec(
        select(User).where(User.disabled == False).order_by(User.created_at)  # noqa: E712
    ).first()
    if owner is not None:
        return owner
    owner = User(email="local@biosphere.local", name="BioSphere")
    session.add(owner)
    session.flush()
    for prop in session.exec(select(Property).where(Property.owner_id.is_(None))).all():  # type: ignore[union-attr]
        prop.owner_id = owner.id
        session.add(prop)
    session.commit()
    session.refresh(owner)
    return owner


def current_user(request: Request, session: SessionDep) -> User:
    """Resolve the signed-in account, or raise 401.

    An expired session is deleted on sight rather than left to a sweep: this is the one
    moment the row is certainly in memory, and a table of dead sessions is a table somebody
    eventually has to explain.

    A token that is not a session may be a capture link (:class:`CaptureGrant`), which acts
    as the property's owner on the few routes a phone capture needs and nowhere else — see
    :data:`_CAPTURE_ROUTES`.
    """
    token = _bearer_token(request)
    if not token:
        if _local_mode():
            return _local_owner(session)
        raise _unauthorised()

    row = session.exec(
        select(AuthSession).where(AuthSession.token_hash == auth_lib.token_digest(token))
    ).first()
    if row is None:
        return _user_from_capture_grant(request, session, token)

    now = datetime.now(UTC)
    if (ensure_utc(row.expires_at) or now) <= now:
        session.delete(row)
        session.commit()
        raise _unauthorised()

    user = session.get(User, row.user_id)
    if user is None or user.disabled:
        raise _unauthorised()

    last_seen = ensure_utc(row.last_seen_at)
    if last_seen is None or now - last_seen > _LAST_SEEN_THROTTLE:
        row.last_seen_at = now
        session.add(row)
        session.commit()

    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require_property(session: Session, property_id: str, user: User) -> Property:
    """The property, if it exists AND belongs to this account.

    A property owned by somebody else answers 404, not 403. A 403 confirms that the id is
    real, which turns this endpoint into a way to test ids against the whole installation;
    the id is short and public tokens are not the only thing people paste into address bars.

    ``user`` is a required positional argument rather than an optional keyword on purpose: a
    call site that forgets it fails loudly at the first request instead of quietly returning
    somebody else's listing.
    """
    prop = session.get(Property, property_id)
    if prop is None or prop.owner_id != user.id:
        raise _not_found("Property", "Ev", "property_not_found")
    return prop


def _owns(session: Session, property_id: str, user: User) -> bool:
    """Whether this account owns the property a row hangs off.

    Every lookup below funnels through here rather than checking ``owner_id`` itself, so
    that ownership is decided in exactly one place. A row whose property has been deleted
    out from under it is not owned by anyone, which is the safe answer.
    """
    prop = session.get(Property, property_id)
    return prop is not None and prop.owner_id == user.id


def require_scene(session: Session, scene_id: str, user: User) -> Scene:
    scene = session.get(Scene, scene_id)
    if scene is None or not _owns(session, scene.property_id, user):
        raise _not_found("Scene", "Sahne", "scene_not_found")
    return scene


def require_hotspot(session: Session, hotspot_id: str, user: User) -> Hotspot:
    hotspot = session.get(Hotspot, hotspot_id)
    if hotspot is None:
        raise _not_found("Hotspot", "Etkileşim noktası", "hotspot_not_found")
    scene = session.get(Scene, hotspot.scene_id)
    if scene is None or not _owns(session, scene.property_id, user):
        raise _not_found("Hotspot", "Etkileşim noktası", "hotspot_not_found")
    return hotspot


def require_scene_photo(session: Session, photo_id: str, user: User) -> ScenePhoto:
    photo = session.get(ScenePhoto, photo_id)
    if photo is None:
        raise _not_found("Photo", "Fotoğraf", "photo_not_found")
    scene = session.get(Scene, photo.scene_id)
    if scene is None or not _owns(session, scene.property_id, user):
        raise _not_found("Photo", "Fotoğraf", "photo_not_found")
    return photo


def require_job(session: Session, job_id: str, user: User) -> Job:
    job = session.get(Job, job_id)
    if job is None or not _owns(session, job.property_id, user):
        raise _not_found("Job", "İş", "job_not_found")
    return job


def scenes_of_property(session: Session, property_id: str) -> list[Scene]:
    """Scenes in the order the player walks them."""
    return list(
        session.exec(
            select(Scene).where(Scene.property_id == property_id).order_by(Scene.order, Scene.created_at)  # type: ignore[arg-type]
        ).all()
    )


def hotspots_of_scene(session: Session, scene_id: str) -> list[Hotspot]:
    return list(
        session.exec(
            select(Hotspot).where(Hotspot.scene_id == scene_id).order_by(Hotspot.created_at)  # type: ignore[arg-type]
        ).all()
    )


def hotspots_by_scene(session: Session, scene_ids: list[str]) -> dict[str, list[Hotspot]]:
    """One query for a whole property; the detail endpoint would otherwise fan out per scene."""
    if not scene_ids:
        return {}
    rows = session.exec(
        select(Hotspot).where(Hotspot.scene_id.in_(scene_ids)).order_by(Hotspot.created_at)  # type: ignore[attr-defined]
    ).all()
    grouped: dict[str, list[Hotspot]] = {sid: [] for sid in scene_ids}
    for row in rows:
        grouped.setdefault(row.scene_id, []).append(row)
    return grouped


def photos_of_scene(session: Session, scene_id: str) -> list[ScenePhoto]:
    """A room's photographs, in the order they are displayed."""
    return list(
        session.exec(
            select(ScenePhoto)
            .where(ScenePhoto.scene_id == scene_id)
            .order_by(ScenePhoto.order, ScenePhoto.created_at)  # type: ignore[arg-type]
        ).all()
    )


def photos_by_scene(session: Session, scene_ids: list[str]) -> dict[str, list[ScenePhoto]]:
    """One query for a whole property, for the same reason :func:`hotspots_by_scene` exists."""
    if not scene_ids:
        return {}
    rows = session.exec(
        select(ScenePhoto)
        .where(ScenePhoto.scene_id.in_(scene_ids))  # type: ignore[attr-defined]
        .order_by(ScenePhoto.order, ScenePhoto.created_at)  # type: ignore[arg-type]
    ).all()
    grouped: dict[str, list[ScenePhoto]] = {sid: [] for sid in scene_ids}
    for row in rows:
        grouped.setdefault(row.scene_id, []).append(row)
    return grouped
