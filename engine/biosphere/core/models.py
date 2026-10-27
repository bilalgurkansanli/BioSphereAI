"""Persistence model — the four tables behind the API contract in ``web/src/lib/types.ts``.

Design notes
------------
*Ids are short URL-safe tokens*, not autoincrement integers: they end up in embed URLs and
in directory names on disk, and sequential integers would leak how many properties exist and let
anyone enumerate them.

*Rich sub-objects are stored as JSON columns.* ``panorama``, ``coverage``, ``metrics`` and
``initial_view`` are written once by the worker and read as whole documents by the UI; there
is no query that filters on their innards, so normalising them into extra tables would buy
nothing and cost a join per scene. They are always **assigned as whole dicts** (never mutated
in place) because SQLAlchemy does not track mutation inside a plain JSON column.

*Timestamps are timezone-aware UTC.* SQLite has no native timestamp type and hands back naive
datetimes, so every read goes through :func:`ensure_utc` before it reaches the API layer,
which serialises with a trailing ``Z``.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

__all__ = [
    "ApiException",
    "User",
    "AuthSession",
    "Property",
    "Scene",
    "ScenePhoto",
    "Hotspot",
    "Job",
    "Walkthrough",
    "AdvisorUsage",
    "new_id",
    "new_public_token",
    "utcnow",
    "ensure_utc",
    "DEFAULT_INITIAL_VIEW",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def new_id() -> str:
    """12-character URL-safe id. 72 bits of entropy — unguessable, still short enough to log."""
    return secrets.token_urlsafe(9)


def new_public_token() -> str:
    """Token that appears in the public embed URL; 128 bits, because it *is* the access control."""
    return secrets.token_urlsafe(16)


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to naive datetimes coming back out of SQLite."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


DEFAULT_INITIAL_VIEW: dict[str, float] = {"yaw": 0.0, "pitch": 0.0, "fov": 75.0}


class ApiException(Exception):
    """A failure that is safe — and useful — to show the end user, in both languages.

    Lives in the model layer rather than the API layer so that storage, the job worker and
    the routes can all raise the same thing without importing FastAPI. ``api.main`` turns it
    into the ``ApiError`` JSON body from ``types.ts``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        message_tr: str,
        *,
        hint: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.message_tr = message_tr
        self.hint = hint
        self.status_code = status_code

    def payload(self) -> dict[str, Any]:
        """The exact ``ApiError`` shape the frontend expects."""
        return {
            "code": self.code,
            "message": self.message,
            "message_tr": self.message_tr,
            "hint": self.hint,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ApiException(code={self.code!r}, status={self.status_code}, message={self.message!r})"


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


class Property(SQLModel, table=True):
    """A property. Owns scenes; the unit that gets published and embedded."""

    __tablename__ = "property"

    id: str = Field(default_factory=new_id, primary_key=True)
    #: Who may see and change it. NULLABLE, and that is a migration decision rather than a
    #: model one: the properties that existed before accounts did have no owner, and there
    #: is no honest way for code to guess whose they are. An ownerless row is invisible to
    #: every API caller — see ``deps.require_property`` — until it is adopted, which happens
    #: once, automatically, when the first account on the installation is created.
    owner_id: str | None = Field(
        default=None,
        sa_column=Column(String(32), ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True),
    )
    title: str = Field(sa_column=Column(String(200), nullable=False))
    description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    address: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    public_token: str = Field(
        default_factory=new_public_token,
        sa_column=Column(String(64), nullable=False, unique=True, index=True),
    )
    published: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, default=False))
    cover_scene_id: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    views: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    #: The agent's figures for the investment assistant (price, rent, aidat, location, the
    #: opt-in) — ``biosphere.invest.inputs.InvestmentInputs``, written whole. A JSON document
    #: rather than twenty columns because nothing filters on its innards and the shape will
    #: grow; nullable, so the upgrade is one ``ADD COLUMN`` and every existing property simply
    #: has the assistant off.
    investment: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Scene(SQLModel, table=True):
    """One ROOM inside a property: a 360 viewpoint, a set of photographs, or both.

    It began as "one 360 viewpoint" and grew into the room because that is what a listing is
    made of — an agent shoots the living room's panorama and also takes four photographs of
    it, and both belong to the same room rather than to two unrelated records. A room with no
    capture is legitimate and reaches ``status = "ready"`` without ever having a job:
    ``panorama`` stays null and :class:`ScenePhoto` carries the content.
    """

    __tablename__ = "scene"

    id: str = Field(default_factory=new_id, primary_key=True)
    property_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("property.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    name: str = Field(sa_column=Column(String(200), nullable=False))
    status: str = Field(default="pending", sa_column=Column(String(16), nullable=False, index=True))
    #: ``video`` | ``photos`` | ``none``. ``none`` is a room created without a capture, which
    #: has no job and no panorama and is ready the moment it exists.
    capture_kind: str = Field(default="video", sa_column=Column(String(16), nullable=False))
    quality_mode: str = Field(default="balanced", sa_column=Column(String(16), nullable=False))
    # ``order`` is a reserved SQL word; the column is named explicitly to keep the generated
    # DDL unambiguous while the API still exposes the field as ``order`` per types.ts.
    order: int = Field(default=0, sa_column=Column("sort_order", Integer, nullable=False, default=0))

    panorama: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    coverage: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    initial_view: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_INITIAL_VIEW), sa_column=Column(JSON, nullable=False)
    )
    error: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    #: The room's 3D area, delivered by the property's walkthrough job: ``{status, url, thumbnail_url,
    #: splat_count, bytes, placed, error}``. Independent of ``panorama`` — a room may have a
    #: 360, a splat, both or neither. Written whole, like every JSON column here.
    walkthrough: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Internal bookkeeping — deliberately absent from the public schema.
    source_paths: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    #: The room's walkthrough video(s); separate from ``source_paths`` because those feed the
    #: stitch and these feed the splat, and reprocessing one must not touch the other.
    walk_source_paths: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    active_job_id: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class ScenePhoto(SQLModel, table=True):
    """An ordinary photograph attached to a room.

    ITS OWN TABLE RATHER THAN A JSON COLUMN ON ``Scene``, and that is the one design decision
    here worth stating. The other rich sub-objects on a scene — ``panorama``, ``coverage``,
    ``metrics`` — are written once by the worker and read as whole documents, which is exactly
    what a JSON column is good at. Photographs are not like that: they are added a few at a
    time, deleted individually and reordered, and SQLAlchemy does not track mutation inside a
    plain JSON column, so every one of those operations would be a read-modify-assign of the
    whole list with a lost update waiting in it the first time two tabs are open.

    The bytes live under the scene's own directory, so deleting a property still takes its
    photographs with it through the same ``delete_tree`` that removes its panoramas.
    """

    __tablename__ = "scene_photo"

    id: str = Field(default_factory=new_id, primary_key=True)
    scene_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("scene.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    #: File name inside ``<scene dir>/photos``. Stored rather than derived so the extension is
    #: whatever the encoder chose.
    filename: str = Field(sa_column=Column(String(128), nullable=False))
    width: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    height: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    caption: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # ``order`` is a reserved SQL word; named explicitly, as on ``Scene``.
    order: int = Field(default=0, sa_column=Column("sort_order", Integer, nullable=False, default=0))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Hotspot(SQLModel, table=True):
    """A clickable marker pinned to a direction on the sphere."""

    __tablename__ = "hotspot"

    id: str = Field(default_factory=new_id, primary_key=True)
    scene_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("scene.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    type: str = Field(default="info", sa_column=Column(String(16), nullable=False))
    yaw: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    pitch: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    title: str = Field(sa_column=Column(String(200), nullable=False))
    body: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    icon: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    target_scene_id: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    media_url: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    accent: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class Job(SQLModel, table=True):
    """One run of the stitch pipeline for one scene.

    The row is the single source of truth for progress: the SSE stream is only a push
    notification of what is already persisted, so a client that reconnects (or connects late)
    can poll ``GET /jobs/{id}`` and see exactly the same state.
    """

    __tablename__ = "job"

    id: str = Field(default_factory=new_id, primary_key=True)
    property_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("property.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    #: ``None`` for a property-level job (the walkthrough), which has no single scene.
    scene_id: str | None = Field(
        default=None,
        sa_column=Column(
            String(32), ForeignKey("scene.id", ondelete="CASCADE"), nullable=True, index=True
        ),
    )
    #: ``stitch`` (one scene's panorama) or ``walkthrough`` (the property's 3D areas).
    kind: str = Field(default="stitch", sa_column=Column(String(16), nullable=False, default="stitch"))
    status: str = Field(default="queued", sa_column=Column(String(16), nullable=False, index=True))
    stage: str = Field(default="upload", sa_column=Column(String(16), nullable=False))
    progress: float = Field(default=0.0, sa_column=Column(Float, nullable=False, default=0.0))
    stage_progress: float = Field(default=0.0, sa_column=Column(Float, nullable=False, default=0.0))
    message: str = Field(default="Queued", sa_column=Column(Text, nullable=False))
    message_tr: str = Field(default="Sırada", sa_column=Column(Text, nullable=False))
    eta_seconds: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    error: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Internal: what to feed the pipeline on (re)run.
    quality_mode: str = Field(default="balanced", sa_column=Column(String(16), nullable=False))
    capture_kind: str = Field(default="video", sa_column=Column(String(16), nullable=False))

    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}


class Walkthrough(SQLModel, table=True):
    """The property's 3D walkthrough: one general video plus one video per room, built together.

    ONE ROW PER PROPERTY, because the rooms are placed in the general walkthrough's frame and a
    room built without it has nowhere to stand. The room videos themselves belong to the
    :class:`Scene` rows (``walk_source_paths``) so a room keeps its 360 and its 3D area side
    by side; this row owns the general video, the job, and the manifest the viewer reads.
    """

    __tablename__ = "walkthrough"

    id: str = Field(default_factory=new_id, primary_key=True)
    property_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("property.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
        )
    )
    status: str = Field(default="pending", sa_column=Column(String(16), nullable=False, index=True))
    quality_mode: str = Field(default="balanced", sa_column=Column(String(16), nullable=False))
    #: The general walkthrough video(s). Empty when the property is a single room.
    base_source_paths: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    #: The rooms, in the order they were given; their videos live on the scenes.
    room_scene_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    #: The served manifest, with media URLs already resolved. Written whole by the worker.
    manifest: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    metrics: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    error: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    active_job_id: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class User(SQLModel, table=True):
    """An account. Owns properties; nothing else in the schema outlives one.

    ``password_hash`` and ``google_sub`` are both nullable and at least one of them is set:
    an account created with an e-mail and a password has the first, one created through
    Google has the second, and an account that has done both has both — which is the point,
    because the same person signing in the other way must land on the same properties rather
    than on a second, empty account. The e-mail address is what joins them, stored folded so
    that ``Ali@`` and ``ali@`` cannot become two rows.
    """

    __tablename__ = "user"

    id: str = Field(default_factory=new_id, primary_key=True)
    email: str = Field(sa_column=Column(String(320), nullable=False, unique=True, index=True))
    name: str = Field(sa_column=Column(String(120), nullable=False))
    #: scrypt, in the format written by :mod:`biosphere.core.auth`. Null for Google-only.
    password_hash: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    #: Google's stable subject id. NOT the e-mail: a Google account can change its address,
    #: and matching on the address alone is how one person's listings end up under another's
    #: login after an address is recycled.
    google_sub: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True, unique=True, index=True)
    )
    avatar_url: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    #: ``owner`` today. Reserved so that adding ``admin`` later is a value, not a migration.
    role: str = Field(default="owner", sa_column=Column(String(16), nullable=False, default="owner"))
    disabled: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, default=False))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    last_login_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class AuthSession(SQLModel, table=True):
    """One signed-in browser.

    Named ``auth_session`` rather than ``session`` because ``session`` is the name of the
    SQLAlchemy object this module is full of, and a table that shadows it makes every
    traceback one degree harder to read.

    The row holds the DIGEST of the token, never the token — see :mod:`biosphere.core.auth`.
    It also holds no user agent and no IP address: the privacy policy says the product does
    not keep them, and the cheapest way to keep that promise true is to have nowhere to put
    them.
    """

    __tablename__ = "auth_session"

    id: str = Field(default_factory=new_id, primary_key=True)
    user_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    token_hash: str = Field(sa_column=Column(String(64), nullable=False, unique=True, index=True))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    expires_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False, index=True))
    last_seen_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )


class CaptureGrant(SQLModel, table=True):
    """A capture link: lets ONE phone upload footage to ONE property without signing in.

    The capture page opens on a different device from the one that made the QR code — that is
    its whole point — and before this table existed the phone met a sign-in form first. An
    agent standing in the client's flat with the camera app half-open does not want to type a
    password on a phone, and the old code admitted as much (``app/cek/layout.tsx``).

    A grant is a bearer token with a deliberately narrow reach, decided in
    :func:`biosphere.api.deps.current_user`, not here: it resolves to the property's owner, but
    only on the handful of routes a capture needs (read the property, stage and assemble an
    upload, follow its job), and only for THIS property. It cannot list properties, read a
    listing, delete anything or reach the AI tools. It expires on its own
    (:data:`biosphere.core.auth.CAPTURE_GRANT_TTL`) and dies with the property.

    Like :class:`AuthSession`, the row holds the DIGEST of the token, never the token: the
    plain value lives only in the QR code the dashboard drew.
    """

    __tablename__ = "capture_grant"

    id: str = Field(default_factory=new_id, primary_key=True)
    property_id: str = Field(
        sa_column=Column(
            String(32), ForeignKey("property.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    #: Who drew the QR. The grant acts as this account, so a grant whose creator no longer
    #: owns the property (or no longer exists) is refused — see ``deps.current_user``.
    created_by: str = Field(
        sa_column=Column(String(32), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    token_hash: str = Field(sa_column=Column(String(64), nullable=False, unique=True, index=True))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    expires_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False, index=True))


class AdvisorUsage(SQLModel, table=True):
    """One billed call the investment assistant made to the hosted model.

    What the daily budget is enforced on and what the dashboard's cost summary adds up, so it is
    a row per CALL (a visitor's question may take two or three: search, compute, answer), with
    the tokens split the way they are billed.

    IT HOLDS NO CONTENT AND NO VISITOR. The visitor's question, the answer and their IP address
    are not here — the conversation itself lives only in the engine's memory and expires with
    it — so the privacy policy's "ziyaretçiye ait hiçbir bilgi tutulmaz" stays true.
    ``conversation_id`` is a random handle that groups the calls of one chat and identifies no
    one.

    ``property_id`` is SET NULL on delete rather than cascading: the money was spent whether or
    not the listing still exists, and the installation-wide daily cap has to keep counting it.
    """

    __tablename__ = "advisor_usage"

    id: str = Field(default_factory=new_id, primary_key=True)
    property_id: str | None = Field(
        default=None,
        sa_column=Column(String(32), ForeignKey("property.id", ondelete="SET NULL"), nullable=True, index=True),
    )
    conversation_id: str = Field(sa_column=Column(String(40), nullable=False, index=True))
    #: The visitor's question number within the conversation, and the call number within it.
    turn: int = Field(default=1, sa_column=Column(Integer, nullable=False, default=1))
    round: int = Field(default=1, sa_column=Column(Integer, nullable=False, default=1))
    #: The model that served the call (a fallback model when one did).
    model: str = Field(sa_column=Column(String(64), nullable=False))
    input_tokens: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    output_tokens: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    cache_read_tokens: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    cache_write_tokens: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    web_search_requests: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    cost_usd: float = Field(default=0.0, sa_column=Column(Float, nullable=False, default=0.0))
    latency_ms: int = Field(default=0, sa_column=Column(Integer, nullable=False, default=0))
    #: The API's ``stop_reason`` (``end_turn``, ``tool_use``, ``refusal`` ...), or ``aborted``
    #: when the visitor left mid-answer and the call was cut off.
    stop_reason: str = Field(default="", sa_column=Column(String(32), nullable=False, default=""))
    created_at: datetime = Field(
        default_factory=utcnow, sa_column=Column(DateTime(timezone=True), nullable=False, index=True)
    )
