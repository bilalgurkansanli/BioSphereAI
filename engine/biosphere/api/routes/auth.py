"""Accounts: register, sign in, sign out, and Google.

Four things here were decided rather than defaulted.

*The wrong e-mail and the wrong password give the same answer.* One message, one status, one
error code, and the same amount of work done before answering. Splitting them is friendlier
by about one second and hands anyone with a list of addresses a way to find out which of
them are customers of an estate-agency product.

*Google is verified by calling Google, not by parsing a token.* The callback exchanges the
authorization code for an access token over TLS with a client secret, then asks
``oauth2/v3/userinfo`` who that token belongs to. That answer comes straight from Google on a
connection we opened, so it needs no signature check — which is the whole reason this file
carries no JWT library and no JWKS cache, each of which is a way to be subtly wrong about
cryptography for no benefit here.

*Accounts are joined on Google's subject id, then on the e-mail address.* A person who
registered with a password and later presses "Google ile giriş" must land on their own
properties. The address is what identifies them the first time; the ``sub`` is what
identifies them afterwards, because an address can be changed and recycled and a ``sub``
cannot.

*The browser never receives the token in a redirect URL.* Google sends the browser back to
the engine; the engine hands the web app a ONE-TIME code with a sixty-second life, and the
web app trades it for the session token in a POST. A token in a query string ends up in
history, in the ``Referer`` header and in every proxy log between here and the user.
"""

from __future__ import annotations

import re
import secrets
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select

from biosphere.api.deps import SESSION_COOKIE, CurrentUser, SessionDep, SettingsDep, current_user
from biosphere.core import auth as auth_lib
from biosphere.core.config import Settings
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException, AuthSession, Property, User, ensure_utc, utcnow

router = APIRouter(prefix="/auth", tags=["auth"])

log = get_logger("biosphere.auth")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

#: Seconds a pending Google sign-in, and the one-time code it produces, stay usable.
_STATE_TTL = 600
_CODE_TTL = 60

#: In-process, because this engine is in-process: one uvicorn worker owns the job queue, the
#: SSE loop and the SQLite connection pool, and that assumption is stated all over
#: ``core/jobs.py``. If it is ever run behind several workers these two dicts move to the
#: database — and so does a great deal else.
_pending_states: dict[str, float] = {}
_pending_codes: dict[str, tuple[str, float]] = {}


def _sweep(now: float) -> None:
    for key, expiry in list(_pending_states.items()):
        if expiry <= now:
            _pending_states.pop(key, None)
    for key, (_token, expiry) in list(_pending_codes.items()):
        if expiry <= now:
            _pending_codes.pop(key, None)


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------


class UserRead(BaseModel):
    id: str
    email: str
    name: str
    avatar_url: str | None = None
    role: str
    has_password: bool
    has_google: bool
    created_at: datetime
    #: True when the dashboard was opened without signing in, in the engine's local mode
    #: (``BIOSPHERE_AUTH_MODE=local``): there is nothing to sign out of.
    local: bool = False


class SessionRead(BaseModel):
    """What a successful sign-in returns. ``token`` is shown exactly once."""

    user: UserRead
    token: str
    expires_at: datetime


#: Deliberately permissive: one "@", something on either side, a dot in the domain, a sane
#: length. Full RFC 5322 validation is a well-known rabbit hole that still cannot tell a
#: deliverable address from an undeliverable one — only sending mail does that. The job here
#: is to catch a typo and to refuse something that is plainly not an address, without adding
#: a dependency to do it.
_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s.]+(\.[^@\s.]+)+$")


def _check_email(value: str) -> str:
    value = value.strip()
    if len(value) > 320 or not _EMAIL_RE.match(value):
        raise ValueError("that does not look like an e-mail address")
    return value


class RegisterRequest(BaseModel):
    email: str
    name: str = Field(min_length=1, max_length=120)
    password: str

    _email = field_validator("email")(_check_email)


class LoginRequest(BaseModel):
    email: str
    password: str

    _email = field_validator("email")(_check_email)


class ExchangeRequest(BaseModel):
    code: str


def _to_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        role=user.role,
        has_password=bool(user.password_hash),
        has_google=bool(user.google_sub),
        created_at=ensure_utc(user.created_at) or utcnow(),
    )


# --------------------------------------------------------------------------------------
# Session plumbing
# --------------------------------------------------------------------------------------


def _bad_credentials() -> ApiException:
    return ApiException(
        "invalid_credentials",
        "That e-mail address and password do not match an account.",
        "Bu e-posta ve şifre bir hesapla eşleşmiyor.",
        status_code=401,
    )


def _issue_session(session: Session, user: User, response: Response, settings: Settings) -> SessionRead:
    token = auth_lib.new_session_token()
    expires_at = auth_lib.session_expiry()
    session.add(
        AuthSession(user_id=user.id, token_hash=auth_lib.token_digest(token), expires_at=expires_at)
    )
    user.last_login_at = utcnow()
    user.updated_at = utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)

    # Also set the cookie. It is only stored by the browser on a same-origin (or
    # credentialed) response, so in the split-origin development setup it is quietly
    # ignored and the header carries the session instead. A deployment that puts the engine
    # behind the site's own domain gets an httpOnly session for free, with no client change.
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(auth_lib.SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=not settings.env.lower().startswith("dev"),
        path="/",
    )
    return SessionRead(user=_to_read(user), token=token, expires_at=expires_at)


def _adopt_ownerless_properties(session: Session, user: User) -> int:
    """Give the first account everything that predates accounts.

    Runs once per installation, on the first user ever created. Any later user starts empty
    — guessing that the second person to register owns the first person's listings would be
    a data leak dressed up as a convenience.
    """
    others = session.exec(select(User).where(User.id != user.id)).first()
    if others is not None:
        return 0
    orphans = list(session.exec(select(Property).where(Property.owner_id.is_(None))).all())  # type: ignore[union-attr]
    for prop in orphans:
        prop.owner_id = user.id
        session.add(prop)
    if orphans:
        session.commit()
        log.info("auth.adopted_ownerless_properties", user=user.id, count=len(orphans))
    return len(orphans)


# --------------------------------------------------------------------------------------
# E-mail and password
# --------------------------------------------------------------------------------------


@router.post("/register", response_model=SessionRead, status_code=201, summary="Create an account.")
def register(payload: RegisterRequest, response: Response, session: SessionDep, settings: SettingsDep) -> SessionRead:
    if not settings.allow_registration:
        raise ApiException(
            "registration_closed",
            "New accounts are not open on this installation.",
            "Bu kurulumda yeni kayıt açık değil.",
            status_code=403,
        )

    email = auth_lib.normalise_email(str(payload.email))
    try:
        auth_lib.check_password_policy(payload.password)
    except auth_lib.PasswordPolicyError as exc:
        raise ApiException("weak_password", exc.message_en, exc.message_tr, status_code=422) from exc

    existing = session.exec(select(User).where(User.email == email)).first()
    if existing is not None:
        # Deliberately explicit. Registration cannot hide that an address is taken — the
        # form has to say something — and pretending otherwise only produces an account the
        # person cannot sign into. The enumeration defence lives on the LOGIN path, where it
        # can actually hold.
        raise ApiException(
            "email_taken",
            "An account already exists for that e-mail address.",
            "Bu e-posta adresiyle bir hesap zaten var.",
            status_code=409,
        )

    user = User(
        email=email,
        name=payload.name.strip(),
        password_hash=auth_lib.hash_password(payload.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    _adopt_ownerless_properties(session, user)
    return _issue_session(session, user, response, settings)


@router.post("/login", response_model=SessionRead, summary="Sign in with an e-mail and password.")
def login(payload: LoginRequest, response: Response, session: SessionDep, settings: SettingsDep) -> SessionRead:
    email = auth_lib.normalise_email(str(payload.email))
    user = session.exec(select(User).where(User.email == email)).first()

    # Called even when there is no user: see `auth.verify_password`.
    ok = auth_lib.verify_password(payload.password, user.password_hash if user else None)
    if user is None or not ok or user.disabled:
        raise _bad_credentials()

    if auth_lib.needs_rehash(user.password_hash):
        user.password_hash = auth_lib.hash_password(payload.password)

    return _issue_session(session, user, response, settings)


@router.post("/logout", status_code=204, summary="End the current session.")
def logout(request: Request, response: Response, session: SessionDep) -> Response:
    """Revokes the presented session and clears the cookie. Never fails.

    A sign-out that can return an error is a sign-out a user cannot rely on; if the token is
    already gone, the desired state has been reached by another route.
    """
    header = request.headers.get("authorization", "")
    token = header.partition(" ")[2].strip() if header.lower().startswith("bearer ") else None
    token = token or request.cookies.get(SESSION_COOKIE)
    if token:
        row = session.exec(
            select(AuthSession).where(AuthSession.token_hash == auth_lib.token_digest(token))
        ).first()
        if row is not None:
            session.delete(row)
            session.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.status_code = 204
    return response


@router.get("/me", response_model=UserRead, summary="The signed-in account.")
def me(request: Request, user: CurrentUser) -> UserRead:
    read = _to_read(user)
    read.local = not request.headers.get("authorization") and not request.cookies.get(SESSION_COOKIE)
    return read


@router.post("/exchange", response_model=SessionRead, summary="Trade a one-time sign-in code for a session.")
def exchange(payload: ExchangeRequest, response: Response, session: SessionDep, settings: SettingsDep) -> SessionRead:
    now = time.time()
    _sweep(now)
    entry = _pending_codes.pop(payload.code, None)
    if entry is None:
        raise ApiException(
            "invalid_code",
            "That sign-in link has expired. Please try again.",
            "Bu giriş bağlantısının süresi doldu. Lütfen tekrar deneyin.",
            status_code=400,
        )
    user_id, expiry = entry
    if expiry <= now:
        raise ApiException(
            "invalid_code",
            "That sign-in link has expired. Please try again.",
            "Bu giriş bağlantısının süresi doldu. Lütfen tekrar deneyin.",
            status_code=400,
        )
    user = session.get(User, user_id)
    if user is None or user.disabled:
        raise _bad_credentials()
    return _issue_session(session, user, response, settings)


# --------------------------------------------------------------------------------------
# Google
# --------------------------------------------------------------------------------------


def _google_enabled(settings: Settings) -> bool:
    return bool(settings.google_client_id.strip() and settings.google_client_secret.strip())


def _google_redirect_uri(settings: Settings) -> str:
    configured = settings.google_redirect_url.strip()
    if configured:
        return configured
    return f"{settings.public_base_url.rstrip('/')}/api/v1/auth/google/callback"


@router.get("/google/status", summary="Whether Google sign-in is configured here.")
def google_status(settings: SettingsDep) -> dict[str, bool]:
    """Lets the login screen decide whether to draw the button.

    A button that leads to a 501 is worse than no button, and whether the operator has
    configured Google is not a secret.
    """
    return {"enabled": _google_enabled(settings)}


@router.get("/google/start", summary="Begin the Google sign-in redirect.")
def google_start(settings: SettingsDep) -> Response:
    if not _google_enabled(settings):
        raise ApiException(
            "google_not_configured",
            "Google sign-in is not configured on this installation.",
            "Bu kurulumda Google ile giriş yapılandırılmamış.",
            status_code=501,
        )
    now = time.time()
    _sweep(now)
    state = secrets.token_urlsafe(24)
    _pending_states[state] = now + _STATE_TTL

    query = urlencode(
        {
            "client_id": settings.google_client_id.strip(),
            "redirect_uri": _google_redirect_uri(settings),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            # Google returns a refresh token only with consent+offline, and this product has
            # no use for one: it never acts on the user's Google account, it only wants to
            # know who they are, once.
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    return Response(status_code=307, headers={"location": f"{GOOGLE_AUTH_URL}?{query}"})


async def _google_identity(code: str, settings: Settings) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id.strip(),
                "client_secret": settings.google_client_secret.strip(),
                "redirect_uri": _google_redirect_uri(settings),
                "grant_type": "authorization_code",
            },
        )
        if token_response.status_code != 200:
            log.warning("auth.google_token_failed", status=token_response.status_code)
            raise ApiException(
                "google_failed",
                "Google could not complete the sign-in.",
                "Google ile giriş tamamlanamadı.",
                status_code=502,
            )
        access_token = token_response.json().get("access_token")
        if not access_token:
            raise ApiException(
                "google_failed",
                "Google could not complete the sign-in.",
                "Google ile giriş tamamlanamadı.",
                status_code=502,
            )
        info = await client.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        if info.status_code != 200:
            raise ApiException(
                "google_failed",
                "Google could not complete the sign-in.",
                "Google ile giriş tamamlanamadı.",
                status_code=502,
            )
        return info.json()


@router.get("/google/callback", include_in_schema=False)
async def google_callback(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> Response:
    """Where Google sends the browser back. Always ends in a redirect to the web app."""
    web = settings.web_base_url.rstrip("/")

    def back(params: dict[str, str]) -> Response:
        return Response(status_code=307, headers={"location": f"{web}/giris?{urlencode(params)}"})

    if error or not code or not state:
        # `access_denied` is the user pressing "cancel" on Google's own screen; it is not a
        # failure and must not be shown as one.
        return back({"google": "cancelled" if error == "access_denied" else "failed"})

    now = time.time()
    _sweep(now)
    if _pending_states.pop(state, None) is None:
        return back({"google": "failed"})

    try:
        identity = await _google_identity(code, settings)
    except ApiException:
        return back({"google": "failed"})

    sub = str(identity.get("sub") or "").strip()
    email = auth_lib.normalise_email(str(identity.get("email") or ""))
    if not sub or not email or identity.get("email_verified") is False:
        # An unverified Google address must never be allowed to claim a password account:
        # that is the classic account takeover for exactly this flow.
        return back({"google": "failed"})

    user = session.exec(select(User).where(User.google_sub == sub)).first()
    if user is None:
        user = session.exec(select(User).where(User.email == email)).first()
        if user is not None:
            user.google_sub = sub
        else:
            if not settings.allow_registration:
                return back({"google": "closed"})
            user = User(
                email=email,
                name=str(identity.get("name") or email.split("@")[0]).strip()[:120],
                google_sub=sub,
                avatar_url=str(identity.get("picture") or "") or None,
            )
        session.add(user)
        session.commit()
        session.refresh(user)
        _adopt_ownerless_properties(session, user)

    if user.disabled:
        return back({"google": "failed"})

    one_time = secrets.token_urlsafe(24)
    _pending_codes[one_time] = (user.id, time.time() + _CODE_TTL)
    return back({"code": one_time})


__all__ = ["router", "current_user", "UserRead", "SessionRead"]
