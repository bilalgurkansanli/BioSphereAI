"""Installation settings the dashboard can change: today, the person's own Anthropic key.

The project ships without a key, so the AI features (listing copy, the investment assistant)
run on whoever runs the engine. These routes let that person paste their key into "Ayarlar"
instead of editing ``engine/.env`` — see :mod:`biosphere.core.api_keys` for where it is kept
and how it ranks against the environment.

The key goes in and never comes back out: reads return whether a key is set, where it came
from, and its last four characters. In ``accounts`` mode only the installation's owner (its
oldest account) may change it, because the key pays for every account's calls.
"""

from __future__ import annotations

import os
from typing import Literal

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlmodel import select

from biosphere.api.deps import CurrentUser, SessionDep, _local_mode
from biosphere.core import api_keys
from biosphere.core.config import get_settings
from biosphere.core.models import ApiException, User

router = APIRouter(prefix="/settings", tags=["settings"])

KeySource = Literal["dashboard", "env", "none"]


class AiSettings(BaseModel):
    configured: bool
    source: KeySource = Field(description="Where the key in use comes from.")
    hint: str = Field(default="", description="`sk-ant-…` plus the last four characters.")
    can_edit: bool
    listing_model: str
    advisor_model: str


class AiKeyIn(BaseModel):
    api_key: str = Field(min_length=1, max_length=512)


class AiKeyCheck(BaseModel):
    ok: bool
    message: str


def _env_key() -> str:
    s = get_settings()
    return (
        (s.listing_api_key or "").strip()
        or (s.advisor_api_key or "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )


def _can_edit(user: User, session: SessionDep) -> bool:
    if _local_mode():
        return True
    owner = session.exec(
        select(User).where(User.disabled == False).order_by(User.created_at)  # noqa: E712
    ).first()
    return owner is not None and owner.id == user.id


def _read(user: User, session: SessionDep) -> AiSettings:
    s = get_settings()
    stored, env = api_keys.stored_api_key(), _env_key()
    key = stored or env
    return AiSettings(
        configured=bool(key),
        source="dashboard" if stored else ("env" if env else "none"),
        hint=api_keys.key_hint(key),
        can_edit=_can_edit(user, session),
        listing_model=s.listing_model,
        advisor_model=s.advisor_model,
    )


def _require_editor(user: User, session: SessionDep) -> None:
    if not _can_edit(user, session):
        raise ApiException(
            "settings_owner_only",
            "Only the installation's owner can change the API key.",
            "API anahtarını yalnızca kurulumun sahibi değiştirebilir.",
            status_code=403,
        )


@router.get("/ai", response_model=AiSettings)
def read_ai_settings(user: CurrentUser, session: SessionDep) -> AiSettings:
    return _read(user, session)


@router.put("/ai", response_model=AiSettings)
def save_ai_key(body: AiKeyIn, user: CurrentUser, session: SessionDep) -> AiSettings:
    _require_editor(user, session)
    if not api_keys.looks_like_key(body.api_key):
        raise ApiException(
            "invalid_api_key",
            "That does not look like an Anthropic API key (it starts with sk-ant-).",
            "Bu bir Anthropic API anahtarına benzemiyor (sk-ant- ile başlar).",
            status_code=422,
        )
    api_keys.save_api_key(body.api_key)
    return _read(user, session)


@router.delete("/ai", response_model=AiSettings)
def clear_ai_key(user: CurrentUser, session: SessionDep) -> AiSettings:
    _require_editor(user, session)
    api_keys.clear_api_key()
    return _read(user, session)


@router.post("/ai/check", response_model=AiKeyCheck)
def check_ai_key(user: CurrentUser, session: SessionDep) -> AiKeyCheck:
    """Ask Anthropic whether the key in use is accepted. Lists models, which costs nothing."""
    key = get_settings().listing_key
    if not key:
        return AiKeyCheck(ok=False, message="Henüz bir API anahtarı girilmedi.")
    try:
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            params={"limit": 1},
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=10.0,
        )
    except httpx.HTTPError:
        return AiKeyCheck(ok=False, message="Anthropic'e ulaşılamadı; internet bağlantısını kontrol edin.")
    if response.status_code == 200:
        return AiKeyCheck(ok=True, message="Anahtar çalışıyor.")
    if response.status_code in (401, 403):
        return AiKeyCheck(ok=False, message="Anthropic bu anahtarı kabul etmedi.")
    return AiKeyCheck(ok=False, message=f"Anthropic beklenmeyen bir yanıt verdi ({response.status_code}).")
