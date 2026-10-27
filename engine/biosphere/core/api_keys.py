"""The Anthropic key a person enters in the dashboard, kept on their own machine.

This is an open-source project: nobody ships a key with it, and nobody's features should run
on somebody else's account. So the key the AI features use is the one the person running the
engine enters under "Ayarlar" — stored in ``<media_root>/config/api_keys.json``, next to the
database and outside the repository, never returned by the API in full.

Precedence, most specific first:

1. the key saved from the dashboard (this module);
2. ``BIOSPHERE_LISTING_API_KEY`` / ``BIOSPHERE_ADVISOR_API_KEY`` in ``engine/.env``;
3. ``ANTHROPIC_API_KEY`` from the environment.

The dashboard wins because it is the one a person can see and change without restarting the
engine; the file is re-read whenever it changes on disk.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

__all__ = ["KEY_PREFIX", "clear_api_key", "key_hint", "looks_like_key", "save_api_key", "stored_api_key"]

#: Every Anthropic API key starts with this. Checked so a pasted e-mail or a half-copied key
#: is refused at the form, not discovered at the first paid call.
KEY_PREFIX = "sk-ant-"

_lock = threading.Lock()
_cache: tuple[Path, float, str] | None = None


def _path() -> Path:
    from biosphere.core.config import get_settings

    return Path(get_settings().media_root) / "config" / "api_keys.json"


def looks_like_key(value: str) -> bool:
    value = value.strip()
    return value.startswith(KEY_PREFIX) and 30 <= len(value) <= 256 and value.isprintable() and " " not in value


def key_hint(value: str) -> str:
    """``sk-ant-…a1b2``: enough to recognise which key is set, never enough to use it."""
    value = value.strip()
    return f"{KEY_PREFIX}…{value[-4:]}" if value else ""


def stored_api_key() -> str:
    """The dashboard-saved key, or ``""``. Cheap: re-reads only when the file changes."""
    global _cache
    try:
        path = _path()
        mtime = path.stat().st_mtime
    except Exception:  # noqa: BLE001 - no file, or settings that cannot load: no key
        return ""
    with _lock:
        if _cache is not None and _cache[0] == path and _cache[1] == mtime:
            return _cache[2]
        try:
            key = str(json.loads(path.read_text(encoding="utf-8")).get("anthropic", "")).strip()
        except Exception:  # noqa: BLE001 - a hand-broken file reads as "no key", not a crash
            key = ""
        _cache = (path, mtime, key)
        return key


def save_api_key(value: str) -> None:
    global _cache
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"anthropic": value.strip()}), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)  # owner-only where the filesystem honours it
    except OSError:
        pass
    os.replace(tmp, path)
    with _lock:
        _cache = None


def clear_api_key() -> None:
    global _cache
    try:
        _path().unlink()
    except FileNotFoundError:
        pass
    with _lock:
        _cache = None
