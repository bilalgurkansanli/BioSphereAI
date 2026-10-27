"""The property's advertisement: one headline and one description, written from the property.

Two routes, and the split between them is the same one the floor plan makes:

``GET  /properties/{id}/listing``   what the panel opens with. Reads the cached copy off the disk
                              and says whether a model could write a new one. Cheap enough
                              for a page load; never calls anything that bills.
``POST /properties/{id}/listing``   writes it. Seconds, a network round trip, and somebody's
                              money — so it happens when a person presses a button and not
                              a moment before.

Why the copy is cached beside the property
--------------------------------------
Because the answer does not change while the property does not. The cache key is the FACTS'
digest plus the locale, the tone and the free instruction, so adding a room, renaming one,
measuring one, or asking in the other language all miss it and everything else hits. A stale
answer to a question nobody asked again is the failure mode this avoids: an agent who opens
the panel a second time should see what they saw the first time, not a second bill.

``refresh`` is the one way past it, and it exists because "write me a different sentence" is
a legitimate thing to want from copy that is not wrong, only not what they wanted.

Applying it is deliberately NOT here
------------------------------------
The generated title and description are a suggestion in a form, and saving them is
``PATCH /properties/{id}`` like any other edit. Giving this endpoint an ``apply`` flag would put a
second writer on ``property.title`` whose validation and audit trail would have to be kept level
with the first one, to save the dashboard a single request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from biosphere.ai import listing as engine
from biosphere.ai import llm
from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, SessionDep, StorageDep, require_property, scenes_of_property
from biosphere.core.logging import get_logger
from biosphere.core.storage import LocalStorage

router = APIRouter(prefix="/properties", tags=["listing"])

log = get_logger("biosphere.api.listing")

_FILENAME = "listing.json"

#: Bumped when the stored shape changes. A file at another version is ignored rather than
#: migrated: it is one model call to replace, and a migration for a cache is a liability.
_CACHE_VERSION = 1

#: One generation at a time per property.
#:
#: For the same reason ``/ai/empty-room`` holds one: the first thing anybody does with a
#: button that takes ten seconds is press it again, and two of these racing are two bills for
#: one paragraph, with the loser overwriting the winner's cache. The second caller waits and
#: then answers off the cache the first one just wrote.
_property_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _cache_path(property_id: str, storage: LocalStorage) -> Path:
    return storage.property_dir(property_id) / _FILENAME


def _cache_key(facts: sc.ListingFacts, payload: sc.GenerateListingRequest) -> str:
    """What has to be unchanged for the stored copy to still be an answer."""
    material = "|".join(
        [
            facts.digest,
            payload.locale,
            payload.tone,
            (payload.instructions or "").strip(),
            llm.model_name() if llm.available() else "offline",
        ]
    )
    return hashlib.blake2b(material.encode("utf-8"), digest_size=8).hexdigest()


def _read_cache(path: Path) -> tuple[str, str, sc.ListingCopy] | None:
    """The stored copy, the key it answers and the facts it was written from. Never raises.

    A corrupt or half-written file is "no cache": the cost of being wrong here is one model
    call, and the cost of raising is a panel that will not open.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != _CACHE_VERSION:
            return None
        return (
            str(payload["key"]),
            str(payload["facts_digest"]),
            sc.ListingCopy.model_validate(payload["copy"]),
        )
    except Exception:  # noqa: BLE001 - see the docstring
        return None


def _write_cache(path: Path, key: str, copy: sc.ListingCopy) -> None:
    document: dict[str, Any] = {
        "version": _CACHE_VERSION,
        "key": key,
        # Kept beside the key because the two answer different questions: the key says
        # "would this request produce the same call", the digest says "is this copy still
        # about this property". The panel opens on the second one, which is why an agent who
        # asked in English yesterday still sees yesterday's English copy today.
        "facts_digest": copy.facts.digest,
        "copy": copy.model_dump(mode="json"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - a full disk on the way out
        log.warning("listing.cache_write_failed", error=str(exc)[:160])


@router.get("/{property_id}/listing", response_model=sc.ListingState, summary="The property's listing copy")
async def get_listing(property_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> sc.ListingState:
    """What the panel opens with. Reads files; writes nothing; calls no model."""
    property = require_property(session, property_id, user)
    scenes = scenes_of_property(session, property_id)
    backend = llm.describe()

    def _load() -> sc.ListingCopy | None:
        cached = _read_cache(_cache_path(property.id, storage))
        if cached is None:
            return None
        _, digest, copy = cached
        # Shown only while it is still about this property. Otherwise the panel would open on a
        # description of a flat that has since gained two rooms — which is worse than an
        # empty panel, because it reads exactly as well.
        facts = engine.collect(property, scenes, storage)
        if digest != facts.digest():
            return None
        return copy.model_copy(update={"cached": True})

    latest = await run_in_threadpool(_load)
    return sc.ListingState(
        available=bool(backend.get("available")),
        model=backend.get("model"),
        reason=backend.get("reason"),
        latest=latest,
    )


@router.post("/{property_id}/listing", response_model=sc.ListingCopy, summary="Write the listing copy")
async def generate_listing(
    property_id: str,
    payload: sc.GenerateListingRequest,
    session: SessionDep,
    storage: StorageDep,
    user: CurrentUser,
) -> sc.ListingCopy:
    """Write the headline and the description for this property.

    Always answers with copy. With no key, no client library or an unreachable service it
    answers with the deterministic composer's version and ``model_backed: false``, and the
    reason is in ``warnings`` — see :mod:`biosphere.ai.listing`.
    """
    property = require_property(session, property_id, user)
    scenes = scenes_of_property(session, property_id)
    # Every row this needs has been read; releasing the transaction before a call that can
    # take half a minute keeps the connection out of the model's hands.
    session.commit()

    path = _cache_path(property.id, storage)

    async with _property_locks[property.id]:

        def _cached() -> sc.ListingCopy | None:
            if payload.refresh:
                return None
            stored = _read_cache(path)
            if stored is None:
                return None
            key, _, copy = stored
            facts = engine.facts_schema(engine.collect(property, scenes, storage))
            if key != _cache_key(facts, payload):
                return None
            return copy.model_copy(update={"cached": True})

        hit = await run_in_threadpool(_cached)
        if hit is not None:
            return hit

        copy = await run_in_threadpool(
            engine.write_listing,
            property,
            scenes,
            storage=storage,
            locale=payload.locale,
            tone=payload.tone,
            instructions=payload.instructions,
        )
        await run_in_threadpool(_write_cache, path, _cache_key(copy.facts, payload), copy)
        log.info(
            "listing.written",
            property=property.id,
            model_backed=copy.model_backed,
            images=copy.images_used,
            warnings=len(copy.warnings),
            elapsed_ms=copy.elapsed_ms,
        )
        return copy
