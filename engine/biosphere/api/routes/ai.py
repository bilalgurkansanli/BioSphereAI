"""AI endpoints: virtual staging, room emptying and single-view measurement.

All three are CPU-bound OpenCV work, so all three run in a worker thread. ``model_backed``
is part of the contract for exactly this reason — the caller can tell whether a neural model
produced the answer or (today) deterministic classical CV did.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from biosphere.ai.empty_room import empty_room as _empty_room
from biosphere.ai.hide_furniture import hide_furniture as _hide_furniture
from biosphere.ai.measure import measure_scene
from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, SessionDep, StorageDep, require_scene
from biosphere.core.models import ApiException, Scene

router = APIRouter(prefix="/ai", tags=["ai"])

#: One emptying at a time per scene.
#:
#: The button that calls this is a TOGGLE, and the first thing a user does with a toggle that
#: does not answer at once is press it again. The result is cached, but the cache is only
#: written at the end — so without this every impatient double-press is two full runs
#: competing for the same cores, and the second one rewrites the files the first is reading.
#:
#: The window this matters in got much wider when the fill became a diffusion model: a room is
#: a dozen or two windows at several seconds of GPU each, so the run is minutes rather than
#: seconds and the second press is not impatience, it is expected. Two of these racing would
#: also be two copies of the same checkpoint on one card.
#: Held for the whole request, so the second caller waits and then answers off the cache.
#:
#: Never pruned, and bounded by the number of scenes that have ever been emptied in this
#: process rather than by anything a caller controls — a scene id has to exist in the
#: database to get this far, so an unknown one is refused by `require_scene` above before it
#: can add an entry. At a few dozen bytes each that is not a leak worth machinery.
_scene_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _require_ready(scene: Scene) -> None:
    if scene.status != "ready" or not scene.panorama:
        raise ApiException(
            "scene_not_ready",
            "This scene has no finished panorama yet.",
            "Bu sahnenin panoraması henüz hazır değil.",
            hint="Wait for processing to finish.",
            status_code=409,
        )


@router.post(
    "/hide-furniture",
    response_model=sc.HideFurnitureResponse,
    summary="Erase furniture from a panorama",
)
async def hide_furniture(
    payload: sc.HideFurnitureRequest, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> sc.HideFurnitureResponse:
    scene = require_scene(session, payload.scene_id, user)
    _require_ready(scene)
    session.commit()
    return await run_in_threadpool(
        _hide_furniture,
        scene,
        payload.polygon,
        storage=storage,
        strength=payload.strength,
    )


@router.post(
    "/empty-room",
    response_model=sc.EmptyRoomResponse,
    summary="Remove every furnishing and deliver the bare room",
)
async def empty_room(
    payload: sc.EmptyRoomRequest, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> sc.EmptyRoomResponse:
    """Strip the scene back to its ceiling, walls and floor.

    Unlike ``/hide-furniture`` this takes no outline: the caller is asking for *everything*,
    which is a question about where the room is rather than about where an object is. See
    ``biosphere.ai.empty_room`` for what that means and where it stops working.
    """
    scene = require_scene(session, payload.scene_id, user)
    _require_ready(scene)
    # Released before the (possibly long) worker call so the lock below is the only thing
    # holding state across it; the scene row is read-only from here on.
    session.commit()
    async with _scene_locks[scene.id]:
        return await run_in_threadpool(
            _empty_room,
            scene,
            storage=storage,
            strength=payload.strength,
            refresh=payload.refresh,
        )


@router.post("/measure", response_model=sc.MeasureResponse, summary="Measure floor distances")
async def measure(payload: sc.MeasureRequest, session: SessionDep, user: CurrentUser) -> sc.MeasureResponse:
    scene = require_scene(session, payload.scene_id, user)
    session.commit()
    return await run_in_threadpool(
        measure_scene,
        scene.id,
        payload.points,
        reference=payload.reference,
        camera_height_cm=payload.camera_height_cm,
    )
