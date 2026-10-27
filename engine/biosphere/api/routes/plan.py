"""The floor plan of a property: read it cheaply, build it deliberately.

Two routes and a hard split between them, which is the whole design of this endpoint:

``GET  /properties/{id}/plan``        assembles the plan out of whatever outlines are already on
                                disk. Pure arithmetic, milliseconds, never loads a model.
``POST /properties/{id}/plan/build``  measures the rooms that have not been measured. Sixteen
                                passes of a scene-parsing model per room, so seconds each.
``PATCH /properties/{id}/plan``       moves rooms by hand. As cheap as the read — it changes the
                                drawing, not the measurement.

That split is what lets an embed on a third-party listing portal show a plan at all. A plan
computed on demand would mean a stranger's page waiting on somebody else's GPU; a plan that
is *published* with the property is a drawing, and a drawing is cheap. It also puts the cost where
the decision is: the agent who finished shooting a flat presses "measure", sees the result,
and can press it again after a reprocess.

An empty plan is a normal answer. Rooms with no outline are still listed with
``pending_scene_ids``, so a client can say "four rooms have not been measured yet" rather
than "there is no plan".
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from biosphere.api import schemas as sc
from biosphere.api.deps import (
    CurrentUser,
    SessionDep,
    StorageDep,
    gpu_lock,
    hotspots_by_scene,
    require_property,
    scenes_of_property,
)
from biosphere.core.logging import get_logger
from biosphere.core.models import Scene
from biosphere.core.storage import LocalStorage
from biosphere.plan import assemble as plan_assemble
from biosphere.plan import cache as plan_cache
from biosphere.plan import placements as plan_placements
from biosphere.plan.types import CAMERA_HEIGHT_M, FloorPlan

router = APIRouter(prefix="/properties", tags=["plan"])

log = get_logger("biosphere.api.plan")


def _can_measure() -> bool:
    """Whether a build on this deployment would measure anything. Loads nothing to decide."""
    from biosphere.ai import backend

    try:
        return str(backend.describe().get("mode", "off")) != "off"
    except Exception:  # noqa: BLE001 - a broken optional dependency is "no", not a 500
        return False


def _to_schema(plan: FloorPlan, *, can_measure: bool) -> sc.FloorPlan:
    return sc.FloorPlan(
        property_id=plan.property_id,
        rooms=[
            sc.PlanRoom(
                scene_id=room.scene_id,
                name=room.name,
                placement=room.placement,
                x=room.x,
                z=room.z,
                rotation=room.rotation,
                polygon=[[x, z] for x, z in room.polygon],
                radius=room.radius,
                area_m2=room.area_m2,
                ready=room.ready,
                measured_fraction=room.measured_fraction,
            )
            for room in plan.rooms
        ],
        doors=[
            sc.PlanDoor(
                from_scene_id=door.from_scene_id,
                to_scene_id=door.to_scene_id,
                x=door.x,
                z=door.z,
                solved=door.solved,
            )
            for door in plan.doors
        ],
        camera_height_m=plan.camera_height_m,
        pending_scene_ids=list(plan.pending_scene_ids),
        bounds=list(plan.bounds),
        warnings=list(plan.warnings),
        can_measure=can_measure,
    )


def plan_of_property(
    property_id: str,
    scenes: list[Scene],
    hotspots: dict[str, list],
    storage: LocalStorage,
) -> FloorPlan:
    """Assemble the plan from what is already on disk. Never runs a model.

    Shared by the private and the public read routes, because a plan an agent looks at and a
    plan a buyer looks at are the same drawing — a second implementation would be the way
    those two drift apart.
    """
    rooms: list[plan_assemble.RoomInput] = []
    pending: list[str] = []
    for scene in scenes:
        outline = plan_cache.read_outline(scene, storage)
        ready = scene.status == "ready" and bool(scene.panorama)
        if outline is None and ready:
            pending.append(scene.id)
        rooms.append(
            plan_assemble.RoomInput(
                scene_id=scene.id, name=scene.name, ready=ready, outline=outline
            )
        )

    portals = [
        plan_assemble.PortalInput(
            from_scene_id=hotspot.scene_id,
            to_scene_id=hotspot.target_scene_id,
            yaw=hotspot.yaw,
            pitch=hotspot.pitch,
        )
        for group in hotspots.values()
        for hotspot in group
        if hotspot.type == "portal" and hotspot.target_scene_id
    ]

    return plan_assemble.assemble(
        property_id,
        rooms,
        portals,
        camera_height_m=CAMERA_HEIGHT_M,
        pending_scene_ids=tuple(pending),
        placements=plan_placements.read_placements(property_id, storage),
    )


@router.get("/{property_id}/plan", response_model=sc.FloorPlan, summary="The property seen from above")
def get_plan(property_id: str, session: SessionDep, storage: StorageDep, user: CurrentUser) -> sc.FloorPlan:
    property = require_property(session, property_id, user)
    scenes = scenes_of_property(session, property.id)
    hotspots = hotspots_by_scene(session, [s.id for s in scenes])
    return _to_schema(plan_of_property(property.id, scenes, hotspots, storage), can_measure=_can_measure())


@router.post(
    "/{property_id}/plan/build",
    response_model=sc.FloorPlan,
    summary="Measure the rooms that have not been measured",
)
async def build_plan(
    property_id: str, payload: sc.BuildPlanRequest, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> sc.FloorPlan:
    """Read every unmeasured room's shape off its panorama, then assemble.

    Under the process-wide :data:`~biosphere.api.deps.gpu_lock` for the same reason every
    other model route is: two of these at once on one card is an out-of-memory error and two
    failed requests, not twice the throughput.

    THIS IS A LONG REQUEST AND IT SHOULD BE A JOB. It carries exactly the debt
    ``/ai/empty-room`` and the restyling routes carry, for the same reason — ``core.jobs`` is
    shaped around one stitch of one scene, with that pipeline's own stage list — and it is
    written down rather than hidden. A flat of eight rooms is under a minute on a card and
    several times that without one.
    """
    property = require_property(session, property_id, user)
    scenes = scenes_of_property(session, property.id)
    hotspots = hotspots_by_scene(session, [s.id for s in scenes])
    session.commit()

    measurable = [s for s in scenes if s.status == "ready" and s.panorama]
    async with gpu_lock:
        for scene in measurable:
            await run_in_threadpool(
                plan_cache.build_outline, scene, storage, refresh=payload.refresh
            )

    plan = plan_of_property(property.id, scenes, hotspots, storage)
    log.info(
        "plan.built",
        property_id=property.id,
        rooms=len(plan.rooms),
        measured=sum(1 for r in plan.rooms if r.measured_fraction > 0.0),
        pending=len(plan.pending_scene_ids),
    )
    return _to_schema(plan, can_measure=_can_measure())


@router.patch(
    "/{property_id}/plan",
    response_model=sc.FloorPlan,
    summary="Move rooms on the plan by hand",
)
def update_plan(
    property_id: str, payload: sc.UpdatePlanRequest, session: SessionDep, storage: StorageDep
,
    user: CurrentUser,
) -> sc.FloorPlan:
    """Put a room where the agent says it is, and give back the plan that follows.

    No model, no measurement — this is the drawing's own state, so it is as cheap as the read
    and answers with the whole recomputed plan rather than an acknowledgement. That matters
    for the caller: a placement is not local, since the walk starts again from every placed
    room, so moving one room can legitimately move the ones beyond it. A client that patched
    its own copy of one room would draw a plan the server does not have.

    Placements survive re-measuring and re-shooting: they are keyed by scene, not by anything
    the solver produced.
    """
    property = require_property(session, property_id, user)
    scenes = scenes_of_property(session, property.id)
    known = {scene.id for scene in scenes}

    changes: dict[str, plan_placements.Placement | None] = {}
    for scene_id, placement in payload.placements.items():
        if scene_id not in known:
            # A scene that is not in this property cannot be placed on its plan. Ignored rather
            # than refused: the alternative is a whole request failing over one stale id in a
            # tab that has been open since before a room was deleted.
            continue
        changes[scene_id] = (
            None
            if placement is None
            else plan_placements.Placement(placement.x, placement.z, placement.rotation)
        )

    # Read, edit and write happen inside `update_placements` under one lock per property: a drag
    # and a turn arriving together would otherwise each write what the other never saw.
    plan_placements.update_placements(property.id, storage, changes=changes, reset=payload.reset)
    hotspots = hotspots_by_scene(session, [s.id for s in scenes])
    session.commit()
    return _to_schema(plan_of_property(property.id, scenes, hotspots, storage), can_measure=_can_measure())
