"""Hotspot update and delete. Creation lives under ``/scenes/{id}/hotspots``."""

from __future__ import annotations

from fastapi import APIRouter, Response

from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, SessionDep, require_hotspot
from biosphere.core.models import ApiException, Scene

router = APIRouter(prefix="/hotspots", tags=["hotspots"])


@router.patch("/{hotspot_id}", response_model=sc.Hotspot, summary="Update a hotspot")
def update_hotspot(
    hotspot_id: str, payload: sc.UpdateHotspotRequest, session: SessionDep
,
    user: CurrentUser,
) -> sc.Hotspot:
    hotspot = require_hotspot(session, hotspot_id, user)
    data = payload.model_dump(exclude_unset=True)

    new_type = data.get("type", hotspot.type)
    new_target = data.get("target_scene_id", hotspot.target_scene_id)
    if new_type in sc.NAVIGATION_TYPES:
        # Validated on every write, not just creation: flipping an info hotspot into one that
        # navigates is exactly how a dangling arrow gets into a published property.
        if not new_target:
            raise ApiException(
                "missing_link_target",
                "A navigation hotspot needs a target scene.",
                "Geçiş noktası için hedef sahne gerekli.",
                status_code=422,
            )
        scene = session.get(Scene, hotspot.scene_id)
        target = session.get(Scene, new_target)
        if target is None or (scene is not None and target.property_id != scene.property_id):
            raise ApiException(
                "invalid_link_target",
                "The target scene does not exist in this property.",
                "Hedef sahne bu evde mevcut değil.",
                status_code=422,
            )

    for key, value in data.items():
        if key == "title" and value is not None:
            value = str(value).strip()
        setattr(hotspot, key, value)
    session.add(hotspot)
    session.commit()
    session.refresh(hotspot)
    return sc.hotspot_to_schema(hotspot)


@router.delete("/{hotspot_id}", status_code=204, summary="Delete a hotspot")
def delete_hotspot(hotspot_id: str, session: SessionDep, user: CurrentUser) -> Response:
    hotspot = require_hotspot(session, hotspot_id, user)
    session.delete(hotspot)
    session.commit()
    return Response(status_code=204)
