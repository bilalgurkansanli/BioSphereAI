"""Unauthenticated read path for the embeddable player.

These responses are the only ones a third-party listing portal ever sees, so they are served
with wide-open CORS (configured in ``api.main``) and carry no private fields beyond what the
player needs to render.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlmodel import select

from biosphere.ai.measure import measure_scene
from biosphere.api import schemas as sc
from biosphere.api.deps import SessionDep, StorageDep, hotspots_by_scene, scenes_of_property
from biosphere.core.models import ApiException, Property


def _property_by_token(session: SessionDep, public_token: str) -> Property:
    """Resolve a property by its embed token.

    The token *is* the credential, so an unknown token gets the same 404 as a deleted property —
    never a 403, which would confirm that the id exists.

    AND SO DOES AN UNPUBLISHED ONE. ``published`` used to be a label the dashboard drew and
    nothing here read, which made "Yayındayken gömme bağlantısı bağlantıya sahip herkese
    açıktır" a promise the engine did not keep: the token is minted when the property is
    created, it is on screen in the embed panel from that moment, and taking a property off
    air changed nothing about who could open it. Every one of the four public endpoints
    resolves through this function, so the check belongs here and only here.
    """
    property = session.exec(
        select(Property).where(Property.public_token == public_token, Property.published.is_(True))  # type: ignore[union-attr]
    ).first()
    if property is None:
        raise ApiException(
            "property_not_found",
            "This property link is not valid.",
            "Bu tur bağlantısı geçerli değil.",
            status_code=404,
        )
    return property


router = APIRouter(prefix="/public", tags=["public"])


@router.get(
    "/properties/{public_token}",
    response_model=sc.PropertyDetail,
    summary="Public property by embed token",
)
def public_property(
    public_token: str, session: SessionDep, storage: StorageDep, response: Response
) -> sc.PropertyDetail:
    """Resolve a property by its embed token and count the view."""
    property = _property_by_token(session, public_token)

    property.views = int(property.views or 0) + 1
    session.add(property)
    session.commit()
    session.refresh(property)

    scenes = scenes_of_property(session, property.id)
    # Embeds are framed by third-party portals; this is the header that permits it, and it
    # replaces X-Frame-Options rather than accompanying it.
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    # Photographs are part of the public payload for the same reason panoramas are: a listing
    # portal embedding this property is exactly who they were uploaded for.
    from biosphere.api.routes.properties import _photos_by_scene
    from biosphere.api.routes.walkthrough import walkthrough_of_property

    return sc.property_detail_to_schema(
        property,
        scenes,
        hotspots_by_scene(session, [s.id for s in scenes]),
        _photos_by_scene(storage, session, scenes),
        walkthrough=walkthrough_of_property(session, property.id),
    )


@router.get(
    "/properties/{public_token}/walkthrough",
    response_model=sc.Walkthrough,
    summary="Public 3D walkthrough by embed token",
)
def public_walkthrough(public_token: str, session: SessionDep, response: Response) -> sc.Walkthrough:
    """The 3D walkthrough manifest alone: what the splat viewer fetches when it opens."""
    from biosphere.api.routes.walkthrough import walkthrough_of_property

    property = _property_by_token(session, public_token)
    walk = walkthrough_of_property(session, property.id)
    if walk is None:
        raise ApiException(
            "walkthrough_not_found",
            "This property has no 3D walkthrough.",
            "Bu evin 3B gezintisi yok.",
            status_code=404,
        )
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return sc.walkthrough_to_schema(walk)


async def _measure_request(request: Request) -> sc.MeasureRequest:
    """The measure body, read as JSON WHATEVER its ``Content-Type`` says.

    The player that calls this is framed on somebody else's listing page, and the public
    prefix's CORS policy answers any origin for GET and HEAD only. A JSON POST from there is
    preceded by a preflight that policy refuses, so the browser never sends it — the measure
    tool was dead in every real embed while the same-origin test passed. The player therefore
    sends a CORS "simple request": no credentials, body labelled ``text/plain``. FastAPI would
    hand a ``text/plain`` body to the model as raw bytes and answer 422, so it is parsed here;
    ``application/json`` still works for every same-origin caller.
    """
    raw = await request.body()
    try:
        return sc.MeasureRequest.model_validate_json(raw or b"{}")
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@router.post(
    "/properties/{public_token}/measure",
    response_model=sc.MeasureResponse,
    summary="Measure floor distances in a public property",
)
def public_measure(
    public_token: str,
    payload: Annotated[sc.MeasureRequest, Depends(_measure_request)],
    session: SessionDep,
    response: Response,
) -> sc.MeasureResponse:
    """Two directions on the floor, one distance in metres, for a viewer on a listing page.

    SAFE TO EXPOSE, and worth checking why rather than taking on trust. This endpoint reads no
    pixels: ``measure_scene`` intersects the marked ray with the floor plane and scales the
    result by the camera height, so the only thing it touches is arithmetic the caller already
    supplied. It cannot leak a panorama that is not already public, cannot start a model, and
    cannot cost the server more than a few microseconds however often it is called.

    It is a POST because the request carries a list of points, not because anything is
    written; nothing here mutates a row, and the view counter deliberately stays with
    ``public_property`` so that measuring a wall does not inflate a listing's view count.

    The scene must belong to THIS property. Without that check the token would be a key to every
    scene in the deployment, since ``measure_scene`` takes a bare scene id.
    """
    property = _property_by_token(session, public_token)
    scene_ids = {s.id for s in scenes_of_property(session, property.id)}
    if payload.scene_id not in scene_ids:
        # The same 404 an unknown token gets, and for the same reason: a different answer
        # here would confirm that a scene id exists somewhere else in the deployment.
        raise ApiException(
            "scene_not_found",
            "This room is not part of this property.",
            "Bu oda bu tura ait değil.",
            status_code=404,
        )
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return measure_scene(
        payload.scene_id,
        payload.points,
        reference=payload.reference,
        camera_height_cm=payload.camera_height_cm,
    )


@router.get(
    "/properties/{public_token}/plan",
    response_model=sc.FloorPlan,
    summary="Public floor plan by embed token",
)
def public_plan(
    public_token: str, session: SessionDep, storage: StorageDep, response: Response
) -> sc.FloorPlan:
    """The plan as it stands, for the embedded player.

    Read-only in the strongest sense: it assembles outlines that are already on disk and never
    measures anything. A viewer on somebody else's listing page cannot start a GPU pass here,
    and a property whose rooms were never measured simply has no plan for them to open — which is
    why the player only offers the button once this answers with a room.
    """
    from biosphere.api.routes.plan import _can_measure, _to_schema, plan_of_property

    property = _property_by_token(session, public_token)
    scenes = scenes_of_property(session, property.id)
    hotspots = hotspots_by_scene(session, [s.id for s in scenes])
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return _to_schema(plan_of_property(property.id, scenes, hotspots, storage), can_measure=_can_measure())
