"""Capture links: the QR code that lets a phone film a property without signing in.

``/cek`` opens on a DIFFERENT device from the dashboard that drew its QR code — that is the
whole point of the page — and until these routes existed the phone met a sign-in form before
it met the camera. The agent is standing in somebody's flat; typing a password on a phone
there is the friction this removes.

What the phone gets is a :class:`~biosphere.core.models.CaptureGrant`: a bearer token that
acts as the property's owner on the handful of routes a capture uses, for this property only,
for :data:`~biosphere.core.auth.CAPTURE_GRANT_TTL`. The scope is enforced where every
credential is resolved — :func:`biosphere.api.deps.current_user` — so these handlers only
mint and revoke.

The response models live here rather than in ``schemas.py``: two shapes that exist for one
pair of routes, and nothing else in the API returns them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlmodel import col, delete, select

from biosphere.api.deps import CurrentUser, SessionDep, require_property
from biosphere.core import auth as auth_lib
from biosphere.core.models import ApiException, CaptureGrant

router = APIRouter(prefix="/properties", tags=["capture"])

#: How many live links one property may have at once.
#:
#: Each press of "QR göster" mints one, and a dashboard left open re-mints when the old one is
#: about to lapse, so a busy day legitimately produces several. Fifty is far past that and
#: still bounds the table if something calls this in a loop. The OLDEST are dropped, never the
#: newest — the newest is the one on somebody's screen.
_MAX_LIVE_GRANTS = 50


class CaptureLink(BaseModel):
    """A freshly minted capture link. ``token`` is shown once and never stored in the clear."""

    token: str = Field(description="Bearer token for the phone; goes into the QR code as `k`.")
    expires_at: datetime
    property_id: str


def _refuse_capture_grant(request: Request) -> None:
    """A capture link cannot mint or revoke capture links.

    ``current_user`` already refuses grants here — these routes are not on its allow-list — so
    this is the second lock on the same door, written where a reader of this file will see it.
    """
    if getattr(request.state, "capture_property_id", None):
        raise ApiException(
            "capture_link_not_allowed",
            "A capture link cannot create other links.",
            "Çekim bağlantısı başka bağlantı oluşturamaz.",
            status_code=403,
        )


@router.post(
    "/{property_id}/capture-link",
    response_model=CaptureLink,
    status_code=201,
    summary="Mint a capture link (QR) for a phone",
)
def create_capture_link(property_id: str, request: Request, session: SessionDep, user: CurrentUser) -> CaptureLink:
    """A new token the dashboard puts into the QR code it draws.

    Expired links of this property are swept on the way in — the only moment this code is
    certain to be looking at them — and the live set is capped (:data:`_MAX_LIVE_GRANTS`).
    """
    _refuse_capture_grant(request)
    prop = require_property(session, property_id, user)
    now = datetime.now(UTC)

    session.exec(  # type: ignore[call-overload]
        delete(CaptureGrant).where(
            col(CaptureGrant.property_id) == prop.id, col(CaptureGrant.expires_at) <= now
        )
    )
    live = list(
        session.exec(
            select(CaptureGrant)
            .where(CaptureGrant.property_id == prop.id)
            .order_by(col(CaptureGrant.created_at).desc())
        ).all()
    )
    for stale in live[_MAX_LIVE_GRANTS - 1 :]:
        session.delete(stale)

    token = auth_lib.new_session_token()
    # Kept aside rather than read back off the row: SQLite returns the column naive after the
    # commit expires the instance, and the phone compares this against its own clock.
    expires_at = auth_lib.capture_grant_expiry(now)
    grant = CaptureGrant(
        property_id=prop.id,
        created_by=user.id,
        token_hash=auth_lib.token_digest(token),
        expires_at=expires_at,
    )
    session.add(grant)
    session.commit()
    return CaptureLink(token=token, expires_at=expires_at, property_id=prop.id)


@router.delete(
    "/{property_id}/capture-links",
    status_code=204,
    summary="Revoke every capture link of a property",
)
def revoke_capture_links(property_id: str, request: Request, session: SessionDep, user: CurrentUser) -> None:
    """Kill every QR code this property has handed out — for a photo of the screen that went
    somewhere it should not have. Phones mid-upload get a 401 and the "scan a new code" copy."""
    _refuse_capture_grant(request)
    prop = require_property(session, property_id, user)
    session.exec(delete(CaptureGrant).where(col(CaptureGrant.property_id) == prop.id))  # type: ignore[call-overload]
    session.commit()
