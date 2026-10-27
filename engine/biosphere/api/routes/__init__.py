"""API routers, mounted under ``/api/v1`` by :func:`biosphere.api.main.create_app`."""

from __future__ import annotations

from biosphere.api.routes.advisor import private_router as advisor_private_router
from biosphere.api.routes.advisor import public_router as advisor_public_router
from biosphere.api.routes.ai import router as ai_router
from biosphere.api.routes.auth import router as auth_router
from biosphere.api.routes.capture import router as capture_router
from biosphere.api.routes.health import router as health_router
from biosphere.api.routes.hotspots import router as hotspots_router
from biosphere.api.routes.jobs import router as jobs_router
from biosphere.api.routes.listing import router as listing_router
from biosphere.api.routes.photos import router as photos_router
from biosphere.api.routes.plan import router as plan_router
from biosphere.api.routes.properties import router as properties_router
from biosphere.api.routes.public import router as public_router
from biosphere.api.routes.restyle import router as restyle_router
from biosphere.api.routes.scenes import router as scenes_router
from biosphere.api.routes.settings import router as settings_router
from biosphere.api.routes.telemetry import router as telemetry_router
from biosphere.api.routes.walkthrough import router as walkthrough_router

__all__ = [
    "walkthrough_router",
    "advisor_private_router",
    "advisor_public_router",
    "ai_router",
    "auth_router",
    "capture_router",
    "PUBLIC_ROUTERS",
    "PRIVATE_ROUTERS",
    "health_router",
    "hotspots_router",
    "jobs_router",
    "listing_router",
    "photos_router",
    "plan_router",
    "public_router",
    "restyle_router",
    "scenes_router",
    "settings_router",
    "telemetry_router",
    "properties_router",
    "ALL_ROUTERS",
]

#: Reachable without an account, and each one is a deliberate decision rather than an
#: oversight:
#:
#: ``health``    a monitor has no login, and the dashboard's engine pill polls it before
#:               anyone has signed in.
#: ``auth``      the door cannot require the key it hands out.
#: ``public``    the embed token IS the credential; this is what a listing portal frames.
#: ``telemetry`` unauthenticated by construction (``sendBeacon`` carries no credential) and
#:               answers 404 outside development. See that module's docstring.
#: ``advisor_public`` the investment assistant on a published listing; token-gated like
#:               ``public``, opted into per property, rate-limited and budget-capped.
PUBLIC_ROUTERS = (
    health_router,
    auth_router,
    public_router,
    advisor_public_router,
    telemetry_router,
)

#: Everything that reads or writes somebody's listings. Mounted behind ``current_user`` in
#: ``api.main`` — as a router-level dependency rather than a per-handler one, because the
#: failure mode of the per-handler version is a new endpoint that silently ships without it.
PRIVATE_ROUTERS = (
    properties_router,
    scenes_router,
    hotspots_router,
    photos_router,
    plan_router,
    listing_router,
    walkthrough_router,
    jobs_router,
    ai_router,
    restyle_router,
    capture_router,
    advisor_private_router,
    settings_router,
)

#: Order matters only for the generated OpenAPI document.
ALL_ROUTERS = (
    health_router,
    auth_router,
    properties_router,
    scenes_router,
    hotspots_router,
    photos_router,
    plan_router,
    listing_router,
    walkthrough_router,
    jobs_router,
    ai_router,
    restyle_router,
    capture_router,
    advisor_private_router,
    settings_router,
    public_router,
    advisor_public_router,
    # Last, and development-only: a diagnostic sink does not belong among the product routes.
    telemetry_router,
)
