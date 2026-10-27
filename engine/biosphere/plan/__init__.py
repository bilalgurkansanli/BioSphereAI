"""The bird's-eye view: a room's floor measured from its panorama, and a flat made of them.

Three modules, split by what could possibly be wrong with them:

``types``     the contracts, and the one assumption the whole feature rests on
              (:data:`~biosphere.plan.types.CAMERA_HEIGHT_M`).
``outline``   panorama -> the distance to the wall in every direction -> a polygon. The half
              with a model in it, and the half whose accuracy ``scripts/bench_floorplan.py``
              measures against rooms of known size.
``assemble``  outlines and doorways -> one drawing. Pure plane geometry, no images.
``cache``     where an outline lives between requests, and what makes it stale.

The assembler's own entry point is ``biosphere.plan.assemble.assemble`` and is deliberately
NOT re-exported here: binding a function called ``assemble`` on the package would shadow the
module of the same name, so ``from biosphere.plan import assemble`` would mean one thing or
the other depending on import order.
"""

from __future__ import annotations

from biosphere.plan.assemble import PortalInput, RoomInput
from biosphere.plan.cache import build_outline, forget_outline, read_outline
from biosphere.plan.outline import (
    outline_from_pano,
    outline_from_surface,
    polygon_from_profile,
    profile_from_surface,
)
from biosphere.plan.types import (
    CAMERA_HEIGHT_M,
    FloorPlan,
    PlanDoor,
    PlanRoom,
    RadialProfile,
    RoomOutline,
)

__all__ = [
    "CAMERA_HEIGHT_M",
    "FloorPlan",
    "PlanDoor",
    "PlanRoom",
    "PortalInput",
    "RadialProfile",
    "RoomInput",
    "RoomOutline",
    "build_outline",
    "forget_outline",
    "outline_from_pano",
    "outline_from_surface",
    "polygon_from_profile",
    "profile_from_surface",
    "read_outline",
]
