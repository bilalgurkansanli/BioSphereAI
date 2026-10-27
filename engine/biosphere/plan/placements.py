"""Where a person put a room, when the arithmetic could not.

The assembler solves what a doorway marked from both sides can prove, guesses what one ring
can only suggest, and refuses to place what nothing reaches at all (see
:mod:`biosphere.plan.assemble`). All three of those are the right answers, and none of them
help the agent standing in front of a plan that has the kitchen a quarter turn out.

So the automatic layout is a FIRST DRAFT and this module is the last word. A placement here
overrides whatever the solver worked out, and it is not a hint or a nudge — the room is where
the person said it is.

Three things follow, and each is the reason a whole class of property becomes plannable:

*Rooms nothing reaches can be placed.* A property with no floor rings at all has no graph to
solve, and until now no plan. Now it has one the moment somebody drags the rooms into place.

*A placed room becomes an anchor of its own.* The walk starts from every placed room as well
as from the automatic one, so correcting the hallway drags everything the hallway leads to
along with it, instead of leaving its neighbours behind at their guessed positions.

*The correction outlives the estimate.* Placements are keyed by scene, not by whatever the
solver happened to produce, so re-measuring a room or shooting the one beyond it does not
throw away the agent's work.

Stored as one file per property rather than a column, for the reason
:mod:`biosphere.plan.cache` gives about the outline: it is written by a pass that may never
run, and the bytes belong with the property's own directory so that deleting the property takes them.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biosphere.core.logging import get_logger
from biosphere.core.storage import LocalStorage

__all__ = [
    "PLACEMENTS_FILENAME",
    "Placement",
    "read_placements",
    "write_placements",
    "update_placements",
]

log = get_logger("biosphere.plan.placements")

PLACEMENTS_FILENAME = "plan.json"

#: Bumped if the meaning of a stored placement changes. Unlike an outline, this file cannot
#: be recomputed from anything — it is somebody's work — so a future version bump has to
#: migrate rather than discard.
_VERSION = 1

#: One lock per property, held across the read-modify-write of its layout.
#:
#: NOT DEFENSIVE PROGRAMMING — this was measured. Every drag and every turn is its own
#: request, FastAPI runs them in a thread pool, and two arriving together both read the file
#: before either writes it: the second write has no idea the first happened and drops it. In
#: the browser that looks like a room springing back to where it was, which is the one thing
#: a direct-manipulation editor must never do.
#:
#: Keyed by property and never pruned, bounded by the number of properties anyone has ever edited in
#: this process — a few dozen bytes each, the same bargain `api.routes.ai` makes per scene.
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


@dataclass(frozen=True, slots=True)
class Placement:
    """One room, put somewhere by hand: metres in the plan frame, and a rotation."""

    x: float
    z: float
    rotation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 3),
            "z": round(float(self.z), 3),
            "rotation": round(float(self.rotation), 5),
        }


def placements_path(property_id: str, storage: LocalStorage) -> Path:
    return storage.property_dir(property_id) / PLACEMENTS_FILENAME


def read_placements(property_id: str, storage: LocalStorage) -> dict[str, Placement]:
    """What a person has decided about this property's layout. Never raises.

    Both read paths go through this — the dashboard's and the embed's — because a plan an
    agent corrected and a plan a buyer sees have to be the same drawing.
    """
    path = placements_path(property_id, storage)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.info("plan.placements_unreadable", property_id=property_id)
        return {}
    if int(payload.get("version", 0)) != _VERSION:
        return {}

    out: dict[str, Placement] = {}
    for scene_id, entry in (payload.get("rooms") or {}).items():
        try:
            out[str(scene_id)] = Placement(
                float(entry["x"]), float(entry["z"]), float(entry["rotation"])
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def write_placements(
    property_id: str, storage: LocalStorage, placements: dict[str, Placement]
) -> None:
    """Replace the stored layout. An empty mapping removes the file entirely.

    Removing rather than writing an empty one matters: "this property has no corrections" and
    "this property was corrected back to nothing" are the same state, and leaving an empty file
    behind would make the second look like a decision somebody made.
    """
    path = placements_path(property_id, storage)
    if not placements:
        path.unlink(missing_ok=True)
        return

    payload: dict[str, Any] = {
        "version": _VERSION,
        "rooms": {scene_id: placement.as_dict() for scene_id, placement in placements.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside and moved into place, so a reader never catches the file halfway through
    # being replaced. `read_placements` would recover from that by returning nothing, which is
    # to say by drawing the plan without anybody's corrections — a worse failure than the
    # rename costs to avoid.
    scratch = path.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(scratch, path)
    log.info("plan.placements_saved", property_id=property_id, rooms=len(placements))


def update_placements(
    property_id: str,
    storage: LocalStorage,
    *,
    changes: dict[str, Placement | None],
    reset: bool = False,
) -> dict[str, Placement]:
    """Apply changes to a property's layout and return what it now is.

    The read, the edit and the write happen under one lock, which is the whole reason this
    exists rather than the caller doing the three steps itself. ``None`` releases a room back
    to the solver; ``reset`` releases all of them first, so one call can clear a flat and pin
    a single room in it.
    """
    with _locks[property_id]:
        current = {} if reset else read_placements(property_id, storage)
        for scene_id, placement in changes.items():
            if placement is None:
                current.pop(scene_id, None)
            else:
                current[scene_id] = placement
        write_placements(property_id, storage, current)
        return current
