"""Restyling a whole sphere, one perspective window at a time.

The photograph case is :mod:`biosphere.ai.restyle`. This module is the part that only exists
because the canvas is equirectangular, and it borrows its shape from
:mod:`biosphere.ai.room_fill`, which solved the same geometry problem for a different question.

Why not simply restyle the canvas
----------------------------------
The same reason ``room_fill`` exists, doubled. On an equirect canvas a skirting board is a sine
wave, a floorboard curves and the wall/floor junction is different in every column; a diffusion
model trained on photographs has never seen a room drawn that way. And a 2:1 canvas four
thousand pixels wide is nothing like the 512-square box the checkpoint was trained in, so it
loses global coherence and draws the same sofa twice along one wall. Through a 90-degree window
the same room is an ordinary photograph, which is the input every part of ``restyle`` was
measured on.

THE HARD PART IS NOT THE PROJECTION, IT IS AGREEMENT
-----------------------------------------------------
``room_fill`` fills holes: most of each window is already photograph, so two neighbouring
windows have plenty of shared evidence and naturally agree. Restyling has no such anchor —
every window is being asked to invent furniture, and sixteen independent inventions of "a
Scandinavian living room" is sixteen different flats meeting at eight seams.

So the windows are not independent, and three things make them agree:

1. **Each window is rendered from the panorama AS IT STANDS**, which already contains
   everything the previous windows furnished. A window's context is its neighbour's answer.

2. **What a neighbour already furnished is PROTECTED in this window, not re-drawn.** That is
   the whole trick, and it is why :func:`restyle.analyse_room` grew ``extra_protect``: the
   overlap arrives as fixed pixels the sampler must continue rather than as a region it may
   reinterpret. Re-drawing it would produce two different sofas in the same place and a seam
   down the middle of the overlap; averaging them would produce a ghost.

3. **They go round in order, and the first one sets the tone.** Window 1 has no neighbours and
   invents freely. Window 2 sees window 1. By the time the ring closes, the last window is
   constrained on both sides — which is the tightest spot in the sweep and therefore the right
   place for the wrap to land.

The poles are protected rather than restyled
---------------------------------------------
Two rows of windows reach about 75 degrees of elevation. Beyond that is the ceiling directly
overhead and the floor directly underfoot, where an equirect pixel covers almost no solid angle
and, in a hand-held capture, the nadir is usually the photographer's own feet. Restyling there
buys nothing and risks a smeared cap, so :data:`POLE_PROTECT_DEG` holds it.

THE GUARD IS AN ALPHA, NOT A THRESHOLD, AND THE FIRST VERSION GOT THAT WRONG. It began as a
feathered ramp folded into the work list, which windows then read through a ``> 127`` test — so
the feather decided *which pixels were restyled at all* rather than *how strongly*. Measured on
the real bedroom, the share of each row that changed ran 97 % at -62 degrees, 52 % at -66 and
0 % at -68: a ragged boundary of fully-restyled pixels next to untouched ones, which is a
dither rather than a blend, and it is visible as a ring.

So the ramp is applied once at the end, as the weight in a composite against the original —
exactly what `restyle.feathered_alpha` does for a window, applied to a latitude instead. The
work list stays binary, which is also what keeps the strike-off crisp.

Resolution is not a quality dial here, it is the difference between working and not
--------------------------------------------------------------------------------------
MEASURED on the same real bedroom, same style, same everything else:

  2048-wide canvas, 512-px windows   115 s.  The room's shape and the agreement between
                                     windows both hold, but half the walls come back as
                                     white smears and two of the eight seam views are
                                     unusable — the curtains and the piled bedding turn
                                     into abstract paint.
  4096-wide canvas, 1024-px windows  122 s.  A room. Wood sideboard, wall clock, olive
                                     cushions, a cactus that is the SAME cactus either side
                                     of the seam it straddles.

Seven seconds apart, because the diffusion pass runs at its own latent size either way and
only the projection either side of it changed. The smearing was never the sampler having a
bad day: at 512 px a 90-degree window off a 2048-wide equirect carries 512 px of real
photograph, and a model given a soft, low-detail wall with no edges in it answers with a soft
low-detail wall. :data:`VIEW_SIZE` is therefore a floor rather than a preference, and callers
that want to go faster should reduce the number of windows, never their size.

Cost
----
Sixteen windows at the photograph path's measured 6.5 s is about two minutes per style on the
reference card, so a five-style scene is ten. That is cached per (scene, style) on disk and it
is the reason this is not offered as five buttons that all light up at once.

**This should be a job, and it is not one yet** — the same debt ``/ai/empty-room`` carries and
for the same reason: the queue in ``core.jobs`` is shaped around a stitch, with a scene id and
the stitch's own stage list, and widening it is a schema change rather than a refactor. Until
then this runs under a long request deadline like emptying does.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from biosphere.ai import restyle, views
from biosphere.core.logging import get_logger

__all__ = [
    "restyle_equirect",
    "PanoRestyleOutput",
    "VIEW_PLAN",
    "VIEW_SIZE",
    "POLE_PROTECT_DEG",
]

log = get_logger("biosphere.ai.restyle_pano")

#: Where the windows point, as ``(pitch_deg, n_yaw)`` — elevation, positive up.
#:
#: TWO ROWS, NOT ``room_fill``'S THREE, AND THE DIFFERENCE IS THE COST OF AGREEMENT. That
#: module's windows mostly contain photograph and are cheap to reconcile; these each cost a
#: full diffusion pass and every extra one is another chance for two neighbours to disagree
#: about what the room contains. At a 90-degree field of view a row centred at 30 degrees of
#: elevation reaches 75, so two rows already cover everything :data:`POLE_PROTECT_DEG` does not
#: hold, and they overlap each other across the horizon band by 30 degrees.
#:
#: Eight yaws is 45 degrees between neighbours against a 90-degree window: every direction is
#: seen by two windows, so nothing is ever furnished only at the edge of one, where a model is
#: least reliable.
VIEW_PLAN: tuple[tuple[float, int], ...] = ((-30.0, 8), (30.0, 8))

#: 1024 px across 90 degrees is 11.4 px per degree, the density a 4096-wide equirect carries.
#: Anything less and the round trip through the window is itself a downscale — see the
#: measurement in the module docstring for what that costs, which is the feature.
VIEW_SIZE = 1024

#: Elevation beyond which the panorama is kept as it was, with a feathered join.
POLE_PROTECT_DEG = 74.0

#: Width of that join, in degrees of elevation.
_POLE_FEATHER_DEG = 12.0

#: How far into a window's edge its contribution fades, as a fraction of its width. Wider than
#: ``room_fill``'s 0.12 on purpose: that module fades between two versions of the same
#: photograph, this one fades between two inventions, and the wider ramp is what stops a
#: brightness or palette difference between neighbours reading as a line down the wall.
_EDGE_FADE = 0.22

#: Above this weight a direction counts as furnished and later windows will not redraw it.
#:
#: HIGHER THAN ``room_fill``'S 0.85, DELIBERATELY. Striking a direction off means the next
#: window must treat it as fixed context, and context that arrived at half strength is a
#: half-faded sofa the neighbour then has to build around. Only the part of a window that
#: landed at nearly full weight is allowed to become somebody else's ground truth.
_SETTLED_ALPHA = 0.92


@dataclass(slots=True)
class PanoRestyleOutput:
    """A restyled sphere and the honest account of how it was made."""

    panorama: np.ndarray
    style_id: str
    windows: int
    elapsed_ms: int
    #: Solid-angle weighted share of the sphere that was regenerated.
    restyled_fraction: float
    #: Its complement, and the number a VIEWER should be told: how much of the sphere is still
    #: the photograph. The poles, everything the segmenter held in each window, and anything no
    #: window reached.
    #:
    #: NOT the mean of each window's own ``protected_fraction``, which was the first version and
    #: was wrong in a way that would have shipped: most of what a pano window protects is what a
    #: NEIGHBOUR already furnished, so that mean read about 46 % and would have been shown under
    #: the API's "kept from the original" wording — describing generated furniture as preserved
    #: property.
    protected_fraction: float
    model_backed: bool = True


def _edge_falloff(size: int) -> np.ndarray:
    """1 in the middle of a window, ramping to 0 at its border."""
    ramp = np.ones(size, np.float32)
    width = max(1, int(round(size * _EDGE_FADE)))
    taper = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp[:width] = taper
    ramp[-width:] = taper[::-1]
    return np.minimum(ramp[:, None], ramp[None, :])


def _pole_guard(height: int, width: int) -> np.ndarray:
    """Per-pixel weight of the RESTYLED canvas: 1 across the band, ramping to 0 at the poles.

    Built from the row's own elevation rather than from a pixel count, so it means the same
    thing on a 2048-wide canvas and an 8192-wide one. Returned as float, because this is a
    blending weight and quantising it to 255 levels would be the same mistake in miniature.
    """
    rows = np.arange(height, dtype=np.float32)
    elevation = 90.0 - 180.0 * (rows + 0.5) / float(height)
    inner = POLE_PROTECT_DEG - _POLE_FEATHER_DEG
    ramp = np.clip(
        (POLE_PROTECT_DEG - np.abs(elevation)) / max(1e-6, POLE_PROTECT_DEG - inner), 0.0, 1.0
    )
    return np.repeat(ramp[:, None], width, axis=1)


def _ring_order(plan: tuple[tuple[float, int], ...]) -> list[tuple[float, float]]:
    """``(yaw, pitch)`` for every window, walked as a ring per row.

    In order rather than easiest-first: ``room_fill`` sorts by how little each window has to do
    because a window with more context does a better job, but here EVERY window has everything
    to do and the only context that exists is the neighbour just finished. Walking the ring is
    what keeps that neighbour adjacent.
    """
    out: list[tuple[float, float]] = []
    for pitch_deg, n_yaw in plan:
        for index in range(n_yaw):
            out.append((360.0 * index / float(n_yaw), pitch_deg))
    return out


def restyle_equirect(
    pano: np.ndarray,
    style: restyle.Style | str,
    *,
    window_dressing: bool = True,
    plan: tuple[tuple[float, int], ...] = VIEW_PLAN,
    view_size: int = VIEW_SIZE,
    progress: Callable[[float, str], None] | None = None,
    **passthrough: Any,
) -> PanoRestyleOutput | None:
    """Furnish a whole equirectangular panorama in ``style``. ``None`` with no backend.

    ``passthrough`` reaches :func:`restyle.restyle_photo` unchanged — ``strength``, ``steps``,
    ``guidance`` and ``control_scales`` are all swept from ``scripts/bench_restyle_pano.py``
    without this function knowing what they mean.
    """
    resolved = style if isinstance(style, restyle.Style) else restyle.style_by_id(style)
    if resolved is None:
        raise ValueError(f"unknown style: {style!r}; expected one of {', '.join(restyle.STYLE_IDS)}")
    if not restyle.available():
        return None

    started = time.perf_counter()
    height, width = pano.shape[:2]
    canvas = pano.copy()

    K = views.intrinsics(view_size, views.VIEW_FOV_DEG)
    falloff = _edge_falloff(view_size)
    settled = (falloff >= _SETTLED_ALPHA).astype(np.uint8) * 255

    # `remaining` is what nobody has furnished yet, and it is BINARY: a window either has work
    # in a direction or it does not. It starts as everything the guard does not fully hold, so
    # no window wastes a diffusion pass on a cap that is about to be blended away.
    guard = _pole_guard(height, width)
    remaining = (guard > 0.0).astype(np.uint8) * 255

    order = _ring_order(plan)
    done = 0

    for index, (yaw_deg, pitch_deg) in enumerate(order):
        rot = views.rotation(yaw_deg, pitch_deg)
        view_mask = views.render_view(remaining, rot, K, view_size, interpolation=cv2.INTER_NEAREST)
        if view_mask.ndim == 3:
            view_mask = view_mask[..., 0]
        editable = (view_mask > 127).astype(np.uint8) * 255
        if not editable.any():
            continue

        # From the canvas as it stands: this is where a neighbour's furniture arrives.
        view = views.render_view(canvas, rot, K, view_size)

        room = restyle.analyse_room(
            view, window_dressing=window_dressing, extra_protect=255 - editable
        )
        if room is None:
            log.warning("restyle_pano.no_depth", yaw=yaw_deg, pitch=pitch_deg)
            return None

        result = restyle.restyle_photo(view, resolved, analysis=room, **passthrough)
        if result is None:
            log.warning("restyle_pano.window_failed", yaw=yaw_deg, pitch=pitch_deg)
            return None

        # Only where this window was asked to invent, and only as strongly as its own edge
        # allows. `room.editable` rather than the raw view mask, so anything the segmenter
        # protected inside the window — a real window, a radiator — is left alone here too.
        alpha = (room.editable > 127).astype(np.float32) * falloff
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.5, view_size / 256.0))
        alpha *= (room.editable > 127).astype(np.float32)
        views.paint_colour(canvas, result.image, alpha, rot, K)

        # Struck off only where this window spoke at nearly full strength; the faded rim stays
        # on the list so the next window finishes it with real context on both sides.
        views.paint_mask(remaining, ((editable > 0) & (settled > 0)).astype(np.uint8) * 255, rot, K, value=0)
        done += 1

        if progress is not None:
            progress((index + 1) / float(len(order)), f"{resolved.id} {index + 1}/{len(order)}")

    leftover = float((remaining > 127).mean())

    # The guard, applied. Everything the windows drew near a pole is faded back toward the
    # photograph here, so the boundary is a gradient in COLOUR rather than a gradient in how
    # many pixels got touched. `rint` for the same reason `restyle_photo` uses it: a bare cast
    # truncates, and a whole band one grey level dark is a band.
    weight = guard[..., None]
    canvas = np.rint(
        np.clip(canvas.astype(np.float32) * weight + pano.astype(np.float32) * (1.0 - weight), 0, 255)
    ).astype(np.uint8)

    elapsed = int((time.perf_counter() - started) * 1000)
    # Solid-angle weighted, because an equirect row near a pole covers almost no sphere and an
    # unweighted share would report the caps as a third of the room.
    weights = _solid_angle_weights(height, width)
    changed = (np.abs(canvas.astype(np.int16) - pano.astype(np.int16)).max(axis=2) > 6).astype(np.float32)
    restyled = float((weights * changed).sum())

    log.info(
        "restyle_pano.done",
        style=resolved.id,
        windows=done,
        seconds=round(elapsed / 1000.0, 1),
        restyled=round(restyled, 4),
        leftover=round(leftover, 5),
    )
    return PanoRestyleOutput(
        panorama=canvas,
        style_id=resolved.id,
        windows=done,
        elapsed_ms=elapsed,
        restyled_fraction=restyled,
        protected_fraction=max(0.0, 1.0 - restyled),
    )


def _solid_angle_weights(height: int, width: int) -> np.ndarray:
    """Per-pixel share of the sphere, summing to 1. Mirrors the engine's own convention."""
    rows = np.arange(height, dtype=np.float64)
    phi = np.pi / 2.0 - np.pi * (rows + 0.5) / float(height)
    band = np.cos(phi)
    band /= band.sum() * width
    return np.repeat(band.astype(np.float32)[:, None], width, axis=1)
