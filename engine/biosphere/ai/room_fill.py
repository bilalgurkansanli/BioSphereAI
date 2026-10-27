"""Rebuilding a room through perspective windows, one at a time.

The problem this replaces
-------------------------
Inpainting the equirect canvas directly produced, in a user's words, something closer to
scribbling over the furniture than to a room. That is a fair description and it has a cause:
the network was asked to continue structure in a projection where structure does not look
like itself. On an equirect canvas a skirting board is a sine wave, a floorboard curves, and
the line where a wall meets a floor is different in every column. LaMa has never seen a room
drawn that way, so it falls back on what it can always do — produce a smooth field that
matches the colours around the hole. Hence fog.

Seen through a 90-degree window the same room is an ordinary photograph. The wall is flat,
the skirting is a straight line, the floorboards are parallel and converge on a vanishing
point. That is the input the model was trained on, and given it, it draws a wall meeting a
floor rather than a cloud.

Why the windows go one at a time
---------------------------------
Because they overlap, and two independent inpaintings of the same wall are two different
walls. Filling them all and averaging gives a ghost; filling them all and picking one gives a
seam down the middle of the overlap.

So each window is rendered from the panorama **as it stands**, which already contains
everything the previous windows rebuilt. A window's job is only what is still missing, and it
does that job with its neighbours' work visible as context — so the floorboards it draws are
a continuation of the floorboards already there rather than a fresh invention. The order is
easiest-first: a window with a little to do has the most context to do it from, and its
result becomes context for the harder ones.

What is left over
------------------
A window contributes at full strength in its middle and fades out toward its edge, and only
the full-strength part is struck off the to-do list. The fade, the poles and anything no
window could reach fall to a final pass on the whole canvas — which is the equirect fill this
module exists to avoid, now facing a few percent of the sphere instead of half of it, where
its softness costs nothing.
"""

from __future__ import annotations

import numpy as np

from biosphere.ai import views
from biosphere.core.logging import get_logger

__all__ = ["fill_by_views", "FILL_VIEW_PLAN", "VIEW_SIZE"]

log = get_logger("biosphere.ai.room_fill")

#: Where the windows point, as ``(pitch_deg, n_yaw)``.
#:
#: Three rows at a 90-degree field of view reach from -80 to +80 degrees of elevation, which
#: is the whole of what may be removed — the poles are protected. Eight yaws per row is 45
#: degrees of overlap between neighbours, so every direction is seen well inside at least one
#: window rather than only at the edge of one.
FILL_VIEW_PLAN: tuple[tuple[float, int], ...] = ((-35.0, 8), (0.0, 8), (35.0, 8))

#: 1024 px across 90 degrees is 11.4 px per degree, the same density a 4096-wide equirect
#: carries. Anything less and the round trip through the window is itself a downscale, which
#: is the blur this module was written to remove.
VIEW_SIZE = 1024

#: How far into a window's edge the contribution fades, as a fraction of its width. The fade
#: is what keeps two neighbours from meeting at a line; it is not free, because faded pixels
#: are not struck off and have to be finished by somebody else.
_EDGE_FADE = 0.12

#: Above this weight a pixel counts as finished and later windows will not revisit it.
_SETTLED_ALPHA = 0.85

#: The mask is grown by this fraction of a window before being handed to the network. LaMa
#: does better when it is not asked to reason right up to an edge it can see is wrong, and
#: the extra ring is background that gets redrawn as itself.
_VIEW_MASK_GROW = 0.006


def _edge_falloff(size: int) -> np.ndarray:
    """1 in the middle of a window, ramping to 0 at its border."""
    ramp = np.ones(size, np.float32)
    width = max(1, int(round(size * _EDGE_FADE)))
    taper = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp[:width] = taper
    ramp[-width:] = taper[::-1]
    return np.minimum(ramp[:, None], ramp[None, :])


#: The scouting pass's own window size, and the canvas it scouts on.
#:
#: Small deliberately: this pass only decides which windows have work and in what order, and
#: rendering two dozen full-size windows before filling any of them would cost more than the
#: ordering saves.
_SCOUT_SIZE = 128
_SCOUT_CANVAS = (512, 256)


def _view_order(hole: np.ndarray) -> list[tuple[float, float, float]]:
    """``(load, yaw, pitch)`` for every window that has anything to do, easiest first.

    IT BUILDS ITS OWN INTRINSICS AND WILL NOT TAKE THE CALLER'S, and that is a scar rather
    than a preference. It used to accept ``K``, and ``fill_by_views`` passed the one it warps
    with — built for :data:`VIEW_SIZE`, 1024 px, principal point at (512, 512). Rendered at 128
    px against that K, the pixel grid 0..127 sits in the far corner of the frame instead of
    around its centre, so each "window" scouted a narrow cone pointing somewhere near the
    corner of the window it was named after.

    MEASURED, on a single block of mask spanning the horizon: nine windows genuinely contained
    it, the mismatched scout found ONE and gave it a load of 0.42 against its true 0.04. The
    consequences are the two things this function exists to get right. Coverage — a room whose
    mask lands where nothing was scouted is reported ``windows=0`` and comes back with the
    furniture still in it, silently, which is how this was found. And order — the loads are the
    whole basis of easiest-first, and a mis-aimed cone returns noise, so the sequential fill was
    seeding its hardest windows off its worst context.

    Intrinsics and render size are one thing, so they are constructed in one place.
    """
    import cv2

    K = views.intrinsics(_SCOUT_SIZE, views.VIEW_FOV_DEG)
    small = cv2.resize(hole, _SCOUT_CANVAS, interpolation=cv2.INTER_AREA)
    plan: list[tuple[float, float, float]] = []
    for pitch_deg, n_yaw in FILL_VIEW_PLAN:
        for index in range(n_yaw):
            yaw_deg = 360.0 * index / float(n_yaw)
            rot = views.rotation(yaw_deg, pitch_deg)
            patch = views.render_view(small, rot, K, _SCOUT_SIZE, interpolation=cv2.INTER_AREA)
            load = float((patch > 8).mean())
            if load > 0.0:
                plan.append((load, yaw_deg, pitch_deg))
    plan.sort()
    return plan


def fill_by_views(
    image: np.ndarray,
    hole: np.ndarray,
    inpaint: object,
    *,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild ``hole`` window by window. Returns ``(panorama, still_missing)``.

    ``inpaint`` is a callable ``(bgr, mask, cyclic=...) -> bgr | None`` — the contract
    :mod:`biosphere.ai.backend` exposes, passed in so this module never imports torch and
    never needs to know whether the work happened here or on another machine.
    """
    import cv2

    pano = image.copy()
    remaining = hole.copy()
    K = views.intrinsics(VIEW_SIZE, views.VIEW_FOV_DEG)
    falloff = _edge_falloff(VIEW_SIZE)
    settled = (falloff >= _SETTLED_ALPHA).astype(np.uint8) * 255
    grow = max(3, int(round(VIEW_SIZE * _VIEW_MASK_GROW)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))

    filled_windows = 0
    for _attempt in range(passes):
        order = _view_order(remaining)
        if not order:
            break
        for _, yaw_deg, pitch_deg in order:
            rot = views.rotation(yaw_deg, pitch_deg)
            view_mask = views.render_view(
                remaining, rot, K, VIEW_SIZE, interpolation=cv2.INTER_NEAREST
            )
            if view_mask.ndim == 3:
                view_mask = view_mask[..., 0]
            if not (view_mask > 127).any():
                continue

            view = views.render_view(pano, rot, K, VIEW_SIZE)
            asked = cv2.dilate((view_mask > 127).astype(np.uint8) * 255, kernel)
            # A window is not cyclic; see `lama.inpaint`.
            painted = inpaint(view, asked, cyclic=False)  # type: ignore[operator]
            if painted is None:
                log.warning("room_fill.inpaint_unavailable", yaw=yaw_deg, pitch=pitch_deg)
                return pano, remaining

            # Only where this window was asked to invent something, and only as strongly as
            # its own edge allows. Everything else in the window is the photograph already.
            alpha = (view_mask > 127).astype(np.float32) * falloff
            alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.5, VIEW_SIZE / 256.0))
            alpha *= (view_mask > 127).astype(np.float32)
            views.paint_colour(pano, painted, alpha, rot, K)

            # Struck off only where this window spoke at full strength; the faded rim stays
            # on the list so a neighbour finishes it with real context on both sides.
            done = ((view_mask > 127) & (settled > 0)).astype(np.uint8) * 255
            views.paint_mask(remaining, done, rot, K, value=0)
            filled_windows += 1

    log.info(
        "room_fill.done",
        windows=filled_windows,
        passes=passes,
        leftover=round(float((remaining > 0).mean()), 5),
    )
    return pano, remaining
