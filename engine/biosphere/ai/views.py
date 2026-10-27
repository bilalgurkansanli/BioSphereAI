"""Perspective windows onto a panorama, and the way back.

Both AI stages need the same thing: an equirectangular sphere resampled the way an ordinary
camera would see it, and their results carried back onto the sphere. This module owns that
geometry so the segmenter and the inpainter cannot drift apart on it — a half-degree of
disagreement between the two would put masks slightly off the objects they were drawn for.

Why anything looks through a window at all
------------------------------------------
An equirect canvas is not a photograph. Straight lines bow, verticals converge toward the
poles, and a sofa spanning ninety degrees of yaw is smeared across a quarter of the image.
Every model worth using was trained on ordinary photographs, so handing one the raw canvas
wastes it — and for an inpainter the waste is not subtle. Asked to rebuild a wall directly on
the equirect, LaMa produces a soft field with no skirting board and no floorboards, because
it has never seen a room drawn that way and there is no straight line in its input to
continue. Asked the same question through a 90-degree window, it draws a wall meeting a floor
at a straight line, which is what a room is.

The convention here is the engine's own: world is +X right, **+Y down**, +Z forward, and
``R`` maps world to camera. See :mod:`biosphere.pano.types`.
"""

from __future__ import annotations

import numpy as np

from biosphere.pano import geometry as geo

__all__ = [
    "VIEW_PIXELS",
    "VIEW_FOV_DEG",
    "intrinsics",
    "rotation",
    "render_view",
    "paint_mask",
    "paint_votes",
    "paint_labels",
    "paint_colour",
    "equirect_window",
]

#: Default window size and field of view.
#:
#: 90 degrees keeps the projection close enough to an ordinary lens that a chair still looks
#: like the chairs in a training set, while overlapping its neighbours enough that an object
#: cut in half by one window is whole in the next. At 1024 px that is 11.4 px per degree,
#: which is exactly what a 4096-wide equirect carries — so nothing is resampled downward on
#: the way out and back.
VIEW_PIXELS = 1024
VIEW_FOV_DEG = 90.0


def intrinsics(size: int, fov_deg: float) -> np.ndarray:
    focal = 0.5 * size / np.tan(0.5 * np.radians(fov_deg))
    return np.array([[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]], np.float64)


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World -> camera for a window aimed at ``(yaw, pitch)``.

    ``+Y`` is down, so looking *up* is a negative rotation about X. Getting that sign wrong
    does not raise anything; it silently mirrors the upper and lower halves of the room.
    """
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    r_yaw = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], np.float64)
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], np.float64)
    return r_pitch @ r_yaw


def render_view(pano: np.ndarray, rot: np.ndarray, K: np.ndarray, size: int, *, interpolation: int | None = None) -> np.ndarray:
    """One rectilinear window onto the sphere."""
    import cv2

    rays = geo.camera_rays(size, size, K)
    world = rays.reshape(-1, 3) @ rot            # R.T @ d, written as a row product
    u, v = geo.equirect_from_rays(world.reshape(size, size, 3), pano.shape[1], pano.shape[0])
    mode = cv2.INTER_LINEAR if interpolation is None else interpolation
    return geo.sample_equirect(pano, u, v, interpolation=mode)


def equirect_window(
    rot: np.ndarray, K: np.ndarray, size: int, width: int, height: int, *, margin: int = 4
) -> tuple[int, int, int, int]:
    """Rows and unwrapped columns of the equirect a window can possibly touch.

    ``(v0, v1, u0, u1)`` with ``u`` NOT wrapped: it may be negative or exceed ``width`` and
    the caller indexes modulo. Without this every projection back would sweep the whole
    canvas — sixteen windows times eight million pixels, for a window that covers a
    twentieth of it.
    """
    edge = np.linspace(0.0, size - 1.0, 96)
    ones = np.ones_like(edge)
    border = np.concatenate(
        [
            np.stack([edge, 0.0 * ones], axis=1),
            np.stack([edge, (size - 1.0) * ones], axis=1),
            np.stack([0.0 * ones, edge], axis=1),
            np.stack([(size - 1.0) * ones, edge], axis=1),
        ]
    )
    rays = geo.unproject_pixels(border[:, 0], border[:, 1], K)
    world = np.asarray(rays).reshape(-1, 3) @ rot
    u, v = geo.equirect_from_rays(world.reshape(1, -1, 3), width, height)
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()

    # Unwrap the columns against the window's own centre so a window straddling the seam
    # stays one interval instead of two.
    centre = np.asarray(np.array([[0.0, 0.0, 1.0]]) @ rot).reshape(1, 1, 3)
    cu, _ = geo.equirect_from_rays(centre, width, height)
    cu = float(np.asarray(cu).ravel()[0])
    u = cu + ((u - cu + width / 2.0) % width) - width / 2.0

    v0 = int(max(0, np.floor(v.min()) - margin))
    v1 = int(min(height, np.ceil(v.max()) + margin + 1))
    u0 = int(np.floor(u.min()) - margin)
    u1 = int(np.ceil(u.max()) + margin + 1)
    # A window that sees a pole spans every column; the border loses that because the pole
    # itself is interior to it.
    if v0 <= 0 or v1 >= height:
        u0, u1 = 0, width
    return v0, v1, u0, u1


def _view_coords(
    rot: np.ndarray, K: np.ndarray, width: int, height: int, window: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each equirect pixel of ``window``: its ``(x, y)`` in the view and whether it lands.

    Returns ``(rows, cols, x, y, inside)`` with ``cols`` already wrapped into the canvas.
    """
    v0, v1, u0, u1 = window
    rows = np.arange(v0, v1)
    cols = np.arange(u0, u1)
    theta = (2.0 * np.pi * ((cols % width) + 0.5) / float(width) - np.pi).astype(np.float32)
    phi = (0.5 * np.pi - np.pi * (rows + 0.5) / float(height)).astype(np.float32)
    rays = geo.rays_from_equirect(theta, phi)                    # (h, w, 3) world
    cam = rays.reshape(-1, 3) @ rot.T                            # R @ d
    z = cam[:, 2]
    in_front = z > 1e-6
    safe = np.where(in_front, z, 1.0)
    x = (K[0, 0] * cam[:, 0] / safe + K[0, 2]).astype(np.float32)
    y = (K[1, 1] * cam[:, 1] / safe + K[1, 2]).astype(np.float32)
    return rows, cols % width, x, y, in_front


def paint_mask(out: np.ndarray, view_mask: np.ndarray, rot: np.ndarray, K: np.ndarray, value: int = 255) -> None:
    """Project a view-space mask onto the equirect canvas, in place."""
    height, width = out.shape[:2]
    size = view_mask.shape[0]
    window = equirect_window(rot, K, size, width, height)
    rows, cols, x, y, in_front = _view_coords(rot, K, width, height, window)

    inside = in_front & (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
    if not inside.any():
        return
    xs = np.clip(x, 0, size - 1).astype(np.int32)
    ys = np.clip(y, 0, size - 1).astype(np.int32)
    hit = inside & (view_mask[ys, xs] > 0)
    if not hit.any():
        return
    grid = hit.reshape(rows.size, cols.size)
    rr, cc = np.nonzero(grid)
    out[rows[rr], cols[cc]] = value


def paint_votes(
    votes: np.ndarray, seen: np.ndarray, view_mask: np.ndarray, rot: np.ndarray, K: np.ndarray
) -> None:
    """Add one window's opinion to a tally: ``seen`` everywhere it looks, ``votes`` where it
    says yes.

    A UNION OVER WINDOWS IS THE WRONG COMBINATION AND IT IS WHY THIS EXISTS. Every direction
    is inside two or three windows, so under a union one false positive anywhere makes a
    pixel furniture for good — and the errors of sixteen windows add up instead of cancelling.
    Measured on a real bedroom, the union put 47 % of the sphere in the mask and left a single
    window 88 % masked, which is a room the inpainter cannot rebuild because almost none of it
    is left to rebuild from. A vote lets the windows that saw a thing clearly outweigh the one
    that caught it edge-on.
    """
    height, width = seen.shape[:2]
    size = view_mask.shape[0]
    window = equirect_window(rot, K, size, width, height)
    rows, cols, x, y, in_front = _view_coords(rot, K, width, height, window)

    inside = in_front & (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
    if not inside.any():
        return
    xs = np.clip(x, 0, size - 1).astype(np.int32)
    ys = np.clip(y, 0, size - 1).astype(np.int32)

    shape = (rows.size, cols.size)
    looked = inside.reshape(shape)
    said_yes = (inside & (view_mask[ys, xs] > 0)).reshape(shape)

    index = np.ix_(rows, cols)
    seen[index] += looked.astype(np.uint16)
    votes[index] += said_yes.astype(np.uint16)


def paint_labels(
    out: np.ndarray, best: np.ndarray, view_labels: np.ndarray, rot: np.ndarray, K: np.ndarray
) -> None:
    """Project a view's per-pixel class ids onto the canvas, keeping the SQUAREST opinion.

    ``best`` holds, per canvas pixel, the cosine between the direction and the optical axis of
    the window that last wrote it; a window only overwrites where it saw the direction more
    head-on. Last-writer-wins would let the order of the sweep decide a pixel's class, and a
    segmenter is least reliable exactly at the edge of its frame, where an object is cut off and
    the perspective is most stretched.
    """
    height, width = out.shape[:2]
    size = view_labels.shape[0]
    window = equirect_window(rot, K, size, width, height)
    rows, cols, x, y, in_front = _view_coords(rot, K, width, height, window)

    inside = in_front & (x >= 0) & (x < size - 1) & (y >= 0) & (y < size - 1)
    if not inside.any():
        return
    # Cosine to the optical axis: the camera-frame z of the unit ray.
    fx, cx = float(K[0, 0]), float(K[0, 2])
    fy, cy = float(K[1, 1]), float(K[1, 2])
    nx, ny = (x - cx) / fx, (y - cy) / fy
    centrality = (1.0 / np.sqrt(1.0 + nx * nx + ny * ny)).astype(np.float32)

    shape = (rows.size, cols.size)
    xs = np.clip(x, 0, size - 1).astype(np.int32)
    ys = np.clip(y, 0, size - 1).astype(np.int32)
    labels = view_labels[ys, xs].reshape(shape)
    score = np.where(inside, centrality, -1.0).reshape(shape)

    index = np.ix_(rows, cols)
    current = best[index]
    better = score > current
    patch = out[index]
    patch[better] = labels[better]
    out[index] = patch
    current[better] = score[better]
    best[index] = current


def paint_colour(
    pano: np.ndarray,
    view_image: np.ndarray,
    view_alpha: np.ndarray,
    rot: np.ndarray,
    K: np.ndarray,
) -> None:
    """Composite a window's pixels back onto the sphere, weighted by ``view_alpha``.

    Bilinear on the way back, because this is a photograph rather than a label: nearest
    neighbour here shows as a stipple along every edge the window reconstructed.
    """
    import cv2

    height, width = pano.shape[:2]
    size = view_image.shape[0]
    window = equirect_window(rot, K, size, width, height)
    rows, cols, x, y, in_front = _view_coords(rot, K, width, height, window)

    inside = in_front & (x >= 0) & (x <= size - 1) & (y >= 0) & (y <= size - 1)
    if not inside.any():
        return

    shape = (rows.size, cols.size)
    map_x = np.where(inside, x, 0.0).reshape(shape).astype(np.float32)
    map_y = np.where(inside, y, 0.0).reshape(shape).astype(np.float32)
    sampled = cv2.remap(view_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    weight = cv2.remap(view_alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    weight = np.where(inside.reshape(shape), weight, 0.0).astype(np.float32)[..., None]

    patch = pano[np.ix_(rows, cols)].astype(np.float32)
    blended = patch * (1.0 - weight) + sampled.astype(np.float32) * weight
    pano[np.ix_(rows, cols)] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
