/**
 * Where a floor ring lands, and which one a tap hit.
 *
 * Separated from `PortalRings.tsx`, which draws them, because this half is arithmetic and the
 * other half is a mesh: a ring in the wrong place is a bug you can only see by standing in the
 * room, whereas the rule that puts it there can be checked at a desk. Nothing here touches
 * React or the renderer, so it runs under the plain-Node test environment.
 */

import * as THREE from "three";
import type { Hotspot } from "@/lib/types";
import { directionFromAngles } from "./useLookControls";

/**
 * Eye height above the floor, in metres.
 *
 * The engine's own default for a hand-held sweep (`ai/measure.DEFAULT_CAMERA_HEIGHT_CM`, 155
 * cm) rounded to where a phone actually rides. It is not a free parameter: it is the only
 * thing converting a direction into a distance, so it sets how far away a ring lands and
 * therefore how big it looks. Too low and every ring crowds the viewer's feet; too high and a
 * doorway ring lands out by the far wall.
 */
export const EYE_HEIGHT = 1.6;

/** Outer radius of a ring, in metres. About the footprint of a person standing. */
export const RING_RADIUS = 0.44;

/**
 * How far away a ring may land before it is dropped.
 *
 * A portal a degree or two below the horizon solves to a floor point tens of metres away —
 * beyond the room, and a sub-pixel ring by the time it is drawn. That is not a portal anybody
 * placed on purpose, it is a misplaced one, and drawing it would put an invisible click target
 * in the middle of a wall.
 */
export const MAX_DISTANCE = 14;

/** Portals nearer the horizon than this have no usable floor point. */
export const MIN_DEPRESSION = 0.035; // ~2 degrees

/** How close a tap has to be, in radians, to count as hitting a ring. */
export const TAP_TOLERANCE = 0.09;

// Reused across calls to keep the per-frame path free of allocation. Each has ONE writer:
// `_dir` belongs to `floorPoint`, `_tap` and `_other` to `hitTest`, so no call can land in
// the middle of another's use of the same vector.
const _dir = new THREE.Vector3();
const _tap = new THREE.Vector3();
const _other = new THREE.Vector3();

/** Where a portal's ring sits, or ``null`` when its ray never reaches the floor. */
export function floorPoint(yaw: number, pitch: number): THREE.Vector3 | null {
  directionFromAngles(yaw, pitch, _dir);
  if (_dir.y > -MIN_DEPRESSION) return null;
  const distance = -EYE_HEIGHT / _dir.y;
  if (distance > MAX_DISTANCE) return null;
  return _dir.clone().multiplyScalar(distance);
}

/**
 * The portal a tap at (yaw, pitch) landed on, or null.
 *
 * Angular rather than planar: the tolerance is a cone around the ring's own direction, which
 * is what "I aimed at that thing" means to a viewer regardless of how far away it is. A ring
 * at the far wall is a smaller target on screen — correctly, because it is further away — and
 * the nearest match wins when two overlap.
 */
export function hitTest(
  portals: Hotspot[],
  yaw: number,
  pitch: number,
  tolerance = TAP_TOLERANCE,
): Hotspot | null {
  directionFromAngles(yaw, pitch, _tap);
  let best: Hotspot | null = null;
  let bestAngle = tolerance;
  for (const portal of portals) {
    if (!floorPoint(portal.yaw, portal.pitch)) continue;
    directionFromAngles(portal.yaw, portal.pitch, _other);
    const angle = _tap.angleTo(_other);
    if (angle < bestAngle) {
      bestAngle = angle;
      best = portal;
    }
  }
  return best;
}

/** A little more than the ring's own extent, so the marker clears the stroke, not kisses it. */
const TAG_LIFT_MARGIN = 0.028; // ~1.6 degrees

/**
 * How far above a ring its editing handle has to sit, in radians of pitch.
 *
 * In the editor a portal is drawn twice: as the ring the visitor will see, and as the small
 * marker every hotspot gets, which is what you click to edit and drag to move. Both are placed
 * at the portal's own direction, so the marker lands exactly in the middle of the ring — and,
 * being a DOM node above the canvas, it swallows every click aimed at it. That is why the ring
 * was not clickable in the editor: aiming at it opened the tag form instead of walking.
 *
 * Rather than take one of the two away, the marker is lifted clear. The amount is not a matter
 * of taste, it is the ring's own size: the highest point of the ring ON SCREEN is its FAR edge,
 * which is further away and therefore nearer the horizon. Clear that edge and the whole annulus
 * is exposed — at every distance, which is the point. A ring at your feet is a wide ellipse and
 * one across the room is a sliver; a fixed offset would clear one and not the other.
 */
export function tagLift(pitch: number): number {
  if (pitch >= -MIN_DEPRESSION) return 0;
  const ground = EYE_HEIGHT / Math.tan(-pitch);
  const farEdge = -Math.atan(EYE_HEIGHT / (ground + RING_RADIUS));
  return farEdge - pitch + TAG_LIFT_MARGIN;
}

/**
 * The ring pitch that a handle dropped at `dropped` was aiming at. Inverse of {@link tagLift}.
 *
 * Subtracting `tagLift(dropped)` is the obvious move and is NOT the inverse, because the lift
 * depends on the pitch being solved for: the lift shrinks as a ring gets further away, so every
 * drag returned a slightly shallower ring than the one you dropped. Measured at 0.66 degrees a
 * drag, which sounds harmless and is not — ten nudges walked a doorway ring a full metre across
 * the floor, always in the same direction, which is exactly the kind of drift nobody attributes
 * to the tool.
 *
 * Solved by fixed-point iteration instead. `tagLift` changes about a quarter as fast as its
 * argument, so every pass cuts the error to roughly a quarter; six of them run once per drag,
 * which is free. Measured over twenty consecutive round trips the ring moves at most 0.007
 * degrees — a fraction of a millimetre on the floor, against the metre it used to walk. The
 * residue is largest for the steepest rings, where the lift varies fastest.
 */
export function unliftPitch(dropped: number): number {
  let pitch = dropped - tagLift(dropped);
  for (let i = 0; i < 6; i += 1) pitch = dropped - tagLift(pitch);
  return pitch;
}
