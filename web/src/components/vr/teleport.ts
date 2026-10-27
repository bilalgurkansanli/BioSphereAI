/**
 * Moving through a Gaussian-splat walkthrough in a headset without walking into garbage.
 *
 * ── WHERE A VISITOR MAY STAND ─────────────────────────────────────────────────────────────
 * A splat reconstruction is only a room near where the camera went. A metre or two off the
 * filmed route the model is needles, floaters and holes — the "unobserved splat garbage" a
 * free teleport lands people in, where the experience falls apart for anyone wearing it. The
 * rule is the walkthrough's own corridor (`walkMath.corridorOf`, `PATH_RADIUS_M`): within a
 * metre of where the phone stood, never bridging a gap in the route longer than a stride. It
 * was measured on the first delivered walk — the old walk box let a visitor reach 161 m² of
 * floor of which only 38 % was within a metre of the route — and a headset puts the EYE at
 * the landing point, which makes the off-route picture worse, not better. Using the same
 * corridor means the 2D walk mode and the headset agree to the centimetre about where a
 * visitor may be.
 *
 * A target just past the edge is pulled back onto it rather than refused, up to `SNAP_SLACK`
 * further: pointing at the floor beside the route and arriving at its edge is what a visitor
 * expects, and a red "no" for missing by twenty centimetres is not.
 *
 * ── WHERE THE FLOOR IS ────────────────────────────────────────────────────────────────────
 * `level.py` puts every area's floor at y = 0 and scales it by assuming the phone rode 1.45 m
 * above it (`ASSUMED_CAMERA_HEIGHT_M`). An area levelled a little off carries the error in its
 * camera heights, so the floor under an area is its median camera height minus that same 1.45
 * — 0 for a well-levelled area, the truth for a slightly mis-levelled one. In `local-floor`
 * space the visitor's real eye height then rides on top, which is right: a tall visitor sees
 * the room from higher up, as they would standing in it.
 *
 * ── THE RIG ───────────────────────────────────────────────────────────────────────────────
 * The headset reports the head relative to a tracking origin; locomotion moves and turns that
 * origin (the "rig"). Teleporting puts the HEAD on the target, not the rig — the visitor may
 * be standing a metre from the middle of their play space — and a snap turn pivots about the
 * head, or the world swings round them like a door.
 */

import type { WalkManifest } from "@/lib/types";
import { PATH_RADIUS_M, corridorOf, drawableAreas, nearestOnCorridor, toViewer, type Corridor } from "@/components/walk/walkMath";
import { rotateY, type Vec3 } from "./gaze";

export type Vec2 = [number, number];

/** `engine/biosphere/splat/level.py: ASSUMED_CAMERA_HEIGHT_M`. */
export const CAPTURE_HEIGHT = 1.45;
/** How far past the corridor's edge a target is pulled back rather than refused, metres. */
export const SNAP_SLACK = 0.6;
/** Teleports shorter than this are not worth a fade. */
export const MIN_TELEPORT = 0.35;

/** Median camera height minus the assumed capture height, held to a sane band. */
export function floorHeightOf(cameras: readonly ArrayLike<number>[]): number {
  if (cameras.length === 0) return 0;
  const heights = cameras.map((c) => toViewer(c)[1]).sort((a, b) => a - b);
  const mid = heights.length >> 1;
  const median = heights.length % 2 ? heights[mid] : (heights[mid - 1] + heights[mid]) / 2;
  const floor = median - CAPTURE_HEIGHT;
  // A metre either way is not a levelling error, it is a broken area; trust y = 0 instead.
  return Math.abs(floor) <= 1 ? floor : 0;
}

/** Where a visitor standing in one area may go, and the floor under each part of it. */
export interface TeleportSpace {
  corridor: Corridor;
  areas: Array<{ key: string; corridor: Corridor; floorY: number }>;
}

/**
 * The corridor a visitor in `activeKey` may teleport along — the walk engine's own rule:
 * every placed area shares one frame, so standing anywhere in the property all of them count;
 * an unplaced room is its own island, opened by name in its own coordinates, and only its own
 * route makes sense there.
 */
export function teleportSpace(manifest: WalkManifest, activeKey: string | null): TeleportSpace {
  const areas = drawableAreas(manifest);
  const active = areas.find((a) => a.key === activeKey);
  const chosen = active && !active.placed ? [active] : areas.filter((a) => a.placed);
  return {
    corridor: corridorOf(chosen),
    areas: chosen.map((a) => ({ key: a.key, corridor: corridorOf([a]), floorY: floorHeightOf(a.cameras) })),
  };
}

/** The floor under (x, z): that of the area whose route passes nearest. */
export function floorAt(space: TeleportSpace, x: number, z: number): number {
  let best = Infinity;
  let floor = 0;
  for (const area of space.areas) {
    const near = nearestOnCorridor(area.corridor, x, z);
    if (near.d < best) {
      best = near.d;
      floor = area.floorY;
    }
  }
  return floor;
}

export interface TeleportVerdict {
  valid: boolean;
  /** Where the visitor would land — the target, or the target pulled back to the corridor. */
  point: Vec2;
  snapped: boolean;
  floorY: number;
}

export function validateTeleport(target: Vec2, space: TeleportSpace, radius = PATH_RADIUS_M, slack = SNAP_SLACK): TeleportVerdict {
  const near = nearestOnCorridor(space.corridor, target[0], target[1]);
  if (!Number.isFinite(near.d)) return { valid: false, point: target, snapped: false, floorY: 0 };
  if (near.d <= radius) return { valid: true, point: target, snapped: false, floorY: floorAt(space, target[0], target[1]) };
  if (near.d <= radius + slack) {
    const k = radius / near.d;
    const point: Vec2 = [near.x + (target[0] - near.x) * k, near.z + (target[1] - near.z) * k];
    return { valid: true, point, snapped: true, floorY: floorAt(space, point[0], point[1]) };
  }
  return { valid: false, point: target, snapped: false, floorY: floorAt(space, target[0], target[1]) };
}

// ---------------------------------------------------------------------------------
// Aiming
// ---------------------------------------------------------------------------------

/** Where a straight ray meets the floor, within `maxDistance` metres, or null. */
export function rayToFloor(origin: Vec3, dir: Vec3, floorY: number, maxDistance = 12): Vec3 | null {
  const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
  const dy = dir[1] / n;
  if (dy > -0.02) return null;
  const t = (floorY - origin[1]) / dy;
  if (t <= 0 || t > maxDistance) return null;
  return [origin[0] + (dir[0] / n) * t, floorY, origin[2] + (dir[2] / n) * t];
}

export interface ArcOptions {
  /** Launch speed, m/s. 7 lands a 45-degree throw from hand height about 6 m away. */
  speed?: number;
  gravity?: number;
  /** Longest flight considered, seconds. */
  maxTime?: number;
  /** Integration step, seconds — also the spacing of the points drawn. */
  step?: number;
}

/**
 * The teleport arc: a thrown ball from the controller, landing on the floor.
 *
 * An arc rather than a straight ray because a straight ray needs the hand pointed almost at
 * the horizon to reach across a room, where a tiny wrist movement is metres on the floor. The
 * parabola maps the comfortable range of a wrist onto the comfortable range of distances.
 */
export function arcToFloor(origin: Vec3, dir: Vec3, floorY: number, options: ArcOptions = {}): { landing: Vec3 | null; points: Vec3[] } {
  const speed = options.speed ?? 7;
  const g = options.gravity ?? 9.81;
  const maxTime = options.maxTime ?? 1.6;
  const step = options.step ?? 0.025;
  const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
  const v: Vec3 = [(dir[0] / n) * speed, (dir[1] / n) * speed, (dir[2] / n) * speed];
  const points: Vec3[] = [[...origin] as Vec3];
  let prev: Vec3 = [...origin] as Vec3;
  for (let t = step; t <= maxTime + 1e-9; t += step) {
    const p: Vec3 = [origin[0] + v[0] * t, origin[1] + v[1] * t - 0.5 * g * t * t, origin[2] + v[2] * t];
    if (p[1] <= floorY) {
      const k = (prev[1] - floorY) / (prev[1] - p[1]);
      const landing: Vec3 = [prev[0] + (p[0] - prev[0]) * k, floorY, prev[2] + (p[2] - prev[2]) * k];
      points.push(landing);
      return { landing, points };
    }
    points.push(p);
    prev = p;
  }
  return { landing: null, points };
}

// ---------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------

/** The tracking origin: position on the floor plane and turn about +Y (PlayCanvas/three sign). */
export interface Rig {
  x: number;
  z: number;
  yaw: number;
}

/** The head's position on the floor plane, from the rig and the head's tracked offset. */
export function headWorld(rig: Rig, headLocal: Vec2): Vec2 {
  const r = rotateY([headLocal[0], 0, headLocal[1]], rig.yaw);
  return [rig.x + r[0], rig.z + r[2]];
}

/** The rig position that puts the head on `target`, keeping the rig's turn. */
export function rigForTeleport(rig: Rig, headLocal: Vec2, target: Vec2): Rig {
  const r = rotateY([headLocal[0], 0, headLocal[1]], rig.yaw);
  return { x: target[0] - r[0], z: target[1] - r[2], yaw: rig.yaw };
}

/** The rig turned by `delta` about the head, which stays exactly where it was. */
export function rigAfterTurn(rig: Rig, headLocal: Vec2, delta: number): Rig {
  const head = headWorld(rig, headLocal);
  return rigForTeleport({ x: rig.x, z: rig.z, yaw: rig.yaw + delta }, headLocal, head);
}

/**
 * A smooth step from `from` to `to`, kept inside the corridor.
 *
 * Slides along the edge instead of stopping dead: if the full step leaves the corridor, each
 * axis is tried alone. A step that ends no further out than it began is always allowed, so a
 * visitor who has physically walked out of the corridor in a large play space can come back.
 */
export function constrainMove(from: Vec2, to: Vec2, space: TeleportSpace, radius = PATH_RADIUS_M): Vec2 {
  if (space.corridor.length < 4) return to;
  const distance = (p: Vec2) => nearestOnCorridor(space.corridor, p[0], p[1]).d;
  const limit = Math.max(radius, distance(from));
  if (distance(to) <= limit) return to;
  const alongX: Vec2 = [to[0], from[1]];
  if (distance(alongX) <= limit) return alongX;
  const alongZ: Vec2 = [from[0], to[1]];
  if (distance(alongZ) <= limit) return alongZ;
  return from;
}
