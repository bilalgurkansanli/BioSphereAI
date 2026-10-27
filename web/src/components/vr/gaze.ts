/**
 * What a ray in a headset is pointing at: a ring on the floor, a tag on the wall, a button on
 * a panel.
 *
 * Plain number tuples rather than three's vectors because two engines ask the same questions —
 * three.js for the 360 player, PlayCanvas for the walkthrough — and because a hit test is the
 * kind of thing that is only ever wrong in a headset, where it cannot be debugged. Here it can
 * be pinned.
 *
 * ── TWO KINDS OF DISTANCE ─────────────────────────────────────────────────────────────────
 * A panorama has no depth: every tag and every ring is drawn "at infinity", at the distance of
 * the photograph itself, so both eyes see it without disparity — the same depth as the wall it
 * sits on (see `pano/VrPanoStage`). Those targets are hit by DIRECTION. A controller is held
 * half a metre from the eye, so its ray is first carried out to the panorama's distance and
 * the direction is taken from the head to that point; at 50 m the parallax that removes is a
 * third of a degree, which is small but is also exactly the size of a far ring.
 *
 * The UI panel is the opposite: a real object at arm's length, hit by intersecting the actual
 * ray with its actual plane.
 */

export type Vec3 = [number, number, number];

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function length(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function normalize(v: Vec3): Vec3 {
  const n = length(v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** The angle between two directions, radians. Robust near 0 and π, unlike acos of a dot. */
export function angleBetween(a: Vec3, b: Vec3): number {
  const cx = a[1] * b[2] - a[2] * b[1];
  const cy = a[2] * b[0] - a[0] * b[2];
  const cz = a[0] * b[1] - a[1] * b[0];
  return Math.atan2(Math.hypot(cx, cy, cz), dot(a, b));
}

/**
 * Rotates about +Y by `angle`, with three's `rotation.y` sign: (x, z) -> (x cos + z sin,
 * -x sin + z cos). The content rotation in `orientation.ts` is this rotation, so a world ray
 * turned by `-contentYaw` is the same ray in panorama coordinates.
 */
export function rotateY(v: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c];
}

/** Unit direction for (yaw, pitch) in the player's convention — `directionFromAngles` in tuples. */
export function directionOf(yaw: number, pitch: number): Vec3 {
  const cp = Math.cos(pitch);
  return [cp * Math.cos(yaw), Math.sin(pitch), cp * Math.sin(yaw)];
}

/**
 * Where a ray from inside a sphere centred on the origin leaves it, as a unit direction from
 * the centre. The way a controller ray is turned into "what direction is it pointing at, as
 * seen from the head" when the targets are on the sphere.
 */
export function directionOnSphere(origin: Vec3, dir: Vec3, radius: number): Vec3 {
  const d = normalize(dir);
  const b = dot(origin, d);
  const c = dot(origin, origin) - radius * radius;
  // Origin inside the sphere (c < 0) means exactly one positive root.
  const t = -b + Math.sqrt(Math.max(0, b * b - c));
  return normalize([origin[0] + d[0] * t, origin[1] + d[1] * t, origin[2] + d[2] * t]);
}

// ---------------------------------------------------------------------------------
// Targets at infinity
// ---------------------------------------------------------------------------------

/** A marker on the panorama: a direction and how far around it still counts, radians. */
export interface AngularTarget {
  id: string;
  dir: Vec3;
  radius: number;
}

/**
 * A floor ring, in the geometry the 2D player already uses: its centre on the floor plane
 * `eyeHeight` below the eye, and its radius in metres (`portalGeometry.RING_RADIUS`).
 */
export interface RingTarget {
  id: string;
  centre: Vec3;
  radius: number;
}

export interface PickOptions {
  /** The target currently hovered; it keeps the hover until the ray is clearly off it. */
  current?: string | null;
  /**
   * How much bigger the current target is than a new one. Without it a ray resting on a
   * target's edge — and a head is never still — flickers the hover on and off, and every
   * flicker restarts a gaze dwell.
   */
  stickiness?: number;
}

export interface Pick {
  id: string;
  /** 0 at the target's centre, 1 at its edge. */
  score: number;
}

/**
 * The marker a direction falls on, nearest-centre first, or null.
 *
 * Score is the miss as a fraction of the target's own radius, so a large marker and a small one
 * compete fairly: aiming at the middle of the small one beats the rim of the large one.
 */
export function pickAngular(dir: Vec3, targets: readonly AngularTarget[], options: PickOptions = {}): Pick | null {
  const stick = options.stickiness ?? 1.3;
  let best: Pick | null = null;
  for (const target of targets) {
    const score = angleBetween(dir, target.dir) / target.radius;
    const limit = target.id === options.current ? stick : 1;
    if (score <= limit && (best === null || score < best.score)) best = { id: target.id, score };
  }
  return best;
}

/** Below this the floor intersection is too far away to mean anything (the 2D player's ~2°). */
const MIN_DEPRESSION = 0.035;

export interface RingPickOptions extends PickOptions {
  eyeHeight: number;
  /**
   * The smallest angular tolerance a ring gets, radians. The floor test alone is exact and
   * unfair: a ring 8 m away is a sliver a degree and a half tall, and a head cannot hold a
   * gaze that still. Near rings are big enough that the floor test dominates.
   */
  minAngle: number;
  /** Slack on the ring's own radius, so its rim counts. */
  radiusScale?: number;
}

/**
 * The floor ring a direction from the eye points at.
 *
 * Two tests, the better of which wins: where the ray meets the floor, measured against the
 * ring in metres (exact, and what "looking at the ring" means up close), and the plain angle
 * to the ring's centre (forgiving, and what it means far away).
 */
export function pickRing(dir: Vec3, rings: readonly RingTarget[], options: RingPickOptions): Pick | null {
  const stick = options.stickiness ?? 1.3;
  const scale = options.radiusScale ?? 1.15;
  const d = normalize(dir);
  let floor: Vec3 | null = null;
  if (d[1] < -MIN_DEPRESSION) {
    const t = -options.eyeHeight / d[1];
    floor = [d[0] * t, -options.eyeHeight, d[2] * t];
  }

  let best: Pick | null = null;
  for (const ring of rings) {
    const angular = angleBetween(d, ring.centre) / options.minAngle;
    const planar = floor
      ? Math.hypot(floor[0] - ring.centre[0], floor[2] - ring.centre[2]) / (ring.radius * scale)
      : Infinity;
    const score = Math.min(angular, planar);
    const limit = ring.id === options.current ? stick : 1;
    if (score <= limit && (best === null || score < best.score)) best = { id: ring.id, score };
  }
  return best;
}

// ---------------------------------------------------------------------------------
// Panels at arm's length
// ---------------------------------------------------------------------------------

/** A flat rectangle in space: its centre, its unit axes, and its half extents. */
export interface Panel {
  centre: Vec3;
  /** Unit vector along the panel's width, left to right. */
  right: Vec3;
  /** Unit vector along its height, bottom to top. */
  up: Vec3;
  halfWidth: number;
  halfHeight: number;
}

export interface PanelHit {
  /** 0 at the left edge, 1 at the right. */
  u: number;
  /** 0 at the TOP edge, 1 at the bottom — the same way down as the canvas it was drawn on. */
  v: number;
  distance: number;
}

/**
 * Where a ray meets a panel, from the front or the back, or null.
 *
 * Either face counts: a panel that follows the head can be glanced from the side while it is
 * catching up, and refusing the click then reads as the panel being broken.
 */
export function rayPanel(origin: Vec3, dir: Vec3, panel: Panel): PanelHit | null {
  const d = normalize(dir);
  const n: Vec3 = [
    panel.right[1] * panel.up[2] - panel.right[2] * panel.up[1],
    panel.right[2] * panel.up[0] - panel.right[0] * panel.up[2],
    panel.right[0] * panel.up[1] - panel.right[1] * panel.up[0],
  ];
  const denom = dot(n, d);
  if (Math.abs(denom) < 1e-6) return null;
  const toCentre: Vec3 = [panel.centre[0] - origin[0], panel.centre[1] - origin[1], panel.centre[2] - origin[2]];
  const t = dot(n, toCentre) / denom;
  if (t <= 0) return null;
  const hit: Vec3 = [origin[0] + d[0] * t - panel.centre[0], origin[1] + d[1] * t - panel.centre[1], origin[2] + d[2] * t - panel.centre[2]];
  const x = dot(hit, panel.right) / panel.halfWidth;
  const y = dot(hit, panel.up) / panel.halfHeight;
  if (Math.abs(x) > 1 || Math.abs(y) > 1) return null;
  return { u: (x + 1) / 2, v: (1 - y) / 2, distance: t };
}

/** A clickable region of a panel, in the panel's own u/v (v down). */
export interface PanelRegion {
  id: string;
  u0: number;
  v0: number;
  u1: number;
  v1: number;
}

export function regionAt(hit: PanelHit, regions: readonly PanelRegion[]): string | null {
  for (const r of regions) {
    if (hit.u >= r.u0 && hit.u <= r.u1 && hit.v >= r.v0 && hit.v <= r.v1) return r.id;
  }
  return null;
}

/**
 * A panel that faces a point and leans back to face it squarely, from its centre and that
 * point. Used to place the body-locked menu below the line of sight, tilted towards the eye.
 */
export function panelFacing(centre: Vec3, eye: Vec3, halfWidth: number, halfHeight: number): Panel {
  const back = normalize([eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2]]);
  // Right is horizontal: the panel never rolls, whatever the head does.
  let right: Vec3 = [back[2], 0, -back[0]];
  if (length(right) < 1e-6) right = [1, 0, 0];
  right = normalize(right);
  const up: Vec3 = [
    back[1] * right[2] - back[2] * right[1],
    back[2] * right[0] - back[0] * right[2],
    back[0] * right[1] - back[1] * right[0],
  ];
  return { centre, right, up: normalize(up), halfWidth, halfHeight };
}
