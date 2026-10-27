/**
 * Pure geometry of the 3D walkthrough. No React, no WebGL, so it runs under vitest's node
 * environment like `planGeometry.ts` does.
 *
 * Two frames meet here. The ENGINE frame is what the manifest speaks: right-handed, +X
 * right, +Y DOWN, +Z forward, floor at y = 0, metres. The VIEWER frame is PlayCanvas's:
 * +Y up, right-handed. The two differ by a half turn about X, which is exactly what
 * `toViewer` / `toEngine` do; nothing else in the viewer ever thinks about it.
 */

import type { WalkArea, WalkFloorPlan, WalkManifest } from "@/lib/types";

export type Vec3 = [number, number, number];

/** Where the visitor's eyes are above the floor, metres. SplatTour uses the same 1.6. */
export const EYE_HEIGHT = 1.6;
/** How close to a wall the walk mode lets the visitor get before the box stops them. */
export const WALL_MARGIN = 0.25;

export function toViewer(p: ArrayLike<number>): Vec3 {
  // `0 - x` rather than `-x`: `-0` is a distinct number that fails deep-equality checks.
  return [p[0], 0 - p[1], 0 - p[2]];
}

export function toEngine(p: ArrayLike<number>): Vec3 {
  return [p[0], 0 - p[1], 0 - p[2]];
}

export interface Box {
  min: Vec3;
  max: Vec3;
}

/** An area's box in the viewer frame (the engine box flips two axes, so min/max swap). */
export function viewerBox(area: WalkArea): Box {
  const a = toViewer(area.bbox.min);
  const b = toViewer(area.bbox.max);
  return {
    min: [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])],
    max: [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])],
  };
}

export function contains(box: Box, p: ArrayLike<number>, margin = 0): boolean {
  return (
    p[0] >= box.min[0] - margin &&
    p[0] <= box.max[0] + margin &&
    p[1] >= box.min[1] - margin &&
    p[1] <= box.max[1] + margin &&
    p[2] >= box.min[2] - margin &&
    p[2] <= box.max[2] + margin
  );
}

export function union(boxes: Box[]): Box | null {
  if (!boxes.length) return null;
  const out: Box = { min: [...boxes[0].min] as Vec3, max: [...boxes[0].max] as Vec3 };
  for (const b of boxes.slice(1)) {
    for (let i = 0; i < 3; i += 1) {
      out.min[i] = Math.min(out.min[i], b.min[i]);
      out.max[i] = Math.max(out.max[i], b.max[i]);
    }
  }
  return out;
}

/**
 * Which area should be drawn for a viewer standing at `p` (viewer frame).
 *
 * The area whose camera path passes closest to the point wins — the part of the walk that
 * was actually filmed from where the visitor stands, which is the part that will look right
 * from there. The base is the fallback, and it also wins ties, so a visitor who has not gone
 * anywhere sees the main walkthrough rather than a fragment of it.
 *
 * NOT the smallest containing box, which is what this used to be. That rule was written for
 * rooms shot separately, whose boxes meet only at doorways. A walk delivered in parts breaks
 * it completely: the parts are stretches of one flat, their boxes overlap almost entirely,
 * and the smallest box wins everywhere — the first real capture opened standing in the main
 * area while a two-second fragment was drawn, so the screen showed that fragment's backdrop
 * and nothing else.
 *
 * Unplaced areas never win by position — they are only ever chosen by name.
 */
export function areaAt(manifest: WalkManifest, p: ArrayLike<number>, current?: string | null): WalkArea {
  const nearest = (area: WalkArea): number => {
    let best = Infinity;
    for (const c of area.cameras) {
      const v = toViewer(c);
      const dx = v[0] - p[0];
      const dz = v[2] - p[2];
      const d = dx * dx + dz * dz;
      if (d < best) best = d;
    }
    return best;
  };
  // `redundant` areas are left out entirely: the delivery measured that the base covers
  // their own viewpoints at least as well, so letting one win here would replace the whole
  // picture with a second reconstruction of the same wall - or, measured on a real property,
  // with an almost black frame where that piece has no geometry at all.
  const drawable = [
    manifest.base,
    ...manifest.rooms.filter((r) => r.placed && !r.redundant && !r.failed && r.url && contains(viewerBox(r), p)),
  ];
  let best = manifest.base;
  let bestDistance = nearest(manifest.base);
  for (const room of drawable) {
    const d = nearest(room);
    if (d < bestDistance) {
      best = room;
      bestDistance = d;
    }
  }
  // Stickiness. The parts of a broken walk overlap, so along an ordinary two-metre stretch
  // the nearest path changes hands again and again: measured on a real property, ten changes,
  // seven of them lasting less than one step. Each one swaps the entire cloud, and to a
  // visitor that reads as being teleported into a slightly different room and back. The area
  // in use therefore keeps its place until another is clearly nearer - distances are squared,
  // so this is a factor of about 1.4 in metres.
  if (current && current !== best.key) {
    const held = drawable.find((a) => a.key === current);
    if (held && nearest(held) <= bestDistance * AREA_STICKINESS) return held;
  }
  return best;
}

/** The point moved back inside `box` (viewer frame), keeping `margin` from its walls. */
export function clampToBox(p: Vec3, box: Box, margin = WALL_MARGIN): Vec3 {
  const out: Vec3 = [p[0], p[1], p[2]];
  for (let i = 0; i < 3; i += 1) {
    const lo = box.min[i] + margin;
    const hi = box.max[i] - margin;
    if (lo > hi) {
      out[i] = (box.min[i] + box.max[i]) / 2;
    } else {
      out[i] = Math.min(Math.max(out[i], lo), hi);
    }
  }
  return out;
}

/** Yaw (about +Y, radians) that makes the viewer look along `forward` (viewer frame, horizontal). */
export function yawOf(forward: ArrayLike<number>): number {
  // PlayCanvas cameras look down their local -Z; yaw 0 therefore looks along world -Z.
  return Math.atan2(-forward[0], -forward[2]);
}

/** How much nearer another area must be before it takes over (squared distances). */
export const AREA_STICKINESS = 2.0;

/**
 * How long a new area must stay the nearest before the cloud is swapped, seconds.
 *
 * Replaying the delivered walk along its own camera path, 16.8 m of walking, counting swaps
 * and how much of each area's own route still draws that area:
 *
 *     dwell     swaps on the base path     own path: room 3 / room 4 / room 5
 *     none                        11              -        -        -
 *     0.35 s                       3            38 %     95 %     43 %
 *     0.60 s                       0            37 %     91 %     39 %
 *     1.00 s                       0            36 %     58 %     31 %
 *
 * Six tenths of a second is where the jolts stop without the rooms starting to disappear:
 * past it the visitor is well inside a room before it is drawn, which trades one fault for
 * a worse picture. Stickiness barely moves this column - it is what stops the flicker
 * between two areas that are both plausible, and the dwell is what stops the crossing.
 */
export const AREA_SWITCH_DWELL_S = 0.6;
/** How long the area being left keeps drawing, so the change dissolves rather than cuts. */
export const AREA_OVERLAP_S = 0.45;

export interface AreaSwitch {
  /** The area waiting to take over, and how long it has been waiting. */
  pending: string | null;
  held: number;
}

/**
 * The area to swap to this frame, or null to keep the one in use.
 *
 * Stickiness alone still changes area the instant the visitor crosses the line, and a
 * doorway is crossed and re-crossed while turning around in it. Making the candidate hold
 * its claim for a third of a second costs nothing anybody can feel - the walk speed is
 * about 1.4 m/s, so it is half a step - and it removes every switch that a shift of weight
 * would otherwise cause. With no area in use yet there is nothing to protect, so the first
 * one is taken immediately.
 */
export function updateAreaSwitch(state: AreaSwitch, candidate: string | null, active: string | null, dt: number): string | null {
  if (!candidate || candidate === active) {
    state.pending = null;
    state.held = 0;
    return null;
  }
  if (active === null) return candidate;
  if (candidate !== state.pending) {
    state.pending = candidate;
    state.held = 0;
  }
  state.held += dt;
  if (state.held < AREA_SWITCH_DWELL_S) return null;
  state.pending = null;
  state.held = 0;
  return candidate;
}

/** Render scale limits and the frame-time band the viewer steers into. */
export const SCALE_MAX = 1.5;
export const SCALE_MIN = 0.55;
export const FRAME_MS_SLOW = 22; // slower than ~45 fps: give up pixels
export const FRAME_MS_FAST = 15; // faster than ~66 fps: take some back

/**
 * The render scale to use next, given the one in use and a smoothed frame time.
 *
 * A splat cloud is almost pure fill rate, so pixels are the only lever that always works:
 * the same scene runs smoothly in a small panel and crawls full-screen on a high-DPI laptop
 * for no other reason. Down steps are larger than up steps (0.85 against 1.08) so a viewer
 * that cannot hold the rate falls to a comfortable scale quickly and then creeps back rather
 * than oscillating between two visibly different resolutions.
 */
export function nextRenderScale(current: number, frameMs: number, ceiling: number): number {
  const top = Math.min(ceiling, SCALE_MAX);
  if (frameMs > FRAME_MS_SLOW) return Math.max(SCALE_MIN, current * 0.85);
  if (frameMs < FRAME_MS_FAST) return Math.min(top, current * 1.08);
  return Math.min(current, top);
}

/** Unit horizontal direction for a yaw, viewer frame. Inverse of `yawOf`. */
export function forwardOf(yaw: number): Vec3 {
  return [-Math.sin(yaw), 0, -Math.cos(yaw)];
}

/** The pose a visitor gets when they arrive in `area`: entry position at eye height, facing in. */
export function entryPose(area: WalkArea): { position: Vec3; yaw: number } {
  const p = toViewer(area.entry.position);
  const f = toViewer(area.entry.forward);
  return { position: [p[0], EYE_HEIGHT, p[2]], yaw: yawOf(f) };
}

/**
 * The camera path of an area in the viewer frame at eye height, for the auto tour.
 * Eye height is forced rather than taken from the walk: the phone was carried at chest
 * height and a tour should not be.
 */
export function pathOf(area: WalkArea): Vec3[] {
  return area.cameras.map((c) => {
    const v = toViewer(c);
    return [v[0], EYE_HEIGHT, v[2]];
  });
}

/**
 * Where the phone pointed at camera `c` (viewer frame, unit), or null when the manifest does
 * not say. A camera entry is `[x, y, z]`, or `[x, y, z, lx, ly, lz]` with the look direction
 * (engine frame) appended - the API's manifest schema is closed, and every reader that takes
 * the first three numbers keeps working.
 */
export function lookOf(c: ArrayLike<number>): Vec3 | null {
  if (c.length < 6) return null;
  const v = toViewer([c[3], c[4], c[5]]);
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 1e-6 ? [v[0] / n, v[1] / n, v[2] / n] : null;
}

/**
 * Yaw and pitch the guided tour should hold at parameter `t` (0..1 along the area's path,
 * the same parameter `splinePoint` takes): the direction the phone was actually pointing
 * there, blended between the two nearest cameras. Null when the area has no look data.
 *
 * WHY. A splat model is right from where - and towards where - it was photographed. A tour
 * that faces down the path tangent looks sideways into rooms the phone only glanced at, and
 * that is exactly where the needles and smears live; following the filmed direction keeps the
 * visitor inside the views the model was trained to reproduce.
 */
export function lookAlong(area: WalkArea, t: number): { yaw: number; pitch: number } | null {
  const n = area.cameras.length;
  if (n === 0) return null;
  const s = Math.min(Math.max(t, 0), 1) * Math.max(n - 1, 0);
  const i = Math.min(Math.floor(s), Math.max(n - 2, 0));
  const u = n > 1 ? s - i : 0;
  const a = lookOf(area.cameras[i]);
  const b = lookOf(area.cameras[Math.min(i + 1, n - 1)]) ?? a;
  if (!a || !b) return null;
  const d: Vec3 = [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  const len = Math.hypot(d[0], d[1], d[2]);
  if (len < 1e-6) return { yaw: yawOf(a), pitch: Math.asin(Math.max(-1, Math.min(1, a[1]))) };
  return { yaw: yawOf(d), pitch: Math.asin(Math.max(-1, Math.min(1, d[1] / len))) };
}

/**
 * Widest horizontal field of view the viewer OPENS at, degrees (the visitor can still widen it).
 *
 * The walk is filmed on a portrait phone: ~63 degrees across, ~95 up and down on the first
 * real capture. The viewer's 70-degree vertical field on a 16:9 screen is 102 degrees across,
 * so half the width of every desktop frame showed what the phone never pointed at from there -
 * the needle-and-smear margins. Rendered side by side at the same poses, an 80-degree cap
 * keeps the room and drops most of the margin (exp/gaze_compare.py); portrait screens are
 * unaffected (70 degrees vertical is ~36 across on a phone).
 */
export const HFOV_OPEN_MAX = 80;

/** The vertical FOV (degrees) to open at on a canvas of `aspect` (width / height). */
export function defaultFov(aspect: number, vfov = 70, hfovMax = HFOV_OPEN_MAX): number {
  if (!(aspect > 0) || !Number.isFinite(aspect)) return vfov;
  const capped = (2 * Math.atan(Math.tan((hfovMax * Math.PI) / 360) / aspect) * 180) / Math.PI;
  return Math.min(vfov, capped);
}

/** Seconds into its video at which camera `c` was filmed, or null when the manifest does not say. */
export function timeOf(c: ArrayLike<number>): number | null {
  return c.length >= 7 && Number.isFinite(c[6]) && c[6] >= 0 ? c[6] : null;
}

/**
 * The order the guided tour visits areas in: every video's parts in the order they were
 * filmed (the base first), so a walk that had to be delivered in pieces is still replayed as
 * the walk it was - entrance, rooms, back to the door - with a fade where a piece could not
 * be placed. A part is `<video key>~<n>`; parts without times keep the manifest's order.
 * Redundant parts (the base already shows that space) and failed ones are skipped; a separate
 * room video is toured only when it was placed, as before.
 */
export function tourOrder(manifest: WalkManifest): string[] {
  const usable = (a: WalkArea) => !a.failed && !!a.url && !a.redundant;
  const videoOf = (key: string) => key.split("~")[0];
  const start = (a: WalkArea) => {
    let t = Infinity;
    for (const c of a.cameras) {
      const s = timeOf(c);
      if (s !== null && s < t) t = s;
    }
    return t;
  };
  const groups = new Map<string, WalkArea[]>();
  for (const a of [manifest.base, ...manifest.rooms]) {
    if (!usable(a)) continue;
    const video = videoOf(a.key);
    const isPartOfAWalk = a.key.includes("~") || video === manifest.base.key;
    if (!isPartOfAWalk && !a.placed) continue;
    groups.set(video, [...(groups.get(video) ?? []), a]);
  }
  const out: string[] = [];
  for (const parts of groups.values()) out.push(...[...parts].sort((a, b) => start(a) - start(b)).map((a) => a.key));
  return out.length ? out : [manifest.base.key];
}

/** Catmull-Rom point on `pts` at parameter `t` in [0, 1] across the whole polyline. */
export function splinePoint(pts: Vec3[], t: number): Vec3 {
  if (pts.length === 0) return [0, EYE_HEIGHT, 0];
  if (pts.length === 1) return [...pts[0]] as Vec3;
  const segs = pts.length - 1;
  const s = Math.min(Math.max(t, 0), 1) * segs;
  const i = Math.min(Math.floor(s), segs - 1);
  const u = s - i;
  const p0 = pts[Math.max(i - 1, 0)];
  const p1 = pts[i];
  const p2 = pts[i + 1];
  const p3 = pts[Math.min(i + 2, pts.length - 1)];
  const out: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k += 1) {
    out[k] =
      0.5 *
      (2 * p1[k] +
        (-p0[k] + p2[k]) * u +
        (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u * u +
        (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u * u * u);
  }
  return out;
}

/** Length of a polyline, metres. */
export function pathLength(pts: Vec3[]): number {
  let n = 0;
  for (let i = 1; i < pts.length; i += 1) {
    n += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
  }
  return n;
}

/** Plan pixel fraction (0..1) of a viewer-frame point, from the manifest's plan mapping. */
export function planFraction(plan: WalkFloorPlan, p: ArrayLike<number>): { x: number; y: number } {
  const e = toEngine(p);
  const u = (e[0] - plan.x0) * plan.px_per_m;
  const v = (plan.z0 - e[2]) * plan.px_per_m;
  return { x: u / Math.max(plan.width, 1), y: v / Math.max(plan.height, 1) };
}

/** Viewer-frame floor point (at eye height) for a plan pixel fraction. Inverse of `planFraction`. */
export function planPoint(plan: WalkFloorPlan, x: number, y: number): Vec3 {
  const ex = plan.x0 + (x * plan.width) / plan.px_per_m;
  const ez = plan.z0 - (y * plan.height) / plan.px_per_m;
  const v = toViewer([ex, 0, ez]);
  return [v[0], EYE_HEIGHT, v[2]];
}

/** Shortest signed difference between two yaws, radians in (-π, π]. */
export function yawDelta(from: number, to: number): number {
  let d = (to - from) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Areas worth drawing: the base plus every room that was delivered. */
export function drawableAreas(manifest: WalkManifest): WalkArea[] {
  const rooms = manifest.rooms.filter((r) => !r.failed && r.url);
  return manifest.base.url ? [manifest.base, ...rooms] : rooms;
}

/**
 * How far from the filmed path a visitor may stand in walk mode, metres.
 *
 * A splat model is only right from near where it was photographed: step a few metres off the
 * path and the picture is made of walls seen edge-on and of splats no camera ever looked at
 * directly - the needles and smears visitors read as "broken". The walk box (the union of the
 * placed areas' padded boxes) is the wrong shape for that. Measured on the first delivered
 * walk (exp/reach.py): the box let the visitor reach 161 m² of floor, and only 38 % of it lies
 * within 1 m of anywhere the phone stood; 41 % is more than 2 m away (p90 5.1 m). The corridor
 * keeps them where the model is trustworthy and still leaves a step either side of the path.
 */
export const PATH_RADIUS_M = 1.0;
/**
 * Consecutive path cameras further apart than this are not joined into a corridor. The path
 * in the manifest is thinned to 64 points, so neighbours are normally a fraction of a metre
 * apart; a gap longer than a stride is a cut in the walk (a part that was placed elsewhere),
 * and joining across it would open a corridor straight through a wall.
 */
export const PATH_MAX_LINK_M = 1.5;

/** Walkable centre lines in the viewer frame: flat [ax, az, bx, bz] per segment. */
export type Corridor = number[];

/** The corridor of `areas`' camera paths (viewer frame; a lone camera is a zero-length segment). */
export function corridorOf(areas: WalkArea[]): Corridor {
  const out: Corridor = [];
  for (const area of areas) {
    const pts = area.cameras.map((c) => toViewer(c));
    for (let i = 0; i < pts.length; i += 1) {
      const a = pts[i];
      const b = pts[i + 1];
      if (b && Math.hypot(b[0] - a[0], b[2] - a[2]) <= PATH_MAX_LINK_M) out.push(a[0], a[2], b[0], b[2]);
      else out.push(a[0], a[2], a[0], a[2]);
    }
  }
  return out;
}

/** The nearest point of the corridor's centre lines to (x, z), and how far it is. */
export function nearestOnCorridor(corridor: Corridor, x: number, z: number): { x: number; z: number; d: number } {
  let best = { x, z, d: Infinity };
  for (let i = 0; i + 3 < corridor.length; i += 4) {
    const ax = corridor[i];
    const az = corridor[i + 1];
    const dx = corridor[i + 2] - ax;
    const dz = corridor[i + 3] - az;
    const len2 = dx * dx + dz * dz;
    const t = len2 > 1e-12 ? Math.min(1, Math.max(0, ((x - ax) * dx + (z - az) * dz) / len2)) : 0;
    const qx = ax + t * dx;
    const qz = az + t * dz;
    const d = Math.hypot(x - qx, z - qz);
    if (d < best.d) best = { x: qx, z: qz, d };
  }
  return best;
}

/**
 * `p` pulled back to within `radius` of the corridor (height untouched). Pushing outwards
 * therefore slides along the corridor's edge the way it would along a wall. An empty
 * corridor (an area with no camera path) constrains nothing.
 */
export function constrainToCorridor(p: Vec3, corridor: Corridor, radius = PATH_RADIUS_M): Vec3 {
  if (corridor.length < 4) return p;
  const near = nearestOnCorridor(corridor, p[0], p[2]);
  if (near.d <= radius) return p;
  const k = radius / near.d;
  return [near.x + (p[0] - near.x) * k, p[1], near.z + (p[2] - near.z) * k];
}
