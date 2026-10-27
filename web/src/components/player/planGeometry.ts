/**
 * Turning a floor plan in metres into a drawing on screen.
 *
 * Separated from `PlanView.tsx` for the same reason `portalGeometry.ts` is separated from
 * `PortalRings.tsx`: a room drawn in the wrong place can only be seen by looking at it,
 * whereas the arithmetic that put it there can be checked at a desk. Nothing here touches
 * React or the DOM, so it runs under the plain-Node test environment.
 *
 * Two coordinate systems, and one of them has its vertical axis the other way up:
 *
 *   PLAN     metres, `+x` right and `+z` forward, exactly the engine's world with the
 *            height dropped. An angle of 0 points along `+z` and grows the way yaw does.
 *   SCREEN   pixels, `+x` right and `+y` DOWN, because that is what SVG is.
 *
 * So the projection is a scale and a flip of `z`, and a plan angle becomes a clockwise
 * screen rotation of the same number of degrees — which is what `rotate()` already does in
 * a y-down space.
 */

import type { FloorPlan, PlanRoom } from "@/lib/types";

/** Scale and origin taking plan metres to screen pixels. */
export interface PlanViewport {
  /** Pixels per metre. */
  scale: number;
  /** Screen position of the plan's origin. */
  originX: number;
  originY: number;
  width: number;
  height: number;
}

/**
 * Breathing room around the drawing, in pixels.
 *
 * Not zero, and not only for looks: room names are drawn at a fixed pixel size at each
 * room's centre, so a room against the edge of the frame has its label clipped. This is
 * about one label's width.
 */
export const PLAN_PADDING = 56;

/**
 * Largest scale the plan will ever be drawn at, in pixels per metre.
 *
 * A property with one small room would otherwise be fitted to the frame and drawn at the size of a
 * wall chart, which reads as a mistake — the plan of a single bedroom should look like the
 * plan of a single bedroom. Above about ninety pixels to the metre a drawing stops gaining
 * anything: the outline's own accuracy is centimetres.
 */
export const MAX_SCALE = 90;

/** Fit `bounds` into a box, centred, never magnified past {@link MAX_SCALE}. */
export function fitViewport(
  bounds: readonly [number, number, number, number],
  width: number,
  height: number,
  padding = PLAN_PADDING,
): PlanViewport {
  const [minX, minZ, maxX, maxZ] = bounds;
  const spanX = Math.max(maxX - minX, 0.5);
  const spanZ = Math.max(maxZ - minZ, 0.5);
  const usableWidth = Math.max(width - 2 * padding, 40);
  const usableHeight = Math.max(height - 2 * padding, 40);
  const scale = Math.min(usableWidth / spanX, usableHeight / spanZ, MAX_SCALE);

  // Centre the *content*, not the coordinate origin: the camera that shot the first room is
  // at (0, 0) and is rarely in the middle of the flat.
  const centreX = (minX + maxX) / 2;
  const centreZ = (minZ + maxZ) / 2;
  return {
    scale,
    originX: width / 2 - centreX * scale,
    originY: height / 2 + centreZ * scale,
    width,
    height,
  };
}

/** One plan point in screen pixels. */
export function toScreen(viewport: PlanViewport, x: number, z: number): [number, number] {
  return [viewport.originX + x * viewport.scale, viewport.originY - z * viewport.scale];
}

/**
 * A distance dragged on screen, in plan metres.
 *
 * The inverse of {@link toScreen} for deltas rather than points, which is what a drag is: the
 * `z` flip comes back, and the origin does not enter into it.
 */
export function screenDeltaToPlan(
  viewport: PlanViewport,
  dx: number,
  dy: number,
): { x: number; z: number } {
  return { x: dx / viewport.scale, z: -dy / viewport.scale };
}

/** An SVG path for a closed ring of plan points. Empty string for anything degenerate. */
export function polygonPath(
  viewport: PlanViewport,
  points: ReadonlyArray<readonly [number, number]>,
): string {
  if (points.length < 3) return "";
  const parts = points.map(([x, z], index) => {
    const [px, py] = toScreen(viewport, x, z);
    return `${index === 0 ? "M" : "L"}${px.toFixed(1)} ${py.toFixed(1)}`;
  });
  return `${parts.join(" ")} Z`;
}

/**
 * Where the viewer is looking, in plan degrees, clockwise from the top of the drawing.
 *
 * The camera's yaw is measured in the room's OWN frame — every panorama starts wherever the
 * photographer happened to be pointing — and `room.rotation` is what the assembler worked
 * out to bring that frame into the plan's. Adding them is the whole conversion, and getting
 * it wrong is the bug that would make the cone point at the wrong wall in every room but the
 * first, which is exactly the room somebody testing would look at.
 */
export function planHeadingDeg(room: Pick<PlanRoom, "rotation">, yaw: number): number {
  return ((yaw + room.rotation) * 180) / Math.PI;
}

/**
 * A scale bar: a round number of metres, and how many pixels it is.
 *
 * Rounds DOWN the available width to the nearest 1-2-5 step, so the bar is always a length
 * somebody can read off ("5 m") rather than the frame's own arbitrary width.
 */
export function scaleBar(viewport: PlanViewport, maxPixels: number): { metres: number; pixels: number } {
  // Clamping `rough` rather than only the magnitude keeps `1 * magnitude <= rough` true by
  // construction, so one of the three steps always matches and the bar is always a number
  // somebody can read. Clamping only inside the logarithm left a gap below a tenth of a
  // metre where nothing matched, and a plan drawn into a collapsed panel fell into it and
  // printed "0.05 m".
  const rough = Math.max(maxPixels / viewport.scale, 0.1);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const metres = [5, 2, 1].map((step) => step * magnitude).find((value) => value <= rough) ?? magnitude;
  return { metres, pixels: metres * viewport.scale };
}

/**
 * Where a room's name goes: the centroid of its outline, not the camera that shot it.
 *
 * The obvious choice is the camera's own position, and it is wrong for the room the viewer is
 * standing in — which is the one room whose name they most want to read. The "you are here"
 * marker is at the camera by definition, so a label there is a label under an arrow. The
 * centroid is also simply a better place for a name: a photographer stands near a doorway or
 * against a wall far more often than in the middle of a room.
 */
export function labelPoint(room: PlanRoom): [number, number] {
  const points = room.polygon;
  if (points.length < 3) return [room.x, room.z];

  let area = 0;
  let cx = 0;
  let cz = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x0, z0] = points[i];
    const [x1, z1] = points[(i + 1) % points.length];
    const cross = x0 * z1 - x1 * z0;
    area += cross;
    cx += (x0 + x1) * cross;
    cz += (z0 + z1) * cross;
  }
  // A degenerate ring has no centroid to speak of; fall back rather than divide by zero.
  if (Math.abs(area) < 1e-9) return [room.x, room.z];
  return [cx / (3 * area), cz / (3 * area)];
}

/** Axis-aligned extent of a room as drawn, in plan metres. */
export interface PlanBounds {
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
}

/** What a room occupies on the plan: its outline, or the disc standing in for one. */
export function roomBounds(room: PlanRoom): PlanBounds {
  if (room.polygon.length < 3) {
    return {
      minX: room.x - room.radius,
      minZ: room.z - room.radius,
      maxX: room.x + room.radius,
      maxZ: room.z + room.radius,
    };
  }
  const xs = room.polygon.map(([x]) => x);
  const zs = room.polygon.map(([, z]) => z);
  return {
    minX: Math.min(...xs),
    minZ: Math.min(...zs),
    maxX: Math.max(...xs),
    maxZ: Math.max(...zs),
  };
}

/**
 * How close two walls have to come, in metres, before a drag snaps them flush.
 *
 * 0.25 m is a comfortable fraction of a room and far more than the outline's own accuracy, so
 * a snap never has to fight a measurement. Below about 0.1 m it stops being reachable with a
 * mouse on a plan of a whole flat; well above 0.3 m it starts catching walls the agent was
 * deliberately leaving apart.
 */
export const SNAP_METRES = 0.25;

/**
 * Nudge a drag so the room lands flush against another one.
 *
 * A flat is a set of rooms that share walls, so a plan assembled by hand and left a few
 * centimetres out looks broken in a way the eye picks up immediately — thin white slivers
 * between rooms that in the building are one partition. Snapping is what makes hand
 * placement produce a drawing rather than a collage.
 *
 * Each axis is decided on its own, and only the SMALLEST available correction is taken, so a
 * room settles against the nearest wall rather than jumping to whichever one was checked
 * first.
 */
export function snapDelta(
  moving: PlanBounds,
  others: PlanBounds[],
  delta: { x: number; z: number },
  threshold = SNAP_METRES,
): { x: number; z: number } {
  const adjust = (low: number, high: number, edges: number[]): number => {
    let best = 0;
    let bestDistance = threshold;
    for (const edge of edges) {
      for (const side of [low, high]) {
        const correction = edge - side;
        if (Math.abs(correction) < bestDistance) {
          bestDistance = Math.abs(correction);
          best = correction;
        }
      }
    }
    return best;
  };

  const movedX = { low: moving.minX + delta.x, high: moving.maxX + delta.x };
  const movedZ = { low: moving.minZ + delta.z, high: moving.maxZ + delta.z };
  return {
    x: delta.x + adjust(movedX.low, movedX.high, others.flatMap((b) => [b.minX, b.maxX])),
    z: delta.z + adjust(movedZ.low, movedZ.high, others.flatMap((b) => [b.minZ, b.maxZ])),
  };
}

/**
 * Where to drop a room that has never been on the plan: clear of everything, to the right.
 *
 * Beside rather than on top, because the first thing that happens to it is a drag and a room
 * that appeared underneath another one cannot be grabbed.
 */
export function parkingSpot(plan: FloorPlan, room: PlanRoom): { x: number; z: number } {
  const [minX, minZ, maxX, maxZ] = plan.bounds;
  const half = room.polygon.length >= 3 ? (maxX - minX) / 2 : room.radius;
  return { x: maxX + Math.max(half, room.radius) + 1.0, z: (minZ + maxZ) / 2 };
}

/** Rooms the plan actually draws, in a stable order: the anchor first, then property order. */
export function drawnRooms(plan: FloorPlan): PlanRoom[] {
  return plan.rooms.filter((room) => room.placement !== "unlinked");
}

/** Rooms that exist but have no place on the drawing. */
export function unlinkedRooms(plan: FloorPlan): PlanRoom[] {
  return plan.rooms.filter((room) => room.placement === "unlinked");
}

/**
 * Whether there is anything worth opening.
 *
 * One room with a measured outline is a plan — that is a bedroom seen from above, which is
 * useful on its own and is what a half-shot property has. A property with nothing measured is not,
 * and the player must not offer a button that opens an empty frame.
 */
export function hasDrawing(plan: FloorPlan | null): boolean {
  return Boolean(plan && drawnRooms(plan).some((room) => room.polygon.length >= 3));
}
