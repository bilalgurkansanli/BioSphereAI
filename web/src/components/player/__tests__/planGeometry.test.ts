/**
 * The arithmetic behind the bird's-eye view.
 *
 * Two things here are worth a test rather than a glance, and both are the kind of mistake
 * that looks fine on the room you happen to be standing in:
 *
 * * `z` points forward in the plan and DOWN the screen in SVG, so the projection flips it.
 *   Get that wrong and the whole flat is mirrored, which reads as a plausible flat.
 * * the heading cone composes the camera's yaw with the room's own rotation, and the anchor
 *   room's rotation is zero — so an implementation that forgot to add it would point
 *   correctly in exactly the room a developer opens first.
 */

import { describe, expect, it } from "vitest";
import type { FloorPlan, PlanRoom } from "@/lib/types";
import {
  MAX_SCALE,
  drawnRooms,
  fitViewport,
  hasDrawing,
  labelPoint,
  parkingSpot,
  planHeadingDeg,
  polygonPath,
  roomBounds,
  scaleBar,
  screenDeltaToPlan,
  snapDelta,
  toScreen,
  unlinkedRooms,
} from "../planGeometry";

function room(overrides: Partial<PlanRoom> = {}): PlanRoom {
  return {
    scene_id: "a",
    name: "Salon",
    placement: "anchor",
    x: 0,
    z: 0,
    rotation: 0,
    polygon: [
      [-2, -2],
      [2, -2],
      [2, 2],
      [-2, 2],
    ],
    radius: 2,
    area_m2: 16,
    ready: true,
    measured_fraction: 1,
    ...overrides,
  };
}

function plan(overrides: Partial<FloorPlan> = {}): FloorPlan {
  return {
    property_id: "t",
    rooms: [room()],
    doors: [],
    camera_height_m: 1.6,
    pending_scene_ids: [],
    bounds: [-2, -2, 2, 2],
    warnings: [],
    can_measure: true,
    ...overrides,
  };
}

describe("fitViewport", () => {
  it("centres the drawing on the frame, not the coordinate origin", () => {
    // A flat that runs off to one side: the camera that shot the first room is at (0, 0)
    // and is nowhere near the middle of it.
    const viewport = fitViewport([0, 0, 12, 4], 600, 400);
    const [cx, cy] = toScreen(viewport, 6, 2);
    expect(cx).toBeCloseTo(300, 6);
    expect(cy).toBeCloseTo(200, 6);
  });

  it("puts +z up the screen, because SVG's y axis points the other way", () => {
    const viewport = fitViewport([-2, -2, 2, 2], 400, 400);
    const [, forward] = toScreen(viewport, 0, 1);
    const [, back] = toScreen(viewport, 0, -1);
    expect(forward).toBeLessThan(back);
  });

  it("fits the long side, so nothing is drawn outside the frame", () => {
    const viewport = fitViewport([-6, -1, 6, 1], 600, 400, 20);
    const [left] = toScreen(viewport, -6, 0);
    const [right] = toScreen(viewport, 6, 0);
    expect(left).toBeGreaterThanOrEqual(20 - 1e-6);
    expect(right).toBeLessThanOrEqual(580 + 1e-6);
  });

  it("does not blow one small room up to wall-chart size", () => {
    const viewport = fitViewport([-1.2, -1.2, 1.2, 1.2], 1400, 900);
    expect(viewport.scale).toBe(MAX_SCALE);
  });

  it("survives a property whose bounds have collapsed to a point", () => {
    const viewport = fitViewport([0, 0, 0, 0], 400, 300);
    expect(Number.isFinite(viewport.scale)).toBe(true);
    expect(viewport.scale).toBeGreaterThan(0);
  });
});

describe("polygonPath", () => {
  it("closes the ring", () => {
    const viewport = fitViewport([-2, -2, 2, 2], 400, 400);
    const path = polygonPath(viewport, room().polygon as ReadonlyArray<readonly [number, number]>);
    expect(path.startsWith("M")).toBe(true);
    expect(path.endsWith("Z")).toBe(true);
    expect(path.match(/L/g)).toHaveLength(3);
  });

  it("draws nothing for a shape that is not one", () => {
    const viewport = fitViewport([-2, -2, 2, 2], 400, 400);
    expect(polygonPath(viewport, [])).toBe("");
    expect(polygonPath(viewport, [[0, 0], [1, 1]])).toBe("");
  });
});

describe("planHeadingDeg", () => {
  it("is the camera's yaw in the room the plan is anchored on", () => {
    expect(planHeadingDeg({ rotation: 0 }, Math.PI / 2)).toBeCloseTo(90, 6);
  });

  it("adds the room's own rotation, so a turned room's cone still points at the wall it faces", () => {
    // The far room's panorama started facing a quarter turn from the plan's frame.
    expect(planHeadingDeg({ rotation: -Math.PI / 2 }, Math.PI / 2)).toBeCloseTo(0, 6);
    expect(planHeadingDeg({ rotation: Math.PI }, 0)).toBeCloseTo(180, 6);
  });
});

describe("labelPoint", () => {
  it("puts the name in the middle of the room, not on top of the you-are-here marker", () => {
    // The photographer stood in a corner; the label must not follow them there.
    const [x, z] = labelPoint(room({ x: -1.8, z: -1.8 }));
    expect(x).toBeCloseTo(0, 6);
    expect(z).toBeCloseTo(0, 6);
  });

  it("falls back to the camera when there is no outline to find a middle of", () => {
    expect(labelPoint(room({ polygon: [], x: 3, z: -1 }))).toEqual([3, -1]);
  });

  it("does not divide by zero on a ring with no area", () => {
    const flat = room({ polygon: [[0, 0], [1, 0], [2, 0]], x: 7, z: 8 });
    expect(labelPoint(flat)).toEqual([7, 8]);
  });
});

describe("scaleBar", () => {
  it("is always a number somebody can read off", () => {
    const viewport = fitViewport([-5, -5, 5, 5], 500, 500);
    for (const width of [40, 80, 140, 300]) {
      const bar = scaleBar(viewport, width);
      const mantissa = bar.metres / 10 ** Math.floor(Math.log10(bar.metres));
      expect([1, 2, 5]).toContain(Math.round(mantissa));
      expect(bar.pixels).toBeLessThanOrEqual(width + 1e-6);
    }
  });

  it("is still one of those numbers when the frame it is drawn in is absurd", () => {
    // A panel measured mid-layout once produced a "0.05 m" bar. The bar has three legal
    // shapes and no fourth, whatever it is handed.
    for (const [span, width] of [[0.5, 4], [400, 12], [3, 1]] as const) {
      const bar = scaleBar(fitViewport([-span, -span, span, span], 300, 300), width);
      const mantissa = bar.metres / 10 ** Math.floor(Math.log10(bar.metres));
      expect([1, 2, 5]).toContain(Math.round(mantissa));
    }
  });
});

describe("what the plan is allowed to offer", () => {
  it("counts an unlinked room out of the drawing but not out of the property", () => {
    const body = plan({
      rooms: [room(), room({ scene_id: "b", name: "Balkon", placement: "unlinked", polygon: [] })],
    });
    expect(drawnRooms(body).map((r) => r.scene_id)).toEqual(["a"]);
    expect(unlinkedRooms(body).map((r) => r.scene_id)).toEqual(["b"]);
  });

  it("offers the button for one measured room, because that is a bedroom seen from above", () => {
    expect(hasDrawing(plan())).toBe(true);
  });

  it("offers nothing when every room is a placeholder", () => {
    expect(hasDrawing(null)).toBe(false);
    expect(hasDrawing(plan({ rooms: [room({ polygon: [] })] }))).toBe(false);
    expect(
      hasDrawing(plan({ rooms: [room({ placement: "unlinked", polygon: [] })] })),
    ).toBe(false);
  });
});

describe("dragging a room by hand", () => {
  it("turns a drag on screen back into metres, with z the other way up", () => {
    const viewport = fitViewport([-2, -2, 2, 2], 400, 400);
    const moved = screenDeltaToPlan(viewport, viewport.scale * 1.5, viewport.scale * 2);
    expect(moved.x).toBeCloseTo(1.5, 6);
    // Dragging DOWN the screen moves a room backwards in the plan.
    expect(moved.z).toBeCloseTo(-2, 6);
  });

  it("measures what a room occupies, outline or placeholder", () => {
    expect(roomBounds(room())).toEqual({ minX: -2, minZ: -2, maxX: 2, maxZ: 2 });
    expect(roomBounds(room({ polygon: [], x: 5, z: 1, radius: 1.5 }))).toEqual({
      minX: 3.5,
      minZ: -0.5,
      maxX: 6.5,
      maxZ: 2.5,
    });
  });

  it("snaps a wall flush when it comes close", () => {
    // A 4 m room at the origin, another whose left wall stands at x = 2.4. Dragged 0.3 m
    // right, this one's right wall lands at 2.3 — a tenth short of flush, which is exactly
    // the sliver that makes a hand-assembled plan look broken.
    const moving = roomBounds(room());
    const other = { minX: 2.4, minZ: -2, maxX: 6.4, maxZ: 2 };
    const snapped = snapDelta(moving, [other], { x: 0.3, z: 0 });
    expect(moving.maxX + snapped.x).toBeCloseTo(2.4, 6);
  });

  it("leaves a deliberate gap alone", () => {
    const snapped = snapDelta(roomBounds(room()), [{ minX: 6, minZ: -2, maxX: 10, maxZ: 2 }], {
      x: 1,
      z: 0,
    });
    expect(snapped).toEqual({ x: 1, z: 0 });
  });

  it("takes the nearest wall, not the first one it checked", () => {
    const others = [
      { minX: 2.6, minZ: -2, maxX: 6, maxZ: 2 },
      { minX: 2.05, minZ: -2, maxX: 6, maxZ: 2 },
    ];
    const snapped = snapDelta(roomBounds(room()), others, { x: 0, z: 0 });
    expect(roomBounds(room()).maxX + snapped.x).toBeCloseTo(2.05, 6);
  });

  it("parks a new room clear of the plan rather than on top of it", () => {
    const body = plan({ bounds: [-2, -2, 2, 2] });
    const spot = parkingSpot(body, room({ polygon: [], radius: 1.2 }));
    expect(spot.x).toBeGreaterThan(2 + 1.2);
    expect(spot.z).toBeCloseTo(0, 6);
  });
});
