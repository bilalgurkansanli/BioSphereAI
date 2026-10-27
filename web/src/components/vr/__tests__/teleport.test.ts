/**
 * Moving through a walkthrough in a headset: where a teleport may land, where the floor is,
 * and that the rig maths puts the HEAD where it was asked to — the visitor is rarely standing
 * in the middle of their play space.
 */

import { describe, expect, it } from "vitest";
import type { WalkArea, WalkManifest } from "@/lib/types";
import { PATH_RADIUS_M, nearestOnCorridor } from "@/components/walk/walkMath";
import {
  CAPTURE_HEIGHT,
  SNAP_SLACK,
  arcToFloor,
  constrainMove,
  floorAt,
  floorHeightOf,
  headWorld,
  rayToFloor,
  rigAfterTurn,
  rigForTeleport,
  teleportSpace,
  validateTeleport,
  type Vec2,
} from "../teleport";

/** Engine frame is +Y down and z flipped: a camera 1.45 m up at viewer (x, z) is [x, -1.45, -z]. */
const cam = (x: number, z: number, height = CAPTURE_HEIGHT) => [x, -height, -z];

/** Cameras every 0.5 m along a polyline in viewer (x, z), as a real thinned path would be. */
function route(points: Vec2[], height = CAPTURE_HEIGHT): number[][] {
  const out: number[][] = [];
  for (let i = 0; i + 1 < points.length; i += 1) {
    const [a, b] = [points[i], points[i + 1]];
    const steps = Math.max(1, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.5));
    for (let k = 0; k < steps; k += 1) out.push(cam(a[0] + ((b[0] - a[0]) * k) / steps, a[1] + ((b[1] - a[1]) * k) / steps, height));
  }
  const last = points[points.length - 1];
  out.push(cam(last[0], last[1], height));
  return out;
}

function area(key: string, cameras: number[][], placed = true, extra: Partial<WalkArea> = {}): WalkArea {
  return {
    key,
    name: key,
    floor_index: 0,
    url: `/${key}.sog`,
    bytes: 1,
    splat_count: 1,
    thumbnail_url: null,
    env_pano_url: null,
    cameras,
    centroid: [0, 0, 0],
    entry: { position: [0, 0, 0], forward: [0, 0, 1] },
    bbox: { min: [-5, -3, -5], max: [5, 0.3, 5] },
    placed,
    warnings: [],
    failed: null,
    ...extra,
  };
}

function manifest(base: WalkArea, rooms: WalkArea[] = []): WalkManifest {
  return { version: 1, frame: {}, base, rooms, floor_plan: null, metrics: {}, warnings: [] };
}

/** An L-shaped corridor: 6 m along +x, then 4 m along +z. Floor at 0. */
const L = manifest(area("base", route([[0, 0], [6, 0], [6, 4]])));
const SPACE = teleportSpace(L, "base");

describe("floor height", () => {
  it("is zero for a well-levelled area", () => {
    expect(floorHeightOf([cam(0, 0), cam(1, 0), cam(2, 0)])).toBeCloseTo(0, 12);
  });

  it("follows an area that was levelled a little off", () => {
    expect(floorHeightOf([cam(0, 0, 1.25), cam(1, 0, 1.3), cam(2, 0, 1.2)])).toBeCloseTo(-0.2, 9);
  });

  it("is not dragged by an outlier pose", () => {
    expect(floorHeightOf([cam(0, 0), cam(1, 0), cam(2, 0), cam(3, 0, 9), cam(4, 0)])).toBeCloseTo(0, 12);
  });

  it("refuses a broken area rather than standing the visitor in mid-air", () => {
    expect(floorHeightOf([cam(0, 0, 4), cam(1, 0, 4.2)])).toBe(0);
    expect(floorHeightOf([])).toBe(0);
  });
});

describe("the space a visitor may teleport in", () => {
  const property = manifest(area("base", route([[0, 0], [4, 0]])), [
    area("kitchen", route([[4, 1], [4, 3]])),
    area("bath", route([[50, 50], [51, 50]]), false),
    area("broken", route([[0, 9], [1, 9]]), true, { failed: "nope" }),
  ]);

  it("covers every placed area while standing in the property", () => {
    const space = teleportSpace(property, "base");
    expect(space.areas.map((a) => a.key).sort()).toEqual(["base", "kitchen"]);
    expect(validateTeleport([4, 2], space).valid).toBe(true);
    // A failed area is not somewhere to stand, however good its route looks.
    expect(validateTeleport([0.5, 9], space).valid).toBe(false);
  });

  it("keeps a visitor in an unplaced room to that room's own route", () => {
    const space = teleportSpace(property, "bath");
    expect(space.areas.map((a) => a.key)).toEqual(["bath"]);
    expect(validateTeleport([50.5, 50], space).valid).toBe(true);
    expect(validateTeleport([2, 0], space).valid).toBe(false);
  });

  it("does not bridge a cut in the route — that would be a corridor through a wall", () => {
    // Two stretches 4 m apart with nothing filmed between them.
    const cut = manifest(area("base", [...route([[0, 0], [2, 0]]), ...route([[6, 0], [8, 0]])]));
    const space = teleportSpace(cut, "base");
    expect(validateTeleport([1, 0], space).valid).toBe(true);
    expect(validateTeleport([7, 0], space).valid).toBe(true);
    // The middle of the gap is 2 m from either end: no segment was drawn across it.
    expect(nearestOnCorridor(space.corridor, 4, 0).d).toBeCloseTo(2, 9);
    expect(validateTeleport([4, 0], space).valid).toBe(false);
  });
});

describe("teleport validation", () => {
  it("accepts a target on or beside the route", () => {
    expect(validateTeleport([3, 0], SPACE).valid).toBe(true);
    expect(validateTeleport([3, PATH_RADIUS_M - 0.1], SPACE)).toMatchObject({ valid: true, snapped: false });
    expect(validateTeleport([6.8, 2], SPACE)).toMatchObject({ valid: true, snapped: false });
  });

  it("pulls a near miss back onto the edge instead of refusing it", () => {
    const verdict = validateTeleport([3, PATH_RADIUS_M + 0.3], SPACE);
    expect(verdict.valid).toBe(true);
    expect(verdict.snapped).toBe(true);
    expect(verdict.point[0]).toBeCloseTo(3, 9);
    expect(verdict.point[1]).toBeCloseTo(PATH_RADIUS_M, 9);
    expect(nearestOnCorridor(SPACE.corridor, verdict.point[0], verdict.point[1]).d).toBeCloseTo(PATH_RADIUS_M, 9);
  });

  it("refuses the unobserved middle of nowhere", () => {
    expect(validateTeleport([3, PATH_RADIUS_M + SNAP_SLACK + 0.1], SPACE).valid).toBe(false);
    expect(validateTeleport([2, 3], SPACE).valid).toBe(false);
    expect(validateTeleport([2, 3], { corridor: [], areas: [] }).valid).toBe(false);
  });

  it("lands on the floor of the area it is in", () => {
    // The loft's cameras ride 0.4 m higher: its floor is 0.4 m up.
    const loft = manifest(area("base", route([[0, 0], [4, 0]])), [area("loft", route([[20, 0], [24, 0]], CAPTURE_HEIGHT + 0.4))]);
    const space = teleportSpace(loft, "base");
    expect(validateTeleport([22, 0.2], space).floorY).toBeCloseTo(0.4, 9);
    expect(validateTeleport([1, 0.2], space).floorY).toBeCloseTo(0, 9);
    expect(floorAt(space, 23, 0)).toBeCloseTo(0.4, 9);
  });
});

describe("aiming", () => {
  it("drops a straight ray onto the floor", () => {
    const hit = rayToFloor([0, 1.6, 0], [0, -1, -1], 0)!;
    expect(hit[0]).toBeCloseTo(0, 12);
    expect(hit[1]).toBe(0);
    expect(hit[2]).toBeCloseTo(-1.6, 12);
  });

  it("finds no floor above the horizon or beyond reach", () => {
    expect(rayToFloor([0, 1.6, 0], [0, 0.1, -1], 0)).toBeNull();
    expect(rayToFloor([0, 1.6, 0], [0, -0.05, -1], 0, 12)).toBeNull();
  });

  it("throws the arc about six metres at 45 degrees from the hand", () => {
    const { landing, points } = arcToFloor([0, 1.2, 0], [0, 1, -1], 0);
    expect(landing).not.toBeNull();
    expect(-landing![2]).toBeGreaterThan(5.5);
    expect(-landing![2]).toBeLessThan(6.5);
    expect(landing![1]).toBe(0);
    // The drawn arc starts at the hand and ends on the landing point.
    expect(points[0]).toEqual([0, 1.2, 0]);
    expect(points[points.length - 1]).toEqual(landing);
  });

  it("lands a downward flick close in front", () => {
    const { landing } = arcToFloor([0, 1.2, 0], [0, -1, -0.3], 0);
    expect(-landing![2]).toBeLessThan(0.6);
  });
});

describe("the rig", () => {
  const rig = { x: 2, z: -1, yaw: 0.7 };
  // The visitor stands 40 cm right of and 30 cm behind the middle of their play space.
  const headLocal: Vec2 = [0.4, 0.3];

  it("teleports the HEAD onto the target, wherever it is in the play space", () => {
    const moved = rigForTeleport(rig, headLocal, [10, 5]);
    const head = headWorld(moved, headLocal);
    expect(head[0]).toBeCloseTo(10, 12);
    expect(head[1]).toBeCloseTo(5, 12);
    expect(moved.yaw).toBe(rig.yaw);
  });

  it("snap-turns about the head, which does not move", () => {
    const before = headWorld(rig, headLocal);
    const turned = rigAfterTurn(rig, headLocal, Math.PI / 6);
    const after = headWorld(turned, headLocal);
    expect(after[0]).toBeCloseTo(before[0], 12);
    expect(after[1]).toBeCloseTo(before[1], 12);
    expect(turned.yaw).toBeCloseTo(rig.yaw + Math.PI / 6, 12);
  });
});

describe("smooth movement", () => {
  it("moves freely along the route", () => {
    expect(constrainMove([1, 0], [1.1, 0.05], SPACE)).toEqual([1.1, 0.05]);
  });

  it("slides along the edge instead of sticking to it", () => {
    // Pushing diagonally out of the corridor keeps the along-corridor half of the step.
    const next = constrainMove([2, PATH_RADIUS_M - 0.05], [2.1, PATH_RADIUS_M + 0.05], SPACE);
    expect(next).toEqual([2.1, PATH_RADIUS_M - 0.05]);
  });

  it("lets a visitor who walked off the route come back, but not go further", () => {
    const outside: Vec2 = [2, 2];
    expect(constrainMove(outside, [2, 1.9], SPACE)).toEqual([2, 1.9]);
    expect(constrainMove(outside, [2, 2.1], SPACE)).toEqual(outside);
  });

  it("constrains nothing in an area with no route at all", () => {
    expect(constrainMove([0, 0], [9, 9], { corridor: [], areas: [] })).toEqual([9, 9]);
  });
});
