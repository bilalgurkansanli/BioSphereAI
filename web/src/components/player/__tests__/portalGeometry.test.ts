/**
 * The rule that puts a floor ring somewhere, measured.
 *
 * Every case here is one a viewer could hit by aiming at the room rather than at a control, so
 * they are written as headings a person would recognise: a ring at a doorway, a ring at the
 * ceiling, a tap that was nearly on target.
 */

import { describe, expect, it } from "vitest";
import type { Hotspot } from "@/lib/types";
import {
  EYE_HEIGHT,
  MAX_DISTANCE,
  RING_RADIUS,
  TAP_TOLERANCE,
  floorPoint,
  hitTest,
  tagLift,
  unliftPitch,
} from "../portalGeometry";

const rad = (deg: number) => (deg * Math.PI) / 180;

function portal(id: string, yawDeg: number, pitchDeg: number): Hotspot {
  return {
    id,
    scene_id: "scene",
    type: "portal",
    yaw: rad(yawDeg),
    pitch: rad(pitchDeg),
    title: id,
    body: null,
    icon: null,
    target_scene_id: "target",
    media_url: null,
    accent: null,
    created_at: "2026-01-01T00:00:00Z",
  };
}

describe("floorPoint", () => {
  it("puts the ring on the floor, never above or below it", () => {
    for (const pitch of [-7, -15, -27, -45, -80, -90]) {
      const hit = floorPoint(rad(30), rad(pitch));
      expect(hit, `pitch ${pitch}`).not.toBeNull();
      expect(hit!.y).toBeCloseTo(-EYE_HEIGHT, 10);
    }
  });

  it("lands at the distance the depression angle implies", () => {
    // The seeded doorway ring: 27 degrees down from a 1.6 m eye is 1.6 / tan(27) away.
    const hit = floorPoint(rad(77), rad(-27))!;
    const ground = Math.hypot(hit.x, hit.z);
    expect(ground).toBeCloseTo(EYE_HEIGHT / Math.tan(rad(27)), 6);
    expect(ground).toBeGreaterThan(3.0);
    expect(ground).toBeLessThan(3.2);
  });

  it("keeps the bearing it was given", () => {
    const hit = floorPoint(rad(77), rad(-27))!;
    // Same convention as `directionFromAngles`: yaw is measured from +X towards +Z.
    expect(Math.atan2(hit.z, hit.x)).toBeCloseTo(rad(77), 10);
  });

  it("refuses a portal on the ceiling instead of clamping it to the floor", () => {
    // Clamping is the tempting bug: it would silently move a misplaced ring to somewhere
    // plausible, and the agent would never learn they had put it on the wall.
    expect(floorPoint(rad(0), rad(10))).toBeNull();
    expect(floorPoint(rad(0), rad(0))).toBeNull();
    expect(floorPoint(rad(0), rad(89))).toBeNull();
  });

  it("refuses a ring that would land beyond the room", () => {
    // A degree below the horizon solves to ~90 m: through the wall, and sub-pixel once drawn,
    // so it would be an invisible click target rather than a portal.
    const grazing = floorPoint(rad(0), rad(-1));
    expect(grazing).toBeNull();

    // The far edge of what is allowed still resolves.
    const justInside = floorPoint(rad(0), -Math.atan(EYE_HEIGHT / (MAX_DISTANCE - 0.5)));
    expect(justInside).not.toBeNull();
    expect(Math.hypot(justInside!.x, justInside!.z)).toBeLessThan(MAX_DISTANCE);
  });

  it("states the shallowest angle a ring can be placed at", () => {
    // Not a rule of its own: it falls out of MAX_DISTANCE and eye height, and it is the number
    // that matters when placing one, because a doorway across a large room is the case that
    // gets refused. atan(1.6 / 14) is about 6.5 degrees below the horizon; anything shallower
    // is dropped. Asserted so that changing either constant has to be a deliberate act.
    const shallowest = Math.atan(EYE_HEIGHT / MAX_DISTANCE);
    expect((shallowest * 180) / Math.PI).toBeCloseTo(6.52, 2);
    expect(floorPoint(rad(0), -shallowest * 1.01)).not.toBeNull();
    expect(floorPoint(rad(0), -shallowest * 0.99)).toBeNull();
  });

  it("puts a straight-down portal at the viewer's feet", () => {
    const hit = floorPoint(rad(0), rad(-90))!;
    expect(Math.hypot(hit.x, hit.z)).toBeCloseTo(0, 6);
  });
});

describe("hitTest", () => {
  const door = portal("door", 77, -27);
  const hall = portal("hall", 200, -35);

  it("takes a tap aimed straight at a ring", () => {
    expect(hitTest([door, hall], door.yaw, door.pitch)?.id).toBe("door");
    expect(hitTest([door, hall], hall.yaw, hall.pitch)?.id).toBe("hall");
  });

  it("forgives a tap a few degrees off", () => {
    expect(hitTest([door], rad(79), rad(-26))?.id).toBe("door");
  });

  it("lets a tap on the floor between rings fall through to the sphere", () => {
    // The look controls own most taps; a portal that claimed them from this far away would
    // make the room feel like it was full of invisible buttons.
    expect(hitTest([door, hall], rad(77), rad(-50))).toBeNull();
    expect(hitTest([door, hall], rad(120), rad(-27))).toBeNull();
  });

  it("gives an overlap to the nearer ring", () => {
    const twin = portal("twin", 80, -27);
    expect(hitTest([door, twin], rad(79.5), rad(-27))?.id).toBe("twin");
    expect(hitTest([door, twin], rad(77.5), rad(-27))?.id).toBe("door");
  });

  it("never hits a ring that was never drawn", () => {
    // A portal above the horizon has no floor point, so it has no ring - and must therefore
    // have no click target either, or the property would have a dead zone nobody can see.
    const onTheWall = portal("wall", 10, 20);
    expect(floorPoint(onTheWall.yaw, onTheWall.pitch)).toBeNull();
    expect(hitTest([onTheWall], onTheWall.yaw, onTheWall.pitch)).toBeNull();
  });

  it("measures the tolerance as an angle, so a far ring is a smaller target on screen", () => {
    // Same angular slack whatever the distance, which is what "I aimed at that" means; the
    // ring itself shrinks with distance because perspective does that for us.
    const near = portal("near", 0, -60);
    const far = portal("far", 0, -8);
    const offBy = TAP_TOLERANCE * 0.6;
    expect(hitTest([near], near.yaw, near.pitch + offBy)?.id).toBe("near");
    expect(hitTest([far], far.yaw, far.pitch + offBy)?.id).toBe("far");
    expect(hitTest([near], near.yaw, near.pitch + TAP_TOLERANCE * 1.4)).toBeNull();
  });

  it("has nothing to say about an empty room", () => {
    expect(hitTest([], rad(0), rad(-30))).toBeNull();
  });
});

describe("tagLift", () => {
  /** The pitch of the ring's highest point on screen: its far edge, being nearest the horizon. */
  function farEdgePitch(pitch: number): number {
    const ground = EYE_HEIGHT / Math.tan(-pitch);
    return -Math.atan(EYE_HEIGHT / (ground + RING_RADIUS));
  }

  it("lifts the editing handle clear of the ring at every distance", () => {
    // The bug this exists for: the handle sat in the middle of the ring and, being a DOM node
    // over the canvas, ate every click meant for it.
    for (const deg of [-7, -12, -20, -27, -40, -60, -80, -89]) {
      const pitch = rad(deg);
      const lifted = pitch + tagLift(pitch);
      expect(lifted, `pitch ${deg}`).toBeGreaterThan(farEdgePitch(pitch));
    }
  });

  it("lifts a near ring further than a far one, because a near ring is bigger", () => {
    // A fixed offset is the obvious implementation and the wrong one: a ring at your feet is a
    // wide ellipse and one across the room is a sliver, so one number cannot clear both.
    const atYourFeet = tagLift(rad(-75));
    const acrossTheRoom = tagLift(rad(-8));
    expect(atYourFeet).toBeGreaterThan(acrossTheRoom * 3);
  });

  it("has nothing to lift where there is no ring", () => {
    expect(tagLift(rad(10))).toBe(0);
    expect(tagLift(rad(0))).toBe(0);
  });

  it("round-trips a drag without walking the ring across the floor", () => {
    // Subtracting the lift measured at the DROP is not the inverse, and the error is one-signed:
    // it returned a slightly shallower ring every time, so a ring nudged ten times drifted a
    // metre. `unliftPitch` solves for the pitch instead.
    for (const deg of [-8, -15, -27, -45, -70]) {
      let pitch = rad(deg);
      for (let i = 0; i < 20; i += 1) pitch = unliftPitch(pitch + tagLift(pitch));
      // Worst measured residue is 0.007 deg over these twenty trips, at the steepest ring. The
      // bound is loose against that and still two orders tighter than the bug it guards: the
      // old one-sided subtraction drifted 13 degrees over the same twenty.
      expect(Math.abs(pitch - rad(deg)), `pitch ${deg}`).toBeLessThan(rad(0.02));
    }
  });
});

