/**
 * What a ray in a headset is pointing at. Each case is something a visitor would do: aim a
 * controller from the hip at a ring across the room, rest a gaze on a ring's rim, poke at a
 * panel from the side.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { directionFromAngles } from "@/components/player/useLookControls";
import { EYE_HEIGHT, RING_RADIUS, TAP_TOLERANCE, floorPoint, hitTest } from "@/components/player/portalGeometry";
import type { Hotspot } from "@/lib/types";
import {
  angleBetween,
  directionOf,
  directionOnSphere,
  normalize,
  panelFacing,
  pickAngular,
  pickRing,
  rayPanel,
  regionAt,
  rotateY,
  type RingTarget,
  type Vec3,
} from "../gaze";

const rad = (deg: number) => (deg * Math.PI) / 180;

function ring(id: string, yawDeg: number, pitchDeg: number): RingTarget {
  const p = floorPoint(rad(yawDeg), rad(pitchDeg))!;
  return { id, centre: [p.x, p.y, p.z], radius: RING_RADIUS };
}

describe("conventions", () => {
  it("agrees with the 2D player's direction for a (yaw, pitch)", () => {
    for (const [y, p] of [
      [0.3, -0.4],
      [-2.1, 0.9],
      [3.1, 0],
    ]) {
      const ours = directionOf(y, p);
      const theirs = directionFromAngles(y, p);
      expect(ours[0]).toBeCloseTo(theirs.x, 12);
      expect(ours[1]).toBeCloseTo(theirs.y, 12);
      expect(ours[2]).toBeCloseTo(theirs.z, 12);
    }
  });

  it("rotates exactly as three's rotation.y does", () => {
    const v: Vec3 = [0.3, -0.2, 0.9];
    for (const a of [0.4, -1.7, 2.9]) {
      const ours = rotateY(v, a);
      const theirs = new THREE.Vector3(...v).applyEuler(new THREE.Euler(0, a, 0));
      expect(ours[0]).toBeCloseTo(theirs.x, 12);
      expect(ours[1]).toBeCloseTo(theirs.y, 12);
      expect(ours[2]).toBeCloseTo(theirs.z, 12);
    }
  });

  it("measures small and opposite angles without losing precision", () => {
    expect(angleBetween([1, 0, 0], [1, 1e-9, 0])).toBeCloseTo(1e-9, 15);
    // acos(dot) cannot resolve either of these: 1e-9 is below the spacing of doubles near 1.
    expect(angleBetween([1, 0, 0], [-1, 1e-9, 0])).toBeCloseTo(Math.PI - 1e-9, 15);
  });
});

describe("controller rays at infinity", () => {
  it("sees what the head would see where the ray meets the panorama", () => {
    // A controller at the hip, 40 cm right and 50 cm below the eye, aimed at a point on the
    // 50 m sphere: the direction from the head must be the point's own direction.
    const hand: Vec3 = [0.4, -0.5, 0.1];
    const target = normalize([0.2, -0.3, -1]);
    const point: Vec3 = [target[0] * 50, target[1] * 50, target[2] * 50];
    const aim = normalize([point[0] - hand[0], point[1] - hand[1], point[2] - hand[2]]);
    const seen = directionOnSphere(hand, aim, 50);
    expect(angleBetween(seen, target)).toBeLessThan(1e-9);
    // Using the bare ray direction instead would be off by about half a degree — the size of
    // a far ring, which is why the sphere step exists.
    expect(angleBetween(aim, target)).toBeGreaterThan(rad(0.4));
  });
});

describe("rings", () => {
  const rings = [ring("door", 77, -27), ring("far", -30, -9), ring("feet", 180, -60)];
  // The headset uses the 2D player's own tap tolerance as the floor of a ring's angular size.
  const opts = { eyeHeight: EYE_HEIGHT, minAngle: TAP_TOLERANCE };

  it("picks the ring the gaze rests on", () => {
    expect(pickRing(directionOf(rad(77), rad(-27)), rings, opts)?.id).toBe("door");
    expect(pickRing(directionOf(rad(-30), rad(-9)), rings, opts)?.id).toBe("far");
    expect(pickRing(directionOf(rad(180), rad(-60)), rings, opts)?.id).toBe("feet");
  });

  it("counts the rim of a near ring, which spans many degrees", () => {
    // The ring at the visitor's feet is 0.92 m away and 0.88 m across: its rim is nearly 15
    // degrees off its centre, far outside any fixed angular tolerance.
    const centre = rings[2].centre;
    const rim: Vec3 = [centre[0] + RING_RADIUS * 0.95, centre[1], centre[2]];
    const offset = angleBetween(normalize(rim), normalize(centre));
    expect(offset).toBeGreaterThan(rad(10));
    expect(pickRing(normalize(rim), rings, opts)?.id).toBe("feet");
  });

  it("gives a far ring a fair chance even though it is a sliver", () => {
    // 1.5 degrees above the far ring's centre: off the floor disc entirely, inside the minimum
    // angular tolerance.
    expect(pickRing(directionOf(rad(-30), rad(-7.5)), rings, opts)?.id).toBe("far");
  });

  it("holds a hovered ring a little longer than it takes a new one", () => {
    // Six degrees above the far ring: past the 5.2-degree tolerance, inside its 1.3x hold.
    const edge = directionOf(rad(-30), rad(-9 + 6));
    expect(pickRing(edge, rings, opts)).toBeNull();
    expect(pickRing(edge, rings, { ...opts, current: "far" })?.id).toBe("far");
  });

  it("finds nothing on the ceiling", () => {
    expect(pickRing(directionOf(0, rad(30)), rings, opts)).toBeNull();
  });

  it("is at least as forgiving as the 2D player's tap at mid-range", () => {
    // Whatever the tap tolerance accepts for a doorway ring, a headset gaze must too —
    // otherwise the same ring is easier to hit with a mouse than with a head.
    const portal: Hotspot = {
      id: "door",
      scene_id: "s",
      type: "portal",
      yaw: rad(77),
      pitch: rad(-27),
      title: "door",
      body: null,
      icon: null,
      target_scene_id: "t",
      media_url: null,
      accent: null,
      created_at: "",
    };
    for (let dy = -4; dy <= 4; dy += 1) {
      for (let dp = -4; dp <= 4; dp += 1) {
        const yaw = rad(77 + dy);
        const pitch = rad(-27 + dp);
        if (hitTest([portal], yaw, pitch)) expect(pickRing(directionOf(yaw, pitch), rings, opts)?.id).toBe("door");
      }
    }
  });
});

describe("angular targets", () => {
  const targets = [
    { id: "a", dir: directionOf(0, 0), radius: rad(3) },
    { id: "b", dir: directionOf(rad(5), 0), radius: rad(3) },
  ];

  it("prefers the nearer centre where two overlap", () => {
    expect(pickAngular(directionOf(rad(2), 0), targets)?.id).toBe("a");
    expect(pickAngular(directionOf(rad(3), 0), targets)?.id).toBe("b");
  });

  it("keeps the current one through a wobble at its edge", () => {
    const wobble = directionOf(rad(-3.4), 0);
    expect(pickAngular(wobble, targets)).toBeNull();
    expect(pickAngular(wobble, targets, { current: "a" })?.id).toBe("a");
  });
});

describe("panels", () => {
  // A menu 1.4 m ahead and 35 degrees below the eye, facing it.
  const eye: Vec3 = [0, 1.6, 0];
  const down = rad(35);
  const centre: Vec3 = [0, 1.6 - 1.4 * Math.sin(down), -1.4 * Math.cos(down)];
  const panel = panelFacing(centre, eye, 0.3, 0.07);

  it("faces the eye squarely and never rolls", () => {
    const toEye = normalize([eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2]]);
    const normal: Vec3 = [
      panel.right[1] * panel.up[2] - panel.right[2] * panel.up[1],
      panel.right[2] * panel.up[0] - panel.right[0] * panel.up[2],
      panel.right[0] * panel.up[1] - panel.right[1] * panel.up[0],
    ];
    expect(angleBetween(normal, toEye)).toBeLessThan(1e-9);
    expect(panel.right[1]).toBeCloseTo(0, 12);
    expect(panel.right[0]).toBeGreaterThan(0.99);
    expect(panel.up[1]).toBeGreaterThan(0);
  });

  it("maps a ray to the panel's own u/v, v counted downwards like the canvas", () => {
    const hit = rayPanel(eye, [centre[0] - eye[0], centre[1] - eye[1], centre[2] - eye[2]], panel);
    expect(hit?.u).toBeCloseTo(0.5, 9);
    expect(hit?.v).toBeCloseTo(0.5, 9);
    expect(hit?.distance).toBeCloseTo(1.4, 9);

    const topLeft: Vec3 = [
      centre[0] - panel.right[0] * 0.25 + panel.up[0] * 0.05,
      centre[1] - panel.right[1] * 0.25 + panel.up[1] * 0.05,
      centre[2] - panel.right[2] * 0.25 + panel.up[2] * 0.05,
    ];
    const corner = rayPanel(eye, [topLeft[0] - eye[0], topLeft[1] - eye[1], topLeft[2] - eye[2]], panel)!;
    expect(corner.u).toBeLessThan(0.2);
    expect(corner.v).toBeLessThan(0.2);
  });

  it("works from a controller held at the hip, and misses past the edge", () => {
    const hand: Vec3 = [0.25, 1.0, -0.2];
    const hit = rayPanel(hand, [centre[0] - hand[0], centre[1] - hand[1], centre[2] - hand[2]], panel);
    expect(hit?.u).toBeCloseTo(0.5, 9);
    const wide: Vec3 = [centre[0] + 0.5, centre[1], centre[2]];
    expect(rayPanel(hand, [wide[0] - hand[0], wide[1] - hand[1], wide[2] - hand[2]], panel)).toBeNull();
  });

  it("does not hit a panel behind the ray", () => {
    expect(rayPanel(eye, [eye[0] - centre[0], eye[1] - centre[1], eye[2] - centre[2]], panel)).toBeNull();
  });

  it("resolves regions", () => {
    const regions = [
      { id: "prev", u0: 0, v0: 0, u1: 0.2, v1: 1 },
      { id: "exit", u0: 0.8, v0: 0, u1: 1, v1: 1 },
    ];
    expect(regionAt({ u: 0.1, v: 0.5, distance: 1 }, regions)).toBe("prev");
    expect(regionAt({ u: 0.9, v: 0.5, distance: 1 }, regions)).toBe("exit");
    expect(regionAt({ u: 0.5, v: 0.5, distance: 1 }, regions)).toBeNull();
  });
});
