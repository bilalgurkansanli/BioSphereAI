/**
 * Head tracking for the phone headset.
 *
 * The poses are built from the physical description ("phone upright in landscape, screen
 * facing me, I face north") rather than typed in as angles, because the angles are where the
 * mistakes live: a Cardboard that turns the wrong way is unwearable, and it looks perfectly
 * correct in any test written from the same misreading of the spec as the code.
 */

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { quatFromDeviceOrientation as captureQuat } from "@/lib/capture/useOrientation";
import { directionFromAngles } from "@/components/player/useLookControls";
import {
  alignmentYaw,
  panoramaYawAt,
  quatFromDeviceOrientation,
  quatFromView,
  rotateByQuat,
  viewOfQuat,
  wrapAngle,
  type Quat,
} from "../orientation";

const RAD = 180 / Math.PI;

/**
 * W3C device orientation angles for a phone whose axes point the given ways in the Earth frame
 * (x east, y north, z up). The rotation matrix's columns ARE the device axes; decomposing it as
 * intrinsic Z-X'-Y'' gives alpha, beta, gamma — exactly what the sensor would report.
 */
function anglesFor(deviceX: THREE.Vector3, deviceY: THREE.Vector3, deviceZ: THREE.Vector3) {
  const m = new THREE.Matrix4().makeBasis(deviceX, deviceY, deviceZ);
  expect(m.determinant(), "the device axes must form a right-handed frame").toBeCloseTo(1, 9);
  const e = new THREE.Euler().setFromRotationMatrix(m, "ZXY");
  return { alpha: e.z * RAD, beta: e.x * RAD, gamma: e.y * RAD };
}

const EAST = new THREE.Vector3(1, 0, 0);
const WEST = new THREE.Vector3(-1, 0, 0);
const NORTH = new THREE.Vector3(0, 1, 0);
const SOUTH = new THREE.Vector3(0, -1, 0);
const UP = new THREE.Vector3(0, 0, 1);
const DOWN = new THREE.Vector3(0, 0, -1);

function cameraAxes(q: Quat) {
  return { forward: rotateByQuat(q, [0, 0, -1]), up: rotateByQuat(q, [0, 1, 0]) };
}

function expectVec(actual: [number, number, number], expected: [number, number, number]) {
  for (let i = 0; i < 3; i += 1) expect(actual[i]).toBeCloseTo(expected[i], 9);
}

/** Three's frame: north is -Z, up is +Y. */
const LOOK_NORTH: [number, number, number] = [0, 0, -1];
const WORLD_UP: [number, number, number] = [0, 1, 0];

describe("device orientation", () => {
  it("looks at the horizon, level, when the phone is upright in portrait", () => {
    // Screen faces the visitor (south), top edge up.
    const a = anglesFor(EAST, UP, SOUTH);
    const { forward, up } = cameraAxes(quatFromDeviceOrientation(a.alpha, a.beta, a.gamma, 0));
    expectVec(forward, LOOK_NORTH);
    expectVec(up, WORLD_UP);
  });

  it("stays level in the headset, in BOTH landscape orientations", () => {
    // Landscape, top edge to the left (screen angle 90): device +X points up, +Y west.
    const left = anglesFor(UP, WEST, SOUTH);
    const a = cameraAxes(quatFromDeviceOrientation(left.alpha, left.beta, left.gamma, 90));
    expectVec(a.forward, LOOK_NORTH);
    expectVec(a.up, WORLD_UP);

    // Landscape the other way up (screen angle 270, which iOS reports as -90).
    const right = anglesFor(DOWN, EAST, SOUTH);
    for (const angle of [270, -90]) {
      const b = cameraAxes(quatFromDeviceOrientation(right.alpha, right.beta, right.gamma, angle));
      expectVec(b.forward, LOOK_NORTH);
      expectVec(b.up, WORLD_UP);
    }
  });

  it("turns the view the way the head turns", () => {
    // Visitor turns to face east: the screen now faces west (back at them) and the phone's top
    // edge, still on their left, points north.
    const east = anglesFor(UP, NORTH, WEST);
    const q = quatFromDeviceOrientation(east.alpha, east.beta, east.gamma, 90);
    expectVec(cameraAxes(q).forward, [1, 0, 0]);
    // In the player's convention north is -90 degrees and east is 0: turning right from north
    // to east is +90 degrees of yaw — "positive turns right", as the engine's contract says.
    const north = viewOfQuat(quatFromView(-Math.PI / 2, 0));
    expect(wrapAngle(viewOfQuat(q).yaw - north.yaw)).toBeCloseTo(Math.PI / 2, 9);
  });

  it("looks down when the visitor looks at the floor", () => {
    // Head tilted 40 degrees down in landscape: the screen tilts back to face partly up.
    const tilt = (40 * Math.PI) / 180;
    const screenNormal = new THREE.Vector3(0, -Math.cos(tilt), Math.sin(tilt));
    const deviceX = new THREE.Vector3(0, Math.sin(tilt), Math.cos(tilt));
    const a = anglesFor(deviceX, WEST, screenNormal);
    expect(viewOfQuat(quatFromDeviceOrientation(a.alpha, a.beta, a.gamma, 90)).pitch).toBeCloseTo(-tilt, 9);
  });

  it("matches three's own Euler maths and the capture flow's expansion, everywhere", () => {
    let seed = 7;
    const random = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    const euler = new THREE.Euler();
    for (let i = 0; i < 300; i += 1) {
      const alpha = random() * 360;
      const beta = random() * 360 - 180;
      const gamma = random() * 180 - 90;
      const angle = [0, 90, 180, 270, -90][i % 5];
      const ours = quatFromDeviceOrientation(alpha, beta, gamma, angle);

      const three = new THREE.Quaternion()
        .setFromEuler(euler.set(beta / RAD, alpha / RAD, -gamma / RAD, "YXZ"))
        .multiply(new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2))
        .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -angle / RAD));
      const capture = captureQuat(alpha, beta, gamma, angle);
      // q and -q are the same rotation.
      const dot = (q: ArrayLike<number>) => Math.abs(ours[0] * q[0] + ours[1] * q[1] + ours[2] * q[2] + ours[3] * q[3]);
      expect(dot([three.x, three.y, three.z, three.w])).toBeCloseTo(1, 9);
      expect(dot(capture)).toBeCloseTo(1, 9);
    }
  });
});

describe("view quaternion", () => {
  it("points exactly where the 2D player's angles point", () => {
    for (const [yaw, pitch] of [
      [0, 0],
      [1.2, 0.3],
      [-2.9, -0.7],
      [Math.PI, 1.2],
    ]) {
      const forward = rotateByQuat(quatFromView(yaw, pitch), [0, 0, -1]);
      const expected = directionFromAngles(yaw, pitch);
      expectVec(forward, [expected.x, expected.y, expected.z]);
      const back = viewOfQuat(quatFromView(yaw, pitch));
      expect(wrapAngle(back.yaw - yaw)).toBeCloseTo(0, 9);
      expect(back.pitch).toBeCloseTo(pitch, 9);
    }
  });

  it("keeps the horizon level", () => {
    const up = rotateByQuat(quatFromView(0.8, 0.4), [1, 0, 0]);
    expect(up[1]).toBeCloseTo(0, 12);
  });
});

describe("yaw alignment", () => {
  it("puts the view the visitor was looking at straight ahead of their head", () => {
    for (const view of [-3, -1, 0, 0.7, 2.5]) {
      for (const head of [-2.2, 0, 1.4, 3.1]) {
        const content = alignmentYaw(view, head);
        expect(wrapAngle(panoramaYawAt(head, content) - view)).toBeCloseTo(0, 9);
        // And it really is three's rotation.y: a direction at panorama yaw `view`, rotated by
        // the content's rotation, lands at world yaw `head`.
        const rotated = directionFromAngles(view, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), content);
        expect(wrapAngle(Math.atan2(rotated.z, rotated.x) - head)).toBeCloseTo(0, 9);
      }
    }
  });

  it("wraps into (-pi, pi]", () => {
    expect(wrapAngle(Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(wrapAngle(-Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(wrapAngle(3 * Math.PI + 0.5)).toBeCloseTo(-Math.PI + 0.5, 12);
  });
});
