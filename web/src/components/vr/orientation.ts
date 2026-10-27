/**
 * Head orientation for the phone headset, and the one rotation that lines a panorama up with a
 * head.
 *
 * Plain quaternion tuples `[x, y, z, w]`, no three.js: the walkthrough's headset mode runs on
 * PlayCanvas and shares this file, and a page that only shows the walkthrough should not pay
 * for a second 3D engine to turn four sensor angles into a rotation.
 *
 * ── DEVICE ORIENTATION → CAMERA ───────────────────────────────────────────────────────────
 * `deviceorientation` reports the phone as intrinsic Z-X'-Y'' angles (alpha, beta, gamma) in an
 * Earth frame that is Z-up. The conversion is the long-standing one from three.js's old
 * DeviceOrientationControls: re-read the angles as a Y-X-Z Euler in a Y-up frame, turn -90
 * degrees about X (the camera looks out of the BACK of the phone, not out of its top edge),
 * then spin about the view axis by the screen angle (in landscape the picture is rotated
 * relative to the hardware). The tests build poses from their physical description and hold
 * this to three's own Euler maths and to `lib/capture/useOrientation.ts`'s independent
 * expansion, so a sign slip in any one of the three is caught by the other two.
 *
 * Nothing is smoothed. A low-pass on head tracking trades jitter for latency, and latency is the
 * one that makes people sick: the picture lagging the head by even 50 ms reads as the room
 * sliding. The sensors are already fused on the device.
 *
 * ── YAW ALIGNMENT ─────────────────────────────────────────────────────────────────────────────
 * A headset's forward is wherever the visitor happened to be facing when it started, and a
 * phone's alpha is relative to wherever the sensor booted. Neither has anything to do with the
 * panorama's yaw 0. So the panorama is turned instead of the head: rotating the content about
 * the vertical by `θ` shows panorama yaw `v` at world yaw `v - θ` (three's `rotation.y`
 * convention), so `θ = v - h` puts the view the visitor was just looking at — `v`, from the 2D
 * player — straight ahead of wherever their head points, `h`. The same rule puts a new room's
 * arrival view in front of them after every doorway.
 */

export type Quat = [number, number, number, number];

const DEG = Math.PI / 180;

export function quatMultiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    ax * bw + aw * bx + ay * bz - az * by,
    ay * bw + aw * by + az * bx - ax * bz,
    az * bw + aw * bz + ax * by - ay * bx,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/** Rotates `v` by `q` (three's `Vector3.applyQuaternion`). */
export function rotateByQuat(q: Quat, v: [number, number, number]): [number, number, number] {
  const [qx, qy, qz, qw] = q;
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return [v[0] + qw * tx + qy * tz - qz * ty, v[1] + qw * ty + qz * tx - qx * tz, v[2] + qw * tz + qx * ty - qy * tx];
}

/** The quaternion of a Y-X-Z Euler (three's `setFromEuler` for order "YXZ"). */
export function quatFromEulerYXZ(x: number, y: number, z: number): Quat {
  const c1 = Math.cos(x / 2);
  const c2 = Math.cos(y / 2);
  const c3 = Math.cos(z / 2);
  const s1 = Math.sin(x / 2);
  const s2 = Math.sin(y / 2);
  const s3 = Math.sin(z / 2);
  return [
    s1 * c2 * c3 + c1 * s2 * s3,
    c1 * s2 * c3 - s1 * c2 * s3,
    c1 * c2 * s3 - s1 * s2 * c3,
    c1 * c2 * c3 + s1 * s2 * s3,
  ];
}

/** -90 degrees about X: from "out of the top edge" to "out of the back". */
const BACK_OF_PHONE: Quat = [-Math.SQRT1_2, 0, 0, Math.SQRT1_2];

/** Camera quaternion for a phone held at (alpha, beta, gamma), screen rotated by `screenAngleDeg`. */
export function quatFromDeviceOrientation(alphaDeg: number, betaDeg: number, gammaDeg: number, screenAngleDeg: number): Quat {
  const device = quatFromEulerYXZ(betaDeg * DEG, alphaDeg * DEG, -gammaDeg * DEG);
  const half = (-screenAngleDeg * DEG) / 2;
  const spin: Quat = [0, 0, Math.sin(half), Math.cos(half)];
  return quatMultiply(quatMultiply(device, BACK_OF_PHONE), spin);
}

/**
 * The camera quaternion for looking along (yaw, pitch) with a level horizon — the drag source
 * of the desktop debug mode, and the same pose `orientCamera` gives the 2D player.
 */
export function quatFromView(yaw: number, pitch: number): Quat {
  return quatFromEulerYXZ(pitch, -(yaw + Math.PI / 2), 0);
}

/** Where a camera quaternion looks, in the player's angle convention (yaw from +X towards +Z). */
export function viewOfQuat(q: Quat): { yaw: number; pitch: number } {
  const f = rotateByQuat(q, [0, 0, -1]);
  return { yaw: Math.atan2(f[2], f[0]), pitch: Math.asin(Math.max(-1, Math.min(1, f[1]))) };
}

/** Wraps an angle to (-π, π]. */
export function wrapAngle(angle: number): number {
  const m = ((((angle + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
  return m === -Math.PI ? Math.PI : m;
}

/** The content rotation that shows panorama yaw `viewYaw` where the head points (`headYaw`). */
export function alignmentYaw(viewYaw: number, headYaw: number): number {
  return wrapAngle(viewYaw - headYaw);
}

/** The panorama yaw under a head pointing at world yaw `headYaw`, given the content rotation. */
export function panoramaYawAt(headYaw: number, contentYaw: number): number {
  return wrapAngle(headYaw + contentYaw);
}

/**
 * The screen's rotation in degrees, from whichever API this browser has.
 *
 * `screen.orientation.angle` everywhere modern; `window.orientation` on iOS Safari before 16.4,
 * which is still a meaningful share of the phones a property link is opened on.
 */
export function readScreenAngle(): number {
  if (typeof window === "undefined") return 0;
  const angle = window.screen?.orientation?.angle;
  if (typeof angle === "number") return angle;
  const legacy = (window as unknown as { orientation?: number }).orientation;
  return typeof legacy === "number" ? legacy : 0;
}
