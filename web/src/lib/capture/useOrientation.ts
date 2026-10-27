"use client";

/**
 * ORIENTATION — one honest yaw/pitch/roll stream out of two incompatible platform APIs.
 * =====================================================================================
 *
 * Everything downstream (the coverage accumulator, the world-locked lattice, the row-pitch
 * detector) trusts this module, so it must never quietly invent a number. Three rules hold
 * throughout:
 *
 *   1. THE QUATERNION IS THE TRUTH. Euler angles are derived for display and for coverage
 *      stamping only. The Z-X'-Y'' triple is singular at beta = 90 deg, where the alpha and
 *      gamma axes coincide and the two are no longer separable — verified: (a=10, b=90, g=0)
 *      and (a=0, b=90, g=-10) both give yaw 350/10, and (a=10, b=90, g=-10) gives yaw 0.
 *      beta = 90 is the phone held UPRIGHT IN PORTRAIT WITH THE CAMERA AT THE HORIZON, i.e.
 *      pitch 0 — the LEVEL row, not the up row. (beta 119 -> pitch +29, beta 61 -> pitch -29,
 *      beta 179 -> pitch +89.) So the ill-conditioned pose is the FIRST row, the one that
 *      sets the loop-closure origin, which is also why phone compasses are visibly jittery
 *      held vertically. Building the quaternion makes the geometry well defined; it cannot
 *      undo the platform's own alpha/gamma noise at that pose, so `setOrigin` averages
 *      instead of sampling. See `setOrigin`.
 *   2. YAW IS CONTINUOUS INTERNALLY. The emitted `yawDeg` is wrapped 0..360 because that is
 *      what the contract's `OrientationSample` declares and what Builder C's
 *      `accumulateYaw` expects, but every filter inside this file runs on an unwrapped
 *      accumulator so a 359 -> 0 crossing can never spike a low-pass or corrupt a span.
 *      `unwrappedYawDeg()` exposes that accumulator for telemetry and diagnostics.
 *   3. SAY WHAT IS KNOWN. `reference` and `headingAccuracyDeg` travel with every sample:
 *      "absolute" means earth-referenced, "relative" means a gyro origin that drifts,
 *      "flow" means there is no orientation sensor at all and yaw is being integrated from
 *      camera optical flow (RAIL MODE) — in which case PITCH AND ROLL ARE NOT MEASURED and
 *      consumers must treat them as unknown rather than as level.
 *
 * PLATFORM ASYMMETRY THIS FILE HIDES
 *   iOS      no `deviceorientationabsolute` (it never fires), a gesture-gated
 *            `DeviceOrientationEvent.requestPermission()`, and a plain `deviceorientation`
 *            whose alpha is relative to an arbitrary start reference and drifts. The only
 *            earth-referenced heading is the non-standard `webkitCompassHeading`, which
 *            runs CLOCKWISE while the spec's alpha runs counter-clockwise.
 *   Android  no permission call, `deviceorientationabsolute` is earth-referenced but
 *            noisier, and plain `deviceorientation` may or may not set `absolute`.
 *
 * -------------------------------------------------------------------------------------
 * THE CAPTURE NEEDS NO TRUE NORTH, SO IT DOES NOT STEER BY THE MAGNETOMETER
 * -------------------------------------------------------------------------------------
 * `setOrigin()` zeroes yaw at `row_level` entry, and every station, every coverage cell and
 * the loop closure itself are defined RELATIVE to that origin. Nothing downstream knows or
 * cares which way is north. So the only thing an earth-referenced heading can contribute to
 * this feature is its errors.
 *
 * And indoors those errors are large and structured. Beside a fridge, a steel stud wall or
 * underfloor heating, `webkitCompassAccuracy` keeps reporting its nominal 10-15 deg while
 * the heading itself is 30-40 deg out, and it SNAPS as you turn past the anomaly. This file
 * used to take the compass whenever `accuracy >= 0` — i.e. whenever it was not the -1
 * "uncalibrated" sentinel — which is a test that an indoor magnetic anomaly passes every
 * time. `accumulateYaw`'s max-step filter protects the row SPAN from the resulting jump, but
 * nothing protects `stampCone` or `updateStations`: during an excursion the dome is painted,
 * and stations are banked, in directions the camera never pointed. That inflates
 * `coveredFraction`, and over-reporting coverage is the one direction the model must never
 * fail in. Worse, the one-euro filter makes it sharper rather than softer — a 35 deg snap
 * reads as an enormous angular speed, which is precisely when that filter opens up.
 *
 * So yaw is ALWAYS built from the relative track: the platform's own gyro/accelerometer
 * fusion, which is smooth, magnetically clean, and drifts a few degrees over a 50 s capture
 * where the magnetometer can be tens of degrees out in a single step. The compass is read
 * only to LABEL the stream (`reference`, `headingAccuracyDeg`) for telemetry and for the
 * calibration prompt. Concretely:
 *
 *   - iOS: `deviceorientation.alpha` drives the quaternion; `webkitCompassHeading` is
 *     recorded and never steers. `reference` becomes "relative", which is the truth.
 *   - Android: plain `deviceorientation` (the RELATIVE orientation sensor — gyroscope plus
 *     accelerometer, no magnetometer) is preferred, and `deviceorientationabsolute`, which
 *     is magnetometer-fused and carries the same indoor anomalies, is used only as a
 *     fallback on devices where the relative stream never fires. This is the exact reverse
 *     of the preference this file used to express.
 *
 * The reviewer's suggested cross-check — compare the frame-to-frame yaw delta against
 * `motion.rotationRate` projected on gravity, and reject updates that disagree — would work,
 * but it imports a second sensor stream into this module in order to reconstruct a signal
 * the platform has already fused for us and published as alpha. Preferring alpha IS that
 * cross-check taken to its conclusion: less noise, no new coupling, and nothing left to
 * disagree with.
 *
 * WHAT THIS COSTS. Gyro integration drifts, and on the absolute-fallback path (a device with
 * no relative stream at all) the heading is still magnetometer-fused and still exposed to
 * the same anomalies — no worse than before, but not fixed either. Both are the right side
 * of the trade: a slow drift moves the whole capture together and loop closure absorbs it,
 * while a 35 deg snap moves one part of it against the rest.
 *
 * The conversion from alpha/beta/gamma to a camera quaternion is the same intrinsic
 * Z-X'-Y'' composition the VR mode uses (`web/src/components/vr/orientation.ts`, whose tests
 * check it against this file), reimplemented on bare tuples so `lib/capture/*`
 * never imports `three`.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { Degrees, FlowSample, Millis, OrientationApi, OrientationReference, OrientationSample, Quat, Vec3 } from "./types";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

// -------------------------------------------------------------------------------------
// Primitives shared by the rest of lib/capture
// -------------------------------------------------------------------------------------

/** `performance.now()` where it exists, `Date.now()` where it does not. Never mixed. */
export function nowMs(): Millis {
  if (typeof performance !== "undefined" && typeof performance.now === "function") return performance.now();
  return Date.now();
}

/**
 * 0..360, and never 360 itself.
 *
 * The upper guard is not pedantry: a yaw of -1e-14 rounds to exactly 360 in double precision,
 * and a consumer binning by `floor(yaw / 5)` would index cell 72 of a 72-column grid. The
 * seam is the one place this arithmetic is ever wrong, so it is closed here rather than in
 * every caller.
 */
export function wrap360(deg: number): Degrees {
  if (!Number.isFinite(deg)) return 0;
  const w = deg % 360;
  const positive = w < 0 ? w + 360 : w;
  return positive >= 360 ? 0 : positive;
}

/** -180..180. The shortest signed difference when applied to (a - b). */
export function wrap180(deg: number): Degrees {
  const w = (deg + 180) % 360;
  return (w < 0 ? w + 360 : w) - 180;
}

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

export function quatConjugate(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (n === 0) return [0, 0, 0, 1];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function quatFromAxisAngle(axis: Vec3, angleRad: number): Quat {
  const h = angleRad / 2;
  const s = Math.sin(h);
  return [axis.x * s, axis.y * s, axis.z * s, Math.cos(h)];
}

/** Unsigned rotation angle between two orientations, degrees. */
export function quatAngleDeg(a: Quat, b: Quat): Degrees {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, dot)) * RAD;
}

/** Shortest-arc spherical interpolation. Port of THREE.Quaternion.slerp on bare tuples. */
export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  if (t <= 0) return a;
  if (t >= 1) return b;

  let [bx, by, bz, bw] = b;
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cos < 0) {
    // Both q and -q are the same rotation; pick the hemisphere that interpolates short.
    cos = -cos;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }
  if (cos >= 1) return a;

  const sqrSin = 1 - cos * cos;
  if (sqrSin <= Number.EPSILON) {
    return quatNormalize([
      (1 - t) * a[0] + t * bx,
      (1 - t) * a[1] + t * by,
      (1 - t) * a[2] + t * bz,
      (1 - t) * a[3] + t * bw,
    ]);
  }

  const sin = Math.sqrt(sqrSin);
  const half = Math.atan2(sin, cos);
  const ra = Math.sin((1 - t) * half) / sin;
  const rb = Math.sin(t * half) / sin;
  return [a[0] * ra + bx * rb, a[1] * ra + by * rb, a[2] * ra + bz * rb, a[3] * ra + bw * rb];
}

/** Rotate a vector by a quaternion. Port of THREE.Vector3.applyQuaternion. */
export function rotateVec(q: Quat, v: Vec3): Vec3 {
  const [qx, qy, qz, qw] = q;
  const tx = 2 * (qy * v.z - qz * v.y);
  const ty = 2 * (qz * v.x - qx * v.z);
  const tz = 2 * (qx * v.y - qy * v.x);
  return {
    x: v.x + qw * tx + qy * tz - qz * ty,
    y: v.y + qw * ty + qz * tx - qx * tz,
    z: v.z + qw * tz + qx * ty - qy * tx,
  };
}

/** Rotate a vector by the inverse of a quaternion (world -> local). */
export function unrotateVec(q: Quat, v: Vec3): Vec3 {
  return rotateVec(quatConjugate(q), v);
}

/**
 * alpha/beta/gamma (degrees) -> camera quaternion, world Y up, -Z north, +X east.
 *
 * `_euler.set(beta, alpha, -gamma, "YXZ")`, then the fixed -90 deg X turn that takes the
 * device frame to the camera frame, then the screen spin that compensates for the phone
 * being held rotated. Identical to `components/vr/orientation.ts`; the Euler-to-quaternion expansion below is
 * three's own YXZ case.
 *
 * At beta = 90 (phone upright in portrait) with alpha = gamma = 0 this returns identity:
 * the rear camera looks along -Z, i.e. at the horizon to the north. At beta = 0 (phone flat
 * on a table, screen up) the camera looks straight down. Both verified by hand.
 */
export function quatFromDeviceOrientation(alphaDeg: number, betaDeg: number, gammaDeg: number, screenAngleDeg: number): Quat {
  const x = betaDeg * DEG;
  const y = alphaDeg * DEG;
  const z = -gammaDeg * DEG;

  const c1 = Math.cos(x / 2);
  const c2 = Math.cos(y / 2);
  const c3 = Math.cos(z / 2);
  const s1 = Math.sin(x / 2);
  const s2 = Math.sin(y / 2);
  const s3 = Math.sin(z / 2);

  const euler: Quat = [
    s1 * c2 * c3 + c1 * s2 * s3,
    c1 * s2 * c3 - s1 * c2 * s3,
    c1 * c2 * s3 - s1 * s2 * c3,
    c1 * c2 * c3 + s1 * s2 * s3,
  ];

  // Device frame -> camera frame.
  const HALF_TURN_X: Quat = [-Math.SQRT1_2, 0, 0, Math.SQRT1_2];
  // Screen rotation, applied on the local side because it rotates the image, not the world.
  const spin = quatFromAxisAngle({ x: 0, y: 0, z: 1 }, -screenAngleDeg * DEG);
  return quatMultiply(quatMultiply(euler, HALF_TURN_X), spin);
}

/**
 * Camera quaternion -> the contract's angles.
 *
 *   yaw   clockwise from the frame's -Z axis, 0..360.  atan2(east, north) = atan2(fx, -fz)
 *   pitch elevation of the optical axis above the horizon, asin(fy)
 *   roll  tilt about the optical axis. Positive = the frame's RIGHT edge is higher than its
 *         left, i.e. the horizon in the preview slopes down to the right.
 *
 * Roll is undefined when the optical axis is vertical (both reference dot products vanish);
 * callers must hold the previous value there. `OrientationTracker` does exactly that.
 */
export function viewAnglesFromQuat(q: Quat): { yawDeg: Degrees; pitchDeg: Degrees; rollDeg: Degrees; degenerateRoll: boolean } {
  const f = rotateVec(q, { x: 0, y: 0, z: -1 });
  const up = rotateVec(q, { x: 0, y: 1, z: 0 });
  const right = rotateVec(q, { x: 1, y: 0, z: 0 });

  const yawDeg = wrap360(Math.atan2(f.x, -f.z) * RAD);
  const pitchDeg = Math.asin(Math.min(1, Math.max(-1, f.y))) * RAD;

  // World up is (0,1,0), so the dot products reduce to the y components.
  const denom = Math.hypot(right.y, up.y);
  const degenerateRoll = denom < 1e-3;
  const rollDeg = degenerateRoll ? 0 : Math.atan2(right.y, up.y) * RAD;

  return { yawDeg, pitchDeg, rollDeg, degenerateRoll };
}

/**
 * The rotation that takes ACCELEROMETER-frame vectors (x right, y top edge, z out of the
 * screen) into the same world frame the camera quaternion lives in.
 *
 * For a rear camera in portrait the camera-local frame IS the device frame (local -Z is out
 * of the back of the phone, local +Y is the top edge), so the only correction is undoing
 * the screen spin that `quatFromDeviceOrientation` applied on the local side. Used by
 * `useMotion` to integrate linear acceleration in a fixed frame instead of a tumbling one.
 */
export function deviceFrameQuat(cameraQuat: Quat, screenAngleDeg: number): Quat {
  const spin = quatFromAxisAngle({ x: 0, y: 0, z: 1 }, -screenAngleDeg * DEG);
  return quatMultiply(cameraQuat, quatConjugate(spin));
}

/** Unit vector pointing along gravity (i.e. DOWN) expressed in accelerometer axes. */
export function deviceGravityDirection(deviceQuat: Quat): Vec3 {
  return unrotateVec(deviceQuat, { x: 0, y: -1, z: 0 });
}

/** `screen.orientation.angle`, with the pre-16.4 iOS fallback and an SSR guard. */
export function readScreenAngleDeg(): Degrees {
  if (typeof window === "undefined") return 0;
  const angle = window.screen?.orientation?.angle;
  if (typeof angle === "number" && Number.isFinite(angle)) return angle;
  const legacy = (window as unknown as { orientation?: number }).orientation;
  return typeof legacy === "number" && Number.isFinite(legacy) ? wrap360(legacy) : 0;
}

// -------------------------------------------------------------------------------------
// The tracker — framework-free, so a test can drive it with a synthetic stream
// -------------------------------------------------------------------------------------

/** One raw reading, in the shape both platforms can be reduced to. */
export interface OrientationReading {
  alpha: number | null;
  beta: number | null;
  gamma: number | null;
  /** Android's `deviceorientationabsolute`, or `event.absolute` on plain events. */
  absolute?: boolean;
  /** iOS `webkitCompassHeading`: degrees CLOCKWISE from magnetic north. */
  compassHeadingDeg?: number | null;
  /** iOS `webkitCompassAccuracy`: typically ~10, and -1 when uncalibrated. */
  compassAccuracyDeg?: number | null;
  screenAngleDeg?: Degrees;
  t?: Millis;
}

export interface OrientationTrackerOptions {
  /**
   * One-euro filter. `minCutoffHz` sets how hard the hand tremor is squashed when the phone
   * is still; `speedCutoffGain` opens the filter up as the phone turns, so the lag never
   * becomes visible.
   *
   * Lag arithmetic, which is the whole reason for an adaptive filter rather than a fixed
   * one: tau = 1/(2*pi*cutoff). At rest cutoff = 1.0 Hz -> tau = 159 ms, invisible because
   * nothing is moving. At the coached 24 deg/s cutoff = 1.0 + 0.08*24 = 2.92 Hz -> tau =
   * 55 ms -> 1.3 deg of yaw lag, a seventh of the 9 deg capture radius. A fixed 1 Hz filter
   * would trail by 3.8 deg there, which reads as the lattice sliding behind the room.
   */
  minCutoffHz?: number;
  speedCutoffGain?: number;
  now?: () => Millis;
  screenAngleDeg?: () => Degrees;
}

export interface OrientationTracker {
  /** Feed one reading. Returns the emitted sample, or null when the reading was unusable. */
  push: (reading: OrientationReading) => OrientationSample | null;
  /**
   * RAIL MODE. Integrates `FlowSample.yawDeltaDeg` into a yaw-only sample. Pitch and roll
   * are carried over from the last real orientation sample (0 if there never was one) and
   * are NOT measurements — `reference` is "flow" precisely so consumers can tell.
   */
  pushFlow: (flow: FlowSample, t?: Millis) => OrientationSample | null;
  read: () => OrientationSample | null;
  readReference: () => OrientationReference;
  /** Continuous, never wraps, survives the 0/360 seam. Zeroed by `setOrigin`. */
  unwrappedYawDeg: () => Degrees;
  headingAccuracyDeg: () => number | null;
  /** True when the magnetometer says it is uncalibrated (-1) or worse than 25 deg. */
  calibrationNeeded: () => boolean;
  /** Milliseconds since the last emitted sample; Infinity before the first. */
  staleMs: (at?: Millis) => number;
  setOrigin: () => void;
  reset: () => void;
}

/**
 * How much trailing history `setOrigin` averages the loop-closure origin over.
 *
 * 500 ms is ~30 samples at either platform's delivery rate, so it divides the origin's
 * standard error by about 5.5, and it is comfortably inside the 1.2 s of stillness the
 * framing gates already demand before recording starts — so it is free.
 */
const ORIGIN_WINDOW_MS = 500;

/**
 * Circular mean of a set of headings, or null when there are none.
 *
 * Done as a vector sum rather than an arithmetic mean because these are angles: averaging
 * 359 and 1 arithmetically gives 180, which would point the origin at the exact opposite of
 * where the phone is looking.
 */
function averageYawDeg(window: readonly { yawDeg: Degrees }[]): Degrees | null {
  if (window.length === 0) return null;
  let sx = 0;
  let sy = 0;
  for (const s of window) {
    sx += Math.cos(s.yawDeg * DEG);
    sy += Math.sin(s.yawDeg * DEG);
  }
  if (sx === 0 && sy === 0) return window[window.length - 1].yawDeg;
  return wrap360(Math.atan2(sy, sx) * RAD);
}

export function createOrientationTracker(options: OrientationTrackerOptions = {}): OrientationTracker {
  const minCutoffHz = options.minCutoffHz ?? 1.0;
  const speedCutoffGain = options.speedCutoffGain ?? 0.08;
  const clock = options.now ?? nowMs;
  const screenAngle = options.screenAngleDeg ?? readScreenAngleDeg;

  const IDENTITY: Quat = [0, 0, 0, 1];

  let smoothed: Quat | null = null;
  let lastRaw: Quat | null = null;
  let lastT: Millis | null = null;
  let originQuat: Quat = IDENTITY;
  let unwrappedYaw = 0;
  let lastYawDeg: number | null = null;
  let lastRollDeg = 0;
  let reference: OrientationReference = "relative";
  let accuracy: number | null = null;
  let sample: OrientationSample | null = null;
  /** Trailing pre-origin headings, so `setOrigin` can average instead of sampling. */
  let originWindow: { t: Millis; yawDeg: Degrees }[] = [];

  function emit(q: Quat, t: Millis, ref: OrientationReference, screenAngleDeg: Degrees): OrientationSample {
    const angles = viewAnglesFromQuat(q);
    // Hold roll through the poles rather than letting it snap to zero: at |pitch| ~ 90 the
    // roll reference vanishes, and a jumping horizon line is worse than a stale one.
    const rollDeg = angles.degenerateRoll ? lastRollDeg : angles.rollDeg;
    lastRollDeg = rollDeg;

    unwrappedYaw += lastYawDeg === null ? 0 : wrap180(angles.yawDeg - lastYawDeg);
    lastYawDeg = angles.yawDeg;

    sample = {
      t,
      q,
      yawDeg: angles.yawDeg,
      pitchDeg: angles.pitchDeg,
      rollDeg,
      reference: ref,
      headingAccuracyDeg: accuracy,
      screenAngleDeg,
    };
    return sample;
  }

  function push(reading: OrientationReading): OrientationSample | null {
    const { alpha, beta, gamma } = reading;
    if (alpha === null || beta === null || gamma === null) return null;
    if (!Number.isFinite(alpha) || !Number.isFinite(beta) || !Number.isFinite(gamma)) return null;

    const t = reading.t ?? clock();
    const screenAngleDeg = reading.screenAngleDeg ?? screenAngle();

    // YAW COMES FROM ALPHA. Always. See the magnetometer note in the file header: the capture
    // is defined relative to `setOrigin()`, so an earth-referenced heading buys nothing and
    // imports every indoor magnetic anomaly into the one number that decides where coverage
    // gets painted. `webkitCompassHeading` (clockwise, where alpha runs counter-clockwise) is
    // read here only so `reference` and `headingAccuracyDeg` can describe the stream honestly.
    const heading = reading.compassHeadingDeg;
    const compassPresent = typeof heading === "number" && Number.isFinite(heading);

    if (compassPresent) {
      // A compass exists, and we are deliberately not steering by it, so the track is
      // relative whatever the magnetometer thinks of itself. The accuracy still travels with
      // the sample: -1 is what drives the figure-8 prompt, and the server reads it too.
      reference = "relative";
      accuracy = reading.compassAccuracyDeg ?? -1;
    } else if (reading.absolute === true) {
      // The absolute fallback: no relative stream on this device, so alpha itself is
      // earth-referenced and the label says so.
      reference = "absolute";
      accuracy = null;
    } else {
      reference = "relative";
      accuracy = null;
    }

    const raw = quatFromDeviceOrientation(alpha, beta, gamma, screenAngleDeg);

    if (smoothed === null || lastRaw === null || lastT === null) {
      smoothed = raw;
    } else {
      // dt is clamped: a backgrounded tab delivers a multi-second gap, and an over-eager
      // sensor can deliver two events in the same millisecond.
      const dt = Math.min(0.25, Math.max(1 / 240, (t - lastT) / 1000));
      const speedDegPerS = quatAngleDeg(lastRaw, raw) / dt;
      const cutoff = minCutoffHz + speedCutoffGain * speedDegPerS;
      const tau = 1 / (2 * Math.PI * cutoff);
      const alphaLp = Math.min(1, Math.max(0, 1 - Math.exp(-dt / tau)));
      smoothed = quatSlerp(smoothed, raw, alphaLp);
    }
    lastRaw = raw;
    lastT = t;

    // Trailing window of the PRE-ORIGIN smoothed heading, for `setOrigin` to average over.
    const preOriginYaw = viewAnglesFromQuat(smoothed).yawDeg;
    originWindow.push({ t, yawDeg: preOriginYaw });
    while (originWindow.length > 1 && t - originWindow[0].t > ORIGIN_WINDOW_MS) originWindow.shift();

    return emit(quatMultiply(originQuat, smoothed), t, reference, screenAngleDeg);
  }

  function pushFlow(flow: FlowSample, t?: Millis): OrientationSample | null {
    if (!flow.valid || !Number.isFinite(flow.yawDeltaDeg)) return null;
    const at = t ?? clock();
    const screenAngleDeg = screenAngle();

    // Yaw only. Rebuilt from scratch each tick about the world up axis, so integration error
    // lives in one scalar instead of drifting the whole orientation.
    const nextYaw = (lastYawDeg ?? 0) + flow.yawDeltaDeg;
    const pitch = sample?.pitchDeg ?? 0;
    const q = quatMultiply(
      quatFromAxisAngle({ x: 0, y: 1, z: 0 }, -nextYaw * DEG),
      quatFromAxisAngle({ x: 1, y: 0, z: 0 }, pitch * DEG),
    );
    reference = "flow";
    accuracy = null;
    return emit(q, at, "flow", screenAngleDeg);
  }

  return {
    push,
    pushFlow,
    read: () => sample,
    readReference: () => reference,
    unwrappedYawDeg: () => unwrappedYaw,
    headingAccuracyDeg: () => accuracy,
    calibrationNeeded: () => accuracy !== null && (accuracy < 0 || accuracy > 25),
    staleMs: (at?: Millis) => (sample === null ? Number.POSITIVE_INFINITY : (at ?? clock()) - sample.t),
    setOrigin: () => {
      // Rotate the whole frame so the current heading becomes 0. Applied on the world side,
      // so `q` and `yawDeg` stay consistent and the lattice lines up with station yaw 0.
      //
      // AVERAGED over the last `ORIGIN_WINDOW_MS`, not sampled. This is called on entry to
      // `row_level`, and the level row is the phone held upright in portrait — beta = 90,
      // where the Z-X'-Y'' chart is exactly singular and the platform's own alpha/gamma
      // separation is at its noisiest (rule 1 in the file header). The instantaneous heading
      // there carries the most noise of any pose in the capture, and it is the one number the
      // entire loop closure and every station yaw is defined against. The framing gates
      // already required 1.2 s of stillness before this runs, so half a second of averaging
      // costs no lag at all and divides the origin's variance by the sample count.
      const current = averageYawDeg(originWindow) ?? (smoothed ? viewAnglesFromQuat(smoothed).yawDeg : 0);
      originQuat = quatFromAxisAngle({ x: 0, y: 1, z: 0 }, current * DEG);
      unwrappedYaw = 0;
      lastYawDeg = null;
      // Re-emit NOW, so `read()` already answers in the new frame. The recorder starts on
      // the same tick this is called, and the first track sample used to be the previous
      // emission -- the pre-origin quaternion, 104 degrees of yaw away from the second.
      if (smoothed !== null && lastT !== null) {
        emit(quatMultiply(originQuat, smoothed), lastT, reference, screenAngle());
      }
    },
    reset: () => {
      smoothed = null;
      lastRaw = null;
      lastT = null;
      originQuat = IDENTITY;
      unwrappedYaw = 0;
      lastYawDeg = null;
      lastRollDeg = 0;
      reference = "relative";
      accuracy = null;
      sample = null;
      originWindow = [];
    },
  };
}

// -------------------------------------------------------------------------------------
// The hook
// -------------------------------------------------------------------------------------

/**
 * How long the magnetometer-fused `deviceorientationabsolute` stream stays suppressed after
 * a plain `deviceorientation` event. One second is ~60 missed events at either platform's
 * rate, so on a device that has a relative stream the absolute one never gets a turn, and on
 * a device that does not it takes over within a second of listening.
 */
const RELATIVE_PREFERENCE_MS = 1000;

type OrientationEventCtor = typeof DeviceOrientationEvent & { requestPermission?: () => Promise<string> };
type CompassEvent = DeviceOrientationEvent & { webkitCompassHeading?: number; webkitCompassAccuracy?: number };

export type PermissionOutcome = "granted" | "denied" | "not-required";

export interface UseOrientationOptions extends OrientationTrackerOptions {
  /**
   * Injected sample source. Receives a push function and returns an unsubscribe. When
   * present the hook never touches `window`, which is how a test drives it with a synthetic
   * stream and how a replay harness re-runs a recorded capture.
   */
  source?: (push: (reading: OrientationReading) => void) => () => void;
  /**
   * RAIL MODE input. When no orientation event arrives within `railTimeoutMs` of listening,
   * and this is supplied, yaw is integrated from camera optical flow at `railHz`.
   */
  flowSource?: () => FlowSample | null;
  /** Attach as soon as permission allows. Default true. */
  autoStart?: boolean;
  /**
   * How often the React `sample`, `reference`, `headingAccuracyDeg` and `calibrationNeeded`
   * fields refresh. `read()` is always current and never involves React.
   *
   * ZERO MEANS NEVER, and that is the setting a consumer driving its own loop wants: every
   * refresh re-renders the component that called the hook, and `push` allocates a fresh
   * sample object each time, so the mirror is a guaranteed state change rather than a
   * bail-out. A caller that only ever reads through `read()` was paying a re-render of its
   * whole tree several times a second for values nothing on screen was showing. Default 4.
   */
  stateHz?: number;
  railTimeoutMs?: number;
  railHz?: number;
}

/** `OrientationApi` plus the things the UI needs in order to be honest about confidence. */
export interface OrientationApiExt extends OrientationApi {
  /** Latest `webkitCompassAccuracy`; null on Android and before the first event. */
  headingAccuracyDeg: number | null;
  /** Accuracy is -1 (uncalibrated) or worse than 25 deg: show the figure-8 prompt. */
  calibrationNeeded: boolean;
  permission: "unknown" | PermissionOutcome;
  /** Continuous yaw, never wrapped. For telemetry and for span arithmetic. */
  unwrappedYawDeg: () => Degrees;
  /** Milliseconds since the last sample; Infinity before the first. Drives `track_lost`. */
  staleMs: () => number;
  /** Attach listeners without asking for permission (Android, or after a grant). */
  start: () => void;
}

export function useOrientation(options: UseOrientationOptions = {}): OrientationApiExt {
  const {
    source,
    flowSource,
    autoStart = true,
    stateHz = 4,
    railTimeoutMs = 1500,
    railHz = 20,
    minCutoffHz,
    speedCutoffGain,
    now,
    screenAngleDeg,
  } = options;

  // Late-bound values live in refs that are SYNCED IN AN EFFECT, never written during render.
  // The listener effect must not re-run just because a caller passed a fresh closure: tearing
  // the sensor stream down mid-capture would restart the smoothing filter and, in rail mode,
  // lose the integrated yaw.
  const sourceRef = useRef(source);
  const flowSourceRef = useRef(flowSource);
  const railRef = useRef({ timeoutMs: railTimeoutMs, hz: railHz });
  const stateHzRef = useRef(stateHz);
  const clockRef = useRef(now ?? nowMs);
  const lastStateAtRef = useRef(0);
  useEffect(() => {
    sourceRef.current = source;
    flowSourceRef.current = flowSource;
    railRef.current = { timeoutMs: railTimeoutMs, hz: railHz };
    stateHzRef.current = stateHz;
    clockRef.current = now ?? nowMs;
  });

  // The tracker is held in state rather than a ref: a ref cannot be lazily initialised during
  // render, and this value is created once and never replaced.
  const [tracker] = useState(() => createOrientationTracker({ minCutoffHz, speedCutoffGain, now, screenAngleDeg }));

  const [sample, setSample] = useState<OrientationSample | null>(null);
  const [reference, setReference] = useState<OrientationReference>("relative");
  const [live, setLive] = useState(false);
  const [permission, setPermission] = useState<"unknown" | PermissionOutcome>("unknown");
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [calibrationNeeded, setCalibrationNeeded] = useState(false);
  const [attached, setAttached] = useState(autoStart);
  /** Mirrors `live` so the latch can be read without making `publish` depend on it. */
  const liveRef = useRef(false);

  /** One place where a fresh sample decides whether React needs to hear about it. */
  const publish = useCallback(
    (next: OrientationSample | null) => {
      if (!next) return;
      const hz = stateHzRef.current;
      const interval = hz > 0 ? 1000 / hz : Number.POSITIVE_INFINITY;
      if (next.t - lastStateAtRef.current >= interval) {
        lastStateAtRef.current = next.t;
        setSample(next);
        setReference(tracker.readReference());
        setAccuracy(tracker.headingAccuracyDeg());
        setCalibrationNeeded(tracker.calibrationNeeded());
      }
      // Latched, not re-set. This one sits outside the throttle because "the sensor has
      // spoken at least once" is not a value that should wait on a publish interval — but it
      // only ever transitions false->true, so calling the setter on every one of ~60 events a
      // second is 59 wasted trips into the scheduler per second for a value that changed once.
      if (!liveRef.current) {
        liveRef.current = true;
        setLive(true);
      }
    },
    [tracker],
  );

  useEffect(() => {
    if (!attached || typeof window === "undefined") return;

    const inject = sourceRef.current;
    const flow = flowSourceRef.current;
    const { timeoutMs: railAfter, hz: rail } = railRef.current;
    const clock = clockRef.current;
    let disposed = false;
    let lastRelativeAt = -Infinity;
    let railTimer: ReturnType<typeof setInterval> | null = null;

    const feed = (reading: OrientationReading) => {
      if (disposed) return;
      publish(tracker.push(reading));
    };

    let detach: () => void;

    if (inject) {
      detach = inject(feed);
    } else if (!("DeviceOrientationEvent" in window)) {
      // No sensor at all. Rail mode is the only route left, and the timer below starts it.
      detach = () => undefined;
    } else {
      const fromEvent = (event: DeviceOrientationEvent, absolute: boolean): OrientationReading => {
        const compass = event as CompassEvent;
        return {
          alpha: event.alpha,
          beta: event.beta,
          gamma: event.gamma,
          absolute: absolute || event.absolute === true,
          compassHeadingDeg: typeof compass.webkitCompassHeading === "number" ? compass.webkitCompassHeading : null,
          compassAccuracyDeg: typeof compass.webkitCompassAccuracy === "number" ? compass.webkitCompassAccuracy : null,
          t: clock(),
        };
      };

      // Android fires BOTH streams. `deviceorientationabsolute` is the magnetometer-fused
      // one, so it is the FALLBACK, not the preference: it is admitted only when the
      // relative stream has gone quiet for a second, which on a device that has one it never
      // does. This is the reverse of the original preference — see the magnetometer note in
      // the file header for why an earth-referenced heading is a liability here and not an
      // asset. iOS never fires this listener at all.
      const onAbsolute = (event: Event) => {
        if (!(event instanceof DeviceOrientationEvent)) return;
        if (clock() - lastRelativeAt < RELATIVE_PREFERENCE_MS) return;
        feed(fromEvent(event, true));
      };
      const onRelative = (event: DeviceOrientationEvent) => {
        lastRelativeAt = clock();
        feed(fromEvent(event, false));
      };

      window.addEventListener("deviceorientationabsolute", onAbsolute);
      window.addEventListener("deviceorientation", onRelative);
      detach = () => {
        window.removeEventListener("deviceorientationabsolute", onAbsolute);
        window.removeEventListener("deviceorientation", onRelative);
      };
    }

    // RAIL MODE. Only ever starts if nothing real arrived, and stops the moment it does.
    if (flow) {
      railTimer = setInterval(() => {
        if (disposed) return;
        if (tracker.staleMs(clock()) < railAfter) return;
        const f = flow();
        if (f) publish(tracker.pushFlow(f, clock()));
      }, Math.max(20, 1000 / Math.max(1, rail)));
    }

    return () => {
      disposed = true;
      detach();
      if (railTimer !== null) clearInterval(railTimer);
    };
  }, [attached, publish, tracker]);

  const start = useCallback(() => setAttached(true), []);

  const request = useCallback(async (): Promise<PermissionOutcome> => {
    if (typeof window === "undefined" || !("DeviceOrientationEvent" in window)) {
      setPermission("not-required");
      setAttached(true);
      return "not-required";
    }
    const ctor = window.DeviceOrientationEvent as OrientationEventCtor;
    if (typeof ctor.requestPermission !== "function") {
      // Android and desktop: no gate, just listen.
      setPermission("not-required");
      setAttached(true);
      return "not-required";
    }
    try {
      // MUST be the first thing that happens: iOS requires a transient user activation, and
      // any await before this line spends it. The caller is contractually inside a click.
      const pending = ctor.requestPermission();
      const result = await pending;
      const outcome: PermissionOutcome = result === "granted" ? "granted" : "denied";
      setPermission(outcome);
      if (outcome === "granted") setAttached(true);
      return outcome;
    } catch {
      // A throw here means no transient activation, or a non-secure context. Both are
      // recoverable by the user, so this is reported as a denial rather than swallowed.
      setPermission("denied");
      return "denied";
    }
  }, []);

  const stop = useCallback(() => {
    setAttached(false);
    setLive(false);
  }, []);

  const setOrigin = useCallback(() => tracker.setOrigin(), [tracker]);
  const read = useCallback(() => tracker.read(), [tracker]);
  const unwrappedYawDeg = useCallback(() => tracker.unwrappedYawDeg(), [tracker]);
  const staleMs = useCallback(() => tracker.staleMs(clockRef.current()), [tracker]);

  return {
    sample,
    read,
    reference,
    live,
    setOrigin,
    request,
    stop,
    headingAccuracyDeg: accuracy,
    calibrationNeeded,
    permission,
    unwrappedYawDeg,
    staleMs,
    start,
  };
}
