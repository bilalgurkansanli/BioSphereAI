/**
 * Builder S self-check: the sensor layer's pure kernels, driven by synthetic streams.
 *
 * Every hook in `lib/capture/{useOrientation,useMotion,useCamera,recorder,telemetry}.ts` is a
 * thin React wrapper over a framework-free core that accepts an injected source, which is
 * what makes this file possible with no DOM, no camera and no phone. These tests are the
 * closest thing to hardware we have.
 */

import { describe, expect, it } from "vitest";
import {
  createOrientationTracker,
  deviceFrameQuat,
  quatFromDeviceOrientation,
  rotateVec,
  unrotateVec,
  viewAnglesFromQuat,
  wrap180,
} from "../useOrientation";
import { createMotionTracker } from "../useMotion";
import { createTranslationEstimator, deviceToWorld } from "../detectors";
import { analyseFrameLuma, pickRearCamera, pxPerDegAt640, quadrantFlow, type FlowFrame } from "../useCamera";
import { createRecorder, type MediaRecorderLike } from "../recorder";
import { createTelemetry } from "../telemetry";
import { THRESHOLDS, type CoverageReport, type OrientationSample, type Quat, type RecordedSegment } from "../types";
import { makeOrientation, quatFromYawPitchRoll } from "./synthetic";

const reading = (alpha: number, beta: number, gamma: number, t: number) => ({ alpha, beta, gamma, screenAngleDeg: 0, t });

describe("orientation geometry", () => {
  it("upright portrait is yaw 0 / pitch 0 / roll 0", () => {
    const a = viewAnglesFromQuat(quatFromDeviceOrientation(0, 90, 0, 0));
    expect(a.yawDeg).toBeCloseTo(0, 4);
    expect(a.pitchDeg).toBeCloseTo(0, 4);
    expect(a.rollDeg).toBeCloseTo(0, 4);
  });

  it("flat on a table points the rear camera straight down", () => {
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(0, 0, 0, 0)).pitchDeg).toBeCloseTo(-90, 4);
  });

  it("alpha runs counter-clockwise, yaw runs clockwise", () => {
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(90, 90, 0, 0)).yawDeg).toBeCloseTo(270, 3);
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(270, 90, 0, 0)).yawDeg).toBeCloseTo(90, 3);
  });

  it("tilting up raises pitch", () => {
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(0, 119, 0, 0)).pitchDeg).toBeCloseTo(29, 3);
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(0, 61, 0, 0)).pitchDeg).toBeCloseTo(-29, 3);
  });

  it("round-trips the test fixtures' own quaternion, so a fixture cannot lie about its angles", () => {
    // `makeOrientation` used to build the MIRROR of this basis (optical axis along +Z, roll
    // of the opposite sign) on the argument that nothing read `q`. The walking gate reads it
    // now — it rotates linear acceleration into the world frame with it — so a fixture whose
    // quaternion disagrees with the angles printed beside it would silently test the wrong
    // rotation.
    for (const [yaw, pitch, roll] of [
      [0, 0, 0],
      [137, 29, 4],
      [312, -29, -6],
      [45, 84, 0],
      [200, -75, 30],
    ]) {
      const a = viewAnglesFromQuat(quatFromYawPitchRoll(yaw, pitch, roll));
      expect(a.yawDeg).toBeCloseTo(yaw, 6);
      expect(a.pitchDeg).toBeCloseTo(pitch, 6);
      expect(a.rollDeg).toBeCloseTo(roll, 6);
    }
  });

  it("reports roll about the optical axis, which is NOT gamma", () => {
    // At beta = 90 the Z-X'-Y'' decomposition is singular: the alpha and gamma axes coincide,
    // so a gamma tilt on an upright phone is a YAW, not a roll. This is exactly why the
    // contract mandates quaternions — the triple is ambiguous here, the rotation is not.
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(0, 90, 8, 0)).yawDeg).toBeCloseTo(352, 3);

    // A phone upright and rolled 10 deg in its own plane reports (90, 80, -90).
    const rolled = viewAnglesFromQuat(quatFromDeviceOrientation(90, 80, -90, 0));
    expect(rolled.rollDeg).toBeCloseTo(10, 6);
    expect(rolled.pitchDeg).toBeCloseTo(0, 6);
    expect(rolled.yawDeg).toBeCloseTo(0, 6);
    expect(viewAnglesFromQuat(quatFromDeviceOrientation(-90, 80, 90, 0)).rollDeg).toBeCloseTo(-10, 6);
  });
});

describe("orientation tracker", () => {
  it("yaw never jumps at the 0/360 seam and unwraps monotonically", () => {
    const tracker = createOrientationTracker({ minCutoffHz: 40, speedCutoffGain: 0 });
    let previous: number | null = null;
    // Sweep a full turn to the right in 1.5 deg steps, crossing the seam twice.
    for (let step = 0; step <= 300; step += 1) {
      const alpha = ((-step * 1.5) % 360 + 360) % 360;
      const sample = tracker.push(reading(alpha, 90, 0, step * 20));
      expect(sample).not.toBeNull();
      if (sample && previous !== null) expect(Math.abs(wrap180(sample.yawDeg - previous))).toBeLessThan(4);
      if (sample) previous = sample.yawDeg;
    }
    // 300 steps x 1.5 deg = 450 deg of continuous yaw, filter lag aside.
    expect(tracker.unwrappedYawDeg()).toBeGreaterThan(440);
    expect(tracker.unwrappedYawDeg()).toBeLessThan(455);
  });

  it("setOrigin makes the current heading yaw 0 and keeps q consistent with yawDeg", () => {
    const tracker = createOrientationTracker({ minCutoffHz: 60, speedCutoffGain: 0 });
    for (let i = 0; i < 40; i += 1) tracker.push(reading(200, 90, 0, i * 16));
    tracker.setOrigin();
    const sample = tracker.push(reading(200, 90, 0, 700));
    expect(sample!.yawDeg).toBeLessThan(360);
    expect(sample?.yawDeg).toBeCloseTo(0, 1);
    expect(viewAnglesFromQuat(sample!.q).yawDeg).toBeCloseTo(sample!.yawDeg, 6);

    const turned = tracker.push(reading(170, 90, 0, 1400));
    expect(turned?.yawDeg).toBeCloseTo(30, 0);
  });

  it("never steers yaw by the magnetometer, however confident the magnetometer sounds", () => {
    // The capture is defined relative to setOrigin(), so true north buys nothing and imports
    // every indoor magnetic anomaly into the number that decides where coverage is painted.
    const tracker = createOrientationTracker({ minCutoffHz: 60, speedCutoffGain: 0 });
    for (let i = 0; i < 30; i += 1) {
      tracker.push({ alpha: 0, beta: 90, gamma: 0, compassHeadingDeg: 90, compassAccuracyDeg: 10, screenAngleDeg: 0, t: i * 16 });
    }
    // Yaw follows ALPHA (0), not the heading (which would have put it at 90).
    expect(tracker.read()?.yawDeg).toBeCloseTo(0, 0);
    // ...and the stream says so rather than claiming to be earth-referenced.
    expect(tracker.readReference()).toBe("relative");
    expect(tracker.headingAccuracyDeg()).toBe(10);
    expect(tracker.calibrationNeeded()).toBe(false);

    tracker.push({ alpha: 0, beta: 90, gamma: 0, compassHeadingDeg: 90, compassAccuracyDeg: -1, screenAngleDeg: 0, t: 600 });
    expect(tracker.calibrationNeeded()).toBe(true);
  });

  it("survives the fridge: a 35-deg heading snap moves yaw not at all", () => {
    // Measured behaviour of `webkitCompassAccuracy` indoors: it keeps reporting its nominal
    // ~12 deg while the heading itself jumps tens of degrees walking past a steel stud wall.
    // Under the old `accuracy >= 0` rule that snap went straight into `stampCone`.
    const tracker = createOrientationTracker({ minCutoffHz: 60, speedCutoffGain: 0 });
    const push = (t: number, headingDeg: number) =>
      tracker.push({ alpha: 30, beta: 90, gamma: 0, compassHeadingDeg: headingDeg, compassAccuracyDeg: 12, screenAngleDeg: 0, t });

    for (let i = 0; i < 40; i += 1) push(i * 16, 100);
    const before = tracker.read()!.yawDeg;
    for (let i = 40; i < 80; i += 1) push(i * 16, 135);
    const after = tracker.read()!.yawDeg;
    expect(Math.abs(wrap180(after - before))).toBeLessThan(0.5);
  });

  it("labels the absolute fallback honestly, for a device with no relative stream", () => {
    const tracker = createOrientationTracker({ minCutoffHz: 60, speedCutoffGain: 0 });
    for (let i = 0; i < 30; i += 1) tracker.push({ alpha: 40, beta: 90, gamma: 0, absolute: true, screenAngleDeg: 0, t: i * 16 });
    expect(tracker.readReference()).toBe("absolute");
    expect(tracker.headingAccuracyDeg()).toBeNull();
  });

  it("averages the loop-closure origin instead of sampling it at the one singular pose", () => {
    // setOrigin runs on entry to `row_level`, which is beta = 90 — where the Z-X'-Y'' chart
    // is exactly singular and the platform's alpha/gamma separation is at its noisiest. A
    // single sample there inherits all of that noise, and every station yaw and the loop
    // closure itself are defined against it.
    const jitter = [+4, -3, +3.5, -4, +2, -2.5, +3, -3.5, +2.5, -2];
    const noisy = createOrientationTracker({ minCutoffHz: 200, speedCutoffGain: 0 });
    for (let i = 0; i < 40; i += 1) noisy.push(reading(100 + jitter[i % jitter.length], 90, 0, i * 16));
    noisy.setOrigin();
    const averaged = noisy.push(reading(100, 90, 0, 700))!.yawDeg;

    // The last sample before setOrigin was 4 deg off; an origin taken from it would carry
    // that whole error into yaw 0. The circular mean of a zero-mean jitter does not.
    expect(Math.abs(wrap180(averaged))).toBeLessThan(1.5);
  });

  it("takes the circular mean, so an origin at the 0/360 seam does not point backwards", () => {
    const tracker = createOrientationTracker({ minCutoffHz: 200, speedCutoffGain: 0 });
    // Alpha either side of the seam: naive arithmetic averaging would land the origin at 180.
    for (let i = 0; i < 40; i += 1) tracker.push(reading(i % 2 === 0 ? 2 : 358, 90, 0, i * 16));
    tracker.setOrigin();
    const zeroed = tracker.push(reading(0, 90, 0, 700))!;
    expect(Math.abs(wrap180(zeroed.yawDeg))).toBeLessThan(4);
  });

  it("rail mode integrates optical flow into yaw and says so", () => {
    const tracker = createOrientationTracker();
    for (let i = 1; i <= 10; i += 1) {
      tracker.pushFlow({ t: i * 200, divergencePx: 1, meanShiftPx: -20, yawDeltaDeg: 2.4, valid: true }, i * 200);
    }
    expect(tracker.readReference()).toBe("flow");
    expect(tracker.read()?.yawDeg).toBeCloseTo(24, 3);
  });

  it("holds roll through the pole instead of snapping it to zero", () => {
    const tracker = createOrientationTracker({ minCutoffHz: 60, speedCutoffGain: 0 });
    for (let i = 0; i < 40; i += 1) tracker.push(reading(90, 80, -90, i * 16));
    expect(Math.abs(tracker.read()!.rollDeg)).toBeGreaterThan(5);
    // Straight up: the roll reference vanishes and a naive atan2 would snap the horizon to 0.
    for (let i = 0; i < 60; i += 1) tracker.push(reading(0, 180, 0, 700 + i * 16));
    expect(Math.abs(tracker.read()!.pitchDeg)).toBeGreaterThan(85);
    expect(Number.isFinite(tracker.read()!.rollDeg)).toBe(true);
  });

  it("never emits a yaw of exactly 360, which would index off the end of a 72-cell grid", () => {
    const tracker = createOrientationTracker({ minCutoffHz: 60, speedCutoffGain: 0 });
    for (let i = 0; i < 40; i += 1) {
      const sample = tracker.push(reading(0, 90, 0, i * 16));
      expect(sample!.yawDeg).toBeGreaterThanOrEqual(0);
      expect(sample!.yawDeg).toBeLessThan(360);
      expect(Math.floor(sample!.yawDeg / 5)).toBeLessThan(72);
    }
    tracker.setOrigin();
    const zeroed = tracker.push(reading(0, 90, 0, 700))!;
    expect(zeroed.yawDeg).toBeLessThan(360);
    expect(zeroed.yawDeg).toBeCloseTo(0, 6);
  });
});

describe("motion translation statistics", () => {
  const run = (
    accel: (t: number) => { x: number; y: number; z: number },
    seconds = 4,
    orientation?: () => OrientationSample | null,
  ) => {
    const tracker = createMotionTracker({ orientation });
    const dt = 1 / 60;
    for (let i = 0; i * dt < seconds; i += 1) {
      const t = i * dt;
      tracker.push({
        t: t * 1000,
        rotationRate: { alpha: 24, beta: 0, gamma: 0 },
        acceleration: accel(t),
        accelerationIncludingGravity: { x: accel(t).x, y: accel(t).y, z: accel(t).z + 9.81 },
        interval: dt * 1000,
      });
    }
    return tracker;
  };

  it("a phone turning on the spot reads as neither walking nor drifting", () => {
    const tracker = run((t) => ({ x: 0.02 * Math.sin(t * 37), y: 0.02 * Math.cos(t * 23), z: 0.02 * Math.sin(t * 11) }));
    const e = tracker.readTranslation();
    expect(e.gateRan).toBe(true);
    expect(e.stepBandRmsMs2).toBeLessThan(THRESHOLDS.stepCadence.warnRmsMs2);
    expect(e.driftMPerS).toBeLessThan(THRESHOLDS.translationDrift.warnMPerS);
    expect(tracker.read()?.omegaDegPerS).toBeCloseTo(24, 6);
  });

  it("footfall impulses at 2 Hz land in the walking band", () => {
    const tracker = run((t) => {
      const phase = (t * 2) % 1;
      const impulse = 3.2 * Math.exp(-((phase - 0.5) ** 2) / (2 * 0.06 ** 2));
      return { x: 0, y: 0, z: impulse };
    });
    expect(tracker.readTranslation().stepBandRmsMs2).toBeGreaterThan(THRESHOLDS.stepCadence.warnRmsMs2);
  });

  it("band-passes each AXIS, not the magnitude — rectification would move a 2 Hz gait to 4 Hz", () => {
    // This module used to filter `|linearAccel|` through one biquad. |sin(2 pi f t)| has its
    // fundamental at 2f, so the dead centre of human walking cadence landed outside the band
    // it was being measured in. A three-axis gait must read its true RMS, not a third of it.
    const amplitude = 1.2;
    const tracker = run(
      (t) => ({
        x: amplitude * 0.42 * Math.sin(2 * Math.PI * 2 * t),
        y: amplitude * Math.sin(2 * Math.PI * 2 * t + 0.7),
        z: amplitude * 0.75 * Math.sin(2 * Math.PI * 2 * t + 1.4),
      }),
      6,
    );
    // The band-passed VECTOR has magnitude ~= the vector amplitude, RMS'd over the window.
    const vectorAmplitude = amplitude * Math.hypot(0.42, 1, 0.75);
    expect(tracker.readTranslation().stepBandRmsMs2).toBeGreaterThan(vectorAmplitude * 0.5);
    expect(tracker.readTranslation().stepBandRmsMs2).toBeGreaterThan(THRESHOLDS.stepCadence.stopRmsMs2);
  });

  it("a body actually translating crosses the drift stop line", () => {
    // 2.5 m/s^2 at 0.5 Hz integrates to a 0.8 m/s peak velocity — a walk, not a sway.
    const tracker = run((t) => ({ x: 2.5 * Math.sin(2 * Math.PI * 0.5 * t), y: 0, z: 0 }));
    const drift = tracker.readTranslation().driftMPerS;
    expect(drift).toBeGreaterThan(THRESHOLDS.translationDrift.stopMPerS);
    // And it must not be absurd: the true peak speed is ~0.8 m/s.
    expect(drift).toBeLessThan(1.5);
  });

  /** Drive one tracker through a turn, with the acceleration specified in the WORLD frame. */
  const turnWith = (worldAccel: (t: number) => { x: number; y: number; z: number }, useOrientation: boolean, omegaDegPerS = 24) => {
    let current: OrientationSample | null = null;
    const tracker = createMotionTracker({ orientation: useOrientation ? () => current : undefined });
    const dt = 1 / 60;
    let peak = 0;
    for (let i = 0; i * dt < 8; i += 1) {
      const t = i * dt;
      current = makeOrientation(t * 1000, (omegaDegPerS * t) % 360, 0);
      // Express the world-frame vector back in accelerometer axes, exactly.
      const accel = unrotateVec(deviceFrameQuat(current.q, current.screenAngleDeg), worldAccel(t));
      tracker.push({
        t: t * 1000,
        rotationRate: { alpha: omegaDegPerS, beta: 0, gamma: 0 },
        acceleration: accel,
        interval: dt * 1000,
      });
      peak = Math.max(peak, tracker.readTranslation().driftMPerS);
    }
    return { peak, estimate: tracker.readTranslation() };
  };

  it("integrates in a fixed frame when an orientation is supplied, and says which it used", () => {
    const push = (t: number) => ({ x: t > 1 && t < 2.2 ? 0.8 : 0, y: 0, z: 0 });
    expect(turnWith(push, false).estimate.worldFrame).toBe(false);
    expect(turnWith(push, true).estimate.worldFrame).toBe(true);
    // A real translation must not read LOWER for having been measured properly.
    expect(turnWith(push, true).peak).toBeGreaterThanOrEqual(turnWith(push, false).peak);
  });

  it("rejects a gravity-removal residual, which the device frame turns into phantom walking", () => {
    // The sharp reason the frame matters. A few percent of gravity left in the signal — the
    // ordinary state of the `derived` accelSource, where gravity is a low-pass estimate — is
    // a CONSTANT in the world frame, which the 0.3 Hz high-pass removes exactly. In the
    // tumbling device frame the same residual becomes an oscillation at the turn rate,
    // 0.067 Hz at the coached 24 deg/s, right where the high-pass cannot help: a phone
    // standing perfectly still manufactures drift in proportion to how fast it is turning.
    const residual = () => ({ x: 0.3, y: 0, z: 0 });
    const deviceFrame = turnWith(residual, false).peak;
    const worldFrame = turnWith(residual, true).peak;
    expect(worldFrame).toBeLessThan(0.02);
    expect(deviceFrame).toBeGreaterThan(worldFrame * 5);
  });

  it("reconstructs the angular rate from orientation when there is no gyroscope", () => {
    // Finding 6: `rotationRate: null` used to give omega 0, speedGrade green and smearPx 0 at
    // any speed, so `too_fast` and `smear` could never fire and the framing gate passed while
    // the phone was being waved.
    let sample: OrientationSample | null = null;
    const tracker = createMotionTracker({ orientation: () => sample });
    const dt = 1 / 60;
    for (let i = 0; i * dt < 2; i += 1) {
      const t = i * dt;
      sample = makeOrientation(t * 1000, 40 * t, 0);
      tracker.push({ t: t * 1000, rotationRate: null, acceleration: { x: 0, y: 0, z: 0 }, interval: dt * 1000 });
    }
    expect(tracker.read()?.rotationRate).toBeNull();
    expect(tracker.readOmegaSource()).toBe("orientation");
    expect(tracker.read()?.omegaDegPerS).toBeCloseTo(40, 0);
  });

  it("holds the derived rate across a repeated orientation sample instead of reporting zero", () => {
    // `devicemotion` and `deviceorientation` are independent streams, so about half the
    // motion events see no new quaternion. A rate that alternated between the truth and 0
    // would halve the speed meter and flicker every grade attached to it.
    let sample: OrientationSample | null = null;
    const tracker = createMotionTracker({ orientation: () => sample });
    for (let i = 0; i < 60; i += 1) {
      const t = i * 8;
      // Orientation updates at half the motion rate.
      if (i % 2 === 0) sample = makeOrientation(t, 40 * (t / 1000), 0);
      tracker.push({ t, rotationRate: null, acceleration: { x: 0, y: 0, z: 0 }, interval: 8 });
    }
    expect(tracker.read()?.omegaDegPerS).toBeCloseTo(40, 0);
  });

  it("reports omegaSource `none` when nothing measured the rate, rather than a confident 0", () => {
    const tracker = createMotionTracker();
    for (let i = 0; i < 30; i += 1) tracker.push({ t: i * 16, rotationRate: null, acceleration: { x: 0, y: 0, z: 0 } });
    expect(tracker.readOmegaSource()).toBe("none");
    expect(tracker.read()?.omegaDegPerS).toBe(0);
  });

  it("late-binds the orientation accessor, because the tracker outlives every render", () => {
    // The hook builds the tracker once, in a useState initialiser. Anything captured there is
    // captured for the whole session, so `useMotion`'s `orientation` option is re-bound from
    // an effect on every render — this is the seam that makes that work.
    let current: OrientationSample | null = null;
    const tracker = createMotionTracker();
    tracker.push({ t: 0, rotationRate: null, acceleration: { x: 0, y: 0, z: 0 } });
    expect(tracker.readTranslation().worldFrame).toBe(false);

    tracker.setOrientation(() => current);
    for (let i = 1; i < 20; i += 1) {
      current = makeOrientation(i * 16, 24 * (i * 16) * 0.001, 0);
      tracker.push({ t: i * 16, rotationRate: null, acceleration: { x: 0.1, y: 0, z: 0 } });
    }
    expect(tracker.readTranslation().worldFrame).toBe(true);
    expect(tracker.readOmegaSource()).toBe("orientation");

    // ...and unbinding it goes back to the device frame rather than a stale attitude.
    tracker.setOrientation(null);
    tracker.push({ t: 400, rotationRate: null, acceleration: { x: 0.1, y: 0, z: 0 } });
    expect(tracker.readTranslation().worldFrame).toBe(false);
  });

  it("no usable acceleration means the gate did not run, not that the user stood still", () => {
    const tracker = createMotionTracker();
    for (let i = 0; i < 60; i += 1) {
      tracker.push({ t: i * 16, rotationRate: { alpha: 10, beta: 0, gamma: 0 }, acceleration: null, accelerationIncludingGravity: null });
    }
    expect(tracker.readAccelSource()).toBe("none");
    expect(tracker.readTranslation().gateRan).toBe(false);
    expect(tracker.read()?.linearAccel).toBeNull();
  });

  it("staleMs is the only thing that can tell a dead stream from a still user", () => {
    const tracker = createMotionTracker({ now: () => 10_000 });
    for (let i = 0; i < 30; i += 1) tracker.push({ t: i * 16, rotationRate: { alpha: 0, beta: 0, gamma: 0 }, acceleration: { x: 0, y: 0, z: 0 } });
    // `read()` keeps handing back the last sample forever; only its age gives the game away.
    expect(tracker.read()).not.toBeNull();
    expect(tracker.staleMs()).toBeGreaterThan(9000);
    expect(tracker.staleMs(500)).toBeCloseTo(500 - 29 * 16, 6);
  });
});

describe("the world-frame rotation the detector duplicates", () => {
  // `detectors.deviceToWorld` deliberately re-implements this rather than importing
  // useOrientation, so that vitest.config.mts's claim — Builder C's modules run under plain
  // Node with no React in the graph — stays true. This is the pin that stops the two copies
  // drifting apart, and it is exact rather than approximate on purpose.
  it("agrees with useOrientation's rotateVec(deviceFrameQuat(...)) at every screen angle", () => {
    const accel = { x: 0.3, y: -0.7, z: 1.1 };
    for (const [yaw, pitch, roll] of [
      [0, 0, 0],
      [137, 29, 4],
      [312, -29, -6],
      [45, 84, 0],
      [200, -75, 30],
    ]) {
      const q: Quat = quatFromYawPitchRoll(yaw, pitch, roll);
      for (const screenAngleDeg of [0, 90, 180, 270]) {
        const expected = rotateVec(deviceFrameQuat(q, screenAngleDeg), accel);
        const actual = deviceToWorld(q, screenAngleDeg, accel);
        expect(actual.x).toBeCloseTo(expected.x, 12);
        expect(actual.y).toBeCloseTo(expected.y, 12);
        expect(actual.z).toBeCloseTo(expected.z, 12);
      }
    }
  });

  it("is a rotation: it preserves length whatever the attitude", () => {
    const accel = { x: 0.3, y: -0.7, z: 1.1 };
    const q: Quat = quatFromYawPitchRoll(137, 29, 4);
    const out = deviceToWorld(q, 90, accel);
    expect(Math.hypot(out.x, out.y, out.z)).toBeCloseTo(Math.hypot(accel.x, accel.y, accel.z), 12);
  });
});

describe("the estimator has exactly one implementation", () => {
  it("gives useMotion.readTranslation() and the detector's own copy the same numbers", () => {
    // The hook runs it at the native sensor rate and the detector runs it over whatever the
    // rAF tick delivered. Fed the identical stream they must agree exactly, or the number the
    // meter shows and the number that stops a capture mean two different things.
    const accel = (t: number) => ({ x: 0.9 * Math.sin(2 * Math.PI * 2 * t), y: 0.4, z: 0.2 * Math.sin(2 * Math.PI * 0.7 * t) });
    const hook = createMotionTracker();
    const bare = createTranslationEstimator();
    const dt = 1 / 60;
    for (let i = 0; i * dt < 5; i += 1) {
      const t = i * dt;
      hook.push({ t: t * 1000, rotationRate: { alpha: 0, beta: 0, gamma: 0 }, acceleration: accel(t), interval: dt * 1000 });
      bare.push({ t: t * 1000, accel: accel(t), attitude: null });
    }
    expect(hook.readTranslation().driftMPerS).toBeCloseTo(bare.read().driftMPerS, 12);
    expect(hook.readTranslation().stepBandRmsMs2).toBeCloseTo(bare.read().stepBandRmsMs2, 12);
  });

  it("is clocked on the sensor, so a host that ticks twice per sample does not integrate twice", () => {
    const accel = (t: number) => ({ x: 0.8 * Math.sin(2 * Math.PI * 0.6 * t), y: 0, z: 0 });
    const once = createTranslationEstimator();
    const twice = createTranslationEstimator();
    const dt = 1 / 60;
    for (let i = 0; i * dt < 5; i += 1) {
      const t = i * dt;
      const input = { t: t * 1000, accel: accel(t), attitude: null };
      once.push(input);
      twice.push(input);
      twice.push(input); // the same sample, handed over again by a faster host
      twice.push(input);
    }
    expect(twice.read().driftMPerS).toBeCloseTo(once.read().driftMPerS, 12);
    expect(twice.read().at).toBe(once.read().at);
  });
});

describe("camera kernels", () => {
  const rgba = (fill: (i: number) => [number, number, number]) => {
    const data = new Uint8ClampedArray(64 * 64 * 4);
    for (let i = 0; i < 64 * 64; i += 1) {
      const [r, g, b] = fill(i);
      data[i * 4] = r;
      data[i * 4 + 1] = g;
      data[i * 4 + 2] = b;
      data[i * 4 + 3] = 255;
    }
    return data;
  };

  it("mean L* matches the OpenCV 0-255 scaling and clipping is counted at the rails", () => {
    expect(analyseFrameLuma(rgba(() => [128, 128, 128])).meanLuma).toBeCloseTo(136.6, 0);
    const white = analyseFrameLuma(rgba(() => [255, 255, 255]));
    expect(white.meanLuma).toBeCloseTo(255, 0);
    expect(white.clippedFraction).toBeCloseTo(1, 6);
    const black = analyseFrameLuma(rgba(() => [0, 0, 0]));
    expect(black.meanLuma).toBeCloseTo(0, 6);
    expect(black.clippedFraction).toBeCloseTo(1, 6);
    expect(analyseFrameLuma(rgba((i) => [20 + (i % 180), 20 + ((i * 7) % 180), 20 + ((i * 13) % 180)])).clippedFraction).toBe(0);
  });

  it("quadrant flow recovers a known shift, in the contract's 640-px units, with the right sign", () => {
    const w = 192;
    const h = 256;
    const noise = (x: number, y: number) => {
      const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
      return (s - Math.floor(s)) * 200 + 20;
    };
    const frameAt = (dx: number, dy: number): FlowFrame => {
      const grey = new Float32Array(w * h);
      for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) grey[y * w + x] = noise(x - dx, y - dy);
      return { grey, canvasW: w, canvasH: h };
    };

    const flow = quadrantFlow(frameAt(0, 0), frameAt(6, 0), 1080, 1920, pxPerDegAt640({ hfovDeg: 43, vfovDeg: 69, pxPerDegAt1080: 23.93, source: "assumed" }), 100);
    expect(flow.valid).toBe(true);
    // 6 canvas px * (640 / 192) = 20 px at the reference width.
    expect(flow.meanShiftPx).toBeCloseTo(20, 0);
    // Content moving right means the camera turned left: yaw decreases.
    expect(flow.yawDeltaDeg).toBeLessThan(0);
    expect(flow.yawDeltaDeg).toBeCloseTo(-20 / 14.18, 1);
    // A pure translation of the whole frame is a pure rotation as far as divergence goes.
    expect(flow.divergencePx).toBeLessThan(1);
  });

  it("a flat frame is reported invalid rather than as zero motion", () => {
    const flat = (): FlowFrame => ({ grey: new Float32Array(192 * 256).fill(90), canvasW: 192, canvasH: 256 });
    expect(quadrantFlow(flat(), flat(), 1080, 1920, 14.18, 0).valid).toBe(false);
  });

  it("picks the physical rear wide and refuses to claim one when it cannot", () => {
    const dev = (deviceId: string, label: string): MediaDeviceInfo =>
      ({ deviceId, label, kind: "videoinput", groupId: "g", toJSON: () => ({}) }) as MediaDeviceInfo;

    const ios = pickRearCamera([
      dev("f", "Front Camera"),
      dev("b", "Back Camera"),
      dev("d", "Back Dual Wide Camera"),
      dev("u", "Back Ultra Wide Camera"),
      dev("t", "Back Telephoto Camera"),
    ]);
    expect(ios.deviceId).toBe("b");
    expect(ios.virtual).toBe(false);

    // "Wide Angle" is the MAIN lens on several Android OEMs and must not be rejected.
    expect(pickRearCamera([dev("a", "Wide Angle Camera (back)")]).deviceId).toBe("a");
    // No labels yet: we cannot prove anything, so we do not.
    expect(pickRearCamera([dev("x", ""), dev("y", "")]).virtual).toBe(true);
  });
});

describe("recorder", () => {
  const fake = () => {
    let onData: (chunk: Blob) => void = () => undefined;
    let onStop: () => void = () => undefined;
    const recorder: MediaRecorderLike & { emit: (bytes: number) => void; state: string } = {
      state: "inactive",
      mimeType: "video/mp4",
      start() {
        recorder.state = "recording";
      },
      stop() {
        recorder.state = "inactive";
        onStop();
      },
      pause() {
        recorder.state = "paused";
      },
      resume() {
        recorder.state = "recording";
      },
      onData: (cb) => {
        onData = cb;
      },
      onStop: (cb) => {
        onStop = cb;
      },
      onError: () => undefined,
      emit: (bytes) => onData(new Blob([new Uint8Array(bytes)])),
    };
    return recorder;
  };

  it("chunks on the timeslice and seals a segment with a measured bitrate", async () => {
    const rec = fake();
    let clock = 0;
    const controller = createRecorder({
      stream: { getVideoTracks: () => [], getTracks: () => [] } as unknown as MediaStream,
      dimensions: () => ({ width: 1080, height: 1920 }),
      factory: () => rec,
      isTypeSupported: () => true,
      now: () => clock,
    });

    await controller.start();
    expect(controller.state).toBe("recording");
    for (let i = 0; i < 4; i += 1) {
      clock += 1000;
      rec.emit(2_500_000);
    }
    expect(controller.chunkCount).toBe(4);

    const segment = await controller.stop("user_finish");
    expect(segment).not.toBeNull();
    expect(segment!.durationMs).toBe(4000);
    expect(segment!.width).toBe(1080);
    expect(segment!.endedBy).toBe("user_finish");
    // 10 MB over 4 s = 20 Mbps, which is the requested rate.
    expect(segment!.bitrateBps).toBeCloseTo(20_000_000, -4);
  });

  it("preflight fails loudly when the platform produces no bytes", async () => {
    const silent = fake();
    silent.emit = () => undefined;
    const controller = createRecorder({
      stream: { getVideoTracks: () => [], getTracks: () => [] } as unknown as MediaStream,
      factory: () => silent,
      isTypeSupported: () => true,
      // Real hardware needs a patient probe; the logic under test does not.
      trialMs: 20,
      preflightBudgetMs: 400,
    });
    // `mimeType` is null, not the codec of the last thing tried. Preflight now walks a
    // ladder of codec/bitrate combinations and reports the one that actually emitted
    // bytes; when none did there is no such format, and naming a codec that produced
    // nothing would be reporting a fiction.
    await expect(controller.preflight()).resolves.toEqual({ ok: false, mimeType: null });
  });

  it("preflight keeps trying after a combination records nothing", async () => {
    // The failure that made a real Android 10 / Chrome 151 phone claim it could not record
    // at all: `isTypeSupported` said yes to every codec, each one constructed happily, and
    // each produced zero bytes — while a recorder built with no options at all worked.
    // Testing only the first rung and giving up is what turned a working phone away.
    const attempts: Array<{ mimeType?: string; videoBitsPerSecond?: number }> = [];
    const controller = createRecorder({
      stream: { getVideoTracks: () => [], getTracks: () => [] } as unknown as MediaStream,
      isTypeSupported: () => true,
      trialMs: 20,
      preflightBudgetMs: 2000,
      factory: (_stream, opts) => {
        attempts.push({ ...opts });
        const rec = fake();
        // Only the bare rung — no codec, no bitrate — produces data on this fake device.
        // The others behave exactly like the real Android failure: they start, they stop,
        // and not one byte ever arrives.
        const silent = opts?.mimeType !== undefined || opts?.videoBitsPerSecond !== undefined;
        const originalStart = rec.start.bind(rec);
        rec.start = (timeslice?: number) => {
          originalStart(timeslice);
          if (!silent) queueMicrotask(() => rec.emit(4096));
        };
        return rec;
      },
    });

    const result = await controller.preflight();
    expect(result.ok).toBe(true);
    expect(attempts.length).toBeGreaterThan(1);
    const last = attempts[attempts.length - 1];
    expect(last.mimeType).toBeUndefined();
    expect(last.videoBitsPerSecond).toBeUndefined();
  });
});

describe("telemetry", () => {
  const coverage: CoverageReport = {
    coveredFraction: 0.89,
    horizonFraction30: 0.99,
    horizonFraction60: 0.98,
    pitchMaxDeg: 63.5,
    pitchMinDeg: -63.5,
    rows: {
      level: { row: "level", filledStations: 24, requiredStations: 22, yawSpanDeg: 371, pitchMinDeg: -2, pitchMaxDeg: 2, complete: true },
      up: { row: "up", filledStations: 22, requiredStations: 22, yawSpanDeg: 368, pitchMinDeg: 27, pitchMaxDeg: 31, complete: true },
      down: { row: "down", filledStations: 11, requiredStations: 22, yawSpanDeg: 180, pitchMinDeg: -31, pitchMaxDeg: -27, complete: false },
    },
    gaps: [],
    complete: false,
    blockedBy: ["row_down"],
  };

  it("aggregates only recorded time and reports the geometry the engine must assert", () => {
    const telemetry = createTelemetry({ now: () => 0 });
    telemetry.setRecordingContext({ screenAngleDeg: 0, requestedBitrateBps: 20_000_000, mimeType: "video/mp4", width: 1080, height: 1920 });
    telemetry.setDevice({ cameraLabel: "Back Camera", virtualLensOnly: false, trackSettings: { width: 1920 }, appliedControls: { lockWhiteBalance: true } });
    telemetry.setCoveredMask(new Uint8Array([1, 0, 1, 1, 0]));

    const frame = (t: number, recording: boolean, pitch: number) => ({
      t,
      orientation: { t, q: [0, 0, 0, 1] as [number, number, number, number], yawDeg: 0, pitchDeg: pitch, rollDeg: 0, reference: "absolute" as const, headingAccuracyDeg: 10, screenAngleDeg: 0 },
      motion: null,
      quality: { t, mediaTime: 0, meanLuma: 150, clippedFraction: 0.01, exposureS: 1 / 60, exposureSource: "estimated" as const, measuredFps: 30, smearPx: 4 },
      flow: null,
      recording,
      activeRow: "level" as const,
    });
    const readout = (omega: number, drift: number) => ({
      omegaDegPerS: omega,
      speedCapDegPerS: 27.8,
      speedGrade: "green" as const,
      smearPx: 4,
      smearGrade: "green" as const,
      pivotRadiusM: 0.18,
      pivotGrade: "green" as const,
      translationMPerS: drift,
      stepBandRmsMs2: 0.1,
      flowDivergencePx: 1,
      walkingGrade: "green" as const,
      rollDeg: 1,
      rollGrade: "green" as const,
      rowPitchErrorDeg: 0,
      rowPitchGrade: "green" as const,
      meanLuma: 150,
      clippedFraction: 0.01,
      exposureGrade: "green" as const,
      measuredFps: 30,
      motionGateDegraded: false,
    });

    // Ignored: not recording.
    telemetry.push(frame(0, false, 80), readout(900, 5));
    for (let i = 0; i < 20; i += 1) telemetry.push(frame(100 + i * 100, true, i < 10 ? 0 : 29), readout(20, 0.05));

    const segments: RecordedSegment[] = [
      { blob: new Blob([new Uint8Array(5_000_000)]), mimeType: "video/mp4", durationMs: 2000, width: 1080, height: 1920, bitrateBps: 20_000_000, startedAt: 0, endedBy: "interrupted" },
      { blob: new Blob([new Uint8Array(5_000_000)]), mimeType: "video/mp4", durationMs: 2000, width: 1080, height: 1920, bitrateBps: 20_000_000, startedAt: 0, endedBy: "user_finish" },
    ];

    const built = telemetry.build(coverage, segments, "gaps");
    expect(built.schema).toBe("biosphere.capture.v1");
    expect(built.recording.segments).toBe(2);
    expect(built.recording.achievedBitrateBps).toBe(20_000_000);
    expect(built.recording.portrait).toBe(true);
    expect(built.sweep.pitchCentreMaxDeg).toBe(29);
    // The pre-recording frame at 80 deg and 900 deg/s must not have leaked in.
    expect(built.sweep.pitchCentreMinDeg).toBe(0);
    expect(built.sweep.p95AngularVelocityDegPerS).toBe(20);
    expect(built.sweep.fractionOverSpeedCap).toBe(0);
    expect(built.sweep.pitchReachMaxDeg).toBeCloseTo(29 + 34.5, 3);
    expect(built.sweep.yawSpanPerRowDeg.down).toBe(180);
    expect(built.sweep.rowCompletion.level).toBe(1);
    expect(built.translation.motionGateRan).toBe(true);
    expect(built.translation.medianPivotRadiusM).toBe(0.18);
    expect(built.coverage.coveredMaskBase64).toBe(Buffer.from([1, 0, 1, 1, 0]).toString("base64"));
    expect(built.track.mediaTime.length).toBe(20);
    expect(built.track.quat.length).toBe(80);
    expect(built.verdict).toBe("gaps");

    const file = telemetry.toFile(built);
    expect(file.name).toBe("motion.json");
    expect(file.type).toBe("application/json");
  });

  it("defaults virtualLensOnly to true when nothing proved the lens", () => {
    const telemetry = createTelemetry();
    expect(telemetry.build(coverage, [], "too_short").device.virtualLensOnly).toBe(true);
  });
});
