import { describe, expect, it } from "vitest";

import {
  MOTION_GATE_FALLBACK,
  STUCK_AFTER_MS,
  THRESHOLDS,
  WARNING_COPY,
  WARNING_PRIORITY,
  adaptiveSpeedCapDegPerS,
  predictedSmearPx,
  type DetectorReadout,
  type SensorFrame,
  type Warning,
  type WarningCode,
} from "@/lib/capture/types";
import { rowTargetPitchDeg } from "@/lib/capture/coverage";
import { blocksCapture, create, gradeAscending, gradeDescending, rankWarnings, worstGrade } from "@/lib/capture/detectors";
import {
  framesFromMotion,
  frozenAfter,
  glidingMotion,
  makeConfig,
  makeFlow,
  makeFrame,
  makeOrientation,
  makeQuality,
  standingMotion,
  turningMotion,
  walkingMotion,
} from "./synthetic";

const config = makeConfig();

/** Push a stream of frames and return the final readout plus the warnings that ever fired. */
function run(frames: SensorFrame[]) {
  const detector = create(config);
  let readout: DetectorReadout | null = null;
  const seen = new Set<WarningCode>();
  const stopped = new Set<WarningCode>();
  let last: Warning[] = [];
  let maxDriftMPerS = 0;
  let maxStepBandRmsMs2 = 0;
  for (const frame of frames) {
    readout = detector.push(frame);
    maxDriftMPerS = Math.max(maxDriftMPerS, readout.translationMPerS);
    maxStepBandRmsMs2 = Math.max(maxStepBandRmsMs2, readout.stepBandRmsMs2);
    last = detector.warnings();
    for (const w of last) {
      seen.add(w.code);
      if (w.severity === "stop") stopped.add(w.code);
    }
  }
  return { detector, readout: readout as DetectorReadout, seen, stopped, last, maxDriftMPerS, maxStepBandRmsMs2 };
}

/** A steady stream at one operating point, for threshold and hysteresis probing. */
function steady(options: {
  ms: number;
  hz?: number;
  omegaDegPerS?: number;
  exposureS?: number;
  meanLuma?: number;
  clippedFraction?: number;
  measuredFps?: number;
  rollDeg?: number;
  pitchDeg?: number;
  startAt?: number;
  flowPx?: number;
  noMotion?: boolean;
  activeRow?: SensorFrame["activeRow"];
}): SensorFrame[] {
  const hz = options.hz ?? 60;
  const dt = 1000 / hz;
  const out: SensorFrame[] = [];
  for (let t = options.startAt ?? 0; t < (options.startAt ?? 0) + options.ms; t += dt) {
    const omega = options.omegaDegPerS ?? 0;
    out.push(
      makeFrame(t, {
        orientation: makeOrientation(t, (omega * t) / 1000, options.pitchDeg ?? 0, { rollDeg: options.rollDeg ?? 0 }),
        motion: options.noMotion === true ? null : turningMotion({ omegaDegPerS: omega, durationMs: 0, startAt: t, sampleHz: hz })[0],
        quality: makeQuality(t, {
          omegaDegPerS: omega,
          exposureS: options.exposureS,
          meanLuma: options.meanLuma,
          clippedFraction: options.clippedFraction,
          measuredFps: options.measuredFps,
        }),
        flow: options.flowPx === undefined ? null : makeFlow(t, options.flowPx),
        activeRow: options.activeRow === undefined ? "level" : options.activeRow,
        recording: true,
      }),
    );
  }
  return out;
}

// =====================================================================================

describe("grading", () => {
  it("puts every THRESHOLDS boundary on the right side of the line", () => {
    const s = THRESHOLDS.angularSpeed;
    expect(gradeAscending(s.greenMaxDegPerS, s.greenMaxDegPerS, s.redDegPerS)).toBe("green");
    expect(gradeAscending(s.greenMaxDegPerS + 0.001, s.greenMaxDegPerS, s.redDegPerS)).toBe("amber");
    expect(gradeAscending(s.redDegPerS, s.greenMaxDegPerS, s.redDegPerS)).toBe("amber");
    expect(gradeAscending(s.redDegPerS + 0.001, s.greenMaxDegPerS, s.redDegPerS)).toBe("red");

    const sm = THRESHOLDS.smear;
    expect(gradeAscending(sm.greenPx, sm.greenPx, sm.amberPx)).toBe("green");
    expect(gradeAscending(sm.amberPx + 0.001, sm.greenPx, sm.amberPx)).toBe("red");

    const p = THRESHOLDS.pivotRadius;
    expect(gradeAscending(p.greenM, p.greenM, p.amberM)).toBe("green");
    expect(gradeAscending(p.amberM + 0.001, p.greenM, p.amberM)).toBe("red");

    const r = THRESHOLDS.roll;
    expect(gradeAscending(r.greenDeg, r.greenDeg, r.amberDeg)).toBe("green");
    expect(gradeAscending(r.amberDeg + 0.001, r.greenDeg, r.amberDeg)).toBe("red");

    const b = THRESHOLDS.brightness;
    expect(gradeDescending(b.warnMeanL, b.warnMeanL, b.blockMeanL)).toBe("green");
    expect(gradeDescending(b.warnMeanL - 0.001, b.warnMeanL, b.blockMeanL)).toBe("amber");
    expect(gradeDescending(b.blockMeanL - 0.001, b.warnMeanL, b.blockMeanL)).toBe("red");
  });

  it("returns the worse of two grades", () => {
    expect(worstGrade("green", "amber")).toBe("amber");
    expect(worstGrade("red", "amber")).toBe("red");
    expect(worstGrade("green", "green")).toBe("green");
  });
});

describe("the contract's own arithmetic", () => {
  it("halves the safe speed when the room darkens", () => {
    expect(adaptiveSpeedCapDegPerS(1 / 120)).toBeCloseTo(31, 0);
    expect(adaptiveSpeedCapDegPerS(1 / 60)).toBeCloseTo(27.8, 1);
    expect(adaptiveSpeedCapDegPerS(1 / 30)).toBeCloseTo(13.9, 1);
    expect(adaptiveSpeedCapDegPerS(1 / 15)).toBe(THRESHOLDS.angularSpeed.adaptiveMinDegPerS);
  });

  it("puts the coached 24 deg/s under the octave-1 smear budget at 1/60 s", () => {
    expect(predictedSmearPx(24, 1 / 60)).toBeLessThan(THRESHOLDS.smear.amberPx);
    expect(predictedSmearPx(24, 1 / 30)).toBeGreaterThan(THRESHOLDS.smear.amberPx);
  });
});

describe("warning ranking", () => {
  it("orders by WARNING_PRIORITY so only one correction is ever shown", () => {
    const codes: WarningCode[] = ["roll", "walking", "row_pitch", "landscape"];
    const warnings = codes.map((code) => ({
      code,
      severity: WARNING_COPY[code].severity,
      since: 0,
      value: 0,
      threshold: 0,
      unit: "deg" as const,
      titleKey: WARNING_COPY[code].title,
      bodyKey: WARNING_COPY[code].body,
      voiceKey: WARNING_COPY[code].voice,
      icon: WARNING_COPY[code].icon,
      haptic: [],
    }));
    expect(rankWarnings(warnings).map((w) => w.code)).toEqual(["landscape", "walking", "row_pitch", "roll"]);
  });

  it("suspends capture on `stop` and only on `stop`", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 3000, meanLuma: 50 })) detector.push(frame);
    const warnings = detector.warnings();
    expect(warnings.some((w) => w.code === "too_dark")).toBe(true);
    expect(blocksCapture(warnings)).toBe(false);
  });

  it("takes severity, copy keys and icon from WARNING_COPY, never from a local literal", () => {
    const { last } = run(steady({ ms: 3000, meanLuma: 50 }));
    const dark = last.find((w) => w.code === "too_dark");
    expect(dark?.severity).toBe(WARNING_COPY.too_dark.severity);
    expect(dark?.titleKey).toBe(WARNING_COPY.too_dark.title);
    expect(dark?.icon).toBe(WARNING_COPY.too_dark.icon);
  });

  it("emits no haptic pattern when haptics are off", () => {
    const detector = create(makeConfig({ haptics: false }));
    for (const frame of steady({ ms: 3000, meanLuma: 50 })) detector.push(frame);
    for (const w of detector.warnings()) expect(w.haptic).toEqual([]);
  });
});

// =====================================================================================

describe("a well-executed turn on the spot", () => {
  // A 0.20 m arc: the radius validate.py's own ladder puts at GOOD_ROT_RESIDUAL_PX = 2.0.
  const { readout, seen, maxDriftMPerS, maxStepBandRmsMs2 } = run(
    framesFromMotion(turningMotion({ omegaDegPerS: 24, pivotRadiusM: 0.2, durationMs: 15_000 })),
  );

  it("raises no warning at all", () => {
    expect(Array.from(seen)).toEqual([]);
  });

  it("grades the coached 24 deg/s green", () => {
    expect(readout.omegaDegPerS).toBeCloseTo(24, 0);
    expect(readout.speedGrade).toBe("green");
  });

  it("recovers the hand-held arc from a_radial / omega^2", () => {
    // Loose on purpose. `r = mean(|a.z|) / mean(omega^2)` averages a MAGNITUDE, so once the
    // radial acceleration approaches the noise floor the rectified mean sits above the true
    // one and the estimate reads high: at 0.20 m and 24 deg/s the signal is 3.6 mg against
    // this fixture's ~3.5 mg RMS tremor, which is the edge of where the contract's own
    // derivation claims validity ("above phone accelerometer noise once averaged over 60
    // samples" is quoted for the 6.3 mg of a 0.35 m arm). The bias is upward, i.e. toward
    // warning too early, which is the safe direction for a quality slope.
    expect(readout.pivotRadiusM as number).toBeGreaterThan(0.15);
    expect(readout.pivotRadiusM as number).toBeLessThan(0.3);
    expect(seen.has("pivot_radius")).toBe(false);
  });

  it("warns at the 0.35 m arc validate.py calls degraded, and never blocks for it", () => {
    const wider = run(framesFromMotion(turningMotion({ omegaDegPerS: 24, pivotRadiusM: 0.35, durationMs: 15_000 })));
    expect(wider.readout.pivotRadiusM).toBeCloseTo(0.35, 1);
    expect(wider.readout.pivotGrade).not.toBe("green");
    expect(wider.seen.has("pivot_radius")).toBe(true);
    // A quality slope, so it warns and never blocks.
    expect(blocksCapture(wider.last)).toBe(false);
  });

  it("stays far clear of BOTH walking thresholds for the whole turn", () => {
    // Centripetal acceleration is DC in the device frame, so the step band sees nothing; in
    // the world frame it rotates at 0.067 Hz, which the 0.3 Hz high-pass takes out of the
    // translation statistic. Measured: 0.03 m/s against a 0.30 stop, 0.04 m/s^2 against 0.70.
    expect(maxDriftMPerS).toBeLessThan(THRESHOLDS.translationDrift.warnMPerS / 2);
    expect(maxStepBandRmsMs2).toBeLessThan(THRESHOLDS.stepCadence.warnRmsMs2 / 2);
    expect(readout.walkingGrade).toBe("green");
  });

  it("ran the inertial gate", () => {
    expect(readout.motionGateDegraded).toBe(false);
  });
});

describe("standing perfectly still", () => {
  const { seen, maxDriftMPerS, maxStepBandRmsMs2 } = run(framesFromMotion(standingMotion({ durationMs: 14_000 })));

  it("measures nothing, on either walking signal", () => {
    // Hand tremor at 8 Hz and a 2 mg noise floor. If either statistic can be moved by this,
    // the thresholds mean nothing.
    expect(maxDriftMPerS).toBeLessThan(0.05);
    expect(maxStepBandRmsMs2).toBeLessThan(0.05);
    expect(seen.has("walking")).toBe(false);
  });
});

describe("a walking capture", () => {
  // The fixture no longer injects the +/-2.5 m/s^2 @ 0.5 Hz "lurch" that used to be the only
  // thing the old drift statistic could see. What is left is a real gait: 2 Hz footfall
  // energy on three axes, at constant travel speed, after three seconds of standing still.
  const { detector, seen, last, maxDriftMPerS, maxStepBandRmsMs2 } = run(
    framesFromMotion(walkingMotion({ durationMs: 14_000 })),
  );

  it("is caught, and caught as a `stop`", () => {
    expect(seen.has("walking")).toBe(true);
    expect(last[0].code).toBe("walking");
    expect(last[0].severity).toBe("stop");
    expect(blocksCapture(last)).toBe(true);
  });

  it("is caught by the STEP CADENCE, which is the only signal a steady gait offers", () => {
    // A body already at travel speed produces no acceleration in the direction of travel —
    // that is physics, not a limitation of the estimator — so the translation statistic
    // legitimately stays under its line and the cadence carries the detection alone.
    // Measured: band 1.09 m/s^2 against a 0.70 stop; drift 0.12 m/s against a 0.30 stop.
    expect(maxStepBandRmsMs2).toBeGreaterThan(THRESHOLDS.stepCadence.stopRmsMs2);
    expect(maxDriftMPerS).toBeGreaterThan(THRESHOLDS.translationDrift.warnMPerS / 2);
  });

  it("reports the walking fraction and the drift statistics for the sidecar", () => {
    const summary = detector.summary();
    expect(summary.translation.walkingFraction).toBeGreaterThan(0.1);
    expect(summary.translation.motionGateRan).toBe(true);
    expect(summary.translation.maxDriftMPerS).toBeGreaterThan(0);
  });

  it("catches a walk that STARTS mid-capture on both signals at once", () => {
    const walkOff = run(framesFromMotion(walkingMotion({ durationMs: 16_000, accelerateSeconds: 1 })));
    expect(walkOff.stopped.has("walking")).toBe(true);
    expect(walkOff.maxStepBandRmsMs2).toBeGreaterThan(THRESHOLDS.stepCadence.stopRmsMs2);
    // The acceleration from rest to 0.8 m/s is the one part of a walk an accelerometer can
    // see, and now it is retained: 0.41 m/s against the 0.30 stop line, where the old leaky
    // double integration reported 0.09.
    expect(walkOff.maxDriftMPerS).toBeGreaterThan(THRESHOLDS.translationDrift.stopMPerS);
  });
});

describe("the silent glide — a user sidestepping past a coffee table", () => {
  // Push off, coast, brake. No footfalls at all, so the step-cadence detector is blind to it
  // by construction; the coasting phase is inertially invisible because a constant velocity
  // produces no acceleration. The push is the only evidence that exists, and the old leaky
  // double integration threw it away inside a second: it measured 0.078 m/s against a 0.30
  // stop line and 0.23 m/s^2 against a 0.70 one, i.e. NO warning of any kind.
  const { seen, stopped, maxDriftMPerS, maxStepBandRmsMs2 } = run(framesFromMotion(glidingMotion({ durationMs: 14_000 })));

  it("is caught, by the translation statistic alone", () => {
    expect(seen.has("walking")).toBe(true);
    expect(stopped.has("walking")).toBe(true);
  });

  it("is invisible to the step-cadence detector, which is why the other signal has to work", () => {
    expect(maxStepBandRmsMs2).toBeLessThan(THRESHOLDS.stepCadence.warnRmsMs2);
  });

  it("clears the stop line with margin rather than grazing it", () => {
    // Measured 0.32 m/s, held above the line for 2.4 s against a 1.5 s sustain.
    expect(maxDriftMPerS).toBeGreaterThan(THRESHOLDS.translationDrift.stopMPerS);
  });
});

describe("what the copy table says about severity", () => {
  it("marks exactly three codes `stop` by default", () => {
    const stops = Object.keys(WARNING_COPY).filter((c) => WARNING_COPY[c as WarningCode].severity === "stop");
    // `landscape` and `lens_changed` are device facts, not measurements of the capture.
    expect(stops.sort()).toEqual(["landscape", "lens_changed", "walking"]);
  });

  it("still ranks a stop first once a warn-level code has been escalated past it", () => {
    // `too_dark` sits at priority 7, below `too_fast` at 3. Once darkness crosses
    // `brightness.blockMeanL` it becomes a stop, and the host reads severity off warnings[0]
    // — so severity has to outrank the table, or a warn would hide a stop.
    const { last } = run(steady({ ms: 4000, meanLuma: 10, omegaDegPerS: 45, exposureS: 1 / 240 }));
    expect(last[0].code).toBe("too_dark");
    expect(last[0].severity).toBe("stop");
    expect(last.some((w) => w.code === "too_fast" && w.severity === "warn")).toBe(true);
  });
});

describe("the band-pass calibration", () => {
  it("measures a 1.73 Hz sinusoid at its true RMS, not at half of it", () => {
    // sqrt(1*3) is the band's geometric centre, where LP3 - LP1 peaks at exactly 0.5. The
    // stop threshold is quoted as the RMS of the WALKING SIGNAL, so an uncorrected filter
    // would need twice the real walking energy before it stopped a capture — fail-open, in
    // the one detector allowed to abort.
    const amplitude = 1.4;
    const hz = 60;
    const frames: SensorFrame[] = [];
    for (let i = 0; i < hz * 6; i += 1) {
      const t = (i / hz) * 1000;
      const a = amplitude * Math.sin(2 * Math.PI * Math.sqrt(3) * (i / hz));
      frames.push(
        makeFrame(t, {
          motion: {
            t,
            rotationRate: { x: 0, y: 0, z: 0 },
            omegaDegPerS: 0,
            linearAccel: { x: 0, y: a, z: 0 },
            accelMagnitude: a,
            gravity: { x: 0, y: -9.81, z: 0 },
            accelSource: "native",
            intervalMs: 1000 / hz,
          },
        }),
      );
    }
    const { readout } = run(frames);
    expect(readout.stepBandRmsMs2).toBeCloseTo(amplitude / Math.SQRT2, 1);
  });

  it("rejects a constant acceleration entirely", () => {
    const hz = 60;
    const frames: SensorFrame[] = [];
    for (let i = 0; i < hz * 5; i += 1) {
      const t = (i / hz) * 1000;
      frames.push(
        makeFrame(t, {
          motion: {
            t,
            rotationRate: { x: 0, y: 0, z: 0 },
            omegaDegPerS: 0,
            linearAccel: { x: 0, y: 0, z: -3 },
            accelMagnitude: 3,
            gravity: { x: 0, y: -9.81, z: 0 },
            accelSource: "native",
            intervalMs: 1000 / hz,
          },
        }),
      );
    }
    const { readout } = run(frames);
    expect(readout.stepBandRmsMs2).toBeLessThan(0.05);
  });
});

// =====================================================================================

describe("the motion gate fails safe", () => {
  it("declares itself degraded when no linear acceleration ever arrives", () => {
    const { readout, detector } = run(steady({ ms: 4000, noMotion: true }));
    expect(readout.motionGateDegraded).toBe(true);
    expect(detector.summary().translation.motionGateRan).toBe(false);
  });

  it("tightens the camera-side stop threshold from 7 px to 5 px when it is degraded", () => {
    const middling = (MOTION_GATE_FALLBACK.flowStopPxWhenDegraded + THRESHOLDS.flowDivergence.stopPx) / 2;
    expect(middling).toBeGreaterThan(MOTION_GATE_FALLBACK.flowStopPxWhenDegraded);
    expect(middling).toBeLessThan(THRESHOLDS.flowDivergence.stopPx);

    // With a healthy inertial gate, 6 px of quadrant divergence is below the 7 px line.
    const healthy = run(steady({ ms: 4000, flowPx: middling, omegaDegPerS: 20 }));
    expect(healthy.seen.has("walking")).toBe(false);

    // With no inertial gate at all, the same 6 px condemns the capture. Absence of evidence
    // of walking must never read as evidence of standing still.
    const degraded = run(steady({ ms: 4000, flowPx: middling, noMotion: true }));
    expect(degraded.seen.has("walking")).toBe(true);
  });

  it("does not invent a drift number it cannot measure", () => {
    const { readout } = run(steady({ ms: 4000, noMotion: true }));
    expect(readout.translationMPerS).toBe(0);
    expect(readout.stepBandRmsMs2).toBe(0);
  });

  it("reconstructs from accelerationIncludingGravity when acceleration is absent", () => {
    // Builder S hands back `accelSource: "derived"` with a reconstructed vector; the gate
    // must treat that as having run, not as missing.
    const motion = walkingMotion({ durationMs: 14_000 }).map((m) => ({ ...m, accelSource: "derived" as const }));
    const { readout, seen } = run(framesFromMotion(motion));
    expect(readout.motionGateDegraded).toBe(false);
    expect(seen.has("walking")).toBe(true);
  });

  it("declares itself degraded when the sensor STREAM dies, not just when the sample is null", () => {
    // The failure that shipped. `motionRef.current.read()` returns the last sample forever, so
    // a `devicemotion` stream that stops — iOS after a background/foreground round trip,
    // Android throttling — hands the detector one frozen sample at every tick. Under the old
    // presence test that read as a perfectly still user for as long as it lasted: drift 0,
    // band 0, walkingGrade green, and telemetry uploading `motionGateRan: true`.
    const live = turningMotion({ omegaDegPerS: 24, durationMs: 12_000 });
    const frames = framesFromMotion(frozenAfter(live, 3000));

    const detector = create(config);
    let degradedAt: number | null = null;
    let finalDegraded = false;
    for (const frame of frames) {
      const readout = detector.push(frame);
      finalDegraded = readout.motionGateDegraded;
      if (degradedAt === null && readout.motionGateDegraded) degradedAt = frame.t;
    }
    expect(degradedAt).not.toBeNull();
    // Two seconds of grace (GATE_STALE_MS), then the gate admits it is not measuring.
    expect(degradedAt as number).toBeGreaterThanOrEqual(3000);
    expect(degradedAt as number).toBeLessThan(3000 + 2500);
    expect(finalDegraded).toBe(true);
    expect(detector.summary().translation.motionGateRan).toBe(true);
  });

  it("tightens the flow threshold once the stream has died, exactly as if it had never existed", () => {
    const middling = (MOTION_GATE_FALLBACK.flowStopPxWhenDegraded + THRESHOLDS.flowDivergence.stopPx) / 2;
    const live = turningMotion({ omegaDegPerS: 24, durationMs: 14_000 });
    const frames = framesFromMotion(frozenAfter(live, 3000), { flow: makeFlow(0, middling) });
    // 6 px is under the healthy 7 px line and over the degraded 5 px one. A user who walks
    // away while the accelerometer is dead must still be caught by the camera.
    expect(run(frames).seen.has("walking")).toBe(true);
  });

  it("does not paint from a FROZEN attitude either — that one banks stations from a dead compass", () => {
    // Orientation is ten times tighter than motion: `updateStations` keeps dwelling on
    // whatever station the frozen sample sits on, and `stampCone` keeps painting a direction
    // the camera is no longer pointing. `track_lost` escalates to `stop` so both suspend.
    const detector = create(config);
    const frozen = makeOrientation(2000, 0, 0);
    let sawStop = false;
    for (let t = 2000; t < 6000; t += 16) {
      detector.push(makeFrame(t, { orientation: frozen, quality: makeQuality(t) }));
      if (detector.warnings().some((w) => w.code === "track_lost" && w.severity === "stop")) sawStop = true;
    }
    expect(sawStop).toBe(true);
    expect(blocksCapture(detector.warnings())).toBe(true);
  });

  it("prefers a host-supplied estimate, and falls back without a seam when it goes stale", () => {
    // `MotionApiExt.readTranslation()` sees every `devicemotion` sample; the detector's own
    // copy sees whichever ones survived the rAF tick. Same code, so the better-sampled one
    // wins while it is fresh — and a host that never wires it, or one whose stream dies, is
    // simply back on the detector's own without anything going quiet.
    const detector = create(config);
    const frames = steady({ ms: 3000, omegaDegPerS: 12 });

    detector.setTranslation({
      driftMPerS: 0.42,
      stepBandRmsMs2: 0.1,
      gateRan: true,
      worldFrame: true,
      windowSeconds: 2,
      at: frames[0].t,
    });
    const supplied = detector.push(frames[0]);
    expect(supplied.translationMPerS).toBeCloseTo(0.42, 6);
    expect(supplied.motionGateDegraded).toBe(false);

    // The same estimate, now four seconds old: past GATE_STALE_MS it is not evidence.
    let last = supplied;
    for (const frame of frames) last = detector.push(frame);
    expect(last.translationMPerS).toBeLessThan(0.1);
    // ...and the detector's own estimator kept running underneath, so the gate is still live.
    expect(last.motionGateDegraded).toBe(false);
  });

  it("keeps a poor heading at `warn`, because yaw no longer comes from the heading", () => {
    const frames = steady({ ms: 3000 }).map((f) => ({
      ...f,
      orientation: makeOrientation(f.t, 0, 0, { headingAccuracyDeg: -1, reference: "absolute" }),
    }));
    const { last } = run(frames);
    const lost = last.find((w) => w.code === "track_lost");
    expect(lost?.severity).toBe("warn");
  });

  it("says nothing about a poor heading on a relative track, which does not use one", () => {
    const frames = steady({ ms: 3000 }).map((f) => ({
      ...f,
      orientation: makeOrientation(f.t, 0, 0, { headingAccuracyDeg: -1, reference: "relative" }),
    }));
    expect(run(frames).seen.has("track_lost")).toBe(false);
  });
});

describe("a phone with no gyroscope", () => {
  // Finding 6: `rotationRate: null` used to give omega 0, `speedGrade: "green"`, smearPx 0 and
  // `smearGrade: "green"` at any speed. `too_fast` and `smear` could never fire, and the
  // framing "hold steady" gate (steadyMaxDegPerS = 6) passed instantly while the phone was
  // being waved.
  const gyroless = (omegaDegPerS: number, ms: number) => {
    const motion = turningMotion({ omegaDegPerS, durationMs: ms, gyroscope: false });
    // Orientation still turns at the real rate: that is the whole point of the fallback.
    return motion.map((m) => {
      const t = m.t;
      return makeFrame(t, {
        motion: m,
        orientation: makeOrientation(t, (omegaDegPerS * t) / 1000, 0),
        quality: makeQuality(t, { exposureS: 1 / 30, omegaDegPerS }),
        activeRow: "level",
      });
    });
  };

  it("reconstructs the rate from successive orientation quaternions", () => {
    const { readout } = run(gyroless(45, 5000));
    expect(readout.omegaDegPerS).toBeCloseTo(45, 0);
  });

  it("can now fire too_fast, which it previously could not at any speed", () => {
    expect(run(gyroless(45, 5000)).seen.has("too_fast")).toBe(true);
    expect(run(gyroless(12, 5000)).seen.has("too_fast")).toBe(false);
  });

  it("can now fire smear in a dim room", () => {
    expect(run(gyroless(45, 5000)).seen.has("smear")).toBe(true);
  });

  it("grades an UNMEASURABLE rate amber, never a confident green 0", () => {
    // No gyroscope and no attitude either: nothing on the phone knows whether it is moving.
    // Amber is the honest reading of "unknown" in a three-colour vocabulary, and it stops the
    // framing gate from starting a recording on a zero nobody measured.
    const frames: SensorFrame[] = [];
    for (let t = 0; t < 3000; t += 16) {
      frames.push(makeFrame(t, { orientation: null, motion: turningMotion({ omegaDegPerS: 0, durationMs: 0, startAt: t, gyroscope: false })[0], quality: makeQuality(t) }));
    }
    const { readout } = run(frames);
    expect(readout.omegaDegPerS).toBe(0);
    expect(readout.speedGrade).toBe("amber");
    expect(readout.smearGrade).toBe("amber");
  });
});

// =====================================================================================

describe("speed and smear are two different walls", () => {
  it("fires too_fast on the fixed extraction wall, not on the adaptive one", () => {
    const { seen, readout } = run(steady({ ms: 4000, omegaDegPerS: 36, exposureS: 1 / 240 }));
    expect(seen.has("too_fast")).toBe(true);
    // Exposure is short enough that smear is fine — the two must be independent.
    expect(readout.smearPx).toBeLessThan(THRESHOLDS.smear.amberPx);
    expect(seen.has("smear")).toBe(false);
  });

  it("stays quiet at 30 deg/s, under the 30.96 deg/s extraction wall", () => {
    const { seen } = run(steady({ ms: 5000, omegaDegPerS: 30, exposureS: 1 / 240 }));
    expect(seen.has("too_fast")).toBe(false);
  });

  it("fires smear in a dim room at a speed that would otherwise be fine", () => {
    const { seen, readout } = run(steady({ ms: 4000, omegaDegPerS: 24, exposureS: 1 / 30, meanLuma: 100 }));
    expect(seen.has("smear")).toBe(true);
    expect(seen.has("too_fast")).toBe(false);
    expect(readout.speedGrade).toBe("green");
    expect(readout.smearGrade).toBe("red");
    expect(readout.speedCapDegPerS).toBeCloseTo(13.9, 1);
  });
});

describe("attitude", () => {
  it("warns on roll past the 5-deg cross-row overlap budget", () => {
    expect(run(steady({ ms: 3000, rollDeg: 3 })).seen.has("roll")).toBe(false);
    expect(run(steady({ ms: 3000, rollDeg: 8 })).seen.has("roll")).toBe(true);
    // Roll leans the horizon line; it never takes over the screen.
    expect(WARNING_COPY.roll.severity).toBe("info");
  });

  it("warns when the frame centre wanders off the active row's elevation", () => {
    // select._ROW_SEPARATION_DEG = 8: a row that wanders merges into its neighbour and the
    // capture reads as two rows instead of three.
    const onRow = run(steady({ ms: 3000, pitchDeg: rowTargetPitchDeg("up"), activeRow: "up" }));
    expect(onRow.seen.has("row_pitch")).toBe(false);
    const wandering = run(steady({ ms: 3000, pitchDeg: rowTargetPitchDeg("up") - 8, activeRow: "up" }));
    expect(wandering.seen.has("row_pitch")).toBe(true);
    expect(wandering.readout.rowPitchErrorDeg).toBeCloseTo(-8, 6);
  });

  it("says nothing about row pitch when no row is being coached", () => {
    expect(run(steady({ ms: 3000, pitchDeg: 60, activeRow: null })).seen.has("row_pitch")).toBe(false);
  });

  it("reports a lost compass", () => {
    const frames = steady({ ms: 3000 }).map((f) => ({
      ...f,
      orientation: makeOrientation(f.t, 0, 0, { headingAccuracyDeg: -1 }),
    }));
    expect(run(frames).seen.has("track_lost")).toBe(true);
    expect(run(steady({ ms: 3000 })).seen.has("track_lost")).toBe(false);
  });
});

describe("exposure", () => {
  it("warns below 80 mean L*, which is 3.2x the engine's own cliff", () => {
    expect(run(steady({ ms: 3000, meanLuma: 120 })).seen.has("too_dark")).toBe(false);
    expect(run(steady({ ms: 3000, meanLuma: 50 })).seen.has("too_dark")).toBe(true);
  });

  it("warns on glare at exactly the point extract._clipping_penalty starts docking", () => {
    expect(THRESHOLDS.clipping.warnFraction).toBe(0.05);
    expect(run(steady({ ms: 3000, clippedFraction: 0.02 })).seen.has("glare")).toBe(false);
    expect(run(steady({ ms: 3000, clippedFraction: 0.12 })).seen.has("glare")).toBe(true);
  });

  it("STOPS below brightness.blockMeanL, so a finger over the lens cannot paint the dome", () => {
    // The scenario: ten seconds of a fifty-second sweep with a finger over the lens while the
    // user re-grips. Luma ~5. Stations keep filling, `stampCone` keeps painting, and
    // `FlowSample.valid` goes false so the flow walking-signal silently drops out too. The
    // whole-capture mean luma is still ~120, so the verdict never notices, and the review
    // screen says "Çekim tam." `blockMeanL` existed for exactly this and had no path to a stop.
    const dim = run(steady({ ms: 3000, meanLuma: 50 }));
    expect(dim.seen.has("too_dark")).toBe(true);
    expect(dim.stopped.has("too_dark")).toBe(false);
    expect(blocksCapture(dim.last)).toBe(false);

    const black = run(steady({ ms: 3000, meanLuma: 5 }));
    expect(black.stopped.has("too_dark")).toBe(true);
    expect(blocksCapture(black.last)).toBe(true);
    const warning = black.last.find((w) => w.code === "too_dark");
    expect(warning?.threshold).toBe(THRESHOLDS.brightness.blockMeanL);
  });

  it("STOPS above clipping.blockFraction, for the ten seconds pointed at a sunlit window", () => {
    const bright = run(steady({ ms: 3000, clippedFraction: 0.08 }));
    expect(bright.seen.has("glare")).toBe(true);
    expect(bright.stopped.has("glare")).toBe(false);

    const blown = run(steady({ ms: 3000, clippedFraction: 0.4 }));
    expect(blown.stopped.has("glare")).toBe(true);
    expect(blocksCapture(blown.last)).toBe(true);
  });

  it("clears the block back down to a warn as the light returns, without flickering", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 3000, meanLuma: 5 })) detector.push(frame);
    expect(blocksCapture(detector.warnings())).toBe(true);
    // 60 L* is clear of the 40 block line and its 47.1 hysteresis exit, and still under the
    // 80 warn line: the stop must clear and the warn must stay.
    for (const frame of steady({ ms: 4000, meanLuma: 60, startAt: 3000 })) detector.push(frame);
    expect(blocksCapture(detector.warnings())).toBe(false);
    expect(detector.warnings().some((w) => w.code === "too_dark")).toBe(true);
  });

  it("reports collapsed frame rate, which on iOS can only be reported and not fixed", () => {
    expect(run(steady({ ms: 4000, measuredFps: 30 })).seen.has("low_fps")).toBe(false);
    expect(run(steady({ ms: 4000, measuredFps: 18 })).seen.has("low_fps")).toBe(true);
  });

  it("grades brightness and clipping together", () => {
    expect(run(steady({ ms: 2000, meanLuma: 150, clippedFraction: 0.01 })).readout.exposureGrade).toBe("green");
    expect(run(steady({ ms: 2000, meanLuma: 150, clippedFraction: 0.2 })).readout.exposureGrade).toBe("red");
    expect(run(steady({ ms: 2000, meanLuma: 30, clippedFraction: 0.01 })).readout.exposureGrade).toBe("red");
  });
});

// =====================================================================================

describe("hysteresis", () => {
  it("does not flicker while a value hovers on its threshold", () => {
    const detector = create(config);
    let transitions = 0;
    let previous = false;
    const hz = 60;
    for (let i = 0; i < hz * 8; i += 1) {
      const t = (i / hz) * 1000;
      // Oscillate either side of the 31 deg/s red line twice a second.
      const omega = THRESHOLDS.angularSpeed.redDegPerS + (i % 30 < 15 ? 2 : -2);
      detector.push(steady({ ms: 1000 / hz, hz, omegaDegPerS: omega, startAt: t, exposureS: 1 / 240 })[0]);
      const active = detector.warnings().some((w) => w.code === "too_fast");
      if (active !== previous) transitions += 1;
      previous = active;
    }
    // A bare comparison would toggle ~16 times over 8 seconds.
    expect(transitions).toBeLessThanOrEqual(2);
  });

  it("holds a warning while the value sits between the exit and enter thresholds", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 3000, omegaDegPerS: 40, exposureS: 1 / 240 })) detector.push(frame);
    expect(detector.warnings().some((w) => w.code === "too_fast")).toBe(true);
    // 28 is below the 31 enter line but above the 26.35 exit line.
    for (const frame of steady({ ms: 3000, omegaDegPerS: 28, exposureS: 1 / 240, startAt: 3000 })) detector.push(frame);
    expect(detector.warnings().some((w) => w.code === "too_fast")).toBe(true);
  });

  it("clears once the value is properly back inside the band", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 3000, omegaDegPerS: 40, exposureS: 1 / 240 })) detector.push(frame);
    expect(detector.warnings().some((w) => w.code === "too_fast")).toBe(true);
    for (const frame of steady({ ms: 4000, omegaDegPerS: 10, exposureS: 1 / 240, startAt: 3000 })) detector.push(frame);
    expect(detector.warnings().some((w) => w.code === "too_fast")).toBe(false);
  });

  it("reports when the condition first held, not when the latch fired", () => {
    const detector = create(config);
    let visibleAt: number | null = null;
    for (const frame of steady({ ms: 4000, omegaDegPerS: 40, exposureS: 1 / 240, startAt: 1000 })) {
      detector.push(frame);
      if (visibleAt === null && detector.warnings().some((w) => w.code === "too_fast")) visibleAt = frame.t;
    }
    const warning = detector.warnings().find((w) => w.code === "too_fast");
    expect(warning).toBeDefined();
    expect(visibleAt).not.toBeNull();
    if (warning !== undefined && visibleAt !== null) {
      // The latch fires 500 ms after the breach starts, but the UI must be able to age the
      // warning honestly, so `since` reports the START of the breach, not the latch.
      expect(warning.since).toBeLessThan(visibleAt);
      expect(visibleAt - warning.since).toBeGreaterThanOrEqual(THRESHOLDS.angularSpeed.redSustainMs);
    }
  });

  it("requires the full sustain before a warning appears at all", () => {
    const detector = create(config);
    let firstAt: number | null = null;
    for (const frame of steady({ ms: 4000, omegaDegPerS: 45, exposureS: 1 / 240 })) {
      detector.push(frame);
      if (firstAt === null && detector.warnings().some((w) => w.code === "too_fast")) firstAt = frame.t;
    }
    expect(firstAt).not.toBeNull();
    expect(firstAt as number).toBeGreaterThanOrEqual(THRESHOLDS.angularSpeed.redSustainMs);
  });
});

// =====================================================================================

describe("external device facts", () => {
  it("latches landscape immediately — it is a fact, not a measurement", () => {
    const detector = create(config);
    detector.push(makeFrame(0));
    detector.setExternalFlags({ landscape: true });
    detector.push(makeFrame(16));
    expect(detector.warnings()[0].code).toBe("landscape");
    expect(detector.warnings()[0].severity).toBe("stop");

    detector.setExternalFlags({ landscape: false });
    detector.push(makeFrame(32));
    expect(detector.warnings().some((w) => w.code === "landscape")).toBe(false);
  });

  it("outranks everything else, because nothing else matters until the phone is upright", () => {
    expect(WARNING_PRIORITY[0]).toBe("landscape");
    const detector = create(config);
    for (const frame of steady({ ms: 3000, meanLuma: 40, omegaDegPerS: 45 })) detector.push(frame);
    detector.setExternalFlags({ landscape: true, lensChanged: true });
    detector.push(makeFrame(3100, { quality: makeQuality(3100, { meanLuma: 40 }) }));
    expect(detector.warnings()[0].code).toBe("landscape");
    expect(detector.warnings()[1].code).toBe("lens_changed");
  });

  it("treats a mid-capture lens change as a stop: the solve fits ONE focal for the file", () => {
    const detector = create(config);
    detector.push(makeFrame(0));
    detector.setExternalFlags({ lensChanged: true });
    detector.push(makeFrame(16));
    expect(blocksCapture(detector.warnings())).toBe(true);
  });
});

describe("stall coaching", () => {
  it("goes quiet until the first station fills, then nags after six seconds", () => {
    const detector = create(config);
    expect(detector.isStalled(999_999)).toBe(false);
    detector.noteStationFilled(1000);
    expect(detector.isStalled(1000 + STUCK_AFTER_MS - 1)).toBe(false);
    expect(detector.isStalled(1000 + STUCK_AFTER_MS)).toBe(true);
    detector.noteStationFilled(9000);
    expect(detector.isStalled(9500)).toBe(false);
  });
});

describe("telemetry aggregates", () => {
  it("accumulates only while recording", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 2000, meanLuma: 90 })) detector.push({ ...frame, recording: false });
    expect(detector.summary().image.meanLuma).toBe(0);
    for (const frame of steady({ ms: 2000, meanLuma: 90, startAt: 2000 })) detector.push(frame);
    expect(detector.summary().image.meanLuma).toBeCloseTo(90, 1);
  });

  it("reports the sweep statistics the sidecar carries", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 6000, omegaDegPerS: 24, exposureS: 1 / 240 })) detector.push(frame);
    const summary = detector.summary();
    expect(summary.sweep.meanAngularVelocityDegPerS).toBeCloseTo(24, 0);
    expect(summary.sweep.p95AngularVelocityDegPerS).toBeCloseTo(24, 0);
    expect(summary.sweep.fractionOverSpeedCap).toBe(0);
    expect(summary.image.fractionOverSmearBudget).toBe(0);
  });

  it("counts the time spent over the extraction wall", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 6000, omegaDegPerS: 45, exposureS: 1 / 240 })) detector.push(frame);
    expect(detector.summary().sweep.fractionOverSpeedCap).toBeGreaterThan(0.8);
  });

  it("withholds a pivot radius it never had enough gated samples to estimate", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 3000, omegaDegPerS: 2 })) detector.push(frame);
    expect(detector.summary().translation.medianPivotRadiusM).toBeNull();
  });

  it("resets everything, so a retake cannot inherit the last take's numbers", () => {
    const detector = create(config);
    for (const frame of steady({ ms: 3000, meanLuma: 40, omegaDegPerS: 45 })) detector.push(frame);
    expect(detector.warnings().length).toBeGreaterThan(0);
    detector.reset();
    expect(detector.warnings()).toEqual([]);
    expect(detector.summary().image.meanLuma).toBe(0);
    expect(detector.summary().translation.motionGateRan).toBe(false);
  });
});

describe("degenerate input", () => {
  it("survives a frame with nothing in it", () => {
    const detector = create(config);
    const readout = detector.push({ t: 0, orientation: null, motion: null, quality: null, flow: null, recording: false, activeRow: null });
    expect(Number.isFinite(readout.omegaDegPerS)).toBe(true);
    expect(readout.pivotRadiusM).toBeNull();
    expect(detector.warnings().every((w) => Number.isFinite(w.value))).toBe(true);
  });

  it("survives a clock that goes backwards", () => {
    const detector = create(config);
    detector.push(makeFrame(5000));
    const readout = detector.push(makeFrame(10));
    expect(Number.isFinite(readout.omegaDegPerS)).toBe(true);
  });

  it("never emits NaN into a readout field", () => {
    const { readout } = run(framesFromMotion(walkingMotion({ durationMs: 4000 })));
    for (const [key, value] of Object.entries(readout)) {
      if (typeof value === "number") expect(Number.isFinite(value), key).toBe(true);
    }
  });
});
