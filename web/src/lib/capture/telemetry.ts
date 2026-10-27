"use client";

/**
 * TELEMETRY — what the phone measured, handed to the engine as a PRIOR.
 * =====================================================================================
 *
 * This is the sidecar that lets the solver pin K instead of re-estimating it per scene, and
 * seed pose instead of solving cold. It rides the existing upload as `motion.json` inside the
 * same `files[]` array `api.scenes.create` already accepts, so nothing in api.ts changes.
 *
 * IT IS A PRIOR AND A DIAGNOSTIC, NEVER TRUTH. The solver is at 0.403 deg pose error and
 * 0.042 deg loop closure — orders of magnitude better than a phone's sensor fusion. Feeding
 * this track to the stitcher as ground truth would make the panorama worse.
 *
 * Two fields here exist purely so the engine can refuse loudly instead of failing silently:
 *   - `recording.screenAngleDeg` together with `width`/`height`: iOS records sensor-native
 *     landscape and applies a rotation tag rather than rotating pixels. An ffmpeg path that
 *     ignores the display matrix solves 69 deg across the short edge and 43 deg down the long
 *     one — the field of view exactly inverted — and produces a plausible-looking but
 *     geometrically wrong panorama with no error anywhere. These fields let the engine assert
 *     the decoded orientation.
 *   - `translation.motionGateRan`: FALSE means the inertial walking gate never ran on this
 *     device. Absence of evidence, not evidence of absence.
 *
 * Aggregates are accumulated only while `SensorFrame.recording` is true, so every "fraction
 * of the capture" in the payload is a fraction of RECORDED time and not of time spent reading
 * the stance card.
 */

import type { DeviceModelHints } from "./device";
import type {
  CameraControlPlan,
  CaptureTelemetry,
  CaptureVerdict,
  CoverageReport,
  DetectorReadout,
  Degrees,
  LensGeometry,
  Millis,
  RecordedSegment,
  RowId,
  SensorFrame,
  ShutterCapRecord,
  TelemetryApi,
} from "./types";
import { COVERAGE_GRID, DEFAULT_LENS, RECORDING, ROW_IDS, TELEMETRY_FILENAME, THRESHOLDS } from "./types";

// -------------------------------------------------------------------------------------
// Small numeric helpers
// -------------------------------------------------------------------------------------

const round = (value: number, digits: number): number => {
  if (!Number.isFinite(value)) return 0;
  const f = 10 ** digits;
  return Math.round(value * f) / f;
};

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * p)));
  return sorted[index];
}

/**
 * Base64 without `btoa`, so this module is identical in a browser, in a worker and in a Node
 * test. The covered mask is 2592 bytes, i.e. 3456 characters — small enough to build directly.
 */
function toBase64(bytes: Uint8Array): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += alphabet[b0 >> 2];
    out += alphabet[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? alphabet[((b1 & 15) << 2) | (b2 >> 6)] : "=";
    out += i + 2 < bytes.length ? alphabet[b2 & 63] : "=";
  }
  return out;
}

function detectPlatform(userAgent: string): CaptureTelemetry["device"]["platform"] {
  if (/android/i.test(userAgent)) return "android";
  // iPadOS reports itself as a Mac; the touch points give it away.
  if (/iphone|ipad|ipod/i.test(userAgent)) return "ios";
  if (/macintosh/i.test(userAgent) && typeof navigator !== "undefined" && navigator.maxTouchPoints > 1) return "ios";
  return "other";
}

// -------------------------------------------------------------------------------------
// API
// -------------------------------------------------------------------------------------

export interface TelemetryOptions {
  lens?: LensGeometry;
  /** Track rate. 10 Hz matches COVERAGE_GRID.stampHz: a 24 deg/s sweep moves 2.4 deg between samples. */
  trackHz?: number;
  now?: () => Millis;
}

/** What the camera layer knows and the telemetry cannot discover for itself. */
export interface TelemetryDeviceInfo {
  cameraLabel: string | null;
  virtualLensOnly: boolean;
  trackSettings: Record<string, unknown>;
  appliedControls: Partial<Record<keyof CameraControlPlan, boolean>>;
}

/** What the recording layer knows: the geometry the engine must assert against. */
export interface TelemetryRecordingContext {
  screenAngleDeg: Degrees;
  requestedBitrateBps: number;
  mimeType: string | null;
  /** Decoder-truth dimensions, from `video.videoWidth/videoHeight`. */
  width: number;
  height: number;
}

/**
 * The camera's own view of itself. `CameraApi` satisfies this structurally, so the page can
 * pass the hook's result straight in and never call `setDevice` by hand. A getter is accepted
 * too, for callers that want the values read at `build()` time rather than at construction.
 */
export interface TelemetryCameraSource {
  label: string | null;
  virtualLensOnly: boolean;
  settings: Record<string, unknown>;
  appliedControls: Partial<Record<keyof CameraControlPlan, boolean>>;
  shutterCap?: ShutterCapRecord | null;
}

export interface TelemetryApiExt extends TelemetryApi {
  setDevice: (info: TelemetryDeviceInfo) => void;
  /** Which phone this is, read once at start-up by `capture/device.ts`. */
  setModelHints: (hints: DeviceModelHints | null) => void;
  setLens: (lens: LensGeometry) => void;
  /** Zoom or the active lens changed mid-capture. The solve fits ONE focal for the file. */
  noteLensChange: () => void;
  setRecordingContext: (context: TelemetryRecordingContext) => void;
  /** The 72x36 covered mask, so the server can render the heat-map the phone showed. */
  setCoveredMask: (mask: Uint8Array) => void;
  /** How many track samples have been accumulated. */
  sampleCount: () => number;
  /**
   * The recorder restarted inside the current segment: drop the wall-clock anchor so the
   * next push re-anchors media time to the new encoder's first frame.
   */
  restartSegment: () => void;
}

export function createTelemetry(
  options: TelemetryOptions = {},
  camera?: TelemetryCameraSource | (() => TelemetryCameraSource | null) | null,
): TelemetryApiExt {
  const trackIntervalMs = 1000 / Math.max(1, options.trackHz ?? COVERAGE_GRID.stampHz);
  const readCamera = (): TelemetryCameraSource | null => (typeof camera === "function" ? camera() : (camera ?? null));

  let lens: LensGeometry = options.lens ?? DEFAULT_LENS;
  let device: TelemetryDeviceInfo | null = null;
  let modelHints: DeviceModelHints | null = null;
  let recordingContext: TelemetryRecordingContext | null = null;
  let coveredMask: Uint8Array | null = null;
  let lensChanged = false;

  // Segment anchoring: `mediaTime` is seconds into the CONCATENATED media, so each segment
  // contributes its own offset plus the wall-clock time since that segment started. The
  // preview element's own rVFC mediaTime cannot be used for this — it counts from when the
  // <video> started playing, not from when MediaRecorder.start() was called.
  let segmentOffsetS = 0;
  let segmentAnchorT: Millis | null = null;

  let lastTrackAt = -Infinity;
  const trackMediaTime: number[] = [];
  const trackQuat: number[] = [];
  const trackOmega: number[] = [];
  const trackSegment: number[] = [];
  let segmentIndex = 0;

  const omegaSamples: number[] = [];
  const pivotSamples: number[] = [];
  const driftSamples: number[] = [];
  const fpsSamples: number[] = [];

  let recordedPushes = 0;
  let overSpeedPushes = 0;
  let walkingPushes = 0;
  let motionGateRan = false;

  let pitchCentreMin = Number.POSITIVE_INFINITY;
  let pitchCentreMax = Number.NEGATIVE_INFINITY;

  let lumaSum = 0;
  let lumaCount = 0;
  let lumaMin = Number.POSITIVE_INFINITY;
  let clippedSum = 0;
  let clippedMax = 0;
  let smearOver = 0;
  let smearCount = 0;

  let maxDrift = 0;

  function push(frame: SensorFrame, readout: DetectorReadout | null): void {
    if (!frame.recording) return;
    if (segmentAnchorT === null) segmentAnchorT = frame.t;

    recordedPushes += 1;

    const omega = readout?.omegaDegPerS ?? frame.motion?.omegaDegPerS ?? 0;
    omegaSamples.push(omega);
    if (omega > THRESHOLDS.angularSpeed.redDegPerS) overSpeedPushes += 1;

    if (frame.orientation) {
      pitchCentreMin = Math.min(pitchCentreMin, frame.orientation.pitchDeg);
      pitchCentreMax = Math.max(pitchCentreMax, frame.orientation.pitchDeg);
    }

    if (frame.quality) {
      lumaSum += frame.quality.meanLuma;
      lumaCount += 1;
      lumaMin = Math.min(lumaMin, frame.quality.meanLuma);
      clippedSum += frame.quality.clippedFraction;
      clippedMax = Math.max(clippedMax, frame.quality.clippedFraction);
      smearCount += 1;
      if (frame.quality.smearPx > THRESHOLDS.smear.amberPx) smearOver += 1;
      if (frame.quality.measuredFps > 0) fpsSamples.push(frame.quality.measuredFps);
    }

    if (readout) {
      if (readout.pivotRadiusM !== null && Number.isFinite(readout.pivotRadiusM)) pivotSamples.push(readout.pivotRadiusM);
      driftSamples.push(readout.translationMPerS);
      maxDrift = Math.max(maxDrift, readout.translationMPerS);
      if (readout.walkingGrade === "red") walkingPushes += 1;
      // Once the gate has run even briefly, it ran. The flag exists to catch hardware that
      // never delivers acceleration at all, not a momentary dropout.
      if (!readout.motionGateDegraded) motionGateRan = true;
    }

    // Track, downsampled. Flat arrays: 1200 samples is 24 KiB this way and ~120 KiB as objects.
    if (frame.orientation && frame.t - lastTrackAt >= trackIntervalMs - 1) {
      lastTrackAt = frame.t;
      trackMediaTime.push(round(segmentOffsetS + (frame.t - segmentAnchorT) / 1000, 3));
      trackQuat.push(
        round(frame.orientation.q[0], 5),
        round(frame.orientation.q[1], 5),
        round(frame.orientation.q[2], 5),
        round(frame.orientation.q[3], 5),
      );
      trackOmega.push(round(omega, 2));
      trackSegment.push(segmentIndex);
    }
  }

  function markSegment(index: number, mediaTimeOffset: number): void {
    segmentIndex = index;
    segmentOffsetS = mediaTimeOffset;
    // Anchored by the first RECORDED push, not by this call: the recorder has not started
    // yet when the stage marks the segment, and the encoder's first frame is later still.
    segmentAnchorT = null;
  }

  function restartSegment(): void {
    segmentAnchorT = null;
  }

  function build(coverage: CoverageReport, segments: RecordedSegment[], verdict: CaptureVerdict): CaptureTelemetry {
    const userAgent = typeof navigator !== "undefined" ? navigator.userAgent : "";
    const live = readCamera();
    const sortedOmega = [...omegaSamples].sort((a, b) => a - b);
    const sortedPivot = [...pivotSamples].sort((a, b) => a - b);
    const sortedDrift = [...driftSamples].sort((a, b) => a - b);

    const totalDurationMs = segments.reduce((sum, s) => sum + s.durationMs, 0);
    const totalBytes = segments.reduce((sum, s) => sum + s.blob.size, 0);
    const longest = segments.reduce<RecordedSegment | null>((best, s) => (best === null || s.durationMs > best.durationMs ? s : best), null);

    const width = recordingContext?.width || longest?.width || 0;
    const height = recordingContext?.height || longest?.height || 0;

    const yawSpanPerRowDeg = {} as Record<RowId, Degrees>;
    const rowCompletion = {} as Record<RowId, number>;
    for (const row of ROW_IDS) {
      const progress = coverage.rows[row];
      yawSpanPerRowDeg[row] = round(progress?.yawSpanDeg ?? 0, 2);
      rowCompletion[row] = progress && progress.requiredStations > 0 ? round(Math.min(1, progress.filledStations / progress.requiredStations), 4) : 0;
    }

    const meanOmega = omegaSamples.length > 0 ? omegaSamples.reduce((a, b) => a + b, 0) / omegaSamples.length : 0;
    const meanFps = fpsSamples.length > 0 ? fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length : 0;
    const minFps = fpsSamples.length > 0 ? Math.min(...fpsSamples) : 0;

    const centreMin = Number.isFinite(pitchCentreMin) ? pitchCentreMin : 0;
    const centreMax = Number.isFinite(pitchCentreMax) ? pitchCentreMax : 0;

    return {
      schema: "biosphere.capture.v1",
      createdAt: new Date().toISOString(),

      device: {
        userAgent,
        platform: detectPlatform(userAgent),
        // An explicit setDevice wins; otherwise the camera source is read here, at build
        // time, so late-applied control locks are reflected. `virtualLensOnly` defaults to
        // TRUE when nothing is known: an unproven lens must never be uploaded as proven.
        cameraLabel: device?.cameraLabel ?? live?.label ?? null,
        virtualLensOnly: device?.virtualLensOnly ?? live?.virtualLensOnly ?? true,
        trackSettings: device?.trackSettings ?? live?.settings ?? {},
        appliedControls: device?.appliedControls ?? live?.appliedControls ?? {},
        shutterCap: live?.shutterCap ?? null,
        modelHints,
      },

      optics: {
        hfovDeg: lens.hfovDeg,
        vfovDeg: lens.vfovDeg,
        lensSource: lens.source,
        lensChangedMidCapture: lensChanged,
      },

      recording: {
        segments: segments.length,
        totalDurationMs: Math.round(totalDurationMs),
        width,
        height,
        screenAngleDeg: recordingContext?.screenAngleDeg ?? 0,
        // Decoder truth, not the constraint: on iOS getSettings() reports the sensor-native
        // landscape pair while the file carries a rotation tag.
        portrait: height > 0 && width > 0 ? height > width : true,
        requestedBitrateBps: recordingContext?.requestedBitrateBps ?? RECORDING.targetBitrateBps,
        achievedBitrateBps: totalDurationMs > 0 ? Math.round((totalBytes * 8) / (totalDurationMs / 1000)) : 0,
        mimeType: recordingContext?.mimeType ?? longest?.mimeType ?? "",
        meanFps: round(meanFps, 2),
        minFps: round(minFps, 2),
      },

      sweep: {
        yawSpanPerRowDeg,
        pitchCentreMinDeg: round(centreMin, 2),
        pitchCentreMaxDeg: round(centreMax, 2),
        // Frame EDGE reach is what poles._CAP_DEG = 60 actually gates on.
        pitchReachMinDeg: round(centreMin - lens.vfovDeg / 2, 2),
        pitchReachMaxDeg: round(centreMax + lens.vfovDeg / 2, 2),
        meanAngularVelocityDegPerS: round(meanOmega, 2),
        p95AngularVelocityDegPerS: round(percentile(sortedOmega, 0.95), 2),
        fractionOverSpeedCap: recordedPushes > 0 ? round(overSpeedPushes / recordedPushes, 4) : 0,
        rowCompletion,
      },

      translation: {
        medianPivotRadiusM: sortedPivot.length > 0 ? round(percentile(sortedPivot, 0.5), 3) : null,
        p90PivotRadiusM: sortedPivot.length > 0 ? round(percentile(sortedPivot, 0.9), 3) : null,
        medianDriftMPerS: round(percentile(sortedDrift, 0.5), 4),
        maxDriftMPerS: round(maxDrift, 4),
        walkingFraction: recordedPushes > 0 ? round(walkingPushes / recordedPushes, 4) : 0,
        motionGateRan,
      },

      image: {
        meanLuma: lumaCount > 0 ? round(lumaSum / lumaCount, 2) : 0,
        minLuma: Number.isFinite(lumaMin) ? round(lumaMin, 2) : 0,
        meanClippedFraction: lumaCount > 0 ? round(clippedSum / lumaCount, 5) : 0,
        maxClippedFraction: round(clippedMax, 5),
        fractionOverSmearBudget: smearCount > 0 ? round(smearOver / smearCount, 4) : 0,
      },

      coverage: {
        coveredFraction: round(coverage.coveredFraction, 4),
        horizonFraction30: round(coverage.horizonFraction30, 4),
        horizonFraction60: round(coverage.horizonFraction60, 4),
        gaps: coverage.gaps,
        coveredMaskBase64: coveredMask ? toBase64(coveredMask) : "",
      },

      track: {
        mediaTime: trackMediaTime,
        quat: trackQuat,
        omega: trackOmega,
        segment: trackSegment,
      },

      verdict,
    };
  }

  function toFile(telemetry: CaptureTelemetry): File {
    const body = JSON.stringify(telemetry);
    if (typeof File === "function") {
      return new File([body], TELEMETRY_FILENAME, { type: "application/json" });
    }
    // Environments with Blob but no File (some workers, older Node). The upload path only
    // reads `name`, `type` and the body, so a tagged Blob is a faithful stand-in.
    const blob = new Blob([body], { type: "application/json" });
    return Object.assign(blob, { name: TELEMETRY_FILENAME, lastModified: Date.now(), webkitRelativePath: "" }) as File;
  }

  function reset(): void {
    segmentOffsetS = 0;
    segmentAnchorT = null;
    lastTrackAt = -Infinity;
    trackMediaTime.length = 0;
    trackQuat.length = 0;
    trackOmega.length = 0;
    trackSegment.length = 0;
    segmentIndex = 0;
    omegaSamples.length = 0;
    pivotSamples.length = 0;
    driftSamples.length = 0;
    fpsSamples.length = 0;
    recordedPushes = 0;
    overSpeedPushes = 0;
    walkingPushes = 0;
    motionGateRan = false;
    pitchCentreMin = Number.POSITIVE_INFINITY;
    pitchCentreMax = Number.NEGATIVE_INFINITY;
    lumaSum = 0;
    lumaCount = 0;
    lumaMin = Number.POSITIVE_INFINITY;
    clippedSum = 0;
    clippedMax = 0;
    smearOver = 0;
    smearCount = 0;
    maxDrift = 0;
    coveredMask = null;
    lensChanged = false;
  }

  return {
    push,
    markSegment,
    restartSegment,
    build,
    toFile,
    reset,
    setModelHints: (hints) => {
      modelHints = hints;
    },

    setDevice: (info) => {
      device = info;
    },
    setLens: (next) => {
      lens = next;
    },
    noteLensChange: () => {
      lensChanged = true;
    },
    setRecordingContext: (context) => {
      recordingContext = context;
    },
    setCoveredMask: (mask) => {
      // Copied: the live grid is mutated in place by the coverage stamper.
      coveredMask = new Uint8Array(mask);
    },
    sampleCount: () => trackMediaTime.length,
  };
}
