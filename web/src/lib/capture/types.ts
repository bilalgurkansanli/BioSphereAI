/**
 * BioSphere AI — guided capture contract.
 * =====================================================================================
 *
 * This file is the single source of truth for the in-app guided capture flow. Four
 * builders (S sensors, C coverage, V visuals, P page/flow) implement against it in
 * parallel and never negotiate directly; if two of them disagree, this file decides.
 *
 * NOTHING HERE IS INVENTED. Every number carries the derivation or the platform fact it
 * came from. Three recon reports fed it:
 *   - the engine-derived numeric targets (read out of engine/biosphere/pano/*.py),
 *   - the browser capability survey (iOS Safari vs Chrome Android),
 *   - the UX design.
 * Where they disagreed, the resolutions are recorded in CONTRACT DECISIONS below.
 *
 * -------------------------------------------------------------------------------------
 * WHY THIS FEATURE EXISTS
 * -------------------------------------------------------------------------------------
 * The stitching engine is geometrically exact on a proper capture (0.403 deg pose error,
 * 0.042 deg loop closure). Real user captures fail for four reasons the camera UI can
 * prevent, all measured:
 *   1. TRANSLATION. Rotational purity residual: 0.98 px nodal, 2.3-2.4 px careful
 *      hand-held, 3.6 px real bedroom capture (ceiling beams visibly bend), 21.2 px
 *      walking (pipeline refuses outright).
 *   2. VERTICAL COVERAGE. The real capture reached only -70.8 to +45.6 deg elevation;
 *      all three gaps sat in the +25..+40 deg band; coverage 0.74 instead of ~0.90.
 *   3. COMPRESSION. WhatsApp re-encoded 1080x1920 to 478x850 at 1816 kbps.
 *   4. BLUR / DROPPED FRAMES / BLOWN-OUT FRAMES.
 * Every threshold below exists to stop one of those four.
 *
 * -------------------------------------------------------------------------------------
 * CONTRACT DECISIONS (where the three reports disagreed)
 * -------------------------------------------------------------------------------------
 * D1. ROW ELEVATIONS: +29 / 0 / -29, not the UX draft's +35 / 0 / -35.
 *     At +/-35 the cross-row vertical overlap is (69-35-1.88)/69 = 0.465, well under the
 *     engine's own design overlap (PipelineConfig.target_overlap = 0.55) on the pose
 *     graph's weakest links — cross-row pairs have neither neighbour_radius nor loop
 *     closure to fall back on. At +/-29 it is 0.552. See ROW_PLAN.
 *
 * D2. YAW STATION SPACING: 15 deg (24 stations/row), not the UX draft's 30 deg (12).
 *     The tightest keyframe bin any quality preset asks for is FAST's 43*0.35 = 15.05
 *     deg, so 15 deg guarantees a candidate frame in every bin for every preset. It is
 *     also exactly 3 cells of coverage.py's 72x36 (5 deg) grid, so the phone's coverage
 *     paint and the server's coverage report share one lattice. Screen clutter is not a
 *     regression: 43 deg hFOV / 15 deg = 2.9 stations visible at once, versus 1.4 at 30
 *     deg spacing — the guidance is denser but each ring is still isolated.
 *
 * D3. SPEED THRESHOLDS: green <= 24, amber 24-28, red > 31 deg/s — not the UX draft's
 *     green 12-38 / amber 38 / red 45. 45 deg/s is ABOVE the extraction wall
 *     (extract_fps 8 x bin 19.35 / 5 candidates = 30.96 deg/s), so the UX numbers would
 *     have coached users straight into blurred keyframes and chain breaks.
 *
 * D4. "GREAT" COVERAGE THRESHOLD: 0.868, not the UX draft's 0.90. The physical ceiling of
 *     the +29/0/-29 plan is sin(63.5 deg) = 0.8949, so a 0.90 gate is unreachable — every
 *     perfect capture would have been graded "gaps". But the gate is not set against the
 *     ceiling either: the coaching calls a 3-deg row error GREEN, and a green row measures
 *     0.8700. A gate above that refuses captures the app itself certified while they were
 *     being shot. See COMPLETENESS.coveredFraction for the measured curve.
 *
 * D5. QUALITY MODE: the guided flow offers `balanced` and `studio` only. FAST is
 *     extract_fps 5 with a 15.05 deg bin, giving omega_max = 15.05 deg/s — a 24-second
 *     turn, 1.6x slower than the coached pace. A capture paced at 24 deg/s delivers only
 *     3.1 candidates per FAST bin and 4.3% of bins fall back to a blurred keyframe.
 *
 * D6. BITRATE: request 20 Mbps (engine's 0.25 bpp floor at 1080x1920x30), not the
 *     browser report's 10-12 Mbps convenience figure. Safari honours a request within
 *     10-15%; Chrome's 2.5 Mbps default does NOT scale with resolution and must always be
 *     overridden explicitly. Accept >= 12 Mbps measured, warn below.
 *
 * D7. `correction` IS NOT A STATE. The UX design listed it among the states; it is a
 *     takeover layer that can own the border, the caption slot and the haptics from
 *     inside any row state. It is modelled as `CaptureSession.activeWarning`, so a
 *     warning never loses the row's progress. Likewise `rotate_portrait` is an interrupt
 *     flag (`CaptureSession.portraitBlocked`), not a node, so nothing is lost on rotation.
 *
 * -------------------------------------------------------------------------------------
 * FILES OUTSIDE THIS CONTRACT THAT MUST NOT BE TOUCHED
 * -------------------------------------------------------------------------------------
 * web/src/lib/{types,api,i18n,utils}.ts, web/src/app/globals.css, next.config.ts,
 * web/src/app/layout.tsx. This file imports from them and never mutates them.
 */

import type { RefObject } from "react";
import type { CaptureKind, Locale, QualityMode } from "@/lib/types";
import type { DeviceModelHints } from "./device";

// =====================================================================================
// 0. ANGLE CONVENTIONS AND PRIMITIVES
// =====================================================================================

/**
 * ALL ANGLES IN THIS CONTRACT ARE DEGREES. The engine's own types.ts uses radians; the
 * three recon reports are entirely in degrees, and mixing the two is exactly the kind of
 * silent bug that produces a plausible-looking but geometrically wrong panorama. Convert
 * once, at the telemetry boundary, using `toEngineAngles`.
 *
 *   yawDeg   0..360, 0 = the heading at the moment recording started (the loop-closure
 *            origin), increasing CLOCKWISE seen from above, i.e. turning to your right.
 *            Chosen to match iOS `webkitCompassHeading`, which also runs clockwise; the
 *            spec's `alpha` runs counter-clockwise, so heading ~= 360 - alpha.
 *   pitchDeg -90..+90, positive = optical axis ABOVE the horizon. Matches the engine.
 *   rollDeg  -180..+180 about the optical axis, 0 = phone's long axis vertical in
 *            portrait. Read from the gravity vector, never from integrated gyro.
 */
export type Degrees = number;

/**
 * Quaternion as [x, y, z, w], THREE.Quaternion's component order.
 *
 * The guidance layer is driven by quaternions and NEVER by raw Euler angles. The spec's
 * intrinsic Z-X'-Y'' triple is singular at beta = 90 deg, where the alpha and gamma axes
 * coincide and the platform can no longer separate them: verified, (alpha 10, beta 90,
 * gamma 0) and (alpha 0, beta 90, gamma -10) both land on the same yaw, and equal and
 * opposite alpha and gamma cancel exactly.
 *
 * beta = 90 is the phone held UPRIGHT IN PORTRAIT WITH THE CAMERA AT THE HORIZON — pitch 0,
 * the LEVEL row. It is not the up row: beta 119 gives pitch +29, beta 61 gives -29, and beta
 * 179 gives +89. (An earlier version of this comment claimed beta = +/-90 was "the phone
 * pointed straight up ... precisely the +25..+40 deg band where all three measured coverage
 * gaps sat". That is false, and it pointed the reasoning at the wrong row.)
 *
 * So the ill-conditioned pose is the FIRST row — the one `setOrigin()` defines the
 * loop-closure origin from, and the reason phone compasses are visibly jittery held
 * vertically. A ring built on Euler jitters and flips there; the quaternion makes the
 * geometry well defined, but nothing can undo the platform's own alpha/gamma noise at that
 * pose, which is why `useOrientation.setOrigin` averages the origin over a window instead of
 * sampling it, and why level-row yaw deserves LESS trust than the other two rows rather than
 * more.
 *
 * Kept as a bare tuple rather than a THREE type so `lib/capture/*` never imports `three`
 * and Builder C's unit tests run in plain Node.
 */
export type Quat = [x: number, y: number, z: number, w: number];

/** Right-handed 3-vector, metres or m/s^2 depending on the field. */
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** `performance.now()` milliseconds. Never `Date.now()` — the clock can step. */
export type Millis = number;

/** Traffic-light grading used by every meter in the UI. */
export type Grade = "green" | "amber" | "red";

// =====================================================================================
// 1. LENS GEOMETRY
// =====================================================================================

/**
 * The optical facts every other number in this file is derived from.
 *
 * hfov 43 deg across the SHORT edge of a portrait phone (26mm-equivalent). Cross-checked
 * against the engine's own focal search: validate.py recovers 38.1 deg on the real
 * Video A (bundle adjustment independently says 38.9) and 44.99 deg on the reference
 * 45-deg portrait sweep. 43 sits inside that band.
 *
 * DO NOT use select.DEFAULT_HFOV_DEG = 67.0 for any UI arithmetic. That is only the
 * engine's pre-solve prior and is wrong by ~1.56x for a portrait phone;
 * solve._prior_focal_px documents this at length.
 *
 * vfov at 1080x1920 = 2*atan(960/1370.9) = 70.0 deg. We plan with 69 as the conservative
 * value. Portrait is MANDATORY: it puts 69-70 deg on the long axis, 1.6x the landscape
 * vfov, and that is the only reason a three-row plan reaches 63.5 deg at all. In
 * landscape the three-row edge reach would be 29 + 21.5 = 50.5 deg, below the 60-deg
 * poles._CAP_DEG line, and no row plan would work.
 */
export interface LensGeometry {
  /** Degrees across the short (horizontal, in portrait) edge. */
  hfovDeg: Degrees;
  /** Degrees down the long (vertical, in portrait) edge. */
  vfovDeg: Degrees;
  /** Pixels per degree at a 1080-px frame width: 540/tan(21.5 deg) / (180/pi) = 23.93. */
  pxPerDegAt1080: number;
  /** How this geometry was obtained; `assumed` is the default and is fine. */
  source: "assumed" | "track-settings" | "engine-reported";
}

/**
 * f(1080) = 540/tan(21.5 deg) = 1370.9 px  ->  1370.9 * pi/180 = 23.93 px/deg.
 * f(640 ref) = 320/tan(21.5 deg) = 812.4 px -> 14.18 px/deg (the engine's 640-px
 * reference width, which is what validate.py's px thresholds are quoted at).
 */
export const DEFAULT_LENS: Readonly<LensGeometry> = {
  hfovDeg: 43,
  vfovDeg: 69,
  pxPerDegAt1080: 23.93,
  source: "assumed",
} as const;

/** The engine quotes its rotational-purity thresholds at a 640-px reference width. */
export const PX_PER_DEG_AT_640 = 14.18;

// =====================================================================================
// 2. THE CAPTURE PLAN — rows, stations, completeness
// =====================================================================================

/** Capture order. Level first (easiest, sets the loop origin), up second (the row users
 *  skip, so it must not be last when neck fatigue has set in), down last (most
 *  comfortable, and it ends with the user looking at the floor near their own feet,
 *  which is a natural stopping posture). */
export const ROW_IDS = ["level", "up", "down"] as const;
export type RowId = (typeof ROW_IDS)[number];

export interface RowPlan {
  id: RowId;
  /** Target elevation of the FRAME CENTRE, degrees. */
  targetPitchDeg: Degrees;
  /** Number of yaw stations in this row. */
  stationCount: number;
  /** Yaw spacing between stations, degrees. */
  stationSpacingDeg: Degrees;
  /** Stations that must fill before the row is considered done. */
  requiredStations: number;
  /** Yaw the row should sweep before its progress arc closes. */
  targetYawSpanDeg: Degrees;
}

/**
 * ROW CENTRES: +29 / 0 / -29. Derived, not assumed.
 *
 * Constraint 1 (overlap): cross-row spacing S <= vfov*(1 - target_overlap)
 *   = 69 * 0.45 = 31.05 deg, mirroring PipelineConfig.target_overlap = 0.55. Vertical
 *   links deserve at least the horizontal overlap, not less: the yaw chain has
 *   neighbour_radius = 3 plus loop closure as backup, a cross-row link has neither.
 * Constraint 2 (roll budget): a roll of rho tips the frame's top edge by
 *   (hfov/2)*tan(rho) = 21.5*tan(rho) at the horizontal extremes, straight out of the
 *   overlap. rho = 5 deg costs 1.88 deg, so S <= 31.05 - 1.88 = 29.17.
 * Constraint 3 (the pole filler's mandate): poles._CAP_DEG = 60.0 — only pixels above
 *   60 deg elevation may be replaced by the stereographic inpaint. Edge reach =
 *   S + vfov/2 must therefore be >= 60.
 *     1 row:             34.5 deg -> a 25.5-deg UNFILLABLE ring. Fails.
 *     2 rows at +/-15.5: 50.0 deg -> a 10.0-deg unfillable ring.  Fails.
 *     3 rows at +/-29:   63.5 deg -> clears 60 by 3.5 deg.        Passes.
 *   THREE is the minimum row count whose residual hole is legally fillable. That, not
 *   coverage percentage, is the reason for three rows.
 *
 * Achieved vertical overlap (69 - 29 - 1.88)/69 = 0.552, at the 0.55 design point.
 * Edge reach 63.5 deg; covered fraction sin(63.5 deg) = 0.8949; residual polar caps
 * 26.5 deg radius each = 5.25% of the sphere apiece, both above the 60-deg line so
 * poles.py fills them without complaint.
 *
 * WHY NOT +/-38: overlap drops to (69-38-1.88)/69 = 0.422, thirteen points under the
 * engine's design overlap, on the graph's weakest links. It buys reach 72.5 and coverage
 * 0.954 vs 0.895 — six points of coverage in a region poles.py synthesises anyway.
 * Wrong trade: solve.MIN_COMPONENT_FRACTION = 0.60 refuses a shattered graph outright,
 * while a 26.5-deg cap is filled silently.
 */
/**
 * How far below the horizon the whole three-row lattice sits. Rows are +21 / -8 / -37, not
 * +29 / 0 / -29.
 *
 * THE SPHERE IS NOT SYMMETRIC IN VALUE. Separation stays at `rowSeparationDeg` — the overlap
 * arithmetic above is untouched, every row simply aims lower by the same amount — so this
 * trades ceiling reach for floor reach one degree at a time. That trade is a product
 * decision, made deliberately and twice: a room's ceiling is usually one flat tone, where an
 * inpainted patch is indistinguishable from a photographed one, while the floor carries the
 * flooring, the rug, the furniture legs and the room's actual scale, where invented pixels
 * read as a smear.
 *
 * WHAT 8 DEGREES COSTS, MEASURED. Frame-edge reach becomes +55.5 / -71.5 instead of the
 * symmetric +/-63.5. The floor gains five degrees of real photography. The ceiling gives up
 * more than it looks:
 *
 *   - The photographed band no longer reaches `poles._CAP_DEG = 60`, so the ring from 55.5
 *     to 60 falls to `poles.fill_uncovered`'s equirect pass rather than the conformal cap.
 *     Not a hole — `fill_uncovered` closes every uncovered region — but invented, not shot.
 *   - The synthesised zenith cap grows from 6.5% of the sphere to 8.8%. ANYTHING REAL UP
 *     THERE IS INVENTED: a chandelier, a ceiling fan, exposed beams, a skylight. This plan
 *     is correct for a flat ceiling and wrong for an interesting one.
 *   - Every gate below moved with it. `pitchMaxDeg` 60 -> 55, `horizonFraction60` 0.98 ->
 *     0.96, `coveredFractionCeiling` 0.8937 -> 0.8856 and the `coveredFraction` gate
 *     0.868 -> 0.855, all measured against the shipped coverage code rather than derived.
 *     The gate had to move: at 0.868 a capture the coaching grades GREEN measures 0.8611
 *     and would be refused, which is the product moving its own goalposts — the exact
 *     failure `coveredFraction` was set to avoid.
 *
 * Do not push this further without re-measuring all four. The next stop is the horizon band
 * the viewer actually looks at: the level row alone already covers only 0.946 of |elev| <= 30
 * at this bias, against 0.99 when it sat on the equator.
 */
export const ROW_BIAS_DEG: Degrees = -8;

export const ROW_PLAN: Readonly<Record<RowId, Readonly<RowPlan>>> = {
  level: {
    id: "level",
    // ROW_BIAS_DEG below the horizon, not on it. The horizon band coverage.py cares about
    // (|elev| <= 30, what the viewer actually looks at) is still covered by this one row's
    // sharpest, least-distorted centre: at vfov 69 a -3 centre spans -37.5..+31.5.
    targetPitchDeg: -8,
    stationCount: 24,
    stationSpacingDeg: 15,
    requiredStations: 22,
    targetYawSpanDeg: 370,
  },
  up: { id: "up", targetPitchDeg: 21, stationCount: 24, stationSpacingDeg: 15, requiredStations: 22, targetYawSpanDeg: 370 },
  down: { id: "down", targetPitchDeg: -37, stationCount: 24, stationSpacingDeg: 15, requiredStations: 22, targetYawSpanDeg: 370 },
} as const;

/**
 * The vertical field of view a frame of this shape actually has.
 *
 * `hfovDeg` is a property of the lens and cannot be read from the browser, so it stays
 * assumed. The RATIO between the two is not a property of the lens at all — it is the frame's
 * aspect ratio, which `getSettings()` reports exactly. Deriving one from the other is
 * therefore free accuracy, and the alternative is a constant that is right for one frame
 * shape and silently wrong for every other.
 *
 * MEASURED, AND THE REASON THIS FUNCTION EXISTS. `DEFAULT_LENS.vfovDeg = 69` is the value
 * for a 9:16 frame, and it was correct for as long as the capture delivered 9:16. Fixing the
 * resolution ladder changed the delivered frame to 4:3 (1080x1440) — more pixels, and a
 * genuinely shorter frame: 55.4 deg at the assumed 43 deg hfov, 61.0 at the 47.7 the engine
 * self-calibrated to on that capture. Nothing noticed. The lattice went on separating its
 * rows by 29 deg as though each frame were 69 deg tall, so cross-row overlap fell from the
 * designed 0.55 to 0.494 — and rows that share too little cannot be held rigid against each
 * other, which arrives in the panorama as straight lines bending across a row boundary.
 */
export function vfovForFrame(hfovDeg: Degrees, shortEdgePx: number, longEdgePx: number): Degrees {
  if (!(shortEdgePx > 0) || !(longEdgePx > 0)) return DEFAULT_LENS.vfovDeg;
  const halfH = (hfovDeg * Math.PI) / 360;
  return (2 * Math.atan(Math.tan(halfH) * (longEdgePx / shortEdgePx)) * 180) / Math.PI;
}

/** The live lens for a delivered frame: assumed hfov, measured shape. */
export function lensForFrame(width: number, height: number): LensGeometry {
  const shortEdge = Math.min(width, height);
  const longEdge = Math.max(width, height);
  return {
    hfovDeg: DEFAULT_LENS.hfovDeg,
    vfovDeg: vfovForFrame(DEFAULT_LENS.hfovDeg, shortEdge, longEdge),
    pxPerDegAt1080: DEFAULT_LENS.pxPerDegAt1080,
    source: "track-settings",
  };
}

/**
 * Runtime recomputation of the row separation, for a frame whose vfov is not 69.
 *
 *   S = min(vfov*0.45 - (hfov/2)*tan(5 deg), 31)
 *
 * At vfov 69 this gives 29.2 -> 29, unchanged. At the 4:3 frame's 55.4 it gives 23.
 *
 * THE `60 - vfov/2` FLOOR IS GONE, AND ITS REMOVAL IS THE POINT. That floor said "never
 * separate the rows so little that the top row stops reaching poles._CAP_DEG = 60", and it
 * OUTRANKED the overlap term — so on a 61-deg frame it pushed the separation UP to 29.5,
 * making the overlap worse in order to protect the pole reach. That is exactly the wrong
 * priority. Overlap is what holds the pose graph rigid: too little and the rows hinge and
 * the whole panorama bends. Pole reach only decides how much of the ceiling is inpainted
 * instead of photographed, on a surface the product has already decided it can concede (see
 * ROW_BIAS_DEG). A bent room is a defect; a synthesised ceiling is a trade.
 */
export function rowSeparationDeg(lens: LensGeometry = DEFAULT_LENS): Degrees {
  const overlapLimited = lens.vfovDeg * 0.45 - (lens.hfovDeg / 2) * Math.tan((5 * Math.PI) / 180);
  return Math.min(31, overlapLimited);
}

/** A single aiming reticle. `${row}:${index}` — stable across a session and a draft. */
export type StationId = `${RowId}:${number}`;

export interface Station {
  id: StationId;
  row: RowId;
  /** 0-based index within the row; yawDeg = index * stationSpacingDeg. */
  index: number;
  yawDeg: Degrees;
  pitchDeg: Degrees;
}

export interface StationState extends Station {
  filled: boolean;
  /** performance.now() at fill, or null. */
  filledAt: Millis | null;
  /** Accumulated dwell towards `stationDwellMs`, reset when the axis leaves the radius. */
  dwellMs: number;
}

/**
 * Completeness criteria. Every one of these is an engine gate, not a preference.
 */
export const COMPLETENESS = {
  /**
   * Solid-angle-weighted covered fraction, computed exactly the way coverage.py does it:
   * 72x36 grid, a cell counts as covered at >= 50% fill, cos(elev) weights.
   *
   * SET AGAINST WHAT THE COACHING CERTIFIES, NOT AGAINST THE PHYSICAL CEILING.
   *
   * The rule this gate exists to obey: a capture the guidance grades GREEN throughout must
   * never be told afterwards that regions are missing. Stations do not start going unfilled
   * until the rows are 9 deg off, so between 2 and 8 deg every ring in every row fills, every
   * row auto-advances, the rail reads 100% — and a gate set above what that capture measures
   * makes the product move its own goalposts. That is the "hala çekmediğin yerler var"
   * report, and it is the failure this number is chosen to make impossible.
   *
   * MEASURED AGAINST THE SHIPPED COVERAGE CODE, on the ROW_BIAS_DEG = -8 lattice:
   *
   *     perfect sweep                    0.8856   <- coveredFractionCeiling
   *     rows held 3 deg off (GREEN)      0.8611   <- the worst the coaching certifies
   *     gate                             0.855    <- 0.006 of real margin under it
   *
   * Symmetric row bias costs about 0.00806 of the sphere per degree, so the gate sits a
   * little under three quarters of a degree below the greenest failing case rather than on
   * top of it. The failing real capture that motivated all of this reported 0.74 and is
   * still refused by a wide margin.
   */
  coveredFraction: 0.855,
  /** The physical ceiling. Never grade against anything above this. See ROW_BIAS_DEG: the
   *  biased lattice reaches +55.5/-71.5. Analytically (sin 71.5 + sin 55.5)/2 = 0.8862;
   *  measured against the shipped coverage code on a perfect sweep, 0.88564. */
  coveredFractionCeiling: 0.8857,
  /** coverage.CoverageReport.horizon_fraction is measured over exactly |elev| <= 30. */
  horizonFraction30: 0.99,
  /**
   * |elev| <= 60 is everything `poles.py`'s conformal cap may NOT repair. It used to be
   * required at 0.98 because the lattice photographed all of it; at ROW_BIAS_DEG = -8 the
   * ring from 55.5 to 60 is no longer reached, which measures 0.9756 on a perfect sweep.
   * 0.96 keeps the gate meaningful — a real gap in the band still trips it — while
   * accepting the 2.4 points the row bias deliberately gave away. `fill_uncovered` closes
   * that ring in the equirect domain, so it arrives inpainted rather than black.
   */
  horizonFraction60: 0.96,
  /**
   * Frame-EDGE elevation reach, centre elevation +/- vfov/2. It is no longer the
   * `poles._CAP_DEG = 60.0` line: ROW_BIAS_DEG = -8 puts the nominal up-row edge at 55.5, so
   * requiring 60 would refuse every correct capture. 55 is what the plan actually reaches,
   * and it still catches the failure this gate was built for — the real bedroom capture hit
   * +45.6, ten degrees short of even this, and is refused.
   */
  pitchMaxDeg: 55,
  pitchMinDeg: -60,
  /** No uncovered arc wider than one 5-deg grid cell anywhere inside |elev| <= 60. */
  maxGapArcDeg: 5,
} as const;

/**
 * Yaw span per row. The wrap-around loop-closure pair is (last keyframe, first keyframe)
 * with separation |span - 360|. For it to verify at the same overlap as an ordinary
 * adjacent pair it needs |span - 360| <= hfov*(1 - 0.55) = 19.35 deg, i.e. span in
 * [340.6, 379.4]. 370 sits inside with 10 deg of overshoot for phase-correlation track
 * error, and gives the seam a second, better-exposed look.
 */
export const YAW_SPAN = {
  targetDeg: 370,
  acceptMinDeg: 360,
  acceptMaxDeg: 380,
  /** solve.MIN_YAW_SPAN_DEG — the engine refuses below this. Never approach it. */
  engineRefuseDeg: 200,
  /** solve.WARN_YAW_SPAN_DEG / validate.WARN_TURN_DEG. */
  engineWarnDeg: 300,
  /**
   * select._ROW_MIN_YAW_SPAN_DEG. A row abandoned under 180 deg is not counted as a row
   * at all: detect_sweep_pattern reads the capture as SINGLE_ROW and per_row_budget
   * doubles.
   */
  rowCountedMinDeg: 180,
} as const;

/**
 * How a station fills.
 *
 * `captureRadiusDeg` 9: great-circle distance from the frame centre to the station. At
 * 15 deg spacing the worst-case yaw distance to the nearest station is 7.5 deg, and the
 * amber row-pitch tolerance is 5 deg, so sqrt(7.5^2 + 5^2) = 9.01. Nine degrees therefore
 * leaves no dead band between stations while keeping the station well inside a 43x69 deg
 * frame. It is an aiming aid, not a frame: the continuous 30 fps recording oversamples
 * the engine's 8 fps extraction by 3.75x regardless.
 *
 * `dwellMs` 250: long enough that a whip-pan cannot bank a station, short enough that a
 * 24 deg/s sweep (which crosses the 18-deg-wide capture window in 750 ms) fills it
 * comfortably without stopping.
 */
export const STATION_RULES = {
  captureRadiusDeg: 9,
  dwellMs: 250,
  /**
   * How long the first ring of the UP and DOWN rows waits before it may fill.
   *
   * `dwellMs` is 250, which is right in the middle of a row: the phone is already moving at
   * a coached speed along a line it has been following, and a quarter of a second of proof
   * is all a station needs. Entering a NEW row is not that. The user has just tilted the
   * phone through 29 degrees to get here, so the first ring is reached at the end of a
   * deliberate re-aim rather than in the middle of a smooth sweep -- the hand is still
   * settling, and 250 ms of it is enough to bank the ring off the blur at the end of the
   * tilt. Reported from a real capture: the first frame of each new row "comes out rushed".
   *
   * Two seconds, and only on the first ring of a row: long enough for the tilt to damp out
   * and for the person to see it happening, short enough that it never becomes the thing
   * they are waiting for. `level` is exempt -- it is entered from a standing start with the
   * framing countdown already behind it, and the user says that one already feels right.
   */
  rowArmMs: 2000,
  /** Stations do NOT fill while any `stop`-severity warning is active — the user must not
   *  be able to bank footage the engine will refuse. See WARNING_BLOCKS_CAPTURE. */
  blockedBySeverity: "stop",
} as const;

/**
 * Optional zenith/nadir bonus. Two extra frames at +/-90, rolled 90 deg apart, close the
 * top cap: a single frame at +90 covers an inscribed 21.5-deg radius, two rolled frames
 * cover 26.5, which raises coverage to 0.947. Opt-in only — poles.py synthesises these
 * acceptably, which is exactly why it must never be a debt.
 */
export const BONUS_POLE_STATIONS: readonly { id: StationId; pitchDeg: Degrees; rollDeg: Degrees }[] = [
  { id: "up:100", pitchDeg: 90, rollDeg: 0 },
  { id: "up:101", pitchDeg: 90, rollDeg: 90 },
  { id: "down:100", pitchDeg: -90, rollDeg: 0 },
  { id: "down:101", pitchDeg: -90, rollDeg: 90 },
] as const;

/** Everything the plan module is configured with. Defaults are the shipped values. */
export interface CaptureConfig {
  lens: LensGeometry;
  rows: Readonly<Record<RowId, Readonly<RowPlan>>>;
  completeness: typeof COMPLETENESS;
  yawSpan: typeof YAW_SPAN;
  station: typeof STATION_RULES;
  recording: RecordingConstraints;
  thresholds: typeof THRESHOLDS;
  /** Guided flow never offers `fast`; see contract decision D5. */
  qualityMode: GuidedQualityMode;
  captureKind: CaptureKind;
  /** Coaching and haptics off/on; the visual channel is designed to work alone. */
  voiceGuide: boolean;
  sounds: boolean;
  haptics: boolean;
  locale: Locale;
}

/** `fast` is excluded by contract decision D5. */
export type GuidedQualityMode = Exclude<QualityMode, "fast">;

/**
 * Total capture budget: 3 rows x 370 deg / 24 deg/s = 46.25 s, plus two ~2 s row
 * transitions ~= 50 s. At extract_fps 8 that is ~400 extracted frames against
 * max_extracted_frames = 900, and 3 * ceil(370/19.35) = 60 keyframes, comfortably inside
 * min_keyframes = 8 / max_keyframes = 90.
 */
export const DURATION_BUDGET = {
  perRowSeconds: 15.4,
  rowTransitionSeconds: 2,
  totalSeconds: 50,
  /** validate.MIN_VIDEO_DURATION_S = 6.0 (and MIN_FRAMES = 6). Below this, refuse. */
  minSeconds: 6,
  /** Bounds the blast radius of a MediaRecorder failure or a backgrounding event. */
  maxSeconds: 120,
} as const;

// =====================================================================================
// 3. RECORDING CONSTRAINTS
// =====================================================================================

/**
 * What the recorder must achieve, and what it must refuse.
 *
 * SILENT DEGRADATION IS THE ENEMY HERE. `ideal` constraints never throw — they hand back
 * the nearest supported mode — and Chrome's default videoBitsPerSecond does not scale
 * with resolution, so a forgotten bitrate gives 1080p at 2.5 Mbps: the same class of
 * damage as the measured WhatsApp 478x850 re-encode, with no error anywhere.
 */
export interface RecordingConstraints {
  /**
   * 1080 on the SHORT edge, portrait, no exceptions.
   * The 8192-px equirect master samples 8192/360 = 22.76 px/deg. Source sampling is
   * width/hfov: 1080/43 = 25.12 px/deg (ratio 1.10 — 1080 portrait is the MINIMUM that
   * fills an 8K master without upsampling), 1920/43 = 44.65, 2160/43 = 50.23. WhatsApp's
   * 478/43 = 11.12 px/deg = 0.49x, i.e. that master was 2x upsampled — measured failure
   * mode 3, arithmetically.
   */
  minShortEdgePx: number;
  /** 4K only pays off at quality_mode = studio, whose max_frame_width is 2560. */
  preferredShortEdgePx: number;
  /**
   * 30 fps supplies 3.75x the 8 fps balanced extraction and 2.5x studio's 12. 24 fps
   * forces a 1/48 s shutter under the 180-degree rule and more blur. 60 fps pays only
   * when the room holds 1/120 s, in which case the adaptive speed cap doubles to 55.6.
   */
  targetFps: number;
  /** Below this the engine starts losing frames to blur; block and coach. */
  minFps: number;
  /**
   * 1080x1920x30 = 62.2 Mpx/s; 0.25 bpp H.264-equivalent = 15.6 Mbps -> ask 20 for
   * margin. Calibration from validate.py's free-homography floor: 0.59 px clean,
   * 1.19-1.26 px on the realistic/harsh fixtures (JPEG included), 1.60 px on
   * WhatsApp-grade Video B at 0.149 bpp. Compression roughly doubles the floor and eats
   * 1.2 px of the 10 px reject budget.
   */
  targetBitrateBps: number;
  /** ~0.19 bpp. Below this, warn loudly and record `bitrateShortfall` in telemetry. */
  minAcceptableBitrateBps: number;
  /**
   * ALWAYS call start(timeslice). Without it everything buffers into one Blob delivered
   * at stop(), and if stop() never cleanly fires — an incoming call ends the track on
   * iOS — the whole 90 s capture is lost.
   */
  timesliceMs: number;
  maxDurationMs: number;
  minDurationMs: number;
  /**
   * isTypeSupported() returning true has historically NOT guaranteed start() succeeds on
   * iOS. Wrap construction AND start() in try/catch and fall down this ladder on throw.
   * Prefer H.264 MP4 on BOTH platforms: it is the one format both produce, both
   * hardware-encode, and the engine's ffmpeg path already accepts. VP9 software encoding
   * will thermally throttle a phone over 90 s.
   */
  mimeLadder: readonly string[];
  /** Portrait is a hard UI gate: screen.orientation.lock() does not exist on iOS Safari
   *  at any version, so there is no API that can force it. */
  requirePortrait: true;
}

export const RECORDING: Readonly<RecordingConstraints> = {
  minShortEdgePx: 1080,
  preferredShortEdgePx: 2160,
  targetFps: 30,
  minFps: 24,
  /**
   * 12 Mbps, down from 20. The original figure came from 0.25 bpp at 1080x1920x30 — right
   * for image quality in the abstract, wrong for the device that has to produce it while
   * also running a camera preview, a 60 Hz guidance overlay and continuous sensor
   * processing. A real mid-range Android phone visibly stuttered at 20 Mbps.
   *
   * 12 Mbps is this contract's own `minAcceptableBitrateBps`, i.e. the rate below which we
   * already said quality suffers — so this spends the whole remaining margin on frame rate,
   * and nothing beyond it. Dropped frames cost keyframes outright, which is a far larger
   * quality loss than the compression difference between 12 and 20 Mbps.
   */
  targetBitrateBps: 12_000_000,
  minAcceptableBitrateBps: 12_000_000,
  timesliceMs: 1000,
  maxDurationMs: 120_000,
  minDurationMs: 6_000,
  mimeLadder: [
    "video/mp4;codecs=avc1.640028",
    "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
    "video/mp4",
    // H.264 in a WebM/Matroska container: the hardware MediaCodec path with the muxer
    // Chrome on Android has shipped for years. Measured on Android 10 / Chrome 151, every
    // mp4 rung above constructed and then produced zero bytes, and the ladder fell straight
    // to software VP9 at 6.8 of the 12 Mbps asked for, heating the phone and softening the
    // very texture the solver needs. These two rungs sit between them.
    "video/webm;codecs=h264",
    "video/x-matroska;codecs=avc1",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ],
  requirePortrait: true,
} as const;

/**
 * Physical rear-lens selection. THE highest-value camera mitigation.
 *
 * On multi-lens iPhones `facingMode:'environment'` resolves to a VIRTUAL device that the
 * OS switches between wide/ultra-wide/telephoto at will. Apple forum thread 776460
 * (iPhone 15 Pro, iOS 18.3.2) documents deviceId, facingMode:{exact} and zoom:1.0 all
 * being silently ignored on such a device. A mid-capture switch from 26mm to 13mm changes
 * the field of view the solver assumes, and — unlike a blurry frame — it does not look
 * like a failure. It looks like a subtly wrong panorama. rotation.py's hfov clamp is
 * [30, 110], so a jump to ~100 deg is not even caught.
 *
 * Two-phase open: getUserMedia once with facingMode:'environment' purely to unlock device
 * labels (labels and real deviceIds appear only AFTER a grant), enumerateDevices(), pick
 * the physical rear wide, stop the phase-1 track, reopen with that exact deviceId.
 *
 * NOTE ON `wide`: the browser report's reject list included /wide-angle/. We deliberately
 * do NOT reject plain "wide" — several Android OEMs label the MAIN lens "Wide Angle", and
 * rejecting it would leave us with no candidate. "Ultra wide" is already caught by
 * /ultra/, which is the case that actually matters.
 */
export const LENS_SELECTION = {
  accept: /back|rear|arka/i,
  reject: /ultra|dual|triple|telephoto|\btele\b|macro|depth|front/i,
  /** deviceIds are regenerated on every page load. NEVER persist them. */
  persistDeviceId: false,
  /** Poll getSettings() at this rate during recording to catch a silent lens switch. */
  configurationPollMs: 500,
} as const;

/**
 * Camera controls, feature-detected from getCapabilities() and each wrapped in try/catch.
 * The published support tables (including the W3C implementation-status doc, stale since
 * Feb 2023) are wrong about iOS torch and will be wrong about something else next
 * release. NEVER version-sniff.
 *
 * Android gets a genuine exposure lock; iOS does not — exposureMode, exposureTime and
 * exposureCompensation are entirely absent from iOS getCapabilities(). Measured failure
 * mode 5 (first and last keyframes blown out on a window, failing to link) cannot be
 * prevented by any API call on iPhone. The substitute is UI: the framing gate makes the
 * user start 40 deg off the light source so auto-exposure settles somewhere survivable.
 */
/** Outcome of `SHUTTER_CAP`, recorded into the sidecar. Times in seconds. */
export interface ShutterCapRecord {
  applied: boolean;
  reason: string;
  exposureBeforeS: number | null;
  exposureAfterS: number | null;
  isoBefore: number | null;
  isoAfter: number | null;
  lumaBefore: number | null;
  lumaAfter: number | null;
}

export interface CameraControlPlan {
  /** ~1 s of auto settle before any lock is applied. */
  settleMs: number;
  /** Android only. Android/Chrome 72+. */
  lockExposure: boolean;
  /** iOS since Safari 17.4, Android always. One of only two controls iOS gives us. */
  lockWhiteBalance: boolean;
  /** Android only. Everything in a room panorama is effectively at infinity. */
  lockFocus: boolean;
  /** Android only (Chrome 87+). A secondary pin against a logical camera drifting wide. */
  pinZoomToMin: boolean;
  /** Both platforms (iOS since 17.x, despite the support tables). Marginal but free. */
  offerTorch: boolean;
  /** Android only. Trade shutter time for gain at constant exposure; see `SHUTTER_CAP`. */
  capExposure: boolean;
}

/**
 * Locking is OFF, and that is a deliberate reversal of the original plan.
 *
 * The idea was sound — auto-exposure drifting through a sweep forces the solver to fit a
 * per-frame gain, and a lens hunting for focus mid-turn blurs keyframes. But the
 * implementation of a "lock" in this API is `exposureMode: "manual"`, which freezes the
 * sensor at whatever value it happens to hold *at that instant*. A camera one second old
 * has not finished converging, so what gets frozen is the dark frame it opened on.
 * Measured on a real Android phone: the preview, and therefore the recording, came out far
 * too dark to use. `focusMode: "manual"` at `focusDistance.max` pins the lens at infinity
 * on the same blind guess, in a room where everything is two to four metres away.
 *
 * The trade is one-sided once you ask which error is recoverable. Exposure and white
 * balance drift ARE recoverable: `engine/biosphere/pano/exposure.py` solves per-image,
 * per-channel gains exactly for this, and that stage is worth about 5 dB. A capture that is
 * uniformly too dark is recoverable by nobody.
 *
 * Zoom pinning stays on: it is not a lock but a LENS CHOICE, and picking the ultra-wide by
 * accident changes the field of view the solver assumes — the one camera error that does
 * not look like an error.
 */
export const CAMERA_CONTROLS: Readonly<CameraControlPlan> = {
  settleMs: 1000,
  lockExposure: false,
  lockWhiteBalance: false,
  lockFocus: false,
  pinZoomToMin: true,
  offerTorch: true,
  capExposure: true,
} as const;

/**
 * THE FOURTH ATTEMPT AT THE SHUTTER, and what makes it different from the three above.
 *
 * Attempt 1 froze whatever the sensor held; 2 asked for a shorter time and a compensating
 * gain in one set and trusted the API's success; 3 added an open-loop luma watchdog that
 * doubled gain and blew out. All three were open loops against a sensor that knows the
 * room. This one is a closed loop that keeps the PRODUCT of the phone's own metering:
 *
 *   1. Only after auto-exposure has settled, read (t_auto, iso_auto). If t_auto is already
 *      inside the budget there is nothing to do.
 *   2. Ask for t_new inside [preferred, max] and iso_new = iso_auto * t_auto / t_new -- the
 *      same total exposure the phone chose -- in ONE advanced constraint set.
 *   3. READ BACK both values. If either did not land within tolerance, go straight back
 *      to continuous auto-exposure. The API reporting success is not evidence.
 *   4. Measure the preview luma before and after; servo the ISO in third-stop steps toward
 *      the auto baseline, a few steps at most; if it still sits more than `abortEv` away,
 *      go back to continuous. The failure mode is a second of odd preview, never a capture.
 *
 * Why it is worth a fourth try: on the reference capture the phone chose 58 ms at ISO 50
 * in a 150-luma room, and 48 % of the frames overshot the smear budget at a mere 9 deg/s.
 * The same exposure at 14 ms needs ISO 210 -- no noise penalty on any modern sensor -- and
 * turns 11 px of smear into 2.5.
 */
export const SHUTTER_CAP = {
  /** Longest exposure to accept, seconds: 11 px of smear at the coached speed. */
  maxExposureS: 0.028,
  /** What to ask for when the ISO headroom allows it. */
  preferredExposureS: 0.014,
  /** Read-back tolerance on exposure time and ISO, as a fraction. */
  readbackTolerance: 0.25,
  /** Servo settling and step sizes. */
  settleMs: 500,
  stepMs: 350,
  maxSteps: 4,
  /** Accept within this many stops of the auto baseline; give up beyond `abortEv`. */
  toleranceEv: 0.35,
  abortEv: 0.7,
} as const;

/**
 * WHAT THE FOURTH ATTEMPT ACTUALLY DID, ON THE USER'S OWN PHONE, 2026-09-06.
 *
 * The closed loop above shipped and a real capture reported, in its own sidecar:
 *
 *     applied: false, reason: "servo_failed"
 *     exposureBefore 1/24 s at ISO 50, luma 138.8
 *     exposureAfter  1/71 s at ISO 149, luma 9.7
 *
 * Read the arithmetic: three times less time against three times the gain should have held
 * the picture where it was. It came back FOURTEEN times darker — 3.84 stops — which means
 * this sensor's reported `iso` is not the gain that is actually applied. Auto-exposure was
 * doing something else as well (an analogue gain, a tone curve, or both) and stopped doing
 * it the moment the mode went manual. The servo then could not climb back: its 1/3-stop
 * steps over four tries reach 1.33 stops, so it correctly gave up and reverted, and the user
 * got the phone's own exposure with about two seconds of settling wasted.
 *
 * A WIDER SERVO IS NOT THE FIX AND WOULD BE THE FIFTH ATTEMPT. Closing 3.84 stops from ISO
 * 149 needs ISO 2100; even where the ceiling allows it, that is a frame of noise, and noise
 * costs the feature detector more than smear does. The through-line from the four attempts
 * is unchanged: `applyConstraints` cannot move time and gain as one atomic trade, and on a
 * sensor whose `iso` does not mean what it says there is nothing to servo against.
 *
 * So the cap stays as it is: it TRIES, it measures, it reverts within two seconds when the
 * phone will not cooperate, and it writes down why. The engine turns that record into the
 * one thing the user can act on — more light, or a slower sweep.
 */
export const SHUTTER_CAP_FIELD_RESULT = {
  device: "vivo Y22s (V2206)",
  stopsLostWhenManual: 3.84,
  outcome: "servo_failed, reverted to the phone's own exposure",
} as const;

/**
 * THE EXPOSURE IS THE PHONE'S. THIS IS A DIAGNOSTIC THRESHOLD, NOT A CONTROL TARGET.
 *
 * MEASURED ON A REAL CAPTURE. The phone's telemetry reported `exposureTime: 400` — 40 ms a
 * frame. At the coached 24 deg/s that is 0.96 deg of rotation smeared into every frame, ~15
 * px at the detector's working scale, several times the octave-0 limit. The frames the
 * solver dropped, the sector where the pose graph went sparse, and the visible seams in the
 * delivered panorama all sat in the same place. The measurement is real and 12 ms is the
 * budget the smear arithmetic wants.
 *
 * WE STILL DO NOT SET IT. Three attempts to take the exposure away from auto-exposure all
 * made the picture worse on the user's own phone, and the third made it worst of all:
 *
 *   1. `exposureMode: "manual"` to hold the metered value — froze whatever the sensor
 *      guessed and came back far too dark.
 *   2. A cap that shortened the exposure and asked for compensating gain in the same
 *      constraint set — the gain does not have to land, and when it did not the frame was
 *      up to 3.3x darker with the API reporting success.
 *   3. The cap plus a luma watchdog that raised gain when the picture went dark — in a dim
 *      room (the phone's own AE chose 1/25 s at ISO 1321) the shortened exposure is dark by
 *      construction, so the watchdog kept doubling gain, overshot correct exposure in a
 *      single coarse step, pinned at the ceiling and left the preview blown to white. It
 *      then stopped, because a blown frame is not a dark one.
 *
 * The through-line: `applyConstraints` has no shutter-priority, no "expose no longer than",
 * and no way to move time and gain as one atomic trade. Every expressible approximation is
 * an open loop against a sensor that knows the room better than we do — and the phone's own
 * camera app, metering the same room, produces a correctly exposed picture every time.
 *
 * So the smear budget is defended by COACHING instead: the sweep-speed prompt already tells
 * the user to slow down when the light is low, which reduces the smear without touching the
 * exposure at all. This constant survives as the threshold the debug HUD flags, so the cost
 * stays visible on the device even though nothing acts on it.
 *
 * Do not add a fourth attempt without a capability the spec does not currently have.
 */
export const MAX_EXPOSURE_SECONDS = 0.012;

// =====================================================================================
// 4. SENSOR SAMPLE TYPES
// =====================================================================================

/** Where a heading came from, and therefore how much to trust it. */
export type OrientationReference =
  /** Earth-referenced. iOS `webkitCompassHeading`, Android `deviceorientationabsolute`. */
  | "absolute"
  /** Gyro-relative to wherever the device was when listening started. Smooth, drifts. */
  | "relative"
  /** No orientation sensor at all — RAIL MODE, yaw comes from camera optical flow. */
  | "flow";

/**
 * One orientation reading. Produced by Builder S's `useOrientation`, consumed by C and V.
 *
 * PLATFORM ASYMMETRY THIS TYPE EXISTS TO HIDE:
 *  - iOS has NO `deviceorientationabsolute` (never fires) and needs a gesture-gated
 *    `DeviceOrientationEvent.requestPermission()`; its plain `deviceorientation` alpha is
 *    relative to an arbitrary start reference and drifts. Absolute heading comes only
 *    from the non-standard `webkitCompassHeading`, which runs CLOCKWISE while the spec's
 *    alpha runs counter-clockwise (heading ~= 360 - alpha).
 *  - Android has no permission call, and `deviceorientationabsolute` is earth-referenced
 *    but noisier.
 * The hook normalises both to ONE quaternion stream, converting alpha/beta/gamma via the
 * spec's intrinsic Z-X'-Y'' order to a matrix and then to a quaternion — the same
 * conversion the VR mode uses (`web/src/components/vr/orientation.ts`, tested against three).
 */
export interface OrientationSample {
  t: Millis;
  /** Camera-frame quaternion. THE source of truth for all guidance geometry. */
  q: Quat;
  /** Derived for display and for coverage stamping ONLY. Never for interpolation. */
  yawDeg: Degrees;
  pitchDeg: Degrees;
  /** From the gravity vector, drift-free. Never from integrated gyro. */
  rollDeg: Degrees;
  reference: OrientationReference;
  /**
   * iOS `webkitCompassAccuracy`, typically ~10 deg; -1 when the magnetometer is
   * uncalibrated. Null on Android. Skip the drift correction and show a figure-8
   * calibration prompt when this is -1 or > 25.
   */
  headingAccuracyDeg: number | null;
  /** `screen.orientation.angle`, needed to un-rotate the device frame. */
  screenAngleDeg: Degrees;
}

/** Whether linear acceleration is real, reconstructed, or unavailable. */
export type AccelSource =
  /** `DeviceMotionEvent.acceleration` present — every iPhone, most Androids. */
  | "native"
  /**
   * MDN is explicit that `acceleration` is simply ABSENT when the hardware cannot remove
   * gravity — true on cheap gyro-less Androids. Reconstructed by low-pass filtering
   * `accelerationIncludingGravity` (one-pole, alpha ~0.8) to estimate gravity and
   * subtracting it.
   */
  | "derived"
  /** Neither usable. The motion gate MUST fail safe: see MOTION_GATE_FALLBACK. */
  | "none";

/**
 * One motion reading. ~60 Hz on iOS; 20-60 Hz on Android (Chrome's Generic Sensor
 * implementation caps at 60 Hz and Android requests SENSOR_DELAY_UI).
 *
 * Human walking cadence is 1.5-2.5 Hz, so even a 20 Hz worst case is 8x Nyquist for step
 * detection. It is NOT enough to integrate for translation: double integration of
 * consumer accelerometer data diverges within about a second. NEVER try to measure
 * parallax by displacement. Detect walking by PATTERN.
 */
export interface MotionSample {
  t: Millis;
  /** deg/s. alpha = z, beta = x, gamma = y. Null on a device with no gyroscope. */
  rotationRate: Vec3 | null;
  /** |rotationRate|, deg/s. 0 when rotationRate is null. The blur meter and speed coach. */
  omegaDegPerS: number;
  /** m/s^2, gravity removed. Null only when accelSource === "none". */
  linearAccel: Vec3 | null;
  /** |linearAccel|, m/s^2. */
  accelMagnitude: number;
  /** m/s^2, the low-passed gravity vector. Source of roll and of absolute pitch. */
  gravity: Vec3 | null;
  accelSource: AccelSource;
  /** `event.interval`, ms. ~16 on iOS. */
  intervalMs: number;
}

/**
 * Per-frame image statistics, sampled off a downscaled canvas drawn from the preview.
 * Driven by `requestVideoFrameCallback` (Safari 15.4+, Chrome 83+) so `mediaTime` keys
 * these to the recorded file and the engine can align the sidecar to its own frames.
 */
export interface FrameQualitySample {
  t: Millis;
  /** `VideoFrameCallbackMetadata.mediaTime`, seconds into the media. */
  mediaTime: number;
  /**
   * Mean CIE L*, 0-255 as OpenCV scales it. The two real videos measured 163 and 144,
   * synthetics 117-132; validate.MIN_MEAN_BRIGHTNESS = 25 is the engine's cliff and is
   * explicitly labelled uncalibrated.
   */
  meanLuma: number;
  /**
   * Fraction of pixels at 0 or 255. extract._clipping_penalty computes
   * excess = max(0, clipped_fraction - 0.05), so 0.05 is the exact point where the engine
   * starts docking a frame's quality score — and quality is what select._best ranks on.
   */
  clippedFraction: number;
  /**
   * Exposure time in seconds. Read from `track.getSettings().exposureTime` on Android
   * (which reports it in 100-microsecond units — divide by 10000); on iOS it is
   * unavailable and must be ESTIMATED from meanLuma and the measured fps. See
   * `estimateExposureSeconds`.
   */
  exposureS: number;
  exposureSource: "reported" | "estimated";
  /** Real delivered frames per second, counted over FPS_WINDOW_MS. Not the constraint. */
  measuredFps: number;
  /**
   * Predicted directional smear at 1080 frame width, px:
   *   smear = omega_gyro (deg/s) * exposureS (s) * 23.93 px/deg.
   * ABSOLUTE and PHYSICAL, and available before the frame is even captured.
   */
  smearPx: number;
}

/**
 * Quadrant optical-flow divergence — the camera-side walking signal, and the only motion
 * signal that survives when the motion sensors are denied (RAIL MODE).
 *
 * Four 96x128 grayscale quadrants are phase-correlated at 5 Hz; the max pairwise shift is
 * normalised to a 640-px width. A pure rotation moves all four quadrants together; a
 * translation moves them by different amounts because they see different depths. That
 * differential is exactly what the engine's rotational-purity residual measures.
 */
export interface FlowSample {
  t: Millis;
  /** Max pairwise quadrant shift, px normalised to a 640-px width. */
  divergencePx: number;
  /** Mean quadrant shift, px at 640 width. Integrated into a yaw estimate in RAIL MODE. */
  meanShiftPx: number;
  /** Sign and magnitude of the yaw implied by meanShiftPx, deg since the last sample. */
  yawDeltaDeg: Degrees;
  /** False when the frame was too dark or too uniform for phase correlation to mean anything. */
  valid: boolean;
}

/** Everything the detectors read, gathered once per tick. */
export interface SensorFrame {
  t: Millis;
  orientation: OrientationSample | null;
  motion: MotionSample | null;
  quality: FrameQualitySample | null;
  flow: FlowSample | null;
  /** True while MediaRecorder is actually running. Detectors idle otherwise. */
  recording: boolean;
  /** Which row is being coached; drives the row-pitch detector's target. */
  activeRow: RowId | null;
}

// =====================================================================================
// 5. THRESHOLDS
// =====================================================================================

/**
 * Every live threshold, with its derivation. Builder C reads these in detectors.ts;
 * Builder V reads them only to draw scale marks. Nobody hard-codes a number anywhere else.
 */
export const THRESHOLDS = {
  /**
   * ANGULAR SPEED. Two independent walls land within 12% of each other:
   *  (a) EXTRACTION. select.select_keyframes uses bin_deg = hfov*(1 - target_overlap):
   *      balanced 43*0.45 = 19.35 deg, fast 15.05, studio 16.34. extract sets
   *      min_sharpness_percentile = 35, so select._best falls back to a blurred frame
   *      only when EVERY candidate in a bin is below the cut: P = 0.35^k.
   *      k=3 -> 4.3%, k=5 -> 0.53%. At ~60 keyframes that is 2.6 vs 0.3 expected blurred
   *      keyframes, and at 19.35 deg spacing the pose graph is effectively a CHAIN (an
   *      i->i+2 pair is 38.7 deg apart at 10% overlap, i->i+3 never verifies, despite
   *      neighbour_radius = 3), so one blurred keyframe is one potential component split
   *      and solve.MIN_COMPONENT_FRACTION = 0.60 refuses a capture that breaks too often.
   *      Take k = 5: omega_max = extract_fps * bin_deg / 5 = 8 * 19.35/5 = 30.96 deg/s.
   *  (b) MOTION BLUR at a plausible indoor 1/60 s: 27.8 deg/s (see `smear`).
   * 24 deg/s clears both, and is the engine's own shipped instruction
   * (validate.WalkingCaptureError: "one full circle in about fifteen seconds").
   */
  angularSpeed: {
    greenMaxDegPerS: 24,
    /** The motion-blur wall at t_exp = 1/60 s. */
    amberMaxDegPerS: 28,
    /** The extraction wall. Sustained breach means that yaw arc must be re-swept. */
    redDegPerS: 31,
    redSustainMs: 500,
    /** Low-pass the |rotationRate| magnitude over this before grading. */
    lowPassSeconds: 0.3,
    /**
     * ADAPTIVE CAP — the single most useful adaptive number in the UI.
     *   omega_max = 0.463 / t_exposure, clamped to [12, 31] deg/s
     * 0.463 deg is the octave-1 smear budget (11.08 px / 23.93 px/deg). Gives 55.6 deg/s
     * at 1/120 s (bright), 27.8 at 1/60 s, 13.9 at 1/30 s. A DIM ROOM MUST BE SWEPT AT
     * HALF SPEED.
     */
    adaptiveNumeratorDeg: 0.463,
    adaptiveMinDegPerS: 12,
    adaptiveMaxDegPerS: 31,
    /**
     * Reference only, never approach: select.estimate_frame_motion clamps any phase
     * correlation shift above 0.45*width to zero as aliasing.
     * 0.45*W = atan(0.9*tan(21.5 deg)) = 19.52 deg of yaw per extracted frame = 156 deg/s
     * at 8 fps. Past this the cumulative motion track integrates zeros and keyframe
     * selection collapses entirely.
     */
    aliasWallDegPerS: 156,
  },

  /**
   * MOTION SMEAR. A directional smear of length L is a box filter with
   * sigma_eq = L/sqrt(12) = 0.2887*L. SIFT's base scale is sigma = 1.6, so the octave it
   * detects at is wiped when sigma_eq exceeds it:
   *   octave 0 survives to L = 1.6/0.2887 = 5.54 px
   *   octave 1 survives to L = 11.08 px
   * Above 11.08 px the matcher runs on octave 2 only: 4x fewer keypoints, 2x worse
   * localisation against a 3.0 px RANSAC gate.
   *
   * DO NOT threshold Laplacian variance absolutely. extract.py rank-normalises it
   * precisely because it has no absolute meaning: the sphere-rendered nodal control
   * measures 8.6 and the unstitchable walkthrough measures 27 — a capture that stitches
   * perfectly is three times "blurrier" than one that cannot be stitched at all.
   */
  smear: {
    greenPx: 5.54,
    amberPx: 11.08,
    pxPerDegAt1080: 23.93,
  },

  /**
   * PIVOT RADIUS — how far the lens orbits the body. A quality slope, never a cliff, so
   * this detector must never block.
   *   r = |a_radial| / omega^2, from linearAccel projected on the optical axis and the
   *   gyro rate, averaged over 1.0 s and gated to omega > 15 deg/s.
   * At 24 deg/s (omega = 0.419 rad/s, omega^2 = 0.176) r = 0.35 m implies
   * a_radial = 61.5 mm/s^2 = 6.3 mg — above phone accelerometer noise (~1-3 mg RMS) once
   * averaged over 60 samples.
   *
   * Ladder interpolation from validate.py's own calibration (arm 0.00 m -> 0.98 px,
   * 0.28 m -> 2.43 px) puts GOOD_ROT_RESIDUAL_PX = 2.0 at
   * r = 0.28 * (2.0-0.98)/(2.43-0.98) = 0.20 m. 0.60 m measures 2.88 px; 1.20 m measures
   * 5.37 px, halfway to the 10.0 px reject.
   *
   * BE HONEST IN THE COPY. validate.py states plainly that no threshold honestly
   * separates a careful hand-held capture (2.34-2.43 px) from the bent-ceiling bedroom
   * turn (3.61 px). Promise "degraded" for r ~ 0.35 m and reserve "good" for r <= 0.20 m.
   */
  pivotRadius: {
    greenM: 0.2,
    /**
     * 0.35 m, not 0.5 m. The comment above already promises "degraded" at r ~ 0.35 m, but
     * the live warning was latched at 0.5 — so the one arc the engine measured as visibly
     * damaging (the real bedroom capture: r ~ 0.35 m, 3.61 px rotational residual, ceiling
     * beams bent) passed the whole capture without the user being told anything. Warning
     * at the boundary the copy already describes is the point at which coaching can still
     * change the outcome; warning at 0.5 m only confirms a capture already spoiled.
     */
    amberM: 0.35,
    gateOmegaDegPerS: 15,
    windowMs: 1000,
  },

  /**
   * TRANSLATION DRIFT — the statistic that actually separates walking from orbiting, and
   * the ONLY detector allowed to abort a capture.
   *
   * Net displacement over a 1.0 s sliding window from high-passed (0.3 Hz) double-
   * integrated linear acceleration. An orbit is zero-mean over a turn; a walk is not.
   *   0.35 m arm at 24 deg/s: chord over 1 s = 2*0.35*sin(12 deg) = 0.146 m/s
   *     -> validate.py measures 2.34 px, accepted.
   *   walking at 0.80 m/s -> 13.70 px, verdict `walking`, refused.
   * Geometric centre sqrt(0.146*0.80) = 0.342; 0.30 is chosen a touch conservative.
   *
   * The double integration diverges over long horizons — that is why the window is 1.0 s
   * and the signal is high-passed. It is a windowed pattern statistic, not odometry.
   */
  translationDrift: {
    warnMPerS: 0.15,
    stopMPerS: 0.3,
    stopSustainMs: 1500,
    windowMs: 1000,
    highPassHz: 0.3,
  },

  /**
   * STEP CADENCE — the second, independent walking signal.
   * Band-pass |linear acceleration| to 1.0-3.0 Hz over a 2 s sliding window. Human
   * walking cadence is 1.5-2.5 Hz and shows periodic peaks of roughly 1-3 m/s^2
   * peak-to-peak; turning on the spot shows high |rotationRate| with near-zero energy in
   * that band. A sinusoid's RMS is peak-to-peak/(2*sqrt(2)), so 1-3 m/s^2 pp maps to
   * 0.35-1.06 m/s^2 RMS: warn at the bottom of the walking range, stop at its middle.
   */
  stepCadence: {
    bandLowHz: 1.0,
    bandHighHz: 3.0,
    windowMs: 2000,
    warnRmsMs2: 0.35,
    stopRmsMs2: 0.7,
  },

  /**
   * QUADRANT FLOW DIVERGENCE — the camera-side walking signal. Survives RAIL MODE.
   * Four 96x128 grayscale quadrants phase-correlated at 5 Hz, max pairwise shift
   * normalised to 640-px width, exceeding 7 px for 0.8 s.
   */
  flowDivergence: {
    stopPx: 7,
    warnPx: 4,
    sustainMs: 800,
    rateHz: 5,
    quadrantWidthPx: 96,
    quadrantHeightPx: 128,
    normaliseToWidthPx: 640,
  },

  /**
   * ROLL about the optical axis. Cost to cross-row overlap is 21.5*tan(rho) deg at the
   * horizontal extremes. The whole budget above the 0.55 line at S = 29 is
   * 69 - 29 - 0.55*69 = 2.05 deg, so rho <= atan(2.05/21.5) = 5.4 deg. At 8 deg the loss
   * is 3.02 deg and overlap falls to 0.536, below design.
   * Read from the gravity vector — absolute, no gyro drift.
   */
  roll: { greenDeg: 3, amberDeg: 5, redDeg: 8, redSustainMs: 500 },

  /**
   * ROW PITCH ERROR from the active row's target elevation.
   * select._ROW_SEPARATION_DEG = 8.0: pitch_clusters merges histogram groups whose
   * centres are closer than 8 deg, and the clustering is density-based — it needs a
   * dwelt-on plateau, which a wandering row does not produce. A row that wanders 8 deg
   * either merges into its neighbour (three rows read as two, per_row_budget doubles) or
   * splits into two spurious clusters.
   */
  rowPitch: { greenDeg: 3, amberDeg: 5, redDeg: 8 },

  /**
   * CLIPPING. 0.05 is exactly where extract._clipping_penalty starts docking a frame's
   * quality score. The measured capture's first and last keyframes failed to link because
   * they pointed at a window and clipped.
   */
  clipping: { warnFraction: 0.05, blockFraction: 0.15, sustainMs: 500 },

  /**
   * BRIGHTNESS, mean CIE L* 0-255. validate.MIN_MEAN_BRIGHTNESS = 25 is the engine's
   * cliff and is explicitly labelled uncalibrated. The two real videos measured 163 and
   * 144, synthetics 117-132. 80 is below everything ever measured and 3.2x above the
   * cliff; 40 is 1.6x above it.
   */
  brightness: { warnMeanL: 80, blockMeanL: 40, engineCliffL: 25 },

  /**
   * DELIVERED FRAME RATE. iOS silently drops the sensor frame rate in low light because
   * auto-exposure lengthens exposure time, and there is no API to prevent it. Measure
   * with requestVideoFrameCallback rather than trusting the constraint. Below 24 fps,
   * block BEFORE recording rather than letting the validator reject it after upload.
   */
  frameRate: { minFps: 24, targetFps: 30, windowMs: 2000 },

  /**
   * FRAMING GATES — must all hold before recording auto-starts. No tap starts the
   * recording: a tap at t=0 jolts the phone, and that is exactly how the measured capture
   * lost its first keyframe to a blown, unlinkable frame.
   */
  framing: {
    /**
     * 15 deg/s, not 6, and held for 0.8 s rather than 1.2.
     *
     * The gate asks "is the operator done aiming?", and 6 deg/s answers a different
     * question: it is a tripod tolerance. A phone held at eye level in a human hand carries
     * several deg/s of tremor on its own, `omegaDegPerS` is the magnitude across all three
     * axes so that tremor adds rather than averages, and the old rule demanded the sum stay
     * under it for 1.2 unbroken seconds. On a real phone the countdown never completed, no
     * matter how still the person tried to be — and there is nothing more discouraging than
     * a product that asks for something and then refuses to acknowledge it.
     *
     * 15 deg/s still separates the two states cleanly, because the thing on the other side
     * is a deliberate turn at the coached 24 deg/s and a careless one at 30+.
     */
    steadyMaxDegPerS: 15,
    steadyHoldMs: 800,
    levelMaxRollDeg: 10,
    levelMaxPitchDeg: 12,
    /** Median luma; deliberately looser than the live `brightness.warnMeanL`. */
    minMedianLuma: 45,
    maxClippedFraction: 0.12,
    /** Suggested start heading offset from the brightest direction found while framing. */
    awayFromLightDeg: 40,
    countdownMs: 3000,
    /** After this long on a held gate, offer "Yine de başla" so an expert is never trapped. */
    forceStartAfterMs: 12_000,
  },
} as const;

/**
 * Adaptive angular-speed ceiling from the live exposure time.
 * omega_max = 0.463 / t_exposure, clamped to [12, 31] deg/s.
 */
export function adaptiveSpeedCapDegPerS(exposureSeconds: number): Degrees {
  const a = THRESHOLDS.angularSpeed;
  if (!Number.isFinite(exposureSeconds) || exposureSeconds <= 0) return a.redDegPerS;
  return Math.min(a.adaptiveMaxDegPerS, Math.max(a.adaptiveMinDegPerS, a.adaptiveNumeratorDeg / exposureSeconds));
}

/** smear_px = omega (deg/s) * t_exposure (s) * 23.93 px/deg, at 1080 frame width. */
export function predictedSmearPx(omegaDegPerS: number, exposureSeconds: number): number {
  return omegaDegPerS * exposureSeconds * THRESHOLDS.smear.pxPerDegAt1080;
}

/**
 * iOS exposes no exposureTime. Estimate it from the delivered frame rate and mean luma:
 * the sensor cannot expose longer than one frame period, and auto-exposure lengthens
 * toward that ceiling as the scene darkens. Deliberately pessimistic (it over-estimates
 * exposure, hence under-estimates the safe speed) because a too-slow coach costs seconds
 * and a too-fast one costs the capture.
 */
export function estimateExposureSeconds(measuredFps: number, meanLuma: number): number {
  const framePeriod = 1 / Math.max(1, measuredFps);
  // Bright room -> ~1/4 of the frame period; dark room -> the whole frame period.
  const darkness = Math.min(1, Math.max(0, (140 - meanLuma) / 140));
  return framePeriod * (0.25 + 0.75 * darkness);
}

// =====================================================================================
// 6. DETECTORS AND LIVE WARNINGS
// =====================================================================================

export type WarningCode =
  /** Walking. The only warning that may abort a capture. */
  | "walking"
  /** Sustained breach of the angular-speed red line. */
  | "too_fast"
  /** Predicted smear over the octave-1 budget — usually means the room is dim. */
  | "smear"
  /** The lens is orbiting the body: elbows out, or turning from the waist. */
  | "pivot_radius"
  /** Roll about the optical axis eating cross-row overlap. */
  | "roll"
  /** Frame centre has wandered off the active row's elevation. */
  | "row_pitch"
  | "too_dark"
  | "glare"
  /** Delivered fps has collapsed; on iOS the cause cannot be fixed, only reported. */
  | "low_fps"
  /** Zoom or active lens changed mid-capture. The solve fits ONE focal for the whole file. */
  | "lens_changed"
  /** Device left portrait. Recording is paused, never stopped. */
  | "landscape"
  /** Orientation degenerated or the compass went uncalibrated. */
  | "track_lost";

export type WarningSeverity =
  /** Furniture. Never takes over; the horizon line simply leans. */
  | "info"
  /** Thick dashed amber frame + one two-word caption. Progress continues. */
  | "warn"
  /** Red corner brackets pulsing inward. STATIONS STOP FILLING. */
  | "stop";

/**
 * Only ONE correction is ever shown, because a person mid-rotation cannot triage.
 * Highest priority first. Builder C's `rankWarnings` returns them sorted by this;
 * Builder V renders `warnings[0]` and nothing else.
 */
export const WARNING_PRIORITY: readonly WarningCode[] = [
  "landscape",
  "lens_changed",
  "walking",
  "too_fast",
  "smear",
  "low_fps",
  "glare",
  "too_dark",
  "pivot_radius",
  "track_lost",
  "row_pitch",
  "roll",
] as const;

/**
 * Severities at which station capture is suspended, so the user cannot bank footage the
 * engine will refuse. `stop` only — an amber warning must never cost earned progress.
 */
export const WARNING_BLOCKS_CAPTURE: readonly WarningSeverity[] = ["stop"] as const;

export interface Warning {
  code: WarningCode;
  severity: WarningSeverity;
  /** performance.now() at which the condition first held continuously. */
  since: Millis;
  /** The measured value that tripped it, in the detector's own unit. */
  value: number;
  /** The threshold it crossed, same unit. */
  threshold: number;
  /** Unit label for debug overlays only; never user-facing. */
  unit: "deg/s" | "px" | "m" | "m/s" | "m/s^2" | "deg" | "fraction" | "fps" | "L*";
  /** Copy key for the two-word verb. Builder P owns the value in copy.ts. */
  titleKey: CaptureCopyKey;
  /** Optional body line. */
  bodyKey: CaptureCopyKey | null;
  /** Spoken line, when the voice guide is on. */
  voiceKey: CaptureCopyKey | null;
  /** Distinct silhouette so the message survives greyscale, glare and a 200 ms glance. */
  icon: WarningIcon;
  /** Android vibration pattern, ms. Empty when haptics are off or unsupported. */
  haptic: readonly number[];
}

export type WarningIcon = "foot-crossed" | "snail" | "sun" | "moon" | "level" | "orbit" | "lens" | "rotate" | "compass";

/**
 * A detector's continuous output, independent of whether it has crossed into a warning.
 * The meters (speed comet, spirit level, parallax bar) read this every frame.
 */
export interface DetectorReadout {
  /** Low-passed |rotationRate|, deg/s. */
  omegaDegPerS: number;
  /** Live ceiling from `adaptiveSpeedCapDegPerS(exposure)`, deg/s. */
  speedCapDegPerS: number;
  speedGrade: Grade;
  smearPx: number;
  smearGrade: Grade;
  /** Metres, or null when omega is below the gate and the estimate is meaningless. */
  pivotRadiusM: number | null;
  pivotGrade: Grade;
  /** m/s of net windowed displacement. */
  translationMPerS: number;
  /** Band RMS in the 1-3 Hz step band, m/s^2. */
  stepBandRmsMs2: number;
  /** Max pairwise quadrant shift, px at 640 width. */
  flowDivergencePx: number;
  walkingGrade: Grade;
  rollDeg: Degrees;
  rollGrade: Grade;
  /** Signed error from the active row's target elevation. */
  rowPitchErrorDeg: Degrees;
  rowPitchGrade: Grade;
  meanLuma: number;
  clippedFraction: number;
  exposureGrade: Grade;
  measuredFps: number;
  /**
   * FAIL-SAFE FLAG. True when neither `acceleration` nor a usable
   * `accelerationIncludingGravity` reconstruction was available, i.e. the inertial
   * walking gate did not run. Absence of evidence of walking MUST NOT be treated as
   * evidence of standing still: the flow detector takes over, the coach falls back to
   * time-based pacing, and telemetry records `motionGateRan: false` so the server knows.
   */
  motionGateDegraded: boolean;
}

/**
 * What the motion gate does when the hardware cannot support it.
 * NEVER let the detector fail open — that would pass exactly the walking captures the
 * pipeline refuses at 21.2 px.
 */
export const MOTION_GATE_FALLBACK = {
  /** No `acceleration`: reconstruct via a one-pole low-pass on the with-gravity signal. */
  gravityLowPassAlpha: 0.8,
  /** No gyro either: disable the inertial gate, rely on flow, tag the upload. */
  tagTelemetryWhenDegraded: true,
  /** With no inertial gate, the flow detector's stop threshold tightens. */
  flowStopPxWhenDegraded: 5,
} as const;

// =====================================================================================
// 7. COVERAGE
// =====================================================================================

/**
 * REPRESENTATION DECISION: an equirectangular bin grid, 72 columns x 36 rows of 5 deg.
 *
 * WHY NOT a subdivided icosahedron (level 3 = 1280 near-equal-area faces) or a spherical
 * cap list: both are geometrically nicer, and both are the wrong choice here. The engine
 * already reports coverage on exactly this lattice — coverage.build_coverage uses a 72x36
 * grid, counts a cell covered at >= 50% fill, and weights by cos(elev). Matching it
 * exactly means the phone's live heat-map and the post-upload coverage report agree cell
 * for cell, so a user who was told "94%" on the phone is told "94%" by the server. A
 * prettier lattice would buy equal-area cells and cost the one property that matters:
 * that the two numbers are THE SAME NUMBER. The area distortion an equirect grid
 * introduces is handled the same way the engine handles it, by cos(elev) weighting.
 *
 * The 15-deg station spacing is exactly 3 grid columns, so stations and cells share the
 * lattice too.
 *
 * MEMORY COST:
 *   cells            72 * 36                        = 2592
 *   fill  Float32Array(2592)                        = 10,368 B
 *   covered Uint8Array(2592)                        =  2,592 B
 *   total                                           = 12,960 B  (12.7 KiB)
 *   stations 72 * ~40 B                             = ~2.9 KiB
 *   orientation track, 10 Hz x 120 s x 5 floats     = 24 KiB as Float32Array
 * Under 40 KiB live, which is nothing on a phone and cheap to flush to IndexedDB every
 * 5 s alongside the recorded chunks.
 *
 * COST PER STAMP: 2592 cell-centre dot products at the 10 Hz stamp rate = 26k ops/s.
 * Implementations SHOULD restrict the iteration to the elevation band
 * [pitch - (vfov/2 + 5), pitch + (vfov/2 + 5)] and the yaw band widened by
 * 1/cos(elev), which cuts it by roughly 6x. Correctness must not depend on that.
 */
export const COVERAGE_GRID = {
  yawCells: 72,
  pitchCells: 36,
  cellDeg: 5,
  /** A cell counts as covered at >= this fill fraction — coverage.build_coverage's rule. */
  coveredAtFill: 0.5,
  /** Stamp rate. Higher adds nothing: a 24 deg/s sweep moves 2.4 deg between stamps. */
  stampHz: 10,
  totalCells: 2592,
  bytes: 12_960,
} as const;

/**
 * The accumulated footprint. Plain data with no methods, so Builder C's functions are
 * pure, trivially testable, and structured-cloneable into IndexedDB.
 *
 * Index order is row-major: `index = pitchIndex * yawCells + yawIndex`, with
 * pitchIndex 0 at elevation -90 and yawIndex 0 at yaw 0 (the capture origin).
 */
export interface CoverageGrid {
  yawCells: number;
  pitchCells: number;
  cellDeg: number;
  /** Fill fraction per cell, 0..1. Float32Array(2592). */
  fill: Float32Array;
  /** 1 when fill >= coveredAtFill. Uint8Array(2592), kept in sync by `stampCone`. */
  covered: Uint8Array;
}

/** Per-row progress, the thing the top rail and the row-complete gate read. */
export interface RowProgress {
  row: RowId;
  filledStations: number;
  requiredStations: number;
  /** Unwrapped yaw swept in this row, degrees. */
  yawSpanDeg: Degrees;
  /** Min/max frame-CENTRE elevation seen while this row was active. */
  pitchMinDeg: Degrees;
  pitchMaxDeg: Degrees;
  complete: boolean;
}

/**
 * The live coverage answer. Mirrors the fields of the engine's `Coverage` (lib/types.ts)
 * so the review screen and the post-upload report speak one vocabulary.
 */
export interface CoverageReport {
  /** Solid-angle-weighted, cos(elev) weights, cells covered at >= 50% fill. */
  coveredFraction: number;
  /** |elev| <= 30. */
  horizonFraction30: number;
  /** |elev| <= 60. */
  horizonFraction60: number;
  /** Frame-EDGE elevation reach = centre elevation +/- vfov/2. The _CAP_DEG gate. */
  pitchMaxDeg: Degrees;
  pitchMinDeg: Degrees;
  rows: Record<RowId, RowProgress>;
  /** Contiguous uncovered runs inside |elev| <= 60, largest first. */
  gaps: CoverageGap[];
  /** True when every criterion in COMPLETENESS is satisfied. */
  complete: boolean;
  /** Which criterion is blocking completion, for the early-finish copy. */
  blockedBy: CompletionBlocker[];
}

export interface CoverageGap {
  yawStartDeg: Degrees;
  yawEndDeg: Degrees;
  pitchStartDeg: Degrees;
  pitchEndDeg: Degrees;
  /** Solid-angle fraction of the sphere this gap represents. */
  solidAngleFraction: number;
  /** Which row would fix it, so the touch-up state can set the right pitch target. */
  nearestRow: RowId;
}

export type CompletionBlocker =
  | "covered_fraction"
  | "horizon_30"
  | "horizon_60"
  | "pitch_max"
  | "pitch_min"
  | "yaw_span_level"
  | "yaw_span_up"
  | "yaw_span_down"
  | "row_level"
  | "row_up"
  | "row_down";

/** Where the bright ring should be, and how to get there. */
export interface StationTarget {
  station: Station;
  /** Great-circle angle from the current view axis, degrees. */
  distanceDeg: Degrees;
  /** Signed shortest yaw delta to the station, -180..180. Positive = turn right. */
  yawDeltaDeg: Degrees;
  /** Signed pitch delta. Positive = tilt up. */
  pitchDeltaDeg: Degrees;
  /** True when the station falls inside the current frame, so no edge chevron is needed. */
  onScreen: boolean;
}

/** Everything persisted to IndexedDB so a dropped call or a crash costs nothing. */
export interface CoverageSnapshot {
  fill: ArrayBuffer;
  covered: ArrayBuffer;
  stations: StationState[];
  rows: Record<RowId, RowProgress>;
  lens: LensGeometry;
}

// =====================================================================================
// 8. THE STATE MACHINE
// =====================================================================================

/**
 * `correction` and `rotate_portrait` are deliberately NOT states — see decision D7.
 * A warning must never cost earned progress, and a rotation must never end a capture.
 */
export const CAPTURE_STATES = [
  /** Full-bleed promise card. No permission is requested here. */
  "intro",
  /** Pre-empts `intro` when IndexedDB holds an unfinished capture under an hour old. */
  "resume_draft",
  /** The tap on "Başla" is consumed synchronously so iOS requestPermission() is legal. */
  "permissions",
  /** Camera denied. A dead end, so we do not pretend otherwise. */
  "permission_denied",
  /** Feet, elbows, don't walk. Reuses SweepDiagram from the dashboard's CaptureGuide. */
  "stance",
  /** STEADY / LEVEL / LIGHT gates, then a 3-2-1 and recording starts by itself. */
  "framing",
  "row_level",
  "row_up",
  "row_down",
  /** Only the missing stations are drawn; typically two or three. */
  "touch_up",
  /** Long-press anywhere, the Bitir pill, visibilitychange, or 6 s of persistent walking. */
  "paused",
  /** 2 s of measuring, so the verdict does not appear to arrive from nowhere. */
  "sweep_check",
  "review",
  /** Offered, never required. */
  "bonus_poles",
  "uploading",
  "upload_failed",
  /** Hands off to the existing JobProgress component, unchanged. */
  "handoff",
  /** Terminal: navigate to the existing wizard upload step. */
  "exit_to_upload",
] as const;
export type CaptureState = (typeof CAPTURE_STATES)[number];

/** The three row states, for exhaustive switches. */
export const ROW_STATES = ["row_level", "row_up", "row_down"] as const;
export type RowState = (typeof ROW_STATES)[number];

export const ROW_STATE_BY_ID: Readonly<Record<RowId, RowState>> = {
  level: "row_level",
  up: "row_up",
  down: "row_down",
} as const;

export const ROW_ID_BY_STATE: Readonly<Record<RowState, RowId>> = {
  row_level: "level",
  row_up: "up",
  row_down: "down",
} as const;

/** True while MediaRecorder should be running. Recording is CONTINUOUS across rows: the
 *  pitch transitions between rows are the footage that ties the rows together in the
 *  solver, so the recorder is never stopped and restarted between them. */
export const RECORDING_STATES: readonly CaptureState[] = ["row_level", "row_up", "row_down", "touch_up", "bonus_poles"] as const;

export type CaptureEventType =
  | "START"
  | "USE_EXISTING_VIDEO"
  | "PERMISSIONS_GRANTED"
  | "PERMISSIONS_DENIED_CAMERA"
  | "PERMISSIONS_DENIED_MOTION"
  | "RETRY_PERMISSIONS"
  | "READY"
  | "BACK"
  | "GATES_PASSED"
  | "FORCE_START"
  | "ROW_COMPLETE"
  | "PAUSE"
  | "RESUME"
  | "FINISH"
  | "RESTART"
  | "DISCARD_DRAFT"
  | "RESUME_DRAFT"
  | "CHECK_DONE"
  | "FILL_GAPS"
  | "ACCEPT"
  | "RETAKE"
  | "WANT_POLES"
  | "SKIP_POLES"
  | "UPLOAD_OK"
  | "UPLOAD_FAILED"
  | "RETRY_UPLOAD";

/**
 * Guards evaluated by the host. Named rather than inlined so the table stays pure data
 * and Builder C can unit-test the machine without React.
 */
export type CaptureGuardName =
  /** An unfinished capture under an hour old exists in IndexedDB. */
  | "hasDraft"
  /** DeviceOrientation granted (or not required). False puts the flow in RAIL MODE. */
  | "motionGranted"
  /** `coverage.complete === false` and at least one gap is reachable. */
  | "hasGaps"
  /** All COMPLETENESS criteria satisfied. */
  | "coverageComplete";

/** `"@resume"` resolves to `CaptureSession.resumeTarget` at reduce time. */
export type TransitionTarget = CaptureState | "@resume";

export interface CaptureTransition {
  on: CaptureEventType;
  to: TransitionTarget;
  /** Taken only when this guard is true. */
  when?: CaptureGuardName;
  /** Taken only when this guard is false. */
  unless?: CaptureGuardName;
}

/**
 * THE TRANSITION TABLE. Exhaustive. Anything not listed is illegal and
 * `nextCaptureState` returns null for it — Builder P must ignore, never throw.
 *
 * Notes that the table cannot express and that therefore bind by contract:
 *  - Orientation changes set `portraitBlocked` and pause the recorder. They do NOT
 *    transition. iOS can change the output dimensions mid-stream, so an in-progress
 *    recording is PAUSED on rotation and a 1.5 s "Sabit tut" re-settle runs before it
 *    resumes.
 *  - `visibilitychange` to hidden fires PAUSE from any recording state; the wake lock is
 *    released by the platform and must be re-acquired on return.
 *  - Warnings never transition. Six seconds of persistent walking fires PAUSE.
 *  - The recorder is NOT stopped between rows; see RECORDING_STATES.
 */
export const CAPTURE_TRANSITIONS: Readonly<Record<CaptureState, readonly CaptureTransition[]>> = {
  intro: [
    { on: "START", to: "permissions" },
    { on: "USE_EXISTING_VIDEO", to: "exit_to_upload" },
    { on: "RESUME_DRAFT", to: "resume_draft", when: "hasDraft" },
  ],
  resume_draft: [
    { on: "RESUME_DRAFT", to: "framing" },
    { on: "RESTART", to: "framing" },
    { on: "DISCARD_DRAFT", to: "intro" },
  ],
  permissions: [
    { on: "PERMISSIONS_GRANTED", to: "stance" },
    // Motion denial is survivable: RAIL MODE keeps every camera-derived meter alive.
    { on: "PERMISSIONS_DENIED_MOTION", to: "stance" },
    { on: "PERMISSIONS_DENIED_CAMERA", to: "permission_denied" },
    { on: "BACK", to: "intro" },
  ],
  permission_denied: [
    { on: "RETRY_PERMISSIONS", to: "permissions" },
    { on: "USE_EXISTING_VIDEO", to: "exit_to_upload" },
  ],
  stance: [
    { on: "READY", to: "framing" },
    { on: "BACK", to: "intro" },
  ],
  framing: [
    { on: "GATES_PASSED", to: "row_level" },
    { on: "FORCE_START", to: "row_level" },
    { on: "BACK", to: "stance" },
  ],
  row_level: [
    { on: "ROW_COMPLETE", to: "row_up" },
    { on: "PAUSE", to: "paused" },
    { on: "FINISH", to: "paused" },
  ],
  row_up: [
    { on: "ROW_COMPLETE", to: "row_down" },
    { on: "PAUSE", to: "paused" },
    { on: "FINISH", to: "paused" },
  ],
  row_down: [
    { on: "ROW_COMPLETE", to: "sweep_check" },
    { on: "PAUSE", to: "paused" },
    { on: "FINISH", to: "paused" },
  ],
  touch_up: [
    { on: "ROW_COMPLETE", to: "sweep_check" },
    { on: "FINISH", to: "sweep_check" },
    { on: "PAUSE", to: "paused" },
  ],
  paused: [
    { on: "RESUME", to: "@resume" },
    { on: "FINISH", to: "sweep_check" },
    { on: "RESTART", to: "framing" },
  ],
  sweep_check: [{ on: "CHECK_DONE", to: "review" }],
  review: [
    { on: "FILL_GAPS", to: "touch_up", when: "hasGaps" },
    { on: "WANT_POLES", to: "bonus_poles" },
    { on: "ACCEPT", to: "uploading" },
    { on: "RETAKE", to: "framing" },
  ],
  bonus_poles: [
    { on: "ROW_COMPLETE", to: "sweep_check" },
    { on: "SKIP_POLES", to: "sweep_check" },
    { on: "PAUSE", to: "paused" },
  ],
  uploading: [
    { on: "UPLOAD_OK", to: "handoff" },
    { on: "UPLOAD_FAILED", to: "upload_failed" },
  ],
  upload_failed: [
    { on: "RETRY_UPLOAD", to: "uploading" },
    { on: "BACK", to: "review" },
  ],
  handoff: [],
  exit_to_upload: [],
} as const;

/**
 * Pure reducer for the machine. Returns the next state, or null when the event is not
 * legal in the current state (ignore it — never throw, never toast).
 *
 * `resumeTarget` is the state that was interrupted by PAUSE; `@resume` resolves to it.
 */
export function nextCaptureState(
  state: CaptureState,
  event: CaptureEventType,
  guards: Partial<Record<CaptureGuardName, boolean>> = {},
  resumeTarget: CaptureState | null = null,
): CaptureState | null {
  for (const transition of CAPTURE_TRANSITIONS[state]) {
    if (transition.on !== event) continue;
    if (transition.when && guards[transition.when] !== true) continue;
    if (transition.unless && guards[transition.unless] === true) continue;
    if (transition.to === "@resume") return resumeTarget ?? "framing";
    return transition.to;
  }
  return null;
}

/** Where the flow starts. `intro` unless a recoverable draft exists. */
export function initialCaptureState(hasDraft: boolean): CaptureState {
  return hasDraft ? "resume_draft" : "intro";
}

/** Permission outcomes, kept separate from the machine so the UI can explain each one. */
export interface PermissionStatus {
  camera: "unknown" | "granted" | "denied" | "unavailable";
  /**
   * iOS 13+ requires DeviceOrientationEvent.requestPermission() from a transient user
   * activation in a secure context. A denial is effectively terminal for the session with
   * no in-page re-prompt, so the recovery copy must name the exact path
   * (Settings > Safari > Motion & Orientation Access) and offer a reload.
   */
  motion: "unknown" | "granted" | "denied" | "not-required" | "unavailable";
  /** True when the page is inside an in-app webview (FBAN/FBAV, Instagram, LinkedInApp,
   *  Line, WeChat) or when navigator.mediaDevices is simply missing. Route to the
   *  "Open in Safari/Chrome" interstitial BEFORE requesting anything. */
  inAppBrowser: boolean;
  /** An active camera stream does not reliably hold the screen awake. Re-acquire on
   *  visibilitychange or a 90 s capture dims mid-sweep. */
  wakeLock: boolean;
  /** Portrait is a hard UI gate; there is no lock() on iOS at any version. */
  portrait: boolean;
}

/** The whole live session. One object, owned by Builder P, read by everybody. */
export interface CaptureSession {
  state: CaptureState;
  /** Set on PAUSE, consumed by `@resume`. */
  resumeTarget: CaptureState | null;
  /** Interrupt flag, not a state. Recording is PAUSED, never stopped. */
  portraitBlocked: boolean;
  /** RAIL MODE: orientation denied or unavailable; yaw comes from optical flow. */
  railMode: boolean;
  permissions: PermissionStatus;
  config: CaptureConfig;
  coverage: CoverageReport;
  stations: StationState[];
  /** The bright ring. Null when the row is complete. */
  target: StationTarget | null;
  readout: DetectorReadout | null;
  /** Ranked; only `[0]` is ever rendered. */
  warnings: Warning[];
  /** Sealed recorder segments. More than one only after an interruption. */
  segments: RecordedSegment[];
  startedAt: Millis | null;
  elapsedMs: number;
}

// =====================================================================================
// 9. THE RESULT HANDED TO UPLOAD
// =====================================================================================

/**
 * One sealed MediaRecorder run. An incoming call ends the track on iOS, so a scene can
 * legitimately consist of several segments. `api.scenes.create` already accepts a
 * `File[]` and appends every entry as `files`, so this rides the existing upload path
 * untouched — but see BUILD PLAN note E1: the ENGINE must concatenate multiple video
 * files into one scene rather than assuming a single file.
 */
export interface RecordedSegment {
  blob: Blob;
  mimeType: string;
  durationMs: number;
  /** Pixel dimensions as recorded. On iOS the track is sensor-native landscape with a
   *  rotation applied, so treat getSettings() width/height as an UNORDERED pair and trust
   *  `video.videoWidth/videoHeight` for what the decoder will produce. */
  width: number;
  height: number;
  /** Measured from blob size over elapsed time, not requested. */
  bitrateBps: number;
  startedAt: Millis;
  /** Why the previous segment ended, for telemetry and for the resume copy. */
  endedBy: "row_complete" | "user_finish" | "interrupted" | "rotation" | "error" | "max_duration";
}

/**
 * Capture telemetry — the sidecar the engine can use as a PRIOR.
 *
 * This is the payload that lets the solver pin K instead of re-estimating it per scene,
 * and seed pose rather than solving cold. It is uploaded as `motion.json` alongside the
 * video in the same `files[]` array.
 *
 * DO NOT feed the orientation track to the stitcher as truth. The solver is already at
 * 0.403 deg pose error and 0.042 deg loop closure — orders of magnitude better than these
 * sensors. It is a prior and a diagnostic, nothing more.
 */
export interface CaptureTelemetry {
  schema: "biosphere.capture.v1";
  createdAt: string;

  device: {
    userAgent: string;
    platform: "ios" | "android" | "other";
    /** The chosen camera's label, e.g. "Back Camera". Evidence of which lens was used. */
    cameraLabel: string | null;
    /** True when only a VIRTUAL (auto-switching) rear device was available — mark the
     *  upload reduced-confidence so the validator can weight it. */
    virtualLensOnly: boolean;
    /** Full `MediaTrackSettings` dump, so the engine sees exactly what was negotiated. */
    trackSettings: Record<string, unknown>;
    /** Which camera controls actually applied, feature-detected. */
    appliedControls: Partial<Record<keyof CameraControlPlan, boolean>>;
    /** What the shutter cap did, in numbers, so the engine can see the exposure it got. */
    shutterCap?: ShutterCapRecord | null;
    /**
     * Which phone this is, as far as the browser will say — see `capture/device.ts`.
     * The engine looks the model up in its own lens table and BOUNDS the focal search with
     * it, because self-calibration on a hand-held sweep is biased long and only long.
     * Absent on captures made before this existed, and empty-ish on iOS, which never
     * reports a model; the screen fingerprint is what identifies an iPhone.
     */
    modelHints?: DeviceModelHints | null;
  };

  optics: {
    hfovDeg: Degrees;
    vfovDeg: Degrees;
    lensSource: LensGeometry["source"];
    /** True when zoom or the active lens changed mid-capture. The solve fits ONE focal
     *  for the whole file, and rotation.py's [30, 110] hfov clamp will NOT catch a jump
     *  to ultrawide — it will silently poison every pose. */
    lensChangedMidCapture: boolean;
  };

  recording: {
    segments: number;
    totalDurationMs: number;
    width: number;
    height: number;
    /** `screen.orientation.angle` at record time, so the engine can assert the decoded
     *  orientation matches and reject loudly rather than solving 69 deg across the short
     *  edge and 43 deg down the long one — the FOV exactly inverted. */
    screenAngleDeg: Degrees;
    portrait: boolean;
    requestedBitrateBps: number;
    achievedBitrateBps: number;
    mimeType: string;
    meanFps: number;
    minFps: number;
  };

  /** The measured shape of the sweep. All degrees. */
  sweep: {
    yawSpanPerRowDeg: Record<RowId, Degrees>;
    /** Frame-CENTRE elevation extremes. */
    pitchCentreMinDeg: Degrees;
    pitchCentreMaxDeg: Degrees;
    /** Frame-EDGE reach = centre +/- vfov/2. This is what the _CAP_DEG = 60 gate reads. */
    pitchReachMinDeg: Degrees;
    pitchReachMaxDeg: Degrees;
    meanAngularVelocityDegPerS: number;
    p95AngularVelocityDegPerS: number;
    /** Fraction of recorded time above THRESHOLDS.angularSpeed.redDegPerS. */
    fractionOverSpeedCap: number;
    rowCompletion: Record<RowId, number>;
  };

  /** The translation evidence, in the two statistics validate.py's ladder separates. */
  translation: {
    /** Median r = |a_radial|/omega^2 over the capture, metres. Null if never gated in. */
    medianPivotRadiusM: number | null;
    p90PivotRadiusM: number | null;
    /** Median net windowed displacement, m/s. */
    medianDriftMPerS: number;
    maxDriftMPerS: number;
    /** Fraction of recorded time the walking gate was tripped. */
    walkingFraction: number;
    /** FALSE means the inertial gate never ran — absence of evidence, not evidence. */
    motionGateRan: boolean;
  };

  image: {
    meanLuma: number;
    minLuma: number;
    meanClippedFraction: number;
    maxClippedFraction: number;
    /** Fraction of frames whose predicted smear exceeded the octave-1 budget. */
    fractionOverSmearBudget: number;
  };

  coverage: {
    coveredFraction: number;
    horizonFraction30: number;
    horizonFraction60: number;
    gaps: CoverageGap[];
    /** Serialised 72x36 covered mask, base64 of the Uint8Array, so the server can render
     *  the same heat-map the phone showed. */
    coveredMaskBase64: string;
  };

  /**
   * Orientation track, downsampled to 10 Hz and keyed to `mediaTime` from
   * requestVideoFrameCallback so the engine can align it to its own extracted frames.
   * Flat arrays rather than objects: 1200 samples is 24 KiB this way and ~120 KiB as JSON
   * objects.
   */
  track: {
    /** Seconds into the media, one entry per sample. */
    mediaTime: number[];
    /** Flattened [x,y,z,w] per sample; length === mediaTime.length * 4. */
    quat: number[];
    /** deg/s per sample. */
    omega: number[];
    /** Per-sample recording segment index, so a consumer can split the track at the
     *  joins instead of guessing from a non-monotonic `mediaTime`. */
    segment?: number[];
  };

  /** What the phone told the user, so the server's verdict can use the same words. */
  verdict: CaptureVerdict;
}

/**
 * The phone's own grade. Deliberately mirrors what
 * engine/biosphere/pano/validate.py will later return, so the two never disagree and the
 * user hears one story from their phone and from the engine.
 */
export type CaptureVerdict = "great" | "gaps" | "walking" | "too_dark" | "too_blurry" | "too_short";

/** Everything handed to the upload step. */
export interface CaptureResult {
  segments: RecordedSegment[];
  telemetry: CaptureTelemetry;
  /** Auto-grabbed at 45-deg intervals during the level row, for the review strip. */
  thumbnails: Blob[];
  verdict: CaptureVerdict;
  qualityMode: GuidedQualityMode;
  captureKind: CaptureKind;
}

/** Engine convention is radians; convert exactly once, here. */
export function toEngineAngles(yawDeg: Degrees, pitchDeg: Degrees): { yaw: number; pitch: number } {
  const wrapped = ((((yawDeg + 180) % 360) + 360) % 360) - 180;
  return { yaw: (wrapped * Math.PI) / 180, pitch: (pitchDeg * Math.PI) / 180 };
}

/** A draft in IndexedDB. Flushed every 5 s so a crash or a dropped call costs nothing. */
export interface CaptureDraft {
  id: string;
  createdAt: number;
  updatedAt: number;
  /** Chunks as they arrived from the timeslice, not a re-muxed blob. */
  chunks: Blob[];
  mimeType: string;
  coverage: CoverageSnapshot;
  partialTelemetry: Partial<CaptureTelemetry>;
  state: CaptureState;
  /** Offered for one hour, auto-expired after 24. */
  expiresAt: number;
}

export const DRAFT_POLICY = {
  flushIntervalMs: 5000,
  offerWithinMs: 60 * 60 * 1000,
  expireAfterMs: 24 * 60 * 60 * 1000,
  storeName: "biosphere-capture-drafts",
} as const;

// =====================================================================================
// 10. MODULE APIS — Builder S and Builder C
// =====================================================================================

/** Builder S: `useOrientation.ts`. */
export interface OrientationApi {
  /** Latest sample, or null before the first event. Read in a rAF loop, not via state. */
  sample: OrientationSample | null;
  /** Mutable ref-style accessor for hot paths, so V never re-renders per sensor event. */
  read: () => OrientationSample | null;
  reference: OrientationReference;
  /** True once at least one event has arrived. */
  live: boolean;
  /**
   * Zeroes the yaw origin. Called once, at the instant recording starts, so yaw 0 is the
   * loop-closure origin and the review screen's compass thumbnails are keyed to it.
   */
  setOrigin: () => void;
  /** iOS gesture-gated request. Must be called synchronously inside a click handler. */
  request: () => Promise<"granted" | "denied" | "not-required">;
  stop: () => void;
}

/** Builder S: `useMotion.ts`. */
export interface MotionApi {
  sample: MotionSample | null;
  read: () => MotionSample | null;
  accelSource: AccelSource;
  /** False when neither acceleration nor a usable reconstruction exists. FAIL SAFE. */
  gateAvailable: boolean;
  request: () => Promise<"granted" | "denied" | "not-required">;
  stop: () => void;
}

/** Builder S: `useCamera.ts`. */
export interface CameraApi {
  stream: MediaStream | null;
  /** Attach to a <video playsInline muted autoPlay> element. */
  videoRef: RefObject<HTMLVideoElement | null>;
  /** Decoder-truth dimensions, after any rotation metadata. */
  size: { width: number; height: number } | null;
  label: string | null;
  virtualLensOnly: boolean;
  settings: Record<string, unknown>;
  appliedControls: Partial<Record<keyof CameraControlPlan, boolean>>;
  /** Fires when getSettings() changes mid-capture; drives the `lens_changed` warning. */
  configurationChanged: boolean;
  error: CameraError | null;
  /** Two-phase open: unlock labels, enumerate, reopen on the physical rear wide. */
  open: () => Promise<void>;
  close: () => void;
  /** Per-frame image statistics off a downscaled canvas, driven by rVFC. */
  readQuality: () => FrameQualitySample | null;
  /** Quadrant phase correlation at THRESHOLDS.flowDivergence.rateHz. */
  readFlow: () => FlowSample | null;
  /** Where the per-frame image analysis actually ran. Diagnostic only. */
  analysisBackend: () => "worker" | "main-thread";
  /** Null while the worker path is healthy; otherwise why it was written off. */
  analysisDemotedReason: () => string | null;
  /** Full-resolution still from the video track via canvas, for the review strip. */
  grabStill: () => Promise<Blob | null>;
  setTorch: (on: boolean) => Promise<boolean>;
}

export type CameraError =
  | "denied"
  | "no-device"
  | "in-app-browser"
  | "insecure-context"
  /** Got a stream but under 1080 on the short edge. Refuse to record; `ideal` never
   *  errors, it just silently hands back the nearest supported mode. */
  | "resolution-too-low"
  | "unknown";

/** Builder S: `recorder.ts`. */
export interface RecorderApi {
  state: "idle" | "recording" | "paused" | "stopped" | "error";
  mimeType: string | null;
  /** Chunks as they arrive on the 1 s timeslice, pushed immediately. */
  chunkCount: number;
  elapsedMs: number;
  /** Measured from accumulated bytes over elapsed time. */
  bitrateBps: number;
  start: () => Promise<void>;
  pause: () => void;
  resume: () => void;
  /** Seals the current segment and returns it. */
  stop: (reason: RecordedSegment["endedBy"]) => Promise<RecordedSegment | null>;
  /**
   * Records ~200 ms into a throwaway recorder during the pre-flight framing step and
   * confirms a non-empty Blob arrives, BEFORE the user has invested any time.
   * isTypeSupported() returning true has historically not guaranteed start() succeeds
   * on iOS.
   */
  preflight: () => Promise<{ ok: boolean; mimeType: string | null }>;
}

/** Builder S: `telemetry.ts`. */
export interface TelemetryApi {
  /** Called at the coverage stamp rate; accumulates the track and the aggregates. */
  push: (frame: SensorFrame, readout: DetectorReadout | null) => void;
  /** Marks the media time of the first frame of a segment, for rVFC alignment. */
  markSegment: (index: number, mediaTimeOffset: number) => void;
  build: (coverage: CoverageReport, segments: RecordedSegment[], verdict: CaptureVerdict) => CaptureTelemetry;
  /** Serialises to the `motion.json` File that rides the existing `files[]` upload. */
  toFile: (telemetry: CaptureTelemetry) => File;
  reset: () => void;
}

/** The filename the sidecar must use. The engine keys off this exact name. */
export const TELEMETRY_FILENAME = "motion.json";

/** Builder C: `coverage.ts` — pure functions, no React, no `three`. */
export interface CoverageApi {
  createGrid: (config?: Pick<typeof COVERAGE_GRID, "yawCells" | "pitchCells" | "cellDeg">) => CoverageGrid;
  /**
   * Stamps a `hfov x vfov` cone centred on the sample's axis onto the grid. Called at
   * COVERAGE_GRID.stampHz and ONLY when no `stop`-severity warning is active.
   * Returns the cells newly crossing `coveredAtFill`.
   */
  stampCone: (grid: CoverageGrid, sample: OrientationSample, lens: LensGeometry) => number;
  /** Solid-angle-weighted report, computed exactly the way coverage.build_coverage does. */
  report: (grid: CoverageGrid, stations: StationState[], lens: LensGeometry) => CoverageReport;
  /** Contiguous uncovered runs inside |elev| <= 60, largest first. */
  findGaps: (grid: CoverageGrid) => CoverageGap[];
  serialize: (grid: CoverageGrid, stations: StationState[], rows: Record<RowId, RowProgress>, lens: LensGeometry) => CoverageSnapshot;
  deserialize: (snapshot: CoverageSnapshot) => { grid: CoverageGrid; stations: StationState[] };
}

/** Builder C: `plan.ts` — pure. */
export interface PlanApi {
  buildStations: (config: CaptureConfig) => StationState[];
  /** Advances dwell, fills stations, returns the ids that filled this tick. */
  updateStations: (stations: StationState[], sample: OrientationSample, now: Millis, blocked: boolean) => StationId[];
  /** The bright ring: nearest uncaptured station in the active row, then any row. */
  nextTarget: (stations: StationState[], sample: OrientationSample, activeRow: RowId | null, lens: LensGeometry) => StationTarget | null;
  rowProgress: (stations: StationState[], row: RowId, yawSpanDeg: Degrees, pitchMinDeg: Degrees, pitchMaxDeg: Degrees) => RowProgress;
  /** Unwrapped yaw accumulation; must survive the 0/360 seam and short reversals. */
  accumulateYaw: (previousDeg: Degrees, currentDeg: Degrees, spanSoFarDeg: Degrees) => Degrees;
  /** Which criteria are blocking `done`, for the early-finish and review copy. */
  blockers: (report: CoverageReport) => CompletionBlocker[];
  verdict: (report: CoverageReport, telemetry: Pick<CaptureTelemetry, "translation" | "image">) => CaptureVerdict;
}

/** Builder C: `detectors.ts` — pure, fed one SensorFrame at a time. */
export interface DetectorsApi {
  /** Stateful only in the sense of holding sliding windows; construct one per session. */
  create: (config: CaptureConfig) => DetectorInstance;
}

export interface DetectorInstance {
  push: (frame: SensorFrame) => DetectorReadout;
  /** Ranked by WARNING_PRIORITY. Only `[0]` is ever rendered. */
  warnings: () => Warning[];
  /** Aggregates for telemetry. */
  summary: () => Pick<CaptureTelemetry, "translation" | "image"> & { sweep: Partial<CaptureTelemetry["sweep"]> };
  reset: () => void;
}

// =====================================================================================
// 11. COMPONENT PROP CONTRACTS — Builder V
// =====================================================================================

/**
 * EVERY V COMPONENT TAKES PLAIN DATA. No hooks from Builder S, no imports from Builder C's
 * modules, no context. V can build and preview all five against hand-written fixtures
 * before S or C exists, and P wires the real data in. This is the whole point of the
 * split.
 *
 * All five must honour `reducedMotion`: it removes the breathe, the comet's motion, the
 * dome's flight and the STOP pulse, and replaces them with static stroke weights,
 * numerals and cross-fades. The world-locked lattice STAYS under reduced motion, because
 * it is registration, not decoration.
 *
 * Colour never carries a message alone. good = a 1.5 px continuous hairline at
 * var(--bio); warn = a 4 px dashed (12/8) frame at var(--amber); stop = 10 px corner
 * brackets at var(--danger). Three different stroke weights and patterns, legible in pure
 * greyscale on a screen in sunlight. Every stroke drawn over the live camera feed sits on
 * a 45% radial scrim — the one thing this feature adds to the design system.
 */

export interface CoverageDomeProps {
  /** 72x36 covered mask. Length must equal COVERAGE_GRID.totalCells. */
  covered: Uint8Array;
  yawCells: number;
  pitchCells: number;
  coveredFraction: number;
  /** Painted as a dashed wireframe band, for the row-transition punctuation. */
  nextRow: RowId | null;
  /** 96 for the 1.2 s row punctuation, 240 for the review hero. */
  size: number;
  /** Review dome auto-rotates; punctuation dome flies in once. */
  variant: "punctuation" | "review";
  /** Degrees; lets the review dome follow a scrubbed thumbnail. */
  yawDeg?: Degrees;
  reducedMotion: boolean;
  label: string;
  /** The live lens. Row latitudes are drawn from `rowTargetPitchDeg`, which depends on it:
   *  a 4:3 frame closes the rows up by 6 deg against the 9:16 ROW_PLAN literals, and a ring
   *  drawn where the station is not is worse than no ring at all. */
  lens?: LensGeometry;
  className?: string;
}

export interface GuidanceOverlayProps {
  /** Camera-frame quaternion. The ONLY orientation input; never Euler. */
  q: Quat;
  /** All stations, filled and not. V draws hollow rings for unfilled, dots for filled. */
  stations: StationState[];
  /** The bright ring. Null when the active row is done. */
  target: StationTarget | null;
  /** Restricts drawing to one row's stations, for `touch_up`. Null draws all. */
  onlyRow: RowId | null;
  /** Draw only these station ids; used by `touch_up` so the screen is nearly empty. */
  onlyStations: StationId[] | null;
  lens: LensGeometry;
  /** Drives the frame instrument: hairline / dashed / corner brackets. */
  severity: WarningSeverity | null;
  /** Leans the horizon line. Amber past THRESHOLDS.roll.amberDeg. */
  rollDeg: Degrees;
  /** RAIL MODE flattens the lattice to a horizon-locked 12-cell strip plus a 3-cell
   *  column, advanced by optical flow. The metaphor bends; it does not break. */
  railMode: boolean;
  reducedMotion: boolean;
  /** Fires when a station enters the capture radius — P uses it for tick + haptic. */
  onStationEnter?: (id: StationId) => void;
  className?: string;
}

export interface SpeedMeterProps {
  /** Low-passed |rotationRate|, deg/s. */
  omegaDegPerS: number;
  /** Live adaptive ceiling. The comet's tail stretches past this. */
  capDegPerS: number;
  grade: Grade;
  /** Predicted smear at 1080 width; shown as a secondary tick when over budget. */
  smearPx: number;
  /** True heading, so the comet's position is the heading and its tail is the rate. */
  headingDeg: Degrees;
  reducedMotion: boolean;
  className?: string;
}

export interface RowPromptProps {
  row: RowId;
  /** 1..3, drives the three-segment rail at the top. */
  rowIndex: number;
  /** Per-row fill fractions for the rail, in ROW_IDS order. */
  rowFractions: Record<RowId, number>;
  /** Two-word caption, already localised by P. */
  title: string;
  cue: string;
  /** Shown after `stuckAfterMs` with no new station: the bright ring grows a tail. */
  stuck: boolean;
  reducedMotion: boolean;
  /** The live lens — see CoverageDomeProps.lens. The rings must sit on the stations. */
  lens?: LensGeometry;
  className?: string;
}

export interface WarningToastProps {
  /** Exactly one, or null. V must never render a stack. */
  warning: Warning | null;
  /** Already localised by P; V never touches the dictionary. */
  title: string;
  body: string | null;
  reducedMotion: boolean;
  className?: string;
}

/** After this long in a row with no station filling, the cue repeats once and the bright
 *  ring grows a directional tail. Six seconds is long enough not to nag and short enough
 *  that a user stuck at eye level in the UP row is rescued before they give up. */
export const STUCK_AFTER_MS = 6000;

// =====================================================================================
// 12. FLOW PROP CONTRACTS — Builder P
// =====================================================================================

export interface CaptureStageProps {
  propertyId: string;
  /** Prefilled from the wizard when the user came from there. */
  defaultSceneName?: string;
  /** Guided flow never offers `fast`; see decision D5. */
  defaultQualityMode?: GuidedQualityMode;
  /** Where "Hazır videom var" goes — the existing wizard upload step, unchanged. */
  onUseExistingVideo: () => void;
  /** Fires with the created scene + job so the page can mount the existing JobProgress. */
  onHandoff: (result: { sceneId: string; jobId: string }) => void;
}

export interface PermissionGateProps {
  status: PermissionStatus;
  /** MUST be invoked synchronously inside the click handler: iOS
   *  DeviceOrientationEvent.requestPermission() requires a transient user activation. */
  onRequest: () => void;
  onUseExistingVideo: () => void;
  onRetry: () => void;
  locale: Locale;
}

export interface ReviewStepProps {
  result: CaptureResult;
  coverage: CoverageReport;
  thumbnails: Blob[];
  sceneName: string;
  onSceneNameChange: (name: string) => void;
  qualityMode: GuidedQualityMode;
  onQualityModeChange: (mode: GuidedQualityMode) => void;
  /**
   * One sentence about this phone, when there is one worth saying, or null.
   *
   * Only ever something the user can act on: "no optical stabilisation, so sweep slowly"
   * changes what they do next. It also carries the LiDAR sentence, which exists because a
   * customer who knows their iPhone has a scanner will otherwise assume the property used it —
   * no browser can read it. See `capture/device.ts`.
   */
  deviceNote?: string | null;
  onFillGaps: () => void;
  onAccept: () => void;
  onRetake: () => void;
  onWantPoles: () => void;
  /** Saves the recorded blob locally. The user's minute is never held hostage. */
  onDownload: () => void;
  locale: Locale;
}

// =====================================================================================
// 13. COPY KEYS — Builder P owns the values in copy.ts
// =====================================================================================

/**
 * `web/src/lib/i18n.ts` is FROZEN. Every capture string lives in
 * `web/src/lib/capture/copy.ts`, in the exact shape of
 * `web/src/components/dashboard/copy.tsx`:
 *
 *   export const captureStrings = { "<key>": { tr: "...", en: "..." }, ... } satisfies Dict;
 *   export function cc(key: CaptureCopyKey, locale: Locale = "tr"): string;
 *
 * Builder P must define a value for EVERY key in this union — the `satisfies
 * Record<CaptureCopyKey, {tr: string; en: string}>` check makes a missing one a compile
 * error rather than a runtime "??key??" leaking onto a phone screen mid-capture.
 *
 * Interpolation: `{n}` and `{band}` are replaced by the caller; keep them in both
 * languages.
 */
export type CaptureCopyKey =
  // intro
  | "capture.intro.title"
  | "capture.intro.body"
  | "capture.intro.start"
  | "capture.intro.have"
  | "capture.intro.how"
  // permissions
  | "capture.perm.title"
  | "capture.perm.body"
  | "capture.perm.allow"
  | "capture.perm.camera.denied"
  | "capture.perm.motion.denied"
  | "capture.perm.inapp"
  | "capture.perm.inapp.body"
  // orientation gate
  | "capture.rotate.title"
  | "capture.rotate.body"
  // stance
  | "capture.stand.title"
  | "capture.stand.1"
  | "capture.stand.2"
  | "capture.stand.3"
  | "capture.stand.ready"
  // framing gates
  | "capture.frame.hold"
  | "capture.frame.level"
  | "capture.frame.dark"
  | "capture.frame.bright"
  | "capture.frame.turnhere"
  | "capture.frame.starting"
  | "capture.frame.force"
  | "capture.frame.slowfps"
  // rows
  | "capture.row.level"
  | "capture.row.up"
  | "capture.row.down"
  | "capture.cue.turn"
  | "capture.cue.up"
  | "capture.cue.down"
  | "capture.cue.back"
  | "capture.cue.closeLoop"
  | "capture.row.done"
  | "capture.row.settle"
  | "capture.row.next"
  | "capture.row.nextUp"
  | "capture.row.nextDown"
  // warnings — one per WarningCode, plus bodies where the verb is not enough
  | "capture.warn.walk"
  | "capture.warn.walk.body"
  | "capture.warn.fast"
  | "capture.warn.fast.body"
  | "capture.warn.smear"
  | "capture.warn.smear.body"
  | "capture.warn.pivot"
  | "capture.warn.pivot.body"
  | "capture.warn.glare"
  | "capture.warn.dark"
  | "capture.warn.tilt"
  | "capture.warn.rowpitch"
  | "capture.warn.lowfps"
  | "capture.warn.lens"
  | "capture.warn.lens.body"
  | "capture.warn.lost"
  // controls / pause
  | "capture.finish"
  | "capture.pause.title"
  | "capture.pause.body"
  | "capture.pause.resume"
  | "capture.pause.finish"
  | "capture.pause.restart"
  | "capture.resume.aim"
  | "capture.early.title"
  | "capture.early.body"
  | "capture.band.ceiling"
  | "capture.band.floor"
  // check + review
  | "capture.check.working"
  | "capture.review.great.title"
  | "capture.review.great.body"
  | "capture.review.gaps.title"
  | "capture.review.gaps.body"
  | "capture.review.walk.title"
  | "capture.review.walk.body"
  | "capture.review.dark.title"
  | "capture.review.dark.body"
  | "capture.review.fill"
  | "capture.review.use"
  | "capture.review.retake"
  | "capture.review.anyway"
  | "capture.review.thumbs"
  | "capture.review.enough"
  // bonus + upload + draft
  | "capture.bonus.poles"
  | "capture.bonus.skip"
  | "capture.save.local"
  | "capture.upload.title"
  | "capture.upload.retry"
  | "capture.upload.slowlink"
  | "capture.draft.title"
  | "capture.draft.body"
  // settings + a11y labels
  | "capture.voice"
  | "capture.sound"
  | "capture.progress.label"
  | "capture.progress.value"
  | "capture.tiles.remaining"
  // spoken lines
  | "capture.say.start"
  | "capture.say.up"
  | "capture.say.down"
  | "capture.say.slow"
  | "capture.say.walk"
  | "capture.say.gap"
  | "capture.say.done";

/**
 * Which copy key each warning uses. Builder C fills `Warning.titleKey` from this so the
 * mapping lives in one place and P cannot drift from it.
 */
export const WARNING_COPY: Readonly<
  Record<WarningCode, { title: CaptureCopyKey; body: CaptureCopyKey | null; voice: CaptureCopyKey | null; icon: WarningIcon; severity: WarningSeverity }>
> = {
  walking: { title: "capture.warn.walk", body: "capture.warn.walk.body", voice: "capture.say.walk", icon: "foot-crossed", severity: "stop" },
  too_fast: { title: "capture.warn.fast", body: "capture.warn.fast.body", voice: "capture.say.slow", icon: "snail", severity: "warn" },
  smear: { title: "capture.warn.smear", body: "capture.warn.smear.body", voice: "capture.say.slow", icon: "snail", severity: "warn" },
  pivot_radius: { title: "capture.warn.pivot", body: "capture.warn.pivot.body", voice: null, icon: "orbit", severity: "warn" },
  roll: { title: "capture.warn.tilt", body: null, voice: null, icon: "level", severity: "info" },
  row_pitch: { title: "capture.warn.rowpitch", body: null, voice: null, icon: "level", severity: "warn" },
  too_dark: { title: "capture.warn.dark", body: null, voice: null, icon: "moon", severity: "warn" },
  glare: { title: "capture.warn.glare", body: null, voice: null, icon: "sun", severity: "warn" },
  low_fps: { title: "capture.warn.lowfps", body: null, voice: null, icon: "moon", severity: "warn" },
  lens_changed: { title: "capture.warn.lens", body: "capture.warn.lens.body", voice: null, icon: "lens", severity: "stop" },
  landscape: { title: "capture.rotate.title", body: "capture.rotate.body", voice: null, icon: "rotate", severity: "stop" },
  track_lost: { title: "capture.warn.lost", body: null, voice: null, icon: "compass", severity: "warn" },
} as const;

/**
 * Haptic patterns, ms, `navigator.vibrate` format. Android only — iOS Safari has no
 * vibration API — so haptics can never be the sole carrier of a message.
 */
export const HAPTICS: Readonly<Record<"stationFilled" | "rowComplete" | "warn" | "stop", readonly number[]>> = {
  stationFilled: [12],
  rowComplete: [18, 60, 18],
  warn: [24, 40, 24],
  stop: [80, 60, 80],
} as const;

// =====================================================================================
// 14. UPLOAD
// =====================================================================================

/**
 * How the result reaches the server. `api.scenes.create(propertyId, files, meta, onProgress,
 * signal)` is reused UNCHANGED: its `files` parameter is a `File[]` and every entry is
 * appended as `files`, so the video segments and the `motion.json` sidecar ride together
 * without touching api.ts.
 *
 * At 20 Mbps a 50 s capture is ~125 MB. Over a typical 5 Mbps LTE uplink that is ~3.5
 * minutes with no resumability, so: keep the recording in IndexedDB until the server
 * acknowledges, check `navigator.connection.effectiveType`, and offer "upload later on
 * Wi-Fi" on 3g-or-slower or when `saveData` is set.
 */
export interface UploadPlan {
  propertyId: string;
  files: File[];
  name: string;
  qualityMode: GuidedQualityMode;
  captureKind: CaptureKind;
  /** Estimated bytes, for the pre-flight "this is big" warning. */
  estimatedBytes: number;
  /** From navigator.connection; null when unsupported. */
  effectiveType: string | null;
  saveData: boolean;
}

export const UPLOAD_POLICY = {
  /** Warn before uploading on these. */
  slowEffectiveTypes: ["slow-2g", "2g", "3g"] as readonly string[],
  /** Keep the blob until the server acknowledges, so a retry needs no re-shoot. */
  retainDraftUntilAck: true,
} as const;
