"use client";

/**
 * The measure tool: tap the floor twice, and be told how far apart those two spots are.
 *
 * ── THE ARITHMETIC IS NOT HERE, AND MUST NOT MOVE HERE ────────────────────────────────
 * `engine/biosphere/ai/measure.py` does the whole of it: a marked (yaw, pitch) is a direction,
 * a direction crossed with the floor plane is a point, and two points have a distance. This
 * file marks directions, draws them, and reports what came back. A second implementation in
 * TypeScript would be a second set of rounding, a second set of edge cases, and — the moment
 * the engine's plane or its axis convention changed — a client that quietly disagreed with the
 * server about how big a room is.
 *
 * Two facts from that module's docstring shape everything below, and both are user-visible:
 *
 *   1. EVERY LENGTH IS SCALED BY ONE NUMBER — the camera's height above the floor. Nothing in
 *      a single panorama recovers it, so it is assumed (155 cm, a phone at chest height on a
 *      hand-held sweep). Get it wrong by 10 % and every distance on screen is wrong by 10 %,
 *      in the same direction, with no visible symptom. That is why the panel carries the
 *      height as an editable field rather than hiding it in a constant: a user who knows they
 *      shot from a tripod at 130 cm can make the whole set correct with one edit, and a user
 *      who does not at least sees which assumption they are trusting.
 *
 *   2. A POINT AT OR ABOVE THE HORIZON HAS NO FLOOR DISTANCE AT ALL. Its ray runs parallel to
 *      or away from the plane. The engine returns `length_cm: null` with a hint instead of a
 *      confidently wrong number, and so this UI shows the hint where the number would have
 *      been. Substituting a dash, or silently dropping the segment, would let a viewer read
 *      the remaining numbers as if the polyline were complete.
 *
 * ── HOW IT IS DRAWN ───────────────────────────────────────────────────────────────────
 * The same way `HotspotLayer` draws tags: HTML and SVG above the canvas, positioned by hand
 * once per frame from the r3f frame loop (`projectMeasure`, called by `PlayerStage.onFrame`),
 * writing straight to element refs. No per-frame React state — a `setState` per mark per frame
 * in a component that also owns a WebGL canvas is the difference between marks welded to the
 * floor and marks that swim across it.
 *
 * The one piece of geometry this file does own is why a STRAIGHT screen-space line between two
 * projected marks is not an approximation. The camera sits at the origin; the two floor points
 * lie on the rays through the two marked directions; a straight 3-D segment projects to a
 * straight image-space segment under a perspective camera; and the marked directions project
 * onto that very same image line, because they are on those same two rays. So joining the two
 * projected marks with a `<line>` draws exactly the floor segment — without this client ever
 * knowing where either floor point is, how far away it is, or what the scale is. That is also
 * why the line can be drawn the instant the second mark lands, before the server has answered:
 * the line asserts nothing about length, only about direction, and the direction is already
 * known.
 *
 * It stops being true the moment an endpoint goes behind the eye — the projection mirrors, and
 * the "line" swings off to the wrong side of the frame. Hence: a segment with either end behind
 * the viewer is hidden outright rather than clipped. Clipping would leave a label sitting on a
 * line whose far end is nowhere on screen, which is a number with nothing to attach it to.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Loader2, Ruler, Trash2, Undo2, X } from "lucide-react";
import { api } from "@/lib/api";
import { t } from "@/lib/i18n";
import { clamp, cn } from "@/lib/utils";
import type { Locale, MeasureResponse } from "@/lib/types";
import { directionFromAngles } from "./useLookControls";

// ---------------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------------
//
// No React and no DOM between here and the next banner, which is what lets
// `__tests__/measureTool.test.ts` check them under the plain-Node test environment. Every one
// of them is a place where a rounding decision turns into a claim about somebody's flat.

/** The engine's own default for a hand-held sweep: `ai/measure.DEFAULT_CAMERA_HEIGHT_CM`. */
export const DEFAULT_CAMERA_HEIGHT_CM = 155;

/**
 * The band the height field will accept, in centimetres.
 *
 * Not a taste judgement — it is the range over which the assumption can still be true. Below
 * 100 cm nobody is holding a phone to sweep a room (that is a camera on the floor, which the
 * capture guide tells people not to do), and above 220 cm the lens is over the head of the
 * tallest person who could hold it. Letting someone type 15 or 1500 would not give them a
 * better measurement, it would give them a plausible-looking set of numbers wrong by a factor
 * of ten, and the numbers are the product.
 */
export const CAMERA_HEIGHT_MIN_CM = 100;
export const CAMERA_HEIGHT_MAX_CM = 220;

/**
 * Mirror of `ai/measure.MIN_DEPRESSION_DEG`.
 *
 * Duplicated ON PURPOSE and for one job only: naming, in the user's own language, WHY a
 * segment came back without a length. The server sends its reason as an English sentence in
 * `hint`, and this product is Turkish first — so the classification is redone here from the
 * pitch that was marked, which is the same input the server classified. It is never used to
 * decide WHETHER a length exists; that answer is only ever the server's.
 */
const MIN_DEPRESSION_DEG = 0.5;
const MIN_DEPRESSION_RAD = (MIN_DEPRESSION_DEG * Math.PI) / 180;

/** Why a marked point cannot reach the floor. */
export type MarkFault = "horizon" | "grazing";

/**
 * Classify one marked pitch the way `ai/measure._hit` does.
 *
 * `pitch >= 0` is at or above the horizon: the ray never meets the plane. Between there and
 * half a degree of depression it meets it more than a hundred camera-heights away, where a
 * pixel of marking error is metres of answer — which the engine also refuses, and rightly.
 */
export function markFault(pitch: number): MarkFault | null {
  if (pitch >= 0) return "horizon";
  if (-pitch <= MIN_DEPRESSION_RAD) return "grazing";
  return null;
}

/**
 * The floor under the error bar, as a fraction of the length.
 *
 * Five per cent is what the ASSUMPTION alone is worth, before any marking error at all: a
 * default of 155 cm against someone who actually held the phone at 148 or 163 — an ordinary
 * spread for "chest height" across adults — is already 5 %. So no measurement from this tool,
 * however cleanly marked, may ever be presented as tighter than that.
 */
const MIN_RELATIVE_ERROR = 0.05;

/**
 * The ceiling, for a set of marks the engine barely trusts.
 *
 * `confidence` collapses towards zero as marks approach the horizon, which is exactly where
 * the floor intersection runs away fastest (`1/sin²(depression)`, per the engine's own
 * `_point_confidence`). 35 % is deliberately large enough to read as "this is a guess" rather
 * than as a measurement with a decoration on it.
 */
const MAX_RELATIVE_ERROR = 0.35;

/** Below this the band is noise, and printing "± 0,00 m" would claim exactness. */
const BAND_FLOOR_CM = 0.5;

/**
 * How wrong a length of `lengthCm` could be, given the engine's `confidence`.
 *
 * Linear between the two bounds above. There is no calibration behind a smarter curve — this
 * tool has not been benched against a tape measure across a set of real rooms — and inventing
 * one would be dressing a guess up as a model. A straight line between two defensible bounds
 * is the most that is honest today; if the numbers ever do get measured, this is the function
 * to replace, and the only one.
 */
export function toleranceCm(lengthCm: number, confidence: number): number {
  const trust = clamp(confidence, 0, 1);
  return (
    Math.abs(lengthCm) * (MIN_RELATIVE_ERROR + (MAX_RELATIVE_ERROR - MIN_RELATIVE_ERROR) * (1 - trust))
  );
}

/**
 * Centimetres as metres, two decimals, in the locale's own notation.
 *
 * Turkish writes the decimal separator as a comma, and `PlanView.metres` already does this for
 * the floor plan — the two readouts sit in the same product and must not disagree about what a
 * number looks like.
 */
export function formatMetres(cm: number, locale: Locale): string {
  const text = (cm / 100).toFixed(2);
  return locale === "tr" ? text.replace(".", ",") : text;
}

/**
 * A length, with the band that says what its digits are worth.
 *
 * TWO DECIMALS AND A ± ARE NOT CONTRADICTORY, they are the whole point. The two decimals are
 * what the geometry actually produced, and they are needed for the numbers to add up — a
 * polyline whose segments are each rounded to 10 cm does not sum to its own total, and a user
 * who checks will trust none of it. The band is what those digits are worth. Printing "3,42 m"
 * alone would be a claim of centimetre accuracy the assumed camera height cannot support;
 * printing "3,4 m" alone would hide the disagreement rather than state it.
 */
export function formatLength(cm: number, confidence: number | null, locale: Locale): string {
  const value = formatMetres(cm, locale);
  if (confidence === null) return `${value} m`;
  const band = toleranceCm(cm, confidence);
  if (band < BAND_FLOOR_CM) return `${value} m`;
  return `${value} ± ${formatMetres(band, locale)} m`;
}

/** The height field's value, brought back into the range the assumption can be true over. */
export function clampCameraHeight(cm: number): number {
  if (!Number.isFinite(cm)) return DEFAULT_CAMERA_HEIGHT_CM;
  return Math.round(clamp(cm, CAMERA_HEIGHT_MIN_CM, CAMERA_HEIGHT_MAX_CM));
}

// ---------------------------------------------------------------------------------
// Per-frame projection
// ---------------------------------------------------------------------------------

/** One marked direction, and the element that draws it. */
export interface MeasureMarkNode {
  yaw: number;
  pitch: number;
  root: HTMLElement | null;
  /** Screen position written by the last projection; read again by the segment pass. */
  x: number;
  y: number;
  /** False when this mark is behind the eye, which makes every segment touching it undrawable. */
  onScreen: boolean;
  lastVisible: boolean | null;
}

/** The line between mark `i` and mark `i + 1`, and the label that sits on it. */
export interface MeasureSegmentNode {
  line: SVGLineElement | null;
  label: HTMLElement | null;
  lastVisible: boolean | null;
}

/**
 * The overlay's elements, as the frame loop sees them.
 *
 * AN OBJECT WITH AN API RATHER THAN TWO BARE ARRAYS, and not for tidiness. This is reached
 * through a ref that the overlay component receives as a prop, and React Compiler forbids a
 * component writing through its own props — with good reason, since a prop that is silently
 * mutated is a value React has been told nothing about. Everything the overlay does to the
 * registry therefore goes through a method, which is exactly the shape `HotspotRegistry` gets
 * for free by being a `Map`. The arrays stay readable, because the projector below is not a
 * component and reads them straight.
 */
export interface MeasureRegistry {
  readonly marks: MeasureMarkNode[];
  readonly segments: MeasureSegmentNode[];
  /** Create or update the node for mark `index`, and attach its element. */
  putMark(index: number, yaw: number, pitch: number, element: HTMLElement | null): void;
  /** Attach segment `index`'s line, or its label. */
  putLine(index: number, element: SVGLineElement | null): void;
  putLabel(index: number, element: HTMLElement | null): void;
  /** Drop everything past the lists React has just drawn. */
  resize(markCount: number): void;
}

export function createMeasureRegistry(): MeasureRegistry {
  const marks: MeasureMarkNode[] = [];
  const segments: MeasureSegmentNode[] = [];

  const segmentAt = (index: number): MeasureSegmentNode => {
    const existing = segments[index];
    if (existing) return existing;
    const created: MeasureSegmentNode = { line: null, label: null, lastVisible: null };
    segments[index] = created;
    return created;
  };

  return {
    marks,
    segments,

    putMark(index, yaw, pitch, element) {
      const node: MeasureMarkNode = marks[index] ?? {
        yaw,
        pitch,
        root: null,
        x: 0,
        y: 0,
        onScreen: false,
        lastVisible: null,
      };
      // A DIFFERENT element means every "last written" value on this node describes a node that
      // has left the document, and `setVisible`'s memo would then short-circuit against it and
      // never write anything to the element that IS on screen — leaving it at the `hidden` it
      // was born with, permanently. The same trap `HotspotLayer.register` documents, and the
      // player remounts this whole overlay every time the tool is switched off and on again.
      if (node.root !== element) node.lastVisible = null;
      node.root = element;
      node.yaw = yaw;
      node.pitch = pitch;
      marks[index] = node;
    },

    putLine(index, element) {
      const node = segmentAt(index);
      if (node.line !== element) node.lastVisible = null;
      node.line = element;
    },

    putLabel(index, element) {
      const node = segmentAt(index);
      if (node.label !== element) node.lastVisible = null;
      node.label = element;
    },

    resize(markCount) {
      marks.length = markCount;
      segments.length = Math.max(markCount - 1, 0);
    },
  };
}

const _dir = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _ndc = new THREE.Vector3();

/**
 * Below this dot product with the view axis a mark is behind us; the same cutoff, and the same
 * reason, as `HotspotLayer`.
 *
 * Marks do NOT get that layer's radial fade towards the frame edge, and that is deliberate. A
 * tag fading out near the rim reads as "there is more over there". A MEASUREMENT fading out
 * reads as the tool being unsure of the mark — which it is not: the mark is exactly where the
 * finger went. It is on screen, or it is behind you, and there is nothing in between.
 */
const FACING_CUTOFF = 0.02;

/**
 * Position every mark, line and label against the current camera.
 *
 * Call once per frame from inside the r3f loop, after the camera pose has been written — the
 * same slot, and the same reason, as `projectHotspots`.
 */
export function projectMeasure(
  registry: MeasureRegistry,
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number,
): void {
  const { marks, segments } = registry;
  if (marks.length === 0 || width === 0 || height === 0) return;
  camera.getWorldDirection(_forward);

  for (const mark of marks) {
    // Holes are normal for exactly one commit: the arrays are truncated to their new length by
    // an effect, and effects run after the ref callbacks that fill them, so a shrink can leave
    // the tail empty until React comes round to it.
    if (!mark) continue;

    directionFromAngles(mark.yaw, mark.pitch, _dir);
    mark.onScreen = _dir.dot(_forward) > FACING_CUTOFF;
    if (!mark.onScreen) {
      setVisible(mark, mark.root, false);
      continue;
    }

    // The mark's direction is a point one unit in front of the eye, so projecting it is
    // exactly the perspective divide the sphere behind it gets.
    _ndc.copy(_dir).project(camera);
    mark.x = (_ndc.x * 0.5 + 0.5) * width;
    mark.y = (-_ndc.y * 0.5 + 0.5) * height;

    setVisible(mark, mark.root, true);
    if (mark.root) {
      mark.root.style.transform = `translate3d(${mark.x.toFixed(2)}px, ${mark.y.toFixed(2)}px, 0)`;
    }
  }

  for (let index = 0; index < segments.length; index += 1) {
    const node = segments[index];
    if (!node) continue;
    const a = marks[index];
    const b = marks[index + 1];
    const visible = Boolean(a && b && a.onScreen && b.onScreen);

    // One memo for the pair: a segment's line and its label are always shown and hidden
    // together, so there is nothing to be gained by tracking them apart and a real chance of
    // a label surviving its own line if they ever fell out of step.
    if (node.lastVisible !== visible) {
      node.lastVisible = visible;
      const css = visible ? "visible" : "hidden";
      if (node.line) node.line.style.visibility = css;
      if (node.label) node.label.style.visibility = css;
    }
    if (!visible || !a || !b) continue;

    if (node.line) {
      node.line.setAttribute("x1", a.x.toFixed(1));
      node.line.setAttribute("y1", a.y.toFixed(1));
      node.line.setAttribute("x2", b.x.toFixed(1));
      node.line.setAttribute("y2", b.y.toFixed(1));
    }
    if (node.label) {
      // The midpoint, not an offset beside it. The label IS the segment's identity, and one
      // floating clear of its line in a room with four other lines has to be traced back.
      const x = (a.x + b.x) * 0.5;
      const y = (a.y + b.y) * 0.5;
      node.label.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0)`;
    }
  }
}

/**
 * Toggle one mark's visibility, remembering what was last written so the hot path only touches
 * an attribute that actually changed.
 *
 * The memo lives on the NODE rather than on the element, which is why `putMark` has to clear it
 * whenever a different element arrives — see the note there.
 */
function setVisible(
  node: { lastVisible: boolean | null },
  element: HTMLElement | SVGElement | null,
  visible: boolean,
): void {
  if (node.lastVisible === visible) return;
  node.lastVisible = visible;
  if (element) element.style.visibility = visible ? "visible" : "hidden";
}

// ---------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------

export type MeasurePhase = "idle" | "working" | "ready" | "failed";

/** One point the user put on the floor. `id` exists only to key the React list. */
export interface MeasureMark {
  id: number;
  yaw: number;
  pitch: number;
}

export interface MeasureState {
  active: boolean;
  /** Marks belonging to the room currently on screen, in the order they were tapped. */
  points: MeasureMark[];
  /**
   * The most recent answer that still describes THIS room at THIS assumed height. It may cover
   * fewer marks than are on screen while the request for the newest one is in flight; `covers`
   * says how many.
   */
  result: MeasureResponse | null;
  covers: number;
  phase: MeasurePhase;
  /** Localised, ready to render. Null unless `phase === "failed"`. */
  message: string | null;
  /** The assumed camera height the current result was computed at, in centimetres. */
  cameraHeightCm: number;
  toggle: () => void;
  add: (point: { yaw: number; pitch: number }) => void;
  undo: () => void;
  clear: () => void;
  /** Ask again at a different assumed height. */
  remeasure: (cameraHeightCm: number) => void;
}

const NO_MARKS: MeasureMark[] = [];

/**
 * Everything the tool remembers, and when it asks the engine.
 *
 * ── WHY IT RE-ASKS ON EVERY TAP ───────────────────────────────────────────────────────
 * `/ai/measure` is exact projective geometry over a handful of angles: no model, no weights,
 * and it does not even open the panorama. It is arithmetic behind an HTTP call, so a request
 * per tap is cheaper than any caching scheme worth writing — and, more importantly, it keeps
 * ONE code path for "a point was added", "a point was taken back" and "the height changed".
 * Three paths would be three chances for the polyline on screen to disagree with the total
 * underneath it.
 *
 * ── WHY MARKS ARE STORED WITH THE ROOM THEY WERE MADE IN ──────────────────────────────
 * The same reasoning as `useEmptyRoom`. Walking into the next room has to drop the marks —
 * they are directions in the old room's sphere, and carried across they would land on whatever
 * happens to lie that way in the new one — and an effect that achieves that by calling
 * `setState` renders twice on every room change, in a component that owns a WebGL canvas.
 * Storing which scene the marks belong to costs one string and makes the reset free. The tool
 * itself STAYS ON across the change: somebody measuring one room is usually about to measure
 * the next.
 */
export function useMeasure(
  sceneId: string | null,
  copy: { failed: string },
  /**
   * The embed token, when this player is on somebody else's page.
   *
   * Two doors, one arithmetic. `/ai/measure` sits behind the restricted CORS policy, so a
   * player framed on a listing portal cannot reach it; `/public/evler/{token}/measure` is the
   * same call under the one prefix that answers arbitrary origins. It is allowed to be public
   * because measuring reads no pixels and starts no model — it intersects a marked ray with
   * the floor plane — and the engine still checks the scene belongs to this token's property.
   */
  publicToken?: string | null,
): MeasureState {
  const [active, setActive] = useState(false);
  const [marked, setMarked] = useState<{ scene: string; marks: MeasureMark[] }>({
    scene: "",
    marks: NO_MARKS,
  });
  const [answer, setAnswer] = useState<{
    scene: string;
    covers: number;
    heightCm: number;
    body: MeasureResponse;
  } | null>(null);
  /**
   * Which exact question failed, rather than a bare "something went wrong" flag.
   *
   * `phase` and `message` are DERIVED from this and from `answer` (see below) instead of being
   * state of their own. A "working" that has to be set from inside the effect that starts the
   * request is a second render for every tap — and React Compiler rightly refuses it — while a
   * "failed" that is not tied to the question that failed survives the next tap and hangs an
   * error over a measurement that has since succeeded.
   */
  const [failure, setFailure] = useState<{ scene: string; count: number; heightCm: number } | null>(
    null,
  );
  const [cameraHeightCm, setCameraHeightCm] = useState(DEFAULT_CAMERA_HEIGHT_CM);

  const nextId = useRef(0);
  /**
   * Which request is still wanted.
   *
   * `api.ai.measure` takes no `AbortSignal` — that wrapper is another agent's file and not mine
   * to widen — so a superseded request is IGNORED rather than cancelled. Which is the right
   * behaviour here in any case: the call costs the server microseconds, and the failure that
   * actually matters is an older answer landing after a newer one and quietly relabelling the
   * polyline. A monotonic sequence number closes exactly that.
   */
  const wanted = useRef(0);

  const points = sceneId !== null && marked.scene === sceneId ? marked.marks : NO_MARKS;

  useEffect(() => {
    if (!active || !sceneId || points.length < 2) return;

    const seq = (wanted.current += 1);
    const asked = { scene: sceneId, count: points.length, heightCm: cameraHeightCm };

    const body = {
      scene_id: sceneId,
      points: points.map((mark) => ({ yaw: mark.yaw, pitch: mark.pitch })),
      camera_height_cm: cameraHeightCm,
    };
    void (publicToken ? api.public.measure(publicToken, body) : api.ai.measure(body))
      .then((body) => {
        if (seq !== wanted.current) return;
        setAnswer({ scene: sceneId, covers: asked.count, heightCm: cameraHeightCm, body });
        setFailure(null);
      })
      .catch(() => {
        if (seq !== wanted.current) return;
        setFailure(asked);
      });

    // Bumping the sequence on cleanup is what makes turning the tool off, clearing the marks,
    // or stepping into another room take any in-flight answer with it.
    return () => {
      wanted.current += 1;
    };
  }, [active, sceneId, points, cameraHeightCm, publicToken]);

  /**
   * The answer, if it is still about what is on screen.
   *
   * `covers <= points.length` rather than `===`, on purpose. Appending a mark cannot change any
   * EARLIER segment — segment i is fixed by marks i and i+1, and both were already placed — so
   * while the request for the new one is in flight the labels already on the picture are still
   * correct and stay put. Demanding an exact match would blank every label on screen for the
   * length of a round trip, once per tap, which reads as the tool losing its place.
   *
   * The height is matched exactly, because changing it changes every length at once.
   */
  const usable =
    answer !== null &&
    answer.scene === sceneId &&
    answer.heightCm === cameraHeightCm &&
    answer.covers <= points.length
      ? answer
      : null;

  /**
   * Where the tool has got to, read off what it has rather than tracked alongside it.
   *
   * "working" is the absence of an answer covering every mark, which is true from the instant
   * the second one lands and needs nothing set anywhere. A failure only counts while it still
   * describes the question on screen — add a mark, take one back, or change the height, and it
   * is about a question nobody is asking any more.
   */
  const failed =
    failure !== null &&
    failure.scene === sceneId &&
    failure.count === points.length &&
    failure.heightCm === cameraHeightCm;
  const phase: MeasurePhase =
    points.length < 2
      ? "idle"
      : failed
        ? "failed"
        : usable !== null && usable.covers === points.length
          ? "ready"
          : "working";
  const message = failed ? copy.failed : null;

  const clear = useCallback(() => {
    setMarked({ scene: sceneId ?? "", marks: NO_MARKS });
    setAnswer(null);
    setFailure(null);
    wanted.current += 1;
  }, [sceneId]);

  const toggle = useCallback(() => {
    // Leaving the tool takes the marks with it. They only mean anything inside it, and coming
    // back later to somebody else's half-drawn polyline is worse than starting over.
    //
    // Written as a branch on the current value rather than as `setActive(on => …)` with the
    // clear inside: an updater is a pure function React may call more than once, and setting
    // another piece of state from inside one is an update during render.
    if (active) clear();
    setActive(!active);
  }, [active, clear]);

  const add = useCallback(
    (point: { yaw: number; pitch: number }) => {
      if (!sceneId) return;
      const mark: MeasureMark = { id: (nextId.current += 1), yaw: point.yaw, pitch: point.pitch };
      setMarked((previous) =>
        previous.scene === sceneId
          ? { scene: sceneId, marks: [...previous.marks, mark] }
          : { scene: sceneId, marks: [mark] },
      );
    },
    [sceneId],
  );

  const undo = useCallback(() => {
    setMarked((previous) =>
      previous.marks.length === 0 ? previous : { ...previous, marks: previous.marks.slice(0, -1) },
    );
  }, []);

  const remeasure = useCallback((height: number) => {
    setCameraHeightCm(clampCameraHeight(height));
  }, []);

  /**
   * Escape, bound to the window rather than to the player.
   *
   * The player's key handling lives in `useLookControls`, which is not mine to extend, and it is
   * bound to the player element — so it only fires while that element itself has focus. Escape
   * has to work from the height field and from the buttons too, and those are inside the player
   * but perfectly capable of holding focus of their own.
   *
   * TWO STEPS, DELIBERATELY. With marks down, Escape clears them; with nothing marked, it puts
   * the tool away. One key that always exited would let a mistyped press throw away a five-point
   * polyline with no way back, and a key that only ever cleared would leave the tool armed with
   * no obvious way out for someone on a keyboard.
   */
  useEffect(() => {
    if (!active) return;
    const hasMarks = points.length > 0;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (hasMarks) clear();
      else setActive(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active, points.length, clear]);

  return {
    active,
    points,
    result: usable?.body ?? null,
    covers: usable?.covers ?? 0,
    phase,
    message,
    cameraHeightCm,
    toggle,
    add,
    undo,
    clear,
    remeasure,
  };
}

// ---------------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------------

/** Localised sentence for each thing the server can say the scale came from. */
const SCALE_KEY = {
  reference: "player.measure.scale.reference",
  camera_height: "player.measure.scale.cameraHeight",
  relative_only: "player.measure.scale.relative",
} as const;

const FAULT_KEY = {
  horizon: "player.measure.hint.horizon",
  grazing: "player.measure.hint.grazing",
} as const;

export interface MeasureToolProps {
  locale: Locale;
  state: MeasureState;
  /** Written to every frame by the player's frame loop; see `projectMeasure`. */
  registry: React.RefObject<MeasureRegistry>;
}

export function MeasureTool({ locale, state, registry }: MeasureToolProps) {
  const { active, points, result, covers, phase, message, cameraHeightCm } = state;

  /**
   * What the height field currently reads, which is NOT what the numbers were computed at.
   *
   * A field that re-measured on every keystroke would fire a request for "1", then "15", then
   * "155" — and the middle one is a room a sixth of its true size, briefly on screen and
   * entirely believable. Worse, a half-typed value cannot be clamped without fighting the
   * person typing it. So the draft is a free string, and "yeniden ölç" is what makes it real.
   *
   * Nothing syncs it back afterwards, because nothing else can move the applied height: the
   * button below is the only caller of `remeasure`, and it writes the clamped value it just
   * applied straight back into the field. An effect mirroring `cameraHeightCm` into here would
   * be a second render per press and a rule violation, for a case that cannot arise.
   */
  const [draft, setDraft] = useState(() => String(cameraHeightCm));

  const parsedDraft = Number.parseInt(draft, 10);
  const draftValid =
    Number.isFinite(parsedDraft) &&
    parsedDraft >= CAMERA_HEIGHT_MIN_CM &&
    parsedDraft <= CAMERA_HEIGHT_MAX_CM;
  const canRemeasure = draftValid && parsedDraft !== cameraHeightCm && points.length >= 2;

  const lineCount = Math.max(points.length - 1, 0);

  /**
   * Keep the registry's arrays the same length as the lists React just drew.
   *
   * Growth needs nothing — the ref callbacks below run during the same commit, before any
   * effect — so this only ever has to TRUNCATE. Doing it here rather than inside the callbacks
   * is what makes a shrink safe: React hands a detached ref its `null` and attaches the
   * survivors in the same commit, so a callback that deleted on `null` would drop nodes that
   * are still on screen. The projector tolerates the one-commit hole this can leave behind.
   */
  useEffect(() => {
    registry.current.resize(points.length);
  }, [registry, points.length]);

  // Plain functions, not `useCallback`. Their identity changes every render, which means React
  // detaches and re-attaches every ref on every render — and that is fine here and handled: the
  // registry keys off the index, and re-attaching the SAME element is a no-op while attaching a
  // different one resets the node's memo. Wrapping them would buy nothing (the compiler already
  // memoizes what it can) and would need `registry.current` in a dependency list, which is a
  // ref read pretending to be a value.
  const registerMark =
    (index: number, yaw: number, pitch: number) => (element: HTMLDivElement | null) => {
      registry.current.putMark(index, yaw, pitch, element);
    };
  const registerLine = (index: number) => (element: SVGLineElement | null) => {
    registry.current.putLine(index, element);
  };
  const registerLabel = (index: number) => (element: HTMLDivElement | null) => {
    registry.current.putLabel(index, element);
  };

  /**
   * What each segment's pill says: a length, or the reason there is not one.
   *
   * The fault is classified from the pitch that was MARKED rather than parsed out of the
   * server's English sentence, so that the sentence can be Turkish. Where our reading and the
   * server's disagree — neither endpoint looks wrong to us, yet no length came back — the
   * server's own words are shown verbatim. That case should not happen; showing the raw hint is
   * how it would be noticed instead of swallowed.
   */
  const labels = useMemo(() => {
    const out: Array<{ text: string; fault: boolean } | null> = [];
    for (let index = 0; index < lineCount; index += 1) {
      const segment = result && index < covers - 1 ? result.segments[index] : null;
      if (!segment) {
        out.push(null);
        continue;
      }
      if (segment.length_cm !== null) {
        out.push({
          text: formatLength(segment.length_cm, result?.confidence ?? null, locale),
          fault: false,
        });
        continue;
      }
      // The engine blames the first bad endpoint of the pair, so look at them in that order.
      const fault = markFault(points[index].pitch) ?? markFault(points[index + 1].pitch);
      // `hint` is additive on the wire (see `schemas.MeasureSegment`) and deliberately absent
      // from `lib/types.ts`, which is another agent's file. Read it defensively rather than
      // widening a shared contract from here.
      const raw = (segment as { hint?: string | null }).hint;
      out.push({
        text: fault ? t(FAULT_KEY[fault], locale) : (raw ?? t("player.measure.failed", locale)),
        fault: true,
      });
    }
    return out;
  }, [result, covers, lineCount, points, locale]);

  /** The first thing that went wrong — repeated in the panel; see there for why. */
  const firstFault = useMemo(() => labels.find((label) => label?.fault) ?? null, [labels]);

  // A total over fewer marks than are on screen is a different number, so it waits for the
  // answer that covers all of them rather than briefly showing a shorter walk.
  const total = result && covers === points.length ? result.total_cm : null;

  const hint =
    points.length === 0
      ? t("player.measure.start", locale)
      : t("player.measure.second", locale);

  if (!active) return null;

  const scaleNote = t(SCALE_KEY[result?.scale_source ?? "camera_height"], locale).replace(
    "{h}",
    String(cameraHeightCm),
  );

  return (
    <>
      {/* THE PICTURE. `pointer-events-none` throughout and without exception: every pixel of
          this sits over the floor the user is trying to tap, and a mark that swallowed the tap
          meant for the spot beside it would look exactly like the tool having stopped working.

          NO z-index, deliberately — the same stacking `HotspotLayer` uses. It is above the
          canvas because it comes after it in the document, and BELOW the player's doorway dim
          at `z-10`, which is what has to be true: a room change clears the marks, but the dim
          closes before the swap, and marks stacked over the dim would be left glowing on the
          black for the length of the transition. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <svg
          aria-hidden
          className="absolute inset-0 h-full w-full"
          // No `viewBox`, so user units are CSS pixels — which is the space the projector
          // already works in. A dark drop shadow on the layer rather than a second, wider
          // stroke under every line: a measurement drawn over a photograph has to survive
          // landing on a white wall and on a dark floor, and one filter is cheaper than
          // doubling the element count.
          style={{ filter: "drop-shadow(0 1px 2px rgb(0 0 0 / 0.8))" }}
        >
          {Array.from({ length: lineCount }, (_, index) => (
            <line
              key={index}
              ref={registerLine(index)}
              stroke="var(--bio)"
              strokeWidth={2}
              strokeLinecap="round"
              style={{ visibility: "hidden" }}
            />
          ))}
        </svg>

        {points.map((mark, index) => (
          <div
            key={mark.id}
            ref={registerMark(index, mark.yaw, mark.pitch)}
            className="absolute top-0 left-0 h-0 w-0"
            style={{ visibility: "hidden", willChange: "transform" }}
          >
            <span
              aria-hidden
              className="absolute grid h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 place-items-center
                         rounded-full border-2 border-bio bg-canvas/80 shadow-[0_1px_3px_rgb(0_0_0_/_0.7)]"
            >
              <span className="h-1 w-1 rounded-full bg-bio" />
            </span>
          </div>
        ))}

        {Array.from({ length: lineCount }, (_, index) => {
          const label = labels[index];
          return (
            <div
              key={index}
              ref={registerLabel(index)}
              className="absolute top-0 left-0 h-0 w-0"
              style={{ visibility: "hidden", willChange: "transform" }}
            >
              {/* Nothing at all until the answer lands. An empty pill sitting on the line
                  would be a promise the server has not made yet; the bare line says only
                  "these two spots", which is true from the moment the second one is tapped. */}
              {label ? (
                <span
                  className={cn(
                    "glass-strong absolute -translate-x-1/2 -translate-y-1/2 rounded-full",
                    "px-2 py-1 text-[11px] leading-snug font-medium tracking-tight tabular-nums",
                    "shadow-[var(--shadow-lift)]",
                    label.fault
                      ? "max-w-[min(15rem,62vw)] text-left text-[10.5px] whitespace-normal text-amber"
                      : "whitespace-nowrap text-ink",
                  )}
                >
                  {label.text}
                </span>
              ) : null}
            </div>
          );
        })}
      </div>

      {/* THE PANEL, top right. The toolbar owns the bottom edge, the room bar the top centre
          and the plan the bottom left; this is the one corner of the frame still free. */}
      <div
        data-player-ui
        // Escape is this tool's own key and has to reach the window listener in `useMeasure`
        // even while the height field holds focus, so it is the one key allowed out. Everything
        // else is stopped here, because `useLookControls` binds `keydown` on the player element
        // with no overlay guard of its own (its pointer and wheel handlers have one; its key
        // handler does not). Without this, an arrow key pressed to nudge the height field spins
        // the room instead AND is `preventDefault`ed, so the nudge never happens — and typing a
        // "-" or a "w" or an "f" into the field zooms out, turns left, or goes fullscreen.
        onKeyDown={(event) => {
          if (event.key !== "Escape") event.stopPropagation();
        }}
        // MIDDLE OF THE LEFT EDGE, which looks like an odd place for a panel until you list
        // what else is on this stage. The scene editor floats its own tag rail down the whole
        // right-hand side, over the picture, so top-right and bottom-right are both unreadable
        // there — measured by putting the panel at top-right and finding "2,48 ± 0,25 m" behind
        // the editor's translucent panel. The top-left corner carries the host's scene chip,
        // the bottom-left is this tool's own rail, and the bottom centre is the toolbar. The
        // left flank is the one region free in the editor, in the dashboard preview and in an
        // embed alike, and a measuring session is short enough that covering it is fair.
        className="absolute top-1/2 left-0 z-30 flex max-w-[min(20rem,86vw)] -translate-y-1/2
                   flex-col px-3"
      >
        <div
          role="group"
          aria-label={t("player.measure", locale)}
          className="glass-strong animate-rise flex flex-col gap-2 rounded-[var(--radius-lg)] p-3
                     shadow-[var(--shadow-lift)]"
        >
          <div className="flex items-start justify-between gap-3">
            <span className="flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-ink-faint uppercase">
              <Ruler size={12} aria-hidden />
              {t("player.measure", locale)}
            </span>
            <button
              type="button"
              onClick={state.toggle}
              aria-label={t("player.measure.off", locale)}
              title={t("player.measure.off", locale)}
              className="-m-1 grid size-6 place-items-center rounded-full p-1 text-ink-faint
                         transition-colors hover:bg-white/[0.06] hover:text-ink focus-visible:text-ink"
            >
              <X size={13} aria-hidden />
            </button>
          </div>

          {/* The readout. `aria-live` because the number arrives with nothing focused — the
              user's attention was on the floor, not on this panel. */}
          <p aria-live="polite" className="flex items-baseline gap-2">
            <span className="text-[19px] leading-none font-semibold tracking-tight text-ink tabular-nums">
              {total !== null ? formatLength(total, result?.confidence ?? null, locale) : "—"}
            </span>
            {points.length > 2 && total !== null ? (
              <span className="text-[11px] text-ink-faint">{t("player.measure.total", locale)}</span>
            ) : null}
            {phase === "working" ? (
              <Loader2
                size={13}
                className="animate-spin text-ink-faint"
                aria-label={t("player.measure.working", locale)}
              />
            ) : null}
          </p>

          {/* A faulted segment's pill is drawn on the line it belongs to — and that line can be
              behind the viewer's head, which is exactly the case where they are left staring at
              a missing total with no explanation anywhere on screen. So the first fault is
              repeated here, where it cannot be turned away from. */}
          {firstFault ? <p className="text-[11px] leading-snug text-amber">{firstFault.text}</p> : null}
          {message ? <p className="text-[11px] leading-snug text-amber">{message}</p> : null}
          {points.length < 2 ? <p className="text-[11px] leading-snug text-ink-muted">{hint}</p> : null}

          {/* WHERE THE SCALE CAME FROM, ALWAYS, and not folded away behind a tooltip. This
              sentence is the difference between a measurement and a number. */}
          <p className="text-[11px] leading-snug text-ink-faint">{scaleNote}</p>

          <div className="flex items-end gap-2">
            <label className="flex min-w-0 flex-col gap-1">
              <span className="text-[10px] tracking-wide text-ink-faint uppercase">
                {t("player.measure.height", locale)}
              </span>
              <span className="flex items-center gap-1">
                <input
                  type="number"
                  inputMode="numeric"
                  min={CAMERA_HEIGHT_MIN_CM}
                  max={CAMERA_HEIGHT_MAX_CM}
                  step={1}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  className={cn(
                    "w-[4.5rem] rounded-[var(--radius-xs)] border bg-white/[0.04] px-2 py-1",
                    "text-[12px] text-ink tabular-nums outline-none focus-visible:border-bio/60",
                    draftValid ? "border-line" : "border-amber/60",
                  )}
                />
                <span className="text-[11px] text-ink-faint">cm</span>
              </span>
            </label>

            <button
              type="button"
              onClick={() => {
                // Apply the clamp here as well as in the hook, and write the result back into
                // the field: a user who typed 90 has to SEE the 100 their numbers were actually
                // computed at, or the panel says one thing and the measurement means another.
                const applied = clampCameraHeight(parsedDraft);
                setDraft(String(applied));
                state.remeasure(applied);
              }}
              disabled={!canRemeasure}
              className={cn(
                "h-[1.85rem] shrink-0 rounded-full border px-3 text-[11.5px] font-medium",
                "transition-all duration-150",
                canRemeasure
                  ? "border-bio/50 bg-bio/12 text-bio active:scale-[0.97]"
                  : "cursor-default border-line text-ink-faint",
              )}
            >
              {t("player.measure.remeasure", locale)}
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={state.undo}
              disabled={points.length === 0}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px]",
                points.length === 0
                  ? "cursor-default text-ink-faint/50"
                  : "text-ink-faint hover:bg-white/[0.06] hover:text-ink",
              )}
            >
              <Undo2 size={12} aria-hidden />
              {t("player.measure.undo", locale)}
            </button>
            <button
              type="button"
              onClick={state.clear}
              disabled={points.length === 0}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px]",
                points.length === 0
                  ? "cursor-default text-ink-faint/50"
                  : "text-ink-faint hover:bg-white/[0.06] hover:text-ink",
              )}
            >
              <Trash2 size={12} aria-hidden />
              {t("player.measure.clear", locale)}
            </button>
          </div>

          {/* NOT under the rail's beta tag, and this line says why. The two tools beside it
              repaint the photograph with something a model invented; this one comes back
              `model_backed: false` on the wire — exact geometry from two directions and one
              assumed height. Lumping it in with them would understate it, and leaving it
              unlabelled beside them would leave a viewer guessing which of the three made
              something up. */}
          <p className="border-t border-line pt-2 text-[10.5px] leading-snug text-ink-faint">
            {t("player.measure.exact", locale)}
          </p>
        </div>
      </div>
    </>
  );
}
