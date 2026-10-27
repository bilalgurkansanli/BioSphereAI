/**
 * The capture state machine, exercised exhaustively.
 *
 * `nextCaptureState` lives in the contract rather than in Builder C's modules, but it is
 * pure and it is the spine every other module hangs off, so it is tested here — where the
 * test runner already is — rather than nowhere.
 */

import { describe, expect, it } from "vitest";

import {
  CAPTURE_STATES,
  CAPTURE_TRANSITIONS,
  RECORDING_STATES,
  ROW_ID_BY_STATE,
  ROW_IDS,
  ROW_STATES,
  ROW_STATE_BY_ID,
  WARNING_COPY,
  WARNING_PRIORITY,
  initialCaptureState,
  nextCaptureState,
  type CaptureEventType,
  type CaptureState,
  type WarningCode,
} from "@/lib/capture/types";

const EVENTS: CaptureEventType[] = [
  "START",
  "USE_EXISTING_VIDEO",
  "PERMISSIONS_GRANTED",
  "PERMISSIONS_DENIED_CAMERA",
  "PERMISSIONS_DENIED_MOTION",
  "RETRY_PERMISSIONS",
  "READY",
  "BACK",
  "GATES_PASSED",
  "FORCE_START",
  "ROW_COMPLETE",
  "PAUSE",
  "RESUME",
  "FINISH",
  "RESTART",
  "DISCARD_DRAFT",
  "RESUME_DRAFT",
  "CHECK_DONE",
  "FILL_GAPS",
  "ACCEPT",
  "RETAKE",
  "WANT_POLES",
  "SKIP_POLES",
  "UPLOAD_OK",
  "UPLOAD_FAILED",
  "RETRY_UPLOAD",
];

const ALL_GUARDS = [
  {},
  { hasDraft: true },
  { hasGaps: true },
  { coverageComplete: true },
  { motionGranted: true },
  { hasDraft: true, hasGaps: true, coverageComplete: true, motionGranted: true },
];

describe("the transition table", () => {
  it("has an entry for every state and no entry for anything else", () => {
    expect(Object.keys(CAPTURE_TRANSITIONS).sort()).toEqual([...CAPTURE_STATES].sort());
  });

  it("never names a state that does not exist", () => {
    for (const state of CAPTURE_STATES) {
      for (const transition of CAPTURE_TRANSITIONS[state]) {
        if (transition.to === "@resume") continue;
        expect(CAPTURE_STATES).toContain(transition.to);
      }
    }
  });

  it("never lists the same event twice unguarded in one state", () => {
    for (const state of CAPTURE_STATES) {
      const unguarded = CAPTURE_TRANSITIONS[state].filter((t) => t.when === undefined && t.unless === undefined);
      expect(new Set(unguarded.map((t) => t.on)).size).toBe(unguarded.length);
    }
  });

  it("uses every declared event somewhere — no dead events in the union", () => {
    const used = new Set<CaptureEventType>();
    for (const state of CAPTURE_STATES) for (const t of CAPTURE_TRANSITIONS[state]) used.add(t.on);
    expect([...EVENTS].filter((e) => !used.has(e))).toEqual([]);
  });
});

describe("nextCaptureState over CAPTURE_STATES x CaptureEventType", () => {
  it("returns a real state or null for all 494 pairs, and never throws", () => {
    let pairs = 0;
    for (const state of CAPTURE_STATES) {
      for (const event of EVENTS) {
        for (const guards of ALL_GUARDS) {
          pairs += 1;
          const next = nextCaptureState(state, event, guards, "row_up");
          if (next !== null) expect(CAPTURE_STATES).toContain(next);
        }
      }
    }
    expect(pairs).toBe(CAPTURE_STATES.length * EVENTS.length * ALL_GUARDS.length);
  });

  it("returns null for an illegal event rather than throwing or guessing", () => {
    expect(nextCaptureState("intro", "UPLOAD_OK")).toBeNull();
    expect(nextCaptureState("review", "GATES_PASSED")).toBeNull();
    expect(nextCaptureState("row_level", "ACCEPT")).toBeNull();
  });

  it("is deterministic", () => {
    for (const state of CAPTURE_STATES) {
      for (const event of EVENTS) {
        const a = nextCaptureState(state, event, { hasDraft: true, hasGaps: true }, "row_down");
        const b = nextCaptureState(state, event, { hasDraft: true, hasGaps: true }, "row_down");
        expect(a).toBe(b);
      }
    }
  });

  it("leaves the terminal states terminal", () => {
    for (const event of EVENTS) {
      expect(nextCaptureState("handoff", event)).toBeNull();
      expect(nextCaptureState("exit_to_upload", event)).toBeNull();
    }
  });
});

describe("guards", () => {
  it("only offers the draft when one exists", () => {
    expect(nextCaptureState("intro", "RESUME_DRAFT", { hasDraft: false })).toBeNull();
    expect(nextCaptureState("intro", "RESUME_DRAFT", { hasDraft: true })).toBe("resume_draft");
    expect(nextCaptureState("intro", "RESUME_DRAFT")).toBeNull();
  });

  it("only offers gap-filling when there are gaps", () => {
    expect(nextCaptureState("review", "FILL_GAPS", { hasGaps: false })).toBeNull();
    expect(nextCaptureState("review", "FILL_GAPS", { hasGaps: true })).toBe("touch_up");
  });

  it("starts at the draft when there is one and at the intro when there is not", () => {
    expect(initialCaptureState(false)).toBe("intro");
    expect(initialCaptureState(true)).toBe("resume_draft");
  });
});

describe("pause and resume", () => {
  it("returns to the row that was interrupted, from every recording row", () => {
    for (const state of ROW_STATES) {
      expect(nextCaptureState(state, "PAUSE")).toBe("paused");
      expect(nextCaptureState("paused", "RESUME", {}, state)).toBe(state);
    }
  });

  it("falls back to framing rather than stranding a session with no resume target", () => {
    expect(nextCaptureState("paused", "RESUME", {}, null)).toBe("framing");
  });

  it("lets a paused capture finish rather than forcing a resume", () => {
    expect(nextCaptureState("paused", "FINISH")).toBe("sweep_check");
    expect(nextCaptureState("paused", "RESTART")).toBe("framing");
  });

  it("survives a pause during touch-up and during the pole bonus", () => {
    expect(nextCaptureState("touch_up", "PAUSE")).toBe("paused");
    expect(nextCaptureState("bonus_poles", "PAUSE")).toBe("paused");
    expect(nextCaptureState("paused", "RESUME", {}, "touch_up")).toBe("touch_up");
  });
});

describe("recording is continuous across rows", () => {
  it("keeps every row state inside RECORDING_STATES", () => {
    for (const state of ROW_STATES) expect(RECORDING_STATES).toContain(state);
    expect(RECORDING_STATES).toContain("touch_up");
    expect(RECORDING_STATES).toContain("bonus_poles");
  });

  it("never leaves the recording set when one row hands over to the next", () => {
    // The pitch transitions between rows are the footage that ties the rows together in the
    // solver, so the recorder must not be stopped and restarted between them.
    expect(nextCaptureState("row_level", "ROW_COMPLETE")).toBe("row_up");
    expect(nextCaptureState("row_up", "ROW_COMPLETE")).toBe("row_down");
    for (const from of ["row_level", "row_up"] as const) {
      const to = nextCaptureState(from, "ROW_COMPLETE") as CaptureState;
      expect(RECORDING_STATES).toContain(to);
    }
  });

  it("leaves the recording set only at the last row", () => {
    const after = nextCaptureState("row_down", "ROW_COMPLETE") as CaptureState;
    expect(after).toBe("sweep_check");
    expect(RECORDING_STATES).not.toContain(after);
  });

  it("maps rows to states and back without loss", () => {
    for (const row of ROW_IDS) expect(ROW_ID_BY_STATE[ROW_STATE_BY_ID[row]]).toBe(row);
    for (const state of ROW_STATES) expect(ROW_STATE_BY_ID[ROW_ID_BY_STATE[state]]).toBe(state);
  });
});

describe("reachability", () => {
  it("can reach every state from the start, so nothing in the union is dead", () => {
    const seen = new Set<CaptureState>();
    const queue: CaptureState[] = [initialCaptureState(false), initialCaptureState(true)];
    while (queue.length > 0) {
      const state = queue.shift() as CaptureState;
      if (seen.has(state)) continue;
      seen.add(state);
      for (const event of EVENTS) {
        for (const guards of ALL_GUARDS) {
          for (const resume of [null, ...ROW_STATES, "touch_up" as const]) {
            const next = nextCaptureState(state, event, guards, resume);
            if (next !== null && !seen.has(next)) queue.push(next);
          }
        }
      }
    }
    expect([...CAPTURE_STATES].filter((s) => !seen.has(s))).toEqual([]);
  });

  it("gives every dead end a way out", () => {
    // A user must never be trapped. `handoff` and `exit_to_upload` are terminal by design —
    // they hand off to screens outside this machine.
    for (const state of CAPTURE_STATES) {
      if (state === "handoff" || state === "exit_to_upload") continue;
      expect(CAPTURE_TRANSITIONS[state].length).toBeGreaterThan(0);
    }
  });

  it("does not pretend a camera denial is recoverable in place", () => {
    expect(nextCaptureState("permissions", "PERMISSIONS_DENIED_CAMERA")).toBe("permission_denied");
    // Motion denial is survivable: RAIL MODE keeps every camera-derived meter alive.
    expect(nextCaptureState("permissions", "PERMISSIONS_DENIED_MOTION")).toBe("stance");
    expect(nextCaptureState("permission_denied", "RETRY_PERMISSIONS")).toBe("permissions");
    expect(nextCaptureState("permission_denied", "USE_EXISTING_VIDEO")).toBe("exit_to_upload");
  });
});

describe("warnings are a layer, not a state", () => {
  it("has no state named for a correction", () => {
    // Contract decision D7: `correction` and `rotate_portrait` are session flags, so a
    // warning never costs earned progress and a rotation never ends a capture.
    for (const state of CAPTURE_STATES) {
      expect(state).not.toBe("correction");
      expect(state).not.toBe("rotate_portrait");
    }
  });

  it("gives every warning code copy, an icon and a severity", () => {
    for (const code of WARNING_PRIORITY) {
      const meta = WARNING_COPY[code];
      expect(meta).toBeDefined();
      expect(meta.title.length).toBeGreaterThan(0);
      expect(meta.icon.length).toBeGreaterThan(0);
      expect(["info", "warn", "stop"]).toContain(meta.severity);
    }
  });

  it("ranks every warning code exactly once", () => {
    const codes = Object.keys(WARNING_COPY) as WarningCode[];
    expect([...WARNING_PRIORITY].sort()).toEqual(codes.sort());
    expect(new Set(WARNING_PRIORITY).size).toBe(WARNING_PRIORITY.length);
  });

  it("gives distinct icons to the warnings a user must tell apart in a 200 ms glance", () => {
    // Colour never carries a message alone, so the silhouettes for the three most urgent
    // corrections must differ.
    const urgent: WarningCode[] = ["landscape", "lens_changed", "walking", "too_fast"];
    const icons = urgent.map((c) => WARNING_COPY[c].icon);
    expect(new Set(icons).size).toBe(icons.length);
  });
});
