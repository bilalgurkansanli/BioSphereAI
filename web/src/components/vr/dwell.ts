/**
 * Gaze-and-hold: the hands-free click.
 *
 * A phone in a VR box has no controller and usually no button, so the only way to say "that
 * one" is to keep looking at it. That is a timer, and every way a naive timer goes wrong is
 * something a visitor feels as the tour misbehaving:
 *
 * - A HEAD IS NEVER STILL. A gaze resting on a ring's rim drops off it for a frame or two at a
 *   time; restarting the count on every drop means the ring can never be chosen. So a target
 *   survives `graceMs` of absence, and only a real departure resets it. The time spent off it
 *   does not count towards the dwell, so the fill pauses rather than jumping when the gaze
 *   comes back.
 * - IT MUST NOT REPEAT. After a ring fires, the visitor is often still looking at the place it
 *   was — or at the same spot in the next room, which may hold another ring. So a target that
 *   has fired cannot fire again until the gaze has left it, and nothing at all arms for
 *   `cooldownMs` after any fire, which covers the doorway fade.
 * - SWITCHING IS A FRESH START. Gaze moving from one ring straight to its neighbour counts the
 *   neighbour from zero; carrying progress across would make whichever ring was looked at
 *   last fire early.
 *
 * Pure state in, state out, clock passed in: the tests drive it with synthetic time.
 */

export interface DwellConfig {
  /** How long a gaze must rest on the target. */
  dwellMs: number;
  /** How long the gaze may leave and return without the count restarting. */
  graceMs: number;
  /** After a fire, how long before anything can arm again. */
  cooldownMs: number;
}

export interface DwellState {
  target: string | null;
  startedAt: number;
  lastSeenAt: number;
  /** When the gaze left the current target, while inside the grace period. */
  absentSince: number | null;
  /** The current target has fired and will not again until the gaze leaves it. */
  fired: boolean;
  cooldownUntil: number;
}

export interface DwellStep {
  /** 0..1 progress towards firing the current target, for the ring that fills. */
  progress: number;
  /** The id that fired on this step, if one did. */
  fired: string | null;
}

export const DWELL_DEFAULTS: DwellConfig = {
  // 1.5 s is Google's Cardboard design guidance for a fuse button, and it is the length below
  // which people start choosing things they only glanced at while looking round a room.
  dwellMs: 1500,
  // About thirteen frames at 60 Hz: covers the tremor at a ring's rim, not a deliberate look
  // away (which takes a head a good deal longer than a fifth of a second to come back from).
  graceMs: 220,
  // Longer than the doorway's fade out (180 ms) plus the swap, so the next room cannot be
  // clicked through the black.
  cooldownMs: 900,
};

export function createDwell(): DwellState {
  return { target: null, startedAt: 0, lastSeenAt: 0, absentSince: null, fired: false, cooldownUntil: 0 };
}

/** Forget whatever was being looked at — a room change, a mode change. */
export function resetDwell(state: DwellState, now: number, cooldownMs = 0): void {
  state.target = null;
  state.fired = false;
  state.startedAt = now;
  state.lastSeenAt = now;
  state.absentSince = null;
  state.cooldownUntil = now + cooldownMs;
}

export function stepDwell(state: DwellState, hovered: string | null, now: number, config: DwellConfig = DWELL_DEFAULTS): DwellStep {
  if (now < state.cooldownUntil) {
    // Nothing counts during the cooldown, but what is being looked at is still TRACKED. The
    // first version cleared the target here, which also cleared "this one has fired" — so a
    // visitor still staring at a ring after it fired had it fire again 2.4 s later, and again
    // after that, every cooldown. Measured in the tests as four fires over a ten-second stare.
    if (hovered !== null && hovered !== state.target) {
      state.target = hovered;
      state.fired = false;
    } else if (hovered === null && state.target !== null && now - state.lastSeenAt > config.graceMs) {
      state.target = null;
      state.fired = false;
    }
    if (hovered !== null) state.lastSeenAt = now;
    state.absentSince = null;
    // A target first seen through the cooldown starts counting the moment it ends.
    state.startedAt = state.cooldownUntil;
    return { progress: 0, fired: null };
  }

  if (hovered !== null && hovered === state.target) {
    if (state.absentSince !== null) {
      // Back inside the grace period: the time away is not dwell time.
      state.startedAt += now - state.absentSince;
      state.absentSince = null;
    }
    state.lastSeenAt = now;
    if (state.fired) return { progress: 0, fired: null };
    const progress = (now - state.startedAt) / config.dwellMs;
    if (progress >= 1) {
      state.fired = true;
      state.cooldownUntil = now + config.cooldownMs;
      return { progress: 1, fired: hovered };
    }
    return { progress, fired: null };
  }

  if (hovered !== null) {
    state.target = hovered;
    state.startedAt = now;
    state.lastSeenAt = now;
    state.absentSince = null;
    state.fired = false;
    return { progress: 0, fired: null };
  }

  // Nothing under the gaze.
  if (state.target !== null && now - state.lastSeenAt <= config.graceMs) {
    if (state.absentSince === null) state.absentSince = now;
    // Held, not advanced: the ring shows where it had got to.
    const held = (state.absentSince - state.startedAt) / config.dwellMs;
    return { progress: state.fired ? 0 : Math.min(1, Math.max(0, held)), fired: null };
  }
  state.target = null;
  state.fired = false;
  state.absentSince = null;
  return { progress: 0, fired: null };
}
