"use client";

/**
 * The 30 Hz publish channel, and why it is not React state.
 *
 * ── THE MEASUREMENT ───────────────────────────────────────────────────────────────────
 * The capture loop publishes one small "hot" object 30 times a second. While that object
 * lived in a `useState` inside `CaptureStage`, every publish re-rendered the WHOLE stage:
 * the overlay, the row prompt, the speed dial, the toast, the clock, the chrome and every
 * conditional branch around them. Profiled on a production build with the render tree
 * driven at exactly the shipped cadence (900 ticks = 30 s of capture, desktop Chrome):
 *
 *     loop maths (detectors + coverage + stations + target)   170 ms
 *     React render + commit                                   348 ms
 *     style recalc + layout                                   140 ms
 *
 * The pure maths — every threshold, every cone stamp, the 2592-cell report — was a fifth
 * of the bill. React reconciliation was the capture's single largest main-thread cost, and
 * on a mid-range Android that multiplier lands on top of a hardware H.264 encode and a live
 * camera preview. The lag then starved `deviceorientation` and `requestVideoFrameCallback`
 * on the same main thread, which is how a slow phone talked itself into `track_lost` and
 * `smear` warnings it had no physical reason to raise.
 *
 * ── WHAT THIS FIXES ───────────────────────────────────────────────────────────────────
 * The hot object moves into a plain mutable store. Consumers subscribe to the SLICE they
 * actually draw, with an equality function that decides what "changed" means for that
 * slice. The result is that a publish wakes only the components whose picture is different:
 *
 *   - the row prompt's rail moves when a station fills, roughly two dozen times per capture,
 *     not thirty times a second;
 *   - the warning toast re-renders on a new warning occurrence, not on the fresh `Warning`
 *     object the detector allocates every tick;
 *   - the recording clock re-renders once a second, because it displays whole seconds;
 *   - `CaptureStage` itself re-renders only on a real state change.
 *
 * Only the world-locked lattice and the speed dial still see every publish, because they
 * are the two things that genuinely move at 24 deg/s — and both write their moving parts
 * through refs rather than through the reconciler.
 *
 * ── WHY NOT A CONTEXT, A REDUCER, OR AN EXTERNAL LIBRARY ──────────────────────────────
 * A context re-renders every consumer on every publish, which is the problem restated. A
 * reducer is still React state. `useSyncExternalStore` is the platform answer and is already
 * used in this codebase for `prefers-reduced-motion`; this is the same shape with a
 * selector and a memoised snapshot on top, which is the part React does not supply.
 */

import { useCallback, useDebugValue, useRef, useSyncExternalStore } from "react";

export interface HotStore<T> {
  /** Bumped on every publish. The snapshot cache is keyed on it. */
  readonly version: number;
  get(): T;
  publish(next: T): void;
  subscribe(listener: () => void): () => void;
}

/**
 * A mutable store with no React inside it.
 *
 * `publish` is called from the rAF loop, so it allocates nothing beyond the caller's own
 * object and notifies synchronously — the subscribers' re-renders have to land in the same
 * frame as the sensor sample that caused them, or the lattice draws a pose the phone has
 * already left.
 */
export function createHotStore<T>(initial: T): HotStore<T> {
  let state = initial;
  let version = 0;
  const listeners = new Set<() => void>();

  return {
    get version() {
      return version;
    },
    get() {
      return state;
    },
    publish(next: T) {
      state = next;
      version += 1;
      // Copied before iterating: a listener that unsubscribes during notification (a
      // component unmounting on the state change its own render triggered) would otherwise
      // mutate the set mid-iteration.
      for (const listener of [...listeners]) listener();
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

/**
 * Subscribe to one slice of the hot state.
 *
 * `isEqual` is the whole point: it decides what counts as a change for THIS consumer, and
 * anything it calls equal costs nothing at all — no render, no reconciliation, no commit.
 * The returned value keeps its identity across an equal publish, so React's own bail-out
 * fires as well.
 *
 * The snapshot is cached against the store's version, because `useSyncExternalStore` calls
 * `getSnapshot` more than once per notification and a selector that allocated a fresh
 * object each call would report a change on every single publish — the exact bug this hook
 * exists to prevent.
 *
 * `selector` and `isEqual` MUST be stable across renders — every caller here passes a
 * module-level function. An inline arrow would rebuild `getSnapshot` on every render, which
 * makes `useSyncExternalStore` re-read and re-subscribe on every render, which is the
 * opposite of what this hook is for. They are therefore taken as plain dependencies rather
 * than smuggled through refs, so the linter enforces it instead of the reviewer.
 */
export function useHotSelector<T, S>(
  store: HotStore<T>,
  selector: (state: T) => S,
  isEqual: (a: S, b: S) => boolean = Object.is,
): S {
  const cache = useRef<{ version: number; value: S } | null>(null);

  const getSnapshot = useCallback(() => {
    const version = store.version;
    const cached = cache.current;
    if (cached !== null && cached.version === version) return cached.value;
    const next = selector(store.get());
    if (cached !== null && isEqual(cached.value, next)) {
      // Same picture: keep the OLD value so React sees no change at all.
      cache.current = { version, value: cached.value };
      return cached.value;
    }
    cache.current = { version, value: next };
    return next;
  }, [store, selector, isEqual]);

  const value = useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
  useDebugValue(value);
  return value;
}

/**
 * A slice that is a fixed-length tuple of primitives, compared element-wise.
 *
 * Every consumer here selects a handful of numbers or strings, so this is the equality
 * function nearly all of them want. `Object.is` rather than `===` so a `NaN` heading — a
 * real thing a phone sensor produces — compares equal to itself and does not re-render the
 * dial forever.
 */
export function tupleEqual(a: readonly unknown[], b: readonly unknown[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (!Object.is(a[i], b[i])) return false;
  return true;
}
