"use client";

/**
 * Two photographs of the same room, one wiped across the other.
 *
 * ── THE HANDLE DOES NOT LIVE IN REACT STATE ───────────────────────────────────────────
 * The same rule the player follows: what changes every frame is written straight to the DOM.
 * A drag is a pointermove stream at the display's refresh rate, and routing each one through
 * `setState` re-renders both `<img>` elements sixty times a second for a value only two
 * inline styles ever read. The position lives in a ref; the two styles are written by hand.
 * React state here holds one thing — which styled image has decoded — because that changes
 * once per style rather than once per frame.
 *
 * ── THE CLIP IS ON THE AFTER IMAGE, NOT THE BEFORE ────────────────────────────────────
 * Both images are stacked; the styled one is clipped from the left so it is revealed on the
 * RIGHT of the handle. That ordering is not arbitrary — it matches how everyone builds this
 * control, and reversing it makes the drag feel like it is pushing the wrong picture.
 *
 * `clip-path: inset(...)` rather than a width-clipped wrapper: it is composited on the GPU and
 * does not re-lay-out the image, so the reveal cannot make the picture reflow mid-drag.
 *
 * ── IT IS A REAL SLIDER, INCLUDING FOR SOMEBODY NOT HOLDING A MOUSE ───────────────────
 * `role="slider"` with arrow keys, Home and End. This is the only control on the page that
 * carries the actual comparison, so leaving it pointer-only would put the whole feature
 * behind a mouse.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/** Arrow-key step, in per cent. Coarse enough to cross the frame without holding the key. */
const KEY_STEP = 2.5;
const KEY_STEP_LARGE = 10;

export interface BeforeAfterSliderProps {
  beforeSrc: string;
  afterSrc: string | null;
  beforeLabel: string;
  afterLabel: string;
  /** Rendered over the frame while `afterSrc` is null — the style is still being generated. */
  pending?: React.ReactNode;
  /** Natural aspect ratio, so the box is the right shape before either image decodes. */
  aspectRatio?: number;
  ariaLabel: string;
  className?: string;
}

export function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeLabel,
  afterLabel,
  pending,
  aspectRatio,
  ariaLabel,
  className,
}: BeforeAfterSliderProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const clipRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<HTMLDivElement>(null);
  const positionRef = useRef(50);
  const draggingRef = useRef(false);

  /**
   * WHICH src has decoded, not WHETHER one has — derived rather than reset.
   *
   * A boolean would have to be cleared by an effect every time `afterSrc` changes, and a
   * `setState` called synchronously from an effect is a cascading render (and a lint error).
   * Storing the src that loaded makes the reset free: the moment a different style arrives,
   * `afterReady` is false because the strings no longer match. Same reasoning as
   * `useEmptyRoom`'s `shownFor`.
   */
  const [readyFor, setReadyFor] = useState<string | null>(null);
  const afterReady = afterSrc !== null && readyFor === afterSrc;

  const paint = useCallback((percent: number) => {
    const clamped = Math.min(100, Math.max(0, percent));
    positionRef.current = clamped;
    if (clipRef.current) clipRef.current.style.clipPath = `inset(0 0 0 ${clamped}%)`;
    if (handleRef.current) {
      handleRef.current.style.left = `${clamped}%`;
      handleRef.current.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }
  }, []);

  /** A new style arrives: recentre so the comparison starts even-handed. A DOM write, not
   *  state — the handle's position never round-trips through React. */
  useEffect(() => {
    paint(50);
  }, [afterSrc, paint]);

  const fromClientX = useCallback((clientX: number) => {
    const frame = frameRef.current;
    if (!frame) return;
    const box = frame.getBoundingClientRect();
    if (box.width <= 0) return;
    return ((clientX - box.left) / box.width) * 100;
  }, []);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!afterSrc) return;
      draggingRef.current = true;
      // Capture on the FRAME, so a drag that leaves the picture keeps steering it instead of
      // sticking wherever the pointer crossed the edge.
      event.currentTarget.setPointerCapture(event.pointerId);
      const next = fromClientX(event.clientX);
      if (next !== undefined) paint(next);
    },
    [afterSrc, fromClientX, paint],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      const next = fromClientX(event.clientX);
      if (next !== undefined) paint(next);
    },
    [fromClientX, paint],
  );

  const endDrag = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const step =
        event.key === "PageUp" || event.key === "PageDown" ? KEY_STEP_LARGE : KEY_STEP;
      let next: number | null = null;
      if (event.key === "ArrowLeft" || event.key === "PageDown") next = positionRef.current - step;
      else if (event.key === "ArrowRight" || event.key === "PageUp") next = positionRef.current + step;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = 100;
      if (next === null) return;
      event.preventDefault();
      paint(next);
    },
    [paint],
  );

  const showAfter = Boolean(afterSrc);

  return (
    <div
      ref={frameRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      className={cn(
        // THE PICTURE SIZES THE FRAME, NOT THE COLUMN. Given the frame its own aspect ratio and
        // a full-width box, a square photograph in a wide column becomes as tall as the column
        // is wide and the bottom half of the comparison lands below the fold — which defeats
        // a control whose entire job is to be looked at as a whole. So the frame hugs the
        // image (`w-fit`), and the image is capped against the viewport height instead.
        "relative mx-auto w-fit select-none overflow-hidden rounded-[var(--radius-lg)] bg-surface-2",
        // `touch-none` or a vertical page scroll steals every horizontal drag on a phone.
        showAfter ? "cursor-ew-resize touch-none" : "cursor-default",
        className,
      )}
    >
      {/* eslint-disable-next-line @next/next/no-img-element -- the engine serves these from
          its own origin at a size it chose; next/image would proxy and re-encode them. */}
      <img
        src={beforeSrc}
        alt={beforeLabel}
        draggable={false}
        // The ratio is carried on the IMAGE so the box is the right shape before it decodes;
        // with `max-h` and `w-auto` the browser derives the width from it, which is what keeps
        // a portrait capture from overflowing the screen.
        style={aspectRatio ? { aspectRatio: String(aspectRatio) } : undefined}
        className="block max-h-[min(68dvh,44rem)] w-auto max-w-full object-cover"
      />

      {showAfter ? (
        <div ref={clipRef} className="absolute inset-0" style={{ clipPath: "inset(0 0 0 50%)" }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={afterSrc ?? undefined}
            alt={afterLabel}
            draggable={false}
            onLoad={() => setReadyFor(afterSrc)}
            className={cn(
              "absolute inset-0 h-full w-full object-cover transition-opacity duration-300",
              afterReady ? "opacity-100" : "opacity-0",
            )}
          />
        </div>
      ) : null}

      {/* Corner captions. The right-hand one names the picture as generated, every time it is
          on screen — this is the only place a viewer is told, and a staged room that is not
          labelled is a photograph of a flat that does not exist. */}
      <Caption className="bottom-3 left-3">{beforeLabel}</Caption>
      {showAfter ? (
        <Caption className="right-3 bottom-3 bg-bio/85 text-canvas">{afterLabel}</Caption>
      ) : null}

      {showAfter ? (
        <div
          ref={handleRef}
          role="slider"
          tabIndex={0}
          aria-label={ariaLabel}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={50}
          aria-orientation="horizontal"
          onKeyDown={onKeyDown}
          className="absolute inset-y-0 z-10 -ml-px w-0.5 bg-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.25)]
                     outline-none focus-visible:bg-bio focus-visible:shadow-[0_0_0_2px_var(--bio)]"
          style={{ left: "50%" }}
        >
          <span
            aria-hidden
            className="absolute top-1/2 left-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 place-items-center
                       rounded-full bg-white/95 text-[11px] font-semibold text-[#111] shadow-[0_2px_10px_rgba(0,0,0,0.35)]"
          >
            ‹ ›
          </span>
        </div>
      ) : null}

      {!showAfter && pending ? (
        <div className="absolute inset-0 grid place-items-center bg-canvas/55 backdrop-blur-[2px]">
          {pending}
        </div>
      ) : null}
    </div>
  );
}

function Caption({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "pointer-events-none absolute rounded-full bg-black/55 px-2.5 py-1 text-[11px]",
        "font-medium text-white/95 backdrop-blur-sm",
        className,
      )}
    >
      {children}
    </span>
  );
}
