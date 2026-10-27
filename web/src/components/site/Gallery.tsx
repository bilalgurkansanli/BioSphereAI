"use client";

import Image from "next/image";
import { useId, useState } from "react";
import { cn } from "@/lib/utils";
import { GALLERY, SHOTS } from "./copy";
import { Section, SectionHeader, useText } from "./primitives";

/**
 * The dashboard and the player, as they look on a fresh install.
 *
 * Every shot was taken against an isolated engine holding only synthetic rooms rendered by
 * `biosphere.synth`, so no real home or person can appear here. All five images sit stacked in
 * one frame and only the chosen one is opaque: switching is instant, and the frame never
 * changes height under the reader.
 */
export function Gallery() {
  const { tx } = useText();
  const [active, setActive] = useState(SHOTS[0].id);
  const baseId = useId();
  const current = SHOTS.find((shot) => shot.id === active) ?? SHOTS[0];

  return (
    <Section id="ekranlar" aria-labelledby="ekranlar-title" className="pb-24">
      <SectionHeader id="ekranlar-title" eyebrow={tx(GALLERY.eyebrow)} title={tx(GALLERY.title)} lead={tx(GALLERY.lead)} />

      <div
        role="tablist"
        aria-label={tx(GALLERY.tabs)}
        className="-mx-4 mt-8 flex gap-1.5 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:flex-wrap sm:px-0"
      >
        {SHOTS.map((shot) => {
          const selected = shot.id === active;
          return (
            <button
              key={shot.id}
              type="button"
              role="tab"
              id={`${baseId}-${shot.id}-tab`}
              aria-selected={selected}
              aria-controls={`${baseId}-panel`}
              onClick={() => setActive(shot.id)}
              className={cn(
                "h-9 shrink-0 rounded-full border px-4 text-[13px] font-medium whitespace-nowrap transition-colors",
                selected
                  ? "border-bio/45 bg-bio/12 text-ink"
                  : "border-line text-ink-muted hover:border-line-strong hover:text-ink",
              )}
            >
              {tx(shot.label)}
            </button>
          );
        })}
      </div>

      <figure
        role="tabpanel"
        id={`${baseId}-panel`}
        aria-labelledby={`${baseId}-${current.id}-tab`}
        className="mt-5"
      >
        <div className="overflow-hidden rounded-xl border border-line-strong bg-canvas-raised shadow-[0_30px_80px_-30px_rgb(0_0_0/0.65)]">
          {/* A browser frame, so a screenshot reads as an application rather than a picture. */}
          <div className="flex items-center gap-3 border-b border-line bg-surface-2/70 px-4 py-2.5">
            <span className="flex gap-1.5" aria-hidden>
              <span className="size-2.5 rounded-full bg-ink-faint/40" />
              <span className="size-2.5 rounded-full bg-ink-faint/40" />
              <span className="size-2.5 rounded-full bg-ink-faint/40" />
            </span>
            <span className="mx-auto min-w-0 truncate rounded-xs bg-canvas/70 px-3 py-1 font-mono text-[11.5px] text-ink-faint sm:min-w-[320px] sm:text-center">
              {current.path}
            </span>
            <span className="hidden w-[42px] sm:block" aria-hidden />
          </div>

          <div className="relative aspect-[16/10] bg-canvas">
            {SHOTS.map((shot) => (
              <Image
                key={shot.id}
                src={shot.src}
                alt={tx(shot.alt)}
                width={shot.width}
                height={shot.height}
                sizes="(min-width: 1152px) 1152px, 100vw"
                aria-hidden={shot.id !== active}
                className={cn(
                  "absolute inset-0 size-full object-cover object-top transition-opacity duration-300",
                  shot.id === active ? "opacity-100" : "opacity-0",
                )}
              />
            ))}
          </div>
        </div>
        <figcaption className="mt-4 max-w-3xl text-[13.5px] leading-relaxed text-ink-muted">{tx(current.caption)}</figcaption>
      </figure>
    </Section>
  );
}
