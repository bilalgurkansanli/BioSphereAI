"use client";

import { Check, Gauge, Rocket, Sparkles, type LucideIcon } from "lucide-react";
import { useRef } from "react";
import { ui, type UiKey } from "./copy";
import { useLocale } from "@/lib/hooks/useLocale";
import { qualityDescription, qualityLabel } from "@/lib/i18n";
import type { QualityMode } from "@/lib/types";
import { cn } from "@/lib/utils";

const MODES: Array<{ mode: QualityMode; icon: LucideIcon; trade: UiKey }> = [
  { mode: "fast", icon: Rocket, trade: "wizard.quality.fast.trade" },
  { mode: "balanced", icon: Gauge, trade: "wizard.quality.balanced.trade" },
  { mode: "studio", icon: Sparkles, trade: "wizard.quality.studio.trade" },
];

export interface QualitySelectorProps {
  value: QualityMode;
  onChange: (mode: QualityMode) => void;
  className?: string;
  label: string;
  /** What is being built: the 360 panorama (default) or the 3D walkthrough, whose modes trade different things. */
  flavour?: "pano" | "walk";
}

export function QualitySelector({ value, onChange, className, label, flavour = "pano" }: QualitySelectorProps) {
  const { locale } = useLocale();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const move = (from: number, delta: number) => {
    const next = (from + delta + MODES.length) % MODES.length;
    onChange(MODES[next].mode);
    refs.current[next]?.focus();
  };

  return (
    <div role="radiogroup" aria-label={label} className={cn("grid gap-3 sm:grid-cols-3", className)}>
      {MODES.map((entry, index) => {
        const selected = entry.mode === value;
        const Icon = entry.icon;
        return (
          <button
            key={entry.mode}
            ref={(node) => {
              refs.current[index] = node;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(entry.mode)}
            onKeyDown={(event) => {
              if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                event.preventDefault();
                move(index, 1);
              } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                event.preventDefault();
                move(index, -1);
              }
            }}
            className={cn(
              "relative flex flex-col items-start gap-1 rounded-md border p-4 text-left transition-all duration-200 ease-out-expo",
              selected
                ? "border-bio/45 bg-bio/5 shadow-[0_0_0_1px_var(--bio-dim)]"
                : "border-line bg-surface-2/40 hover:border-line-strong hover:bg-surface-2",
            )}
          >
            <span className="flex w-full items-center justify-between">
              <Icon className={cn("size-4", selected ? "text-bio" : "text-ink-faint")} strokeWidth={1.8} aria-hidden />
              <span
                className={cn(
                  "grid size-4 place-items-center rounded-full border transition-colors",
                  selected ? "border-bio bg-bio text-canvas" : "border-line-strong",
                )}
                aria-hidden
              >
                {selected ? <Check className="size-3" strokeWidth={3} /> : null}
              </span>
            </span>

            <span className="mt-1.5 text-sm font-semibold tracking-tight text-ink">
              {qualityLabel(entry.mode, locale)}
            </span>
            <span className="font-mono text-[11px] text-ink-faint">
              {flavour === "walk" ? ui(`walk.quality.${entry.mode}.desc` as UiKey, locale) : qualityDescription(entry.mode, locale)}
            </span>
            <span className="mt-1.5 text-[12px] leading-relaxed text-ink-muted">
              {flavour === "walk" ? ui(`walk.quality.${entry.mode}.trade` as UiKey, locale) : ui(entry.trade, locale)}
            </span>
          </button>
        );
      })}
    </div>
  );
}
