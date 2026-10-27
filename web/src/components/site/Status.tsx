"use client";

import {
  CircleCheck,
  CirclePause,
  FileText,
  FlaskConical,
  Footprints,
  Glasses,
  Globe,
  KeyRound,
  Map,
  QrCode,
  Smartphone,
  Sparkles,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";
import { AUTHOR } from "@/lib/project";
import { cn } from "@/lib/utils";
import { STATUS, STATUS_GROUPS, type StatusItem, type StatusTone } from "./copy";
import { Section, SectionHeader, useText } from "./primitives";

/**
 * The core of the page: what works, what needs a key, what is experimental and what is parked.
 *
 * A table in spirit — every component has a row, every row has a state — laid out as grouped
 * rows so it survives a phone screen. The parked row is written in the first person on
 * purpose: it is a decision, not a defect report, and it reads more honestly as one.
 */

const TONE: Record<StatusTone, { icon: LucideIcon; badge: string; dot: string; bar: string }> = {
  stable: {
    icon: CircleCheck,
    badge: "border-bio/35 bg-bio/10 text-bio",
    dot: "bg-bio",
    bar: "bg-bio",
  },
  key: {
    icon: KeyRound,
    badge: "border-sky/35 bg-sky/10 text-sky",
    dot: "bg-sky",
    bar: "bg-sky",
  },
  experimental: {
    icon: FlaskConical,
    badge: "border-amber/35 bg-amber/10 text-amber",
    dot: "bg-amber",
    bar: "bg-amber",
  },
  parked: {
    icon: CirclePause,
    badge: "border-line-strong bg-surface-2 text-ink-muted",
    dot: "bg-ink-faint",
    bar: "bg-ink-faint/70",
  },
};

const ITEM_ICON: Record<StatusItem["icon"], LucideIcon> = {
  capture: Smartphone,
  player: Globe,
  plan: Map,
  accounts: QrCode,
  listing: FileText,
  invest: TrendingUp,
  vr: Glasses,
  walk: Footprints,
  staging: Sparkles,
};

function Badge({ tone, label }: { tone: StatusTone; label: string }) {
  const Icon = TONE[tone].icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11.5px] font-semibold tracking-wide whitespace-nowrap",
        TONE[tone].badge,
      )}
    >
      <Icon className="size-3.5" strokeWidth={2} aria-hidden />
      {label}
    </span>
  );
}

export function Status() {
  const { tx } = useText();

  return (
    <Section id="durum" aria-labelledby="durum-title" className="pb-24">
      <SectionHeader id="durum-title" eyebrow={tx(STATUS.eyebrow)} title={tx(STATUS.title)} lead={tx(STATUS.lead)} />

      {/* The summary: one bar, one segment per component, so the proportions read at a glance. */}
      <div className="mt-10" aria-label={tx(STATUS.summary)} role="group">
        <div className="flex h-2 gap-1 overflow-hidden rounded-full">
          {STATUS_GROUPS.flatMap((group) =>
            group.items.map((item) => (
              <span
                key={item.title.en}
                className={cn("h-full flex-1 first:rounded-l-full last:rounded-r-full", TONE[group.tone].bar)}
              />
            )),
          )}
        </div>
        <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-2 text-[13px] text-ink-muted">
          {STATUS_GROUPS.map((group) => (
            <li key={group.tone} className="flex items-center gap-2">
              <span className={cn("size-2 rounded-full", TONE[group.tone].dot)} aria-hidden />
              <span className="font-semibold text-ink tabular-nums">{group.items.length}</span>
              {tx(group.label)}
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-10 space-y-10">
        {STATUS_GROUPS.map((group) => (
          <div key={group.tone} className="grid grid-cols-1 gap-4 lg:grid-cols-[200px_minmax(0,1fr)] lg:gap-8">
            <div className="flex items-center gap-3 lg:sticky lg:top-24 lg:flex-col lg:items-start lg:self-start lg:pt-5">
              <Badge tone={group.tone} label={tx(group.label)} />
              <p className="text-[13px] text-ink-faint">{tx(group.blurb)}</p>
            </div>

            <ul
              className={cn(
                "divide-y divide-line overflow-hidden rounded-lg border",
                "border-line bg-surface/40",
              )}
            >
              {group.items.map((item) => {
                const Icon = ITEM_ICON[item.icon];
                const parked = group.tone === "parked";
                return (
                  <li key={item.title.en} className="grid grid-cols-1 gap-4 p-5 sm:grid-cols-[auto_minmax(0,1fr)] sm:gap-5 sm:p-6">
                    <span
                      className={cn(
                        "grid size-10 place-items-center rounded-sm border border-line bg-surface-2",
                        "text-ink-muted",
                      )}
                    >
                      <Icon className="size-[18px]" strokeWidth={1.7} aria-hidden />
                    </span>
                    <div className="min-w-0">
                      <h3 className="text-[16px] font-semibold tracking-tight text-ink">{tx(item.title)}</h3>
                      {parked ? (
                        <blockquote className="mt-2.5 max-w-3xl border-l-2 border-line-strong pl-4 text-[14px] leading-relaxed text-ink-muted">
                          <p>{tx(item.body)}</p>
                          <footer className="mt-2 text-[12.5px] text-ink-faint">
                            {tx(STATUS.signature)} · <span className="font-mono">{AUTHOR.handle}</span>
                          </footer>
                        </blockquote>
                      ) : (
                        <p className="mt-2 max-w-3xl text-[14px] leading-relaxed text-ink-muted">{tx(item.body)}</p>
                      )}
                      <ul className="mt-3.5 flex flex-wrap gap-1.5">
                        {item.facts.map((fact) => (
                          <li
                            key={fact.en}
                            className="rounded-xs border border-line bg-canvas-raised px-2 py-0.5 font-mono text-[11.5px] text-ink-muted"
                          >
                            {tx(fact)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </Section>
  );
}
