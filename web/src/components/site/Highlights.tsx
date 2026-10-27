"use client";

import { HIGHLIGHTS, METRICS, TECHNIQUES } from "./copy";
import { Section, SectionHeader, useText } from "./primitives";

/** Before and after for the reference capture, in seconds. Measured on the same laptop. */
const BEFORE_S = 1278;
const AFTER_S = 170;

/**
 * The numbers, each with what it measures. The speed-up gets a picture because a ratio is
 * the one number here that a bar explains better than a sentence.
 */
export function Highlights() {
  const { tx, locale } = useText();
  const format = (n: number) => n.toLocaleString(locale === "tr" ? "tr-TR" : "en-GB");
  const speedup = (BEFORE_S / AFTER_S).toLocaleString(locale === "tr" ? "tr-TR" : "en-GB", {
    maximumFractionDigits: 1,
  });
  const [iou, ...rest] = METRICS;

  return (
    <Section id="muhendislik" aria-labelledby="muhendislik-title" className="pb-24">
      <SectionHeader
        id="muhendislik-title"
        eyebrow={tx(HIGHLIGHTS.eyebrow)}
        title={tx(HIGHLIGHTS.title)}
        lead={tx(HIGHLIGHTS.lead)}
      />

      <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-12">
        <figure className="glass rounded-lg p-6 sm:col-span-2 lg:col-span-8">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <figcaption className="text-[13px] font-medium text-ink-muted">{tx(HIGHLIGHTS.speedTitle)}</figcaption>
            <p className="font-mono text-[12px] text-bio">≈ {speedup}×</p>
          </div>

          <div className="mt-6 space-y-4">
            {[
              { label: tx(HIGHLIGHTS.speedBefore), seconds: BEFORE_S, now: false },
              { label: tx(HIGHLIGHTS.speedAfter), seconds: AFTER_S, now: true },
            ].map((row) => {
              // The bar is the track's full width for the old time and its share of it for the
              // new one. The label rides inside a long bar and just past the end of a short one,
              // so it never wraps and never sits on top of the fill it is describing.
              const share = row.seconds / BEFORE_S;
              const inside = share > 0.5;
              return (
                <div key={row.label} className="grid grid-cols-[52px_minmax(0,1fr)] items-center gap-3 sm:grid-cols-[72px_minmax(0,1fr)]">
                  <span className="text-[12.5px] text-ink-faint">{row.label}</span>
                  <div className="relative h-8">
                    <div
                      className={
                        row.now
                          ? "bio-gradient absolute inset-y-0 left-0 rounded-xs shadow-[0_0_24px_-6px_var(--bio)]"
                          : "absolute inset-y-0 left-0 rounded-xs border border-line bg-surface-3/70"
                      }
                      style={{ width: `${Math.max(4, share * 100)}%` }}
                    />
                    <span
                      className={
                        "absolute inset-y-0 flex items-center font-mono whitespace-nowrap tabular-nums " +
                        (row.now ? "text-[15px] font-semibold text-ink" : "text-[13px] text-ink-muted")
                      }
                      style={inside ? { right: 12 } : { left: `calc(${Math.max(4, share * 100)}% + 12px)` }}
                    >
                      {row.now ? "≈ " : ""}
                      {format(row.seconds)} {tx(HIGHLIGHTS.seconds)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-6 text-[13px] leading-relaxed text-ink-muted">{tx(HIGHLIGHTS.speedNote)}</p>
        </figure>

        <div className="glass flex flex-col rounded-lg p-6 sm:col-span-2 lg:col-span-4">
          <p className="text-[13px] font-medium text-ink-muted">{tx(iou.label)}</p>
          <p className="mt-auto pt-6 text-[clamp(2rem,4vw,2.6rem)] leading-none font-semibold tracking-[-0.03em] text-ink tabular-nums">
            {tx(iou.value)}
          </p>
          <p className="mt-3 text-[13px] leading-relaxed text-ink-muted">{tx(iou.note)}</p>
        </div>

        {rest.map((metric) => (
          <div key={metric.label.en} className="rounded-lg border border-line bg-surface/40 p-5 lg:col-span-3">
            <p className="text-[clamp(1.6rem,3vw,2rem)] leading-none font-semibold tracking-[-0.03em] text-ink tabular-nums">
              {tx(metric.value)}
            </p>
            <p className="mt-3 text-[13px] font-medium text-ink">{tx(metric.label)}</p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-muted">{tx(metric.note)}</p>
          </div>
        ))}
      </div>

      <h3 className="mt-14 text-[13px] font-semibold tracking-wide text-ink uppercase">{tx(HIGHLIGHTS.techniquesTitle)}</h3>
      <ul className="mt-4 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
        {TECHNIQUES.map((technique, index) => (
          <li key={technique.title.en} className="border-t border-line-strong pt-4">
            <span className="font-mono text-[11px] tracking-widest text-bio/80">{String(index + 1).padStart(2, "0")}</span>
            <p className="mt-2 text-[15px] font-semibold tracking-tight text-ink">{tx(technique.title)}</p>
            <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">{tx(technique.body)}</p>
          </li>
        ))}
      </ul>
    </Section>
  );
}
