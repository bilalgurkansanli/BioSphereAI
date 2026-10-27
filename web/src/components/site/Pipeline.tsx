"use client";

import { AppWindow, ArrowUpRight, ChevronRight, RefreshCcw, Server } from "lucide-react";
import { DOCS } from "@/lib/project";
import { cn } from "@/lib/utils";
import { LANE_360, LANE_3D, PIPELINE, STACK, type PipelineNode } from "./copy";
import { Section, SectionHeader, useText } from "./primitives";

/**
 * The two pipelines as a diagram built from HTML rather than an image: it follows the theme
 * tokens, it reflows to a vertical timeline on a phone, and every label in it is real text.
 */

function Lane({
  label,
  nodes,
  accent,
  tag,
}: {
  label: string;
  nodes: PipelineNode[];
  accent: "bio" | "violet";
  tag?: string;
}) {
  const { tx } = useText();
  const last = nodes.length - 1;

  return (
    <div>
      <div className="mb-3 flex items-center gap-2.5">
        <span
          className={cn("size-2 rounded-full", accent === "bio" ? "bg-bio" : "bg-violet")}
          aria-hidden
        />
        <h3 className="text-[13px] font-semibold tracking-wide text-ink uppercase">{label}</h3>
        {tag ? (
          <span className="rounded-full border border-amber/35 bg-amber/10 px-2 py-0.5 text-[10.5px] font-semibold tracking-wide text-amber uppercase">
            {tag}
          </span>
        ) : null}
      </div>

      <ol className="grid grid-cols-1 gap-2 lg:grid-cols-6 lg:gap-3">
        {nodes.map((node, index) => (
          <li
            key={node.title.en}
            className="relative flex items-start gap-3 rounded-md border border-line bg-surface-2/50 p-3 lg:block lg:p-3.5"
          >
            <span
              className={cn(
                "grid size-7 shrink-0 place-items-center rounded-full border font-mono text-[10.5px] font-semibold lg:block lg:size-auto lg:rounded-none lg:border-0 lg:bg-transparent lg:tracking-widest",
                accent === "bio"
                  ? "border-bio/30 bg-bio/10 text-bio lg:text-bio/80"
                  : "border-violet/30 bg-violet/10 text-violet lg:text-violet/80",
              )}
            >
              {String(index + 1).padStart(2, "0")}
            </span>
            <div className="min-w-0 lg:mt-2">
              <p className="text-[14px] leading-snug font-semibold tracking-tight text-ink">{tx(node.title)}</p>
              <p className="mt-1 text-[12.5px] leading-snug text-ink-muted">{tx(node.detail)}</p>
              <p
                className={cn(
                  "mt-2 inline-block rounded-xs px-1.5 py-0.5 font-mono text-[11px]",
                  accent === "bio" ? "bg-bio/8 text-bio" : "bg-violet/10 text-violet",
                )}
              >
                {node.tech}
              </p>
            </div>

            {index < last ? (
              <>
                {/* Wide: an arrow in the gap between two cards. */}
                <span
                  className="absolute top-1/2 -right-[16px] z-10 hidden size-5 -translate-y-1/2 place-items-center rounded-full border border-line-strong bg-canvas text-ink-faint lg:grid"
                  aria-hidden
                >
                  <ChevronRight className="size-3" strokeWidth={2.2} />
                </span>
                {/* Narrow: a timeline stroke down to the next card. */}
                <span className="absolute -bottom-2 left-[25.5px] h-2 w-px bg-line-strong lg:hidden" aria-hidden />
              </>
            ) : null}
          </li>
        ))}
      </ol>
    </div>
  );
}

export function Pipeline() {
  const { tx } = useText();

  return (
    <Section id="nasil" aria-labelledby="nasil-title" className="pb-24">
      <SectionHeader
        id="nasil-title"
        eyebrow={tx(PIPELINE.eyebrow)}
        title={tx(PIPELINE.title)}
        lead={tx(PIPELINE.lead)}
        aside={
          <a
            href={DOCS.architecture}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-[13px] font-medium text-bio hover:underline"
          >
            {tx(PIPELINE.docs)}
            <ArrowUpRight className="size-3.5" aria-hidden />
          </a>
        }
      />

      <div className="glass mt-10 space-y-8 rounded-xl p-4 sm:p-6 lg:p-8">
        <Lane label={tx(PIPELINE.lane360)} nodes={LANE_360} accent="bio" />

        <div className="flex gap-3 rounded-md border border-dashed border-line-strong p-4 text-[13px] leading-relaxed text-ink-muted sm:items-center">
          <span className="grid size-8 shrink-0 place-items-center rounded-sm border border-line bg-surface-2 text-bio">
            <RefreshCcw className="size-4" strokeWidth={1.8} aria-hidden />
          </span>
          <p>
            <span className="font-semibold text-ink">{tx(PIPELINE.laneGate)}.</span> {tx(PIPELINE.gate)}
          </p>
        </div>

        <div className="border-t border-line pt-8">
          <Lane label={tx(PIPELINE.lane3d)} nodes={LANE_3D} accent="violet" tag={tx(PIPELINE.experimental)} />
        </div>

        <dl className="grid grid-cols-1 gap-3 border-t border-line pt-6 text-[12.5px] sm:grid-cols-2">
          <div className="flex flex-col gap-1">
            <dt className="flex items-center gap-1.5 font-semibold text-ink">
              <Server className="size-3.5 text-ink-faint" aria-hidden />
              {tx(PIPELINE.stackEngine)}
            </dt>
            <dd className="font-mono text-ink-muted">{STACK.engine}</dd>
          </div>
          <div className="flex flex-col gap-1">
            <dt className="flex items-center gap-1.5 font-semibold text-ink">
              <AppWindow className="size-3.5 text-ink-faint" aria-hidden />
              {tx(PIPELINE.stackWeb)}
            </dt>
            <dd className="font-mono text-ink-muted">{STACK.web}</dd>
          </div>
        </dl>
      </div>
    </Section>
  );
}
