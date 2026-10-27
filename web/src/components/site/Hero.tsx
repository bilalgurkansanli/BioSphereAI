"use client";

import { ArrowDown, Laptop, Scale, Terminal } from "lucide-react";
import { buttonStyles } from "@/components/ui/Button";
import { CLONE_URL, REPO_URL } from "@/lib/project";
import { HERO } from "./copy";
import { HeroPano } from "./HeroPano";
import { CopyButton, GitHubMark, useText } from "./primitives";

/**
 * What it is, in one line, and the thing itself beside it.
 *
 * Three children in two orders: on a wide screen the words sit left and the sphere right; on
 * a phone the order is claim → sphere → the rest, because the sphere is the one thing that has
 * to be seen before somebody decides to keep scrolling.
 */
export function Hero() {
  const { tx } = useText();
  const clone = `git clone ${CLONE_URL}`;

  const badges = [
    { icon: Scale, label: tx(HERO.badges.license) },
    { icon: Laptop, label: tx(HERO.badges.local) },
    { icon: Terminal, label: tx(HERO.badges.stack) },
  ];

  return (
    <section className="relative">
      <div className="grid-bg pointer-events-none absolute inset-x-0 top-0 -z-10 h-[860px]" aria-hidden />
      <div
        className="pointer-events-none absolute top-[-340px] left-1/2 -z-10 size-[780px] -translate-x-1/2 rounded-full
                   bg-[radial-gradient(circle,var(--bio-dim)_0%,transparent_62%)] opacity-80 blur-2xl"
        aria-hidden
      />

      <div className="mx-auto grid w-full max-w-6xl grid-cols-1 items-center gap-x-14 gap-y-10 px-4 pt-10 pb-20 sm:px-6 sm:pt-14 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] lg:grid-rows-[auto_auto] lg:gap-y-0 lg:pt-20 lg:pb-28">
        <div className="animate-rise lg:col-start-1 lg:row-start-1 lg:self-end">
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="group inline-flex items-center gap-2 rounded-full border border-line bg-surface-2/60 py-1 pr-3 pl-1 text-[12px] font-medium text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
          >
            <span className="grid size-5 place-items-center rounded-full bg-bio/15 text-bio">
              <GitHubMark className="size-3" />
            </span>
            {tx(HERO.eyebrow)}
            <span className="text-ink-faint transition-transform group-hover:translate-x-0.5" aria-hidden>
              →
            </span>
          </a>

          {/* Two lines by construction: what goes in, what comes out. The accent and the tail
              never break inside themselves, so a narrow column moves the tail to a line of its
              own instead of splitting "360° sanal tur" across two. */}
          <h1 className="mt-6 text-[clamp(1.85rem,4.4vw,3.15rem)] leading-[1.08] font-semibold tracking-[-0.035em] text-ink">
            <span className="block">{tx(HERO.titleLead)}</span>
            <span className="block">
              <span className="bio-text whitespace-nowrap">{tx(HERO.titleAccent)}</span>{" "}
              <span className="whitespace-nowrap text-ink-muted">{tx(HERO.titleTail)}</span>
            </span>
          </h1>
        </div>

        <HeroPano className="lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:self-center" />

        <div className="animate-rise lg:col-start-1 lg:row-start-2 lg:self-start">
          <p className="mt-0 max-w-xl text-[15.5px] leading-relaxed text-ink-muted sm:text-base lg:mt-6">
            {tx(HERO.body)}
          </p>

          <ul className="mt-6 flex flex-wrap gap-2">
            {badges.map(({ icon: Icon, label }) => (
              <li
                key={label}
                className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface/50 px-2.5 py-1 text-[12px] font-medium text-ink-muted"
              >
                <Icon className="size-3.5 text-bio" strokeWidth={1.8} aria-hidden />
                {label}
              </li>
            ))}
          </ul>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <a href={REPO_URL} target="_blank" rel="noreferrer" className={buttonStyles("primary", "lg")}>
              <GitHubMark className="size-[18px]" />
              {tx(HERO.ctaPrimary)}
            </a>
            <a href="/rehber" className={buttonStyles("outline", "lg")}>
              {tx(HERO.ctaSecondary)}
              <ArrowDown className="size-4" aria-hidden />
            </a>
          </div>

          <div className="mt-6 flex max-w-xl items-center gap-2 rounded-md border border-line bg-canvas-raised/80 py-1.5 pr-1.5 pl-3.5">
            <span className="font-mono text-[12.5px] text-bio select-none" aria-hidden>
              $
            </span>
            <code
              aria-label={tx(HERO.cloneLabel)}
              className="min-w-0 flex-1 overflow-x-auto font-mono text-[12.5px] whitespace-nowrap text-ink-muted [scrollbar-width:none]"
            >
              {clone}
            </code>
            <CopyButton value={clone} />
          </div>
        </div>
      </div>
    </section>
  );
}
