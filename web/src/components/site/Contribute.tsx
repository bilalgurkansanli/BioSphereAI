"use client";

import { ArrowUpRight, Container, Eraser, Footprints, Glasses, Languages, Terminal, type LucideIcon } from "lucide-react";
import { buttonStyles } from "@/components/ui/Button";
import { DOCS, REPO_URL } from "@/lib/project";
import { CONTRIBUTE, HERO, IDEAS } from "./copy";
import { GitHubMark, Section, SectionHeader, useText } from "./primitives";

const ICON: Record<(typeof IDEAS)[number]["icon"], LucideIcon> = {
  eraser: Eraser,
  glasses: Glasses,
  footprints: Footprints,
  terminal: Terminal,
  container: Container,
  languages: Languages,
};

/** Where help would matter most, and the two doors in: the guide and the issue tracker. */
export function Contribute() {
  const { tx } = useText();

  return (
    <Section id="katki" aria-labelledby="katki-title" className="pb-28">
      <SectionHeader id="katki-title" eyebrow={tx(CONTRIBUTE.eyebrow)} title={tx(CONTRIBUTE.title)} lead={tx(CONTRIBUTE.lead)} />

      <ul className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {IDEAS.map((idea) => {
          const Icon = ICON[idea.icon];
          return (
            <li
              key={idea.title.en}
              className="group flex flex-col rounded-lg border border-line bg-surface/40 p-5 transition-colors hover:border-line-strong"
            >
              <div className="flex items-center justify-between">
                <span className="grid size-9 place-items-center rounded-sm border border-line bg-surface-2 text-bio">
                  <Icon className="size-4" strokeWidth={1.7} aria-hidden />
                </span>
                <span className="font-mono text-[11px] text-ink-faint">{idea.tag}</span>
              </div>
              <h3 className="mt-4 text-[15px] font-semibold tracking-tight text-ink">{tx(idea.title)}</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">{tx(idea.body)}</p>
            </li>
          );
        })}
      </ul>

      <div className="relative mt-10 overflow-hidden rounded-xl border border-line-strong p-6 sm:p-8">
        <div
          className="pointer-events-none absolute -top-28 -left-20 size-72 rounded-full bg-[radial-gradient(circle,var(--bio-dim),transparent_65%)] blur-xl"
          aria-hidden
        />
        <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="font-mono text-[12px] text-ink-faint">{REPO_URL.replace(/^https:\/\//, "")}</p>
            <p className="mt-2 text-[clamp(1.25rem,2.6vw,1.6rem)] leading-snug font-semibold tracking-[-0.02em] text-ink">
              {tx(HERO.titleLead)} <span className="bio-text">{tx(HERO.titleAccent)}</span>
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <a href={REPO_URL} target="_blank" rel="noreferrer" className={buttonStyles("primary", "md")}>
              <GitHubMark className="size-4" />
              {tx(HERO.ctaPrimary)}
            </a>
            <a href={DOCS.contributing} target="_blank" rel="noreferrer" className={buttonStyles("outline", "md")}>
              {tx(CONTRIBUTE.guide)}
              <ArrowUpRight className="size-3.5" aria-hidden />
            </a>
            <a href={DOCS.issues} target="_blank" rel="noreferrer" className={buttonStyles("ghost", "md")}>
              {tx(CONTRIBUTE.issues)}
              <ArrowUpRight className="size-3.5" aria-hidden />
            </a>
          </div>
        </div>
      </div>
    </Section>
  );
}
