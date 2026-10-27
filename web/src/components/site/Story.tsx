"use client";

import { Check } from "lucide-react";
import { STORY } from "./copy";
import { Section, SectionHeader, useText } from "./primitives";

/** Why the code is public: a short, first-person account. No competitor is named. */
export function Story() {
  const { tx } = useText();

  return (
    <Section id="neden" aria-labelledby="neden-title" className="pb-24">
      <SectionHeader id="neden-title" eyebrow={tx(STORY.eyebrow)} title={tx(STORY.title)} />

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] lg:gap-14">
        <div className="space-y-4 text-[15.5px] leading-[1.75] text-ink-muted">
          {STORY.paragraphs.map((paragraph, index) => (
            <p key={index} className={index === STORY.paragraphs.length - 1 ? "text-ink" : undefined}>
              {tx(paragraph)}
            </p>
          ))}
        </div>

        <ul className="glass h-fit space-y-3.5 rounded-lg p-6">
          {STORY.facts.map((fact) => (
            <li key={fact.en} className="flex gap-3 text-[14px] leading-snug text-ink-muted">
              <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-bio/12 text-bio">
                <Check className="size-3" strokeWidth={2.4} aria-hidden />
              </span>
              {tx(fact)}
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}
