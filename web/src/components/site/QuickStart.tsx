"use client";

import { ArrowRight, ExternalLink, Info, ShieldAlert } from "lucide-react";
import Link from "next/link";
import { useId, useState } from "react";
import { DOCS } from "@/lib/project";
import { cn } from "@/lib/utils";
import { COMMANDS, QUICKSTART, REQUIREMENTS, URLS } from "./copy";
import { CodeBlock, Section, SectionHeader, useText } from "./primitives";

type Tab = "scripts" | "manual";

/** Requirements, two ways to install, and where everything ends up. */
export function QuickStart() {
  const { tx, locale } = useText();
  const [tab, setTab] = useState<Tab>("scripts");
  const tabsId = useId();

  const tabs: { id: Tab; label: string }[] = [
    { id: "scripts", label: tx(QUICKSTART.tabScripts) },
    { id: "manual", label: tx(QUICKSTART.tabManual) },
  ];

  return (
    <Section id="kurulum" aria-labelledby="kurulum-title" className="pb-24">
      <SectionHeader
        id="kurulum-title"
        eyebrow={tx(QUICKSTART.eyebrow)}
        title={tx(QUICKSTART.title)}
        lead={tx(QUICKSTART.lead)}
        aside={
          <Link href="/rehber" className="inline-flex items-center gap-1.5 rounded-full border border-bio/40 bg-bio/10 px-4 py-2 text-[13.5px] font-medium text-bio transition-colors hover:bg-bio/15">
            {locale === "tr" ? "Adım adım rehber" : "Step-by-step guide"} <ArrowRight className="size-3.5" aria-hidden />
          </Link>
        }
      />

      <div className="mt-10 grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-10">
        {/* ── Requirements ──────────────────────────────────────────────────────────── */}
        <div>
          <h3 className="text-[13px] font-semibold tracking-wide text-ink uppercase">{tx(QUICKSTART.requirements)}</h3>
          <div className="mt-4 overflow-hidden rounded-lg border border-line">
            <table className="w-full text-left text-[13px]">
              <thead className="bg-surface-2/70 text-[11.5px] tracking-wide text-ink-faint uppercase">
                <tr>
                  <th scope="col" className="px-4 py-2.5 font-semibold">
                    {tx(QUICKSTART.colTool)}
                  </th>
                  <th scope="col" className="px-4 py-2.5 font-semibold">
                    {tx(QUICKSTART.colVersion)}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {REQUIREMENTS.map((row) => (
                  <tr key={row.tool} className="align-top">
                    <th scope="row" className="px-4 py-3 font-normal">
                      <span className="block font-semibold text-ink">{row.tool}</span>
                      <span className="mt-0.5 block text-[12.5px] text-ink-muted">{tx(row.why)}</span>
                    </th>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span
                        className={cn(
                          "inline-block rounded-xs px-1.5 py-0.5 font-mono text-[12px]",
                          row.optional ? "bg-surface-2 text-ink-muted" : "bg-bio/10 text-bio",
                        )}
                      >
                        {tx(row.version)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-5 flex gap-2.5 text-[13px] leading-relaxed text-ink-muted">
            <Info className="mt-0.5 size-4 shrink-0 text-ink-faint" aria-hidden />
            <span>
              {tx(QUICKSTART.firstAccount)}{" "}
              <a
                href={DOCS.envExample}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 font-mono text-[12.5px] text-bio hover:underline"
              >
                engine/.env.example
                <ExternalLink className="size-3" aria-hidden />
              </a>
            </span>
          </p>
        </div>

        {/* ── Install ───────────────────────────────────────────────────────────────── */}
        <div className="min-w-0">
          <div role="tablist" aria-label={tx(QUICKSTART.title)} className="flex gap-1 border-b border-line">
            {tabs.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                id={`${tabsId}-${item.id}-tab`}
                aria-selected={tab === item.id}
                aria-controls={`${tabsId}-${item.id}-panel`}
                onClick={() => setTab(item.id)}
                className={cn(
                  "-mb-px border-b-2 px-3 py-2.5 text-[13.5px] font-medium transition-colors",
                  tab === item.id
                    ? "border-bio text-ink"
                    : "border-transparent text-ink-faint hover:text-ink-muted",
                )}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div
            role="tabpanel"
            id={`${tabsId}-${tab}-panel`}
            aria-labelledby={`${tabsId}-${tab}-tab`}
            className="mt-5 space-y-4"
          >
            {tab === "scripts" ? (
              <>
                <CodeBlock code={tx(COMMANDS.scripts)} title="PowerShell" />
                <div className="rounded-md border border-amber/25 bg-amber/[0.06] p-4">
                  <p className="text-[13px] leading-relaxed text-ink-muted">{tx(QUICKSTART.policy)}</p>
                  <CodeBlock code={tx(COMMANDS.policy)} title="PowerShell" className="mt-3" />
                </div>
              </>
            ) : (
              <>
                <CodeBlock code={tx(COMMANDS.engine)} title={`${tx(QUICKSTART.engine)} · engine/`} />
                <CodeBlock code={tx(COMMANDS.web)} title={`${tx(QUICKSTART.web)} · web/`} />
              </>
            )}
          </div>

          <div className="mt-6">
            <p className="text-[12px] font-semibold tracking-wide text-ink-faint uppercase">{tx(QUICKSTART.open)}</p>
            <ul className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-3">
              {URLS.map((url) => (
                <li key={url.href}>
                  <span className="flex h-full flex-col gap-1 rounded-md border border-line bg-surface/40 px-3.5 py-2.5">
                    <span className="text-[12px] text-ink-faint">{tx(QUICKSTART[url.label])}</span>
                    <span className="font-mono text-[12.5px] break-all text-ink">{url.href.replace(/^https?:\/\//, "")}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="mt-10 flex gap-3 rounded-lg border border-line bg-surface/30 p-5 text-[13.5px] leading-relaxed text-ink-muted">
        <ShieldAlert className="mt-0.5 size-4 shrink-0 text-amber" strokeWidth={1.8} aria-hidden />
        <p>{tx(QUICKSTART.operator)}</p>
      </div>
    </Section>
  );
}
