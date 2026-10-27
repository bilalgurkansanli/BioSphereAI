"use client";

import { ArrowUpRight } from "lucide-react";
import Link from "next/link";
import { useEffect } from "react";
import { Wordmark } from "@/components/dashboard/Logo";
import { buttonStyles } from "@/components/ui/Button";
import { useLocale } from "@/lib/hooks/useLocale";
import { REPO_URL, SITE_ONLY } from "@/lib/project";
import { cn } from "@/lib/utils";
import { HEADER, NAV, pick } from "./copy";
import { GitHubMark } from "./primitives";

/**
 * The public page's header: the project, its sections, the language, and the repository.
 *
 * Sticky and frosted, because the page is long and the one action that matters — the
 * repository — should never be more than a glance away. The dashboard link is deliberately
 * quiet: it is for somebody who already runs an installation, not an invitation to sign up.
 */
export function SiteHeader() {
  const { locale, setLocale } = useLocale();

  // The document says which language it is in. The server renders Turkish; a visitor who
  // chose English gets `lang="en"` once the stored choice is read.
  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-canvas/75 backdrop-blur-xl backdrop-saturate-150">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between gap-3 px-4 sm:px-6">
        <Link href="/" aria-label={pick(HEADER.home, locale)} className="shrink-0">
          <Wordmark />
        </Link>

        <nav aria-label={pick(HEADER.nav, locale)} className="hidden items-center gap-1 md:flex">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="rounded-sm px-3 py-1.5 text-[13.5px] text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
            >
              {pick(item.label, locale)}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          {SITE_ONLY ? null : (
            <Link
              href="/evler"
              className="hidden items-center gap-1 px-2 text-[13px] text-ink-faint transition-colors hover:text-ink lg:inline-flex"
            >
              {pick(HEADER.dashboard, locale)}
              <ArrowUpRight className="size-3.5" aria-hidden />
            </Link>
          )}

          <div
            role="group"
            aria-label={pick(HEADER.language, locale)}
            className="flex items-center gap-0.5 rounded-full border border-line bg-surface-2/60 p-0.5"
          >
            {(["tr", "en"] as const).map((code) => (
              <button
                key={code}
                type="button"
                lang={code}
                aria-pressed={locale === code}
                onClick={() => setLocale(code)}
                className={cn(
                  "rounded-full px-2.5 py-1 text-[11px] font-semibold tracking-wide uppercase transition-colors",
                  locale === code ? "bg-surface-3 text-ink" : "text-ink-faint hover:text-ink-muted",
                )}
              >
                {code}
              </button>
            ))}
          </div>

          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            aria-label={pick(HEADER.github, locale)}
            className={buttonStyles("secondary", "sm", "px-2.5 sm:px-3")}
          >
            <GitHubMark className="size-4" />
            <span className="hidden sm:inline">{pick(HEADER.githubShort, locale)}</span>
          </a>
        </div>
      </div>
    </header>
  );
}
