"use client";

import Link from "next/link";
import { LogoMark } from "@/components/dashboard/Logo";
import { useLocale } from "@/lib/hooks/useLocale";
import { AUTHOR, DOCS, REPO_URL, SITE_ONLY } from "@/lib/project";
import { FOOTER, NAV, pick, type Text } from "./copy";
import { GitHubMark } from "./primitives";

/**
 * The public footer: where the code lives, what licence it is under, and who wrote it.
 *
 * No company block and no legal pages. There is no company: this is an open-source project,
 * and whoever runs a public instance of it is the operator who owes their visitors a privacy
 * notice — the last line says exactly that.
 */

interface FooterLink {
  label: Text;
  href: string;
  external?: boolean;
}

const PROJECT_LINKS: FooterLink[] = [
  { label: FOOTER.repo, href: REPO_URL, external: true },
  { label: FOOTER.readme, href: DOCS.readme, external: true },
  { label: FOOTER.architecture, href: DOCS.architecture, external: true },
  { label: FOOTER.contributing, href: DOCS.contributing, external: true },
  { label: FOOTER.license, href: DOCS.license, external: true },
];

export function SiteFooter() {
  const { locale } = useLocale();

  const column = (id: string, title: Text, links: FooterLink[]) => (
    <nav aria-labelledby={id}>
      <h2 id={id} className="text-[12px] font-semibold tracking-wide text-ink uppercase">
        {pick(title, locale)}
      </h2>
      <ul className="mt-3 space-y-2 text-[13px] text-ink-muted">
        {links.map((link) => (
          <li key={link.href}>
            {link.external ? (
              <a href={link.href} target="_blank" rel="noreferrer" className="transition-colors hover:text-ink">
                {pick(link.label, locale)}
              </a>
            ) : link.href.startsWith("#") ? (
              <a href={link.href} className="transition-colors hover:text-ink">
                {pick(link.label, locale)}
              </a>
            ) : (
              <Link href={link.href} className="transition-colors hover:text-ink">
                {pick(link.label, locale)}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );

  return (
    <footer className="mt-auto border-t border-line bg-canvas-raised/40">
      <div className="mx-auto grid w-full max-w-6xl grid-cols-2 gap-x-6 gap-y-10 px-4 py-12 sm:px-6 lg:grid-cols-[2fr_1fr_1fr]">
        <div className="col-span-2 lg:col-span-1">
          <span className="flex items-center gap-2 text-ink">
            <LogoMark className="size-5 text-bio" />
            <span className="text-[15px] font-semibold tracking-tight">BioSphere AI</span>
          </span>
          <p className="mt-3 max-w-xs text-[13px] leading-relaxed text-ink-muted">{pick(FOOTER.tagline, locale)}</p>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="mt-5 inline-flex items-center gap-2 rounded-full border border-line px-3 py-1.5 text-[12.5px] text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
          >
            <GitHubMark className="size-3.5" />
            <span className="font-mono">{REPO_URL.replace(/^https:\/\//, "")}</span>
          </a>
          {SITE_ONLY ? null : (
            <p className="mt-4 text-[12.5px]">
              <Link href="/evler" className="text-ink-faint transition-colors hover:text-ink">
                {pick(FOOTER.dashboard, locale)} →
              </Link>
            </p>
          )}
        </div>

        {column("footer-project", FOOTER.project, PROJECT_LINKS)}
        {column(
          "footer-page",
          FOOTER.page,
          NAV.map((item) => ({ label: item.label, href: item.href })),
        )}
      </div>

      <div className="border-t border-line">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 py-5 text-[12px] text-ink-faint sm:px-6 md:flex-row md:items-center md:justify-between">
          <p>
            <a href={DOCS.license} target="_blank" rel="noreferrer" className="transition-colors hover:text-ink-muted">
              {pick(FOOTER.license, locale)}
            </a>{" "}
            · BioSphere AI ·{" "}
            <a href={AUTHOR.url} target="_blank" rel="noreferrer" className="transition-colors hover:text-ink-muted">
              {AUTHOR.handle}
            </a>
          </p>
          <p className="md:text-right">{pick(FOOTER.operator, locale)}</p>
        </div>
      </div>
    </footer>
  );
}
