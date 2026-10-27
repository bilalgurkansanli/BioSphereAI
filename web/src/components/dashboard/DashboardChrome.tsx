"use client";

import { Images, Plus, Settings, Sofa, type LucideIcon } from "lucide-react";
import { MotionConfig } from "motion/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Toaster, toast } from "sonner";
import type { CSSProperties, ReactNode } from "react";
import { AccountMenu } from "./AccountMenu";
import { Breadcrumbs, CrumbProvider } from "./Breadcrumbs";
import { LogoMark } from "./Logo";
import { ui } from "./copy";
import { Tooltip } from "@/components/ui/Tooltip";
import { useEngineHealth } from "@/lib/hooks/useEngineHealth";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import type { EngineStatus } from "@/lib/store";
import { cn } from "@/lib/utils";
import type { Locale } from "@/lib/types";

interface NavItem {
  key: "properties" | "new" | "decor" | "settings";
  href: string | null;
  icon: LucideIcon;
  label: (locale: Locale) => string;
  match: (pathname: string) => boolean;
}

const NAV: NavItem[] = [
  {
    key: "properties",
    href: "/evler",
    icon: Images,
    label: (l) => t("nav.properties", l),
    match: (p) => p.startsWith("/evler"),
  },
  {
    key: "new",
    href: "/new",
    icon: Plus,
    label: (l) => t("nav.new", l),
    match: (p) => p === "/new",
  },
  {
    key: "decor",
    href: "/dekor",
    icon: Sofa,
    label: (l) => ui("decor.nav", l),
    match: (p) => p.startsWith("/dekor"),
  },
  {
    key: "settings",
    href: "/ayarlar",
    icon: Settings,
    label: (l) => t("nav.settings", l),
    match: (p) => p.startsWith("/ayarlar"),
  },
];

const DOT: Record<EngineStatus, string> = {
  checking: "bg-ink-faint",
  online: "bg-success",
  degraded: "bg-amber",
  offline: "bg-danger",
};

const PILL: Record<EngineStatus, string> = {
  checking: "border-line text-ink-faint",
  online: "border-success/25 text-success",
  degraded: "border-amber/25 text-amber",
  offline: "border-danger/25 text-danger",
};

function EnginePill() {
  const { locale } = useLocale();
  const engine = useEngineHealth();

  const headline = ui(`chrome.engine.${engine.status}`, locale);

  return (
    <Tooltip
      side="bottom"
      content={
        <span className="block space-y-1.5">
          <span className="block font-semibold text-ink">{headline}</span>
          {engine.issues.length > 0 ? (
            <span className="block space-y-1">
              {engine.issues.map((issue) => (
                <span key={issue} className="block text-ink-muted">
                  {issue}
                </span>
              ))}
            </span>
          ) : engine.info ? (
            <span className="block text-ink-muted">{ui("chrome.engine.ok", locale)}</span>
          ) : null}
          {engine.info ? (
            <span className="block border-t border-line pt-1.5 font-mono text-[11px] text-ink-faint">
              <span className="block">
                {ui("chrome.engine.version", locale)} {engine.info.version}
              </span>
              <span className="block">
                {ui("chrome.engine.gpu", locale)}: {engine.info.gpu ?? ui("chrome.engine.gpu.none", locale)}
              </span>
              <span className="block">
                {ui("chrome.engine.disk", locale)}: {engine.info.media_root_free_gb.toFixed(1)} GB
              </span>
            </span>
          ) : null}
        </span>
      }
    >
      <button
        type="button"
        onClick={engine.refresh}
        aria-label={`${ui("chrome.engine", locale)}: ${headline}`}
        className={cn(
          "flex items-center gap-2 rounded-full border bg-surface-2/60 px-2.5 py-1.5 text-[12px] font-medium transition-colors hover:bg-surface-3",
          PILL[engine.status],
        )}
      >
        <span className="relative flex size-1.5">
          {engine.status !== "online" ? (
            <span className={cn("animate-breathe absolute inline-flex size-full rounded-full", DOT[engine.status])} />
          ) : null}
          <span className={cn("relative inline-flex size-1.5 rounded-full", DOT[engine.status])} />
        </span>
        <span className="hidden sm:inline">{headline}</span>
      </button>
    </Tooltip>
  );
}

function LocaleToggle() {
  const { locale, setLocale } = useLocale();

  return (
    <div
      role="group"
      aria-label={ui("chrome.locale", locale)}
      className="flex items-center gap-0.5 rounded-full border border-line bg-surface-2/60 p-0.5"
    >
      {(["tr", "en"] as const).map((code) => (
        <button
          key={code}
          type="button"
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
  );
}

interface NavButtonProps {
  item: NavItem;
  compact: boolean;
  /** Supplied by `Tooltip` when it wraps the rail variant. */
  "aria-describedby"?: string;
}

function NavButton({ item, compact, ...rest }: NavButtonProps) {
  const pathname = usePathname();
  const { locale } = useLocale();
  const active = item.match(pathname);
  const label = item.label(locale);
  const Icon = item.icon;

  const inner = (
    <>
      <Icon className="size-[18px]" strokeWidth={1.8} aria-hidden />
      {compact ? <span className="text-[10px] font-medium">{label}</span> : null}
      {active ? (
        <span
          className={cn(
            "bio-gradient absolute rounded-full",
            compact ? "inset-x-5 top-0 h-0.5" : "top-1/2 left-0 h-6 w-0.5 -translate-y-1/2",
          )}
          aria-hidden
        />
      ) : null}
    </>
  );

  const className = cn(
    "relative flex items-center justify-center transition-colors",
    compact ? "h-full flex-1 flex-col gap-1 pt-1" : "size-11 rounded-md",
    active ? "text-ink" : "text-ink-faint hover:text-ink-muted",
    !compact && active && "bg-surface-2",
    !compact && !active && "hover:bg-surface-2/60",
  );

  if (!item.href) {
    return (
      <button
        type="button"
        className={className}
        aria-label={compact ? undefined : label}
        onClick={() => toast(ui("chrome.settings.soon", locale))}
        {...rest}
      >
        {inner}
      </button>
    );
  }

  return (
    <Link
      href={item.href}
      className={className}
      aria-label={compact ? undefined : label}
      aria-current={active ? "page" : undefined}
      {...rest}
    >
      {inner}
    </Link>
  );
}

export function DashboardChrome({ children }: { children: ReactNode }) {
  const { locale } = useLocale();

  return (
    <MotionConfig reducedMotion="user">
      <CrumbProvider>
        {/* Rail — md and up. */}
        <aside
          aria-label={ui("chrome.menu", locale)}
          className="fixed inset-y-0 left-0 z-40 hidden w-16 flex-col items-center gap-1 border-r border-line bg-canvas-raised/80 py-4 backdrop-blur-xl md:flex"
        >
          <Link
            href="/"
            aria-label={t("app.name", locale)}
            className="mb-4 grid size-10 place-items-center rounded-md border border-line bg-surface-2 text-bio transition-colors hover:border-line-strong"
          >
            <LogoMark className="size-5" />
          </Link>
          {NAV.map((item) => (
            <Tooltip key={item.key} side="right" content={item.label(locale)}>
              <NavButton item={item} compact={false} />
            </Tooltip>
          ))}
        </aside>

        <div className="md:pl-16">
          <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line bg-canvas/80 px-4 backdrop-blur-xl sm:px-6">
            <Link href="/" className="text-bio md:hidden" aria-label={t("app.name", locale)}>
              <LogoMark className="size-5" />
            </Link>
            <Breadcrumbs />
            <div className="ml-auto flex items-center gap-2">
              <LocaleToggle />
              <EnginePill />
              <AccountMenu />
            </div>
          </header>

          <main className="pb-16 md:pb-0">{children}</main>
        </div>

        {/* Bottom bar — under md. */}
        <nav
          aria-label={ui("chrome.menu", locale)}
          className="fixed inset-x-0 bottom-0 z-40 flex h-16 items-stretch border-t border-line bg-canvas-raised/90 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl md:hidden"
        >
          {NAV.map((item) => (
            <NavButton key={item.key} item={item} compact />
          ))}
        </nav>

        <Toaster
          theme="dark"
          position="bottom-right"
          offset={16}
          toastOptions={{
            classNames: {
              toast: "!glass-strong !rounded-md !shadow-lift",
              title: "!text-ink !text-[13px] !font-medium",
              description: "!text-ink-muted !text-[12px]",
              actionButton: "!bg-bio !text-canvas",
              cancelButton: "!bg-surface-3 !text-ink-muted",
            },
          }}
          style={
            {
              "--normal-bg": "var(--surface-2)",
              "--normal-text": "var(--ink)",
              "--normal-border": "var(--line-strong)",
              "--error-bg": "var(--surface-2)",
              "--error-text": "var(--danger)",
              "--success-bg": "var(--surface-2)",
              "--success-text": "var(--success)",
            } as CSSProperties
          }
        />
      </CrumbProvider>
    </MotionConfig>
  );
}
