"use client";

import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ui } from "./copy";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";

/**
 * A breadcrumb built from the URL alone would read "Turlar / 8f3c-…". Pages that know the
 * human name for a dynamic segment register it here, and the trail picks it up. The
 * registry is a context rather than global state so it unmounts with the route.
 */
interface CrumbRegistry {
  labels: Record<string, string>;
  set: (path: string, label: string | null) => void;
}

const CrumbContext = createContext<CrumbRegistry>({ labels: {}, set: () => {} });

export function CrumbProvider({ children }: { children: ReactNode }) {
  const [labels, setLabels] = useState<Record<string, string>>({});

  const set = useCallback((path: string, label: string | null) => {
    setLabels((current) => {
      if (label === null) {
        if (!(path in current)) return current;
        const next = { ...current };
        delete next[path];
        return next;
      }
      if (current[path] === label) return current;
      return { ...current, [path]: label };
    });
  }, []);

  const value = useMemo(() => ({ labels, set }), [labels, set]);
  return <CrumbContext.Provider value={value}>{children}</CrumbContext.Provider>;
}

/** Registers a human label for the given pathname while the component is mounted. */
export function useCrumbLabel(path: string, label: string | null | undefined) {
  const { set } = useContext(CrumbContext);

  useEffect(() => {
    if (!label) return;
    set(path, label);
    return () => set(path, null);
  }, [path, label, set]);
}

interface Crumb {
  href: string;
  label: string;
  current: boolean;
}

export function Breadcrumbs() {
  const pathname = usePathname();
  const { labels } = useContext(CrumbContext);
  const { locale } = useLocale();

  const crumbs = useMemo<Crumb[]>(() => {
    const segments = pathname.split("/").filter(Boolean);
    const out: Crumb[] = [];
    let href = "";

    segments.forEach((segment, index) => {
      href += `/${segment}`;
      // `scenes` is a URL container, not a place you can navigate to.
      if (segment === "scenes") return;

      const known: Record<string, string> = {
        properties: t("nav.properties", locale),
        evler: t("nav.properties", locale),
        new: t("nav.new", locale),
        dekor: ui("decor.nav", locale),
        ayarlar: t("nav.settings", locale),
      };

      out.push({
        href,
        label: known[segment] ?? labels[href] ?? `${segment.slice(0, 6)}…`,
        current: index === segments.length - 1,
      });
    });

    return out;
  }, [pathname, labels, locale]);

  return (
    <nav aria-label={ui("chrome.breadcrumb", locale)} className="min-w-0">
      <ol className="flex min-w-0 items-center gap-1 text-[13px]">
        {crumbs.map((crumb, index) => (
          <li key={crumb.href} className="flex min-w-0 items-center gap-1">
            {index > 0 ? <ChevronRight className="size-3.5 shrink-0 text-ink-faint" aria-hidden /> : null}
            {crumb.current ? (
              <span aria-current="page" className="truncate font-medium text-ink">
                {crumb.label}
              </span>
            ) : (
              <Link
                href={crumb.href}
                // Ancestors give up their width first so the current page keeps its name
                // readable on a phone.
                className="max-w-24 truncate text-ink-faint transition-colors hover:text-ink-muted sm:max-w-none"
              >
                {crumb.label}
              </Link>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
