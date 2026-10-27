"use client";

/**
 * The few pieces every section of the public page is built from.
 *
 * Client components, because the page follows the visitor's language choice and that choice
 * lives in the browser (`useLocale`). The server still renders every word in Turkish first,
 * so the HTML is complete without JavaScript and a crawler reads the primary language.
 */

import { Check, Copy } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useLocale } from "@/lib/hooks/useLocale";
import { cn } from "@/lib/utils";
import { QUICKSTART, pick, type Text } from "./copy";

/** `pick` bound to the visitor's language. */
export function useText() {
  const { locale } = useLocale();
  return { locale, tx: (text: Text) => pick(text, locale) };
}

/**
 * The GitHub mark. Lucide dropped brand icons, and a repository link without the mark reads as
 * a link to anywhere.
 */
export function GitHubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={cn("size-4", className)} aria-hidden>
      <path d="M12 .5C5.65.5.5 5.65.5 12.02c0 5.09 3.29 9.4 7.86 10.93.58.1.79-.25.79-.56v-1.97c-3.2.7-3.87-1.54-3.87-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.43-2.7 5.4-5.27 5.69.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.53 11.53 0 0 0 23.5 12.02C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}

export function Section({
  id,
  children,
  className,
  "aria-labelledby": labelledBy,
}: {
  id?: string;
  children: ReactNode;
  className?: string;
  "aria-labelledby"?: string;
}) {
  return (
    <section
      id={id}
      aria-labelledby={labelledBy}
      className={cn("mx-auto w-full max-w-6xl scroll-mt-24 px-4 sm:px-6", className)}
    >
      {children}
    </section>
  );
}

export function SectionHeader({
  id,
  eyebrow,
  title,
  lead,
  aside,
}: {
  id: string;
  eyebrow: string;
  title: string;
  lead?: string;
  aside?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-5 border-t border-line pt-10 sm:pt-12 lg:flex-row lg:items-end lg:justify-between">
      <div className="max-w-2xl">
        <p className="font-mono text-[11.5px] font-medium tracking-[0.14em] text-bio uppercase">{eyebrow}</p>
        <h2
          id={id}
          className="mt-3 text-[clamp(1.6rem,3.4vw,2.35rem)] leading-[1.1] font-semibold tracking-[-0.03em] text-ink"
        >
          {title}
        </h2>
        {lead ? <p className="mt-4 text-[15px] leading-relaxed text-ink-muted">{lead}</p> : null}
      </div>
      {aside ? <div className="shrink-0">{aside}</div> : null}
    </div>
  );
}

/**
 * Copies a string and says so for a moment. Falls back to a hidden textarea where the async
 * clipboard is not available (an http origin on the LAN is not a secure context).
 */
export function CopyButton({ value, className }: { value: string; className?: string }) {
  const { tx } = useText();
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      const area = document.createElement("textarea");
      area.value = value;
      area.setAttribute("readonly", "");
      area.style.cssText = "position:fixed;opacity:0;pointer-events:none";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
      } finally {
        area.remove();
      }
    }
    setCopied(true);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), 1600);
  }, [value]);

  const label = tx(copied ? QUICKSTART.copied : QUICKSTART.copy);

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={label}
      title={label}
      className={cn(
        "grid size-8 shrink-0 place-items-center rounded-sm border border-line bg-surface-2/80 text-ink-faint",
        "transition-colors hover:border-line-strong hover:text-ink",
        copied && "border-bio/40 text-bio hover:text-bio",
        className,
      )}
    >
      {copied ? <Check className="size-3.5" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
      <span className="sr-only" aria-live="polite">
        {copied ? label : ""}
      </span>
    </button>
  );
}

/**
 * A shell snippet with a copy button. Comment lines are dimmed so the commands stand out, and
 * the copied text is exactly what is shown — comments included, which every shell ignores.
 */
export function CodeBlock({ code, title, className }: { code: string; title?: string; className?: string }) {
  return (
    <div className={cn("overflow-hidden rounded-md border border-line bg-canvas-raised", className)}>
      <div className="flex items-center justify-between gap-3 border-b border-line bg-surface/60 py-1.5 pr-1.5 pl-4">
        <span className="truncate font-mono text-[11.5px] text-ink-faint">{title ?? "shell"}</span>
        <CopyButton value={code} />
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[12.5px] leading-[1.75]">
        <code>
          {code.split("\n").map((line, index) => {
            const hash = line.indexOf("#");
            const isComment = line.trimStart().startsWith("#");
            return (
              <span key={index} className="block whitespace-pre">
                {isComment ? (
                  <span className="text-ink-faint">{line}</span>
                ) : hash > 0 && /\s#\s/.test(line) ? (
                  <>
                    <span className="text-ink">{line.slice(0, hash)}</span>
                    <span className="text-ink-faint">{line.slice(hash)}</span>
                  </>
                ) : (
                  <span className="text-ink">{line || " "}</span>
                )}
              </span>
            );
          })}
        </code>
      </pre>
    </div>
  );
}
