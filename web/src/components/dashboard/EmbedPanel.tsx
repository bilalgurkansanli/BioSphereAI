"use client";

import { ExternalLink } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ui } from "./copy";
import { CopyButton } from "@/components/ui/CopyButton";
import { Switch } from "@/components/ui/Switch";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import type { Property } from "@/lib/types";
import { cn } from "@/lib/utils";

const SITE_FROM_ENV = (process.env.NEXT_PUBLIC_SITE_URL ?? "").replace(/\/+$/, "");

/**
 * What the frame may use. VR needs all of it: `xr-spatial-tracking` for a headset (WebXR is
 * refused outright in a frame without it), the motion sensors for a phone in a VR box —
 * `magnetometer` as well, for browsers that only deliver the compass-referenced orientation
 * event — and `screen-wake-lock`, or the phone dims inside the box. Where a portal strips it,
 * the player offers the same room in a new tab instead (`components/vr/capabilities.ts`).
 */
const ALLOW = "fullscreen; xr-spatial-tracking; accelerometer; gyroscope; magnetometer; screen-wake-lock";

function plainSnippet(src: string): string {
  return (
    `<iframe src="${src}" width="100%" height="560" style="border:0" ` +
    `allow="${ALLOW}" allowfullscreen loading="lazy"></iframe>`
  );
}

/** Some portals strip fixed heights, so the 16:9 padding box is offered as an option. */
function responsiveSnippet(src: string): string {
  return [
    `<div style="position:relative;width:100%;padding-top:56.25%">`,
    `  <iframe src="${src}" style="position:absolute;inset:0;width:100%;height:100%;border:0"`,
    `    allow="${ALLOW}" allowfullscreen loading="lazy"></iframe>`,
    `</div>`,
  ].join("\n");
}

export interface EmbedPanelProps {
  property: Pick<Property, "public_token" | "published">;
  className?: string;
  /** Drops the heading when the panel is already inside a titled container. */
  bare?: boolean;
}

export function EmbedPanel({ property, className, bare = false }: EmbedPanelProps) {
  const { locale } = useLocale();
  const [responsive, setResponsive] = useState(false);
  const [origin, setOrigin] = useState(SITE_FROM_ENV);

  // Falls back to the browser's origin when the site URL was not configured at build time.
  useEffect(() => {
    if (!SITE_FROM_ENV) setOrigin(window.location.origin);
  }, []);

  const src = `${origin}/embed/${property.public_token}`;
  const snippet = useMemo(() => (responsive ? responsiveSnippet(src) : plainSnippet(src)), [responsive, src]);

  return (
    // `min-w-0` keeps the snippet's single long line from widening whatever contains it.
    <div className={cn("min-w-0 space-y-3.5", className)}>
      {!bare ? <h2 className="text-sm font-semibold tracking-tight text-ink">{t("embed.title", locale)}</h2> : null}

      <div className="overflow-hidden rounded-md border border-line bg-surface-2/60">
        {/* The snippet scrolls sideways, so the copy button sits in its own bar rather
            than floating over code that slides underneath it. */}
        <div className="flex items-center justify-between gap-2 border-b border-line px-3 py-2">
          <span className="font-mono text-[10px] tracking-widest text-ink-faint uppercase">html</span>
          <CopyButton
            value={snippet}
            label={t("embed.copy", locale)}
            copiedLabel={t("embed.copied", locale)}
            failedLabel={ui("common.copyFailed", locale)}
          />
        </div>
        <pre className="max-h-40 overflow-auto p-3 font-mono text-[11px] leading-relaxed whitespace-pre text-ink-muted">
          {snippet}
        </pre>
      </div>

      <Switch
        checked={responsive}
        onChange={setResponsive}
        label={ui("embed.responsive", locale)}
        description={ui("embed.responsive.desc", locale)}
      />

      <div className="flex items-center gap-2 rounded-md border border-line bg-surface-2/40 py-1.5 pr-1.5 pl-3">
        <span className="min-w-0 flex-1">
          <span className="block text-[11px] text-ink-faint">{ui("embed.link", locale)}</span>
          <span className="block truncate font-mono text-[11px] text-ink-muted">{src}</span>
        </span>
        <CopyButton
          value={src}
          label={t("embed.copy", locale)}
          copiedLabel={t("embed.copied", locale)}
          iconOnly
        />
        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          aria-label={ui("embed.open", locale)}
          className="grid size-8 shrink-0 place-items-center rounded-sm text-ink-faint transition-colors hover:bg-surface-3 hover:text-ink"
        >
          <ExternalLink className="size-3.5" aria-hidden />
        </a>
      </div>

      <p className="text-[12px] leading-relaxed text-ink-faint">{t("embed.hint", locale)}</p>

      {!property.published ? (
        <p className="rounded-sm border border-amber/20 bg-amber/5 px-3 py-2 text-[12px] leading-relaxed text-amber">
          {ui("embed.draft", locale)}
        </p>
      ) : null}
    </div>
  );
}
