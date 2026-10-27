"use client";

import { CircleCheck, CircleSlash, Info } from "lucide-react";
import { ui, type UiKey } from "./copy";
import { Badge } from "@/components/ui/Badge";
import { absoluteUrl } from "@/lib/api";
import { useLocale } from "@/lib/hooks/useLocale";
import { t } from "@/lib/i18n";
import type { Coverage, CoverageGap, Locale } from "@/lib/types";

/** Everything above this reads as a complete sphere to a human looking at the property. */
const COMPLETE_AT = 0.97;

function directionKey(yawDeg: number): UiKey {
  const yaw = ((yawDeg + 180) % 360 + 360) % 360 - 180;
  if (Math.abs(yaw) <= 45) return "dir.front";
  if (yaw > 45 && yaw <= 135) return "dir.right";
  if (yaw < -45 && yaw >= -135) return "dir.left";
  return "dir.back";
}

/**
 * Gaps arrive as four angles. An agent standing in the room needs a sentence, so the
 * midpoint becomes a direction they can turn towards and the span becomes "how big".
 */
function describeGap(gap: CoverageGap, locale: Locale): string {
  const yawMid = (gap.yaw_start_deg + gap.yaw_end_deg) / 2;
  const pitchMid = (gap.pitch_start_deg + gap.pitch_end_deg) / 2;
  const span = Math.max(Math.abs(gap.yaw_end_deg - gap.yaw_start_deg), Math.abs(gap.pitch_end_deg - gap.pitch_start_deg));

  const vertical: UiKey = pitchMid > 20 ? "dir.up" : pitchMid < -20 ? "dir.down" : "dir.level";

  return `${ui(directionKey(yawMid), locale)}, ${ui(vertical, locale)} ~${Math.round(span)}°`;
}

function Row({ label, value, tone }: { label: string; value: string; tone?: "ok" | "warn" }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-[12px] text-ink-muted">{label}</span>
      <span
        className={
          tone === "warn"
            ? "font-mono text-[12px] text-amber tabular-nums"
            : tone === "ok"
              ? "font-mono text-[12px] text-bio tabular-nums"
              : "font-mono text-[12px] text-ink tabular-nums"
        }
      >
        {value}
      </span>
    </div>
  );
}

export function CoveragePanel({ coverage }: { coverage: Coverage | null }) {
  const { locale } = useLocale();

  if (!coverage) {
    return <p className="text-[13px] text-ink-faint">{ui("coverage.none", locale)}</p>;
  }

  const complete = coverage.covered_fraction >= COMPLETE_AT || coverage.gaps.length === 0;
  const map = absoluteUrl(coverage.map_url);

  return (
    <div className="space-y-5">
      {map ? (
        <figure>
          <div className="overflow-hidden rounded-md border border-line bg-canvas">
            {/* Equirect heat-map from the engine origin — never resized by the optimiser. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={map} alt={ui("coverage.map", locale)} loading="lazy" className="block w-full" />
          </div>
          <figcaption className="mt-1.5 text-[11px] leading-relaxed text-ink-faint">
            {ui("coverage.map.legend", locale)}
          </figcaption>
        </figure>
      ) : null}

      <div
        className={
          complete
            ? "flex gap-2.5 rounded-md border border-bio/25 bg-bio/5 p-3"
            : "flex gap-2.5 rounded-md border border-amber/20 bg-amber/5 p-3"
        }
      >
        {complete ? (
          <CircleCheck className="mt-px size-4 shrink-0 text-bio" aria-hidden />
        ) : (
          <Info className="mt-px size-4 shrink-0 text-amber" aria-hidden />
        )}
        <p className={complete ? "text-[12px] leading-relaxed text-bio" : "text-[12px] leading-relaxed text-amber"}>
          {complete ? t("coverage.complete", locale) : t("coverage.reshoot", locale)}
        </p>
      </div>

      <div className="divide-y divide-line rounded-md border border-line bg-surface-2/40 px-3 py-1">
        <Row
          label={ui("coverage.covered", locale)}
          value={`%${Math.round(coverage.covered_fraction * 100)}`}
          tone={coverage.covered_fraction >= COMPLETE_AT ? "ok" : "warn"}
        />
        <Row
          label={ui("coverage.horizon", locale)}
          value={`%${Math.round(coverage.horizon_fraction * 100)}`}
          tone={coverage.horizon_fraction >= 0.98 ? "ok" : "warn"}
        />
        <Row label={ui("coverage.yawSpan", locale)} value={`${Math.round(coverage.yaw_span_deg)}°`} />
        <Row
          label={ui("coverage.pitchRange", locale)}
          value={`${Math.round(coverage.pitch_min_deg)}° … ${Math.round(coverage.pitch_max_deg)}°`}
        />
      </div>

      <div className="flex flex-wrap gap-2">
        <Badge tone={coverage.zenith_covered ? "bio" : "neutral"} dot>
          {ui("coverage.zenith", locale)} · {ui(coverage.zenith_covered ? "coverage.yes" : "coverage.no", locale)}
        </Badge>
        <Badge tone={coverage.nadir_covered ? "bio" : "neutral"} dot>
          {ui("coverage.nadir", locale)} · {ui(coverage.nadir_covered ? "coverage.yes" : "coverage.no", locale)}
        </Badge>
      </div>

      {!complete && coverage.gaps.length > 0 ? (
        <div>
          <h4 className="mb-2 flex items-center gap-2 text-[12px] font-semibold text-ink">
            <CircleSlash className="size-3.5 text-amber" aria-hidden />
            {ui("coverage.gapsTitle", locale)}
            <span className="text-ink-faint">
              ({coverage.gaps.length} {t("coverage.gaps", locale)})
            </span>
          </h4>
          <ul className="space-y-1.5">
            {coverage.gaps.slice(0, 6).map((gap, index) => (
              <li
                key={`${gap.yaw_start_deg}-${gap.pitch_start_deg}-${index}`}
                className="flex items-center gap-2 rounded-sm border border-line bg-surface-2/40 px-2.5 py-1.5 text-[12px] text-ink-muted"
              >
                <span className="size-1.5 shrink-0 rounded-full bg-amber" aria-hidden />
                {describeGap(gap, locale)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
