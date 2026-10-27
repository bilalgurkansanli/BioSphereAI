"use client";

import { TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { ui, type UiKey } from "./copy";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { useLocale } from "@/lib/hooks/useLocale";
import { stageLabel, t, type StringKey } from "@/lib/i18n";
import { JOB_STAGES, type DeviceInfo, type JobStage, type Locale, type StitchMetrics } from "@/lib/types";
import { formatDuration } from "@/lib/utils";

type Verdict = "great" | "ok" | "poor";

const VERDICT_TONE: Record<Verdict, BadgeTone> = { great: "success", ok: "sky", poor: "amber" };

/** Lower is better: the two thresholds are the excellent and the acceptable ceiling. */
function lowerIsBetter(value: number, great: number, ok: number): Verdict {
  if (value <= great) return "great";
  if (value <= ok) return "ok";
  return "poor";
}

/** Inside the sweet spot is excellent, inside the tolerated band is fine. */
function insideBand(value: number, great: [number, number], ok: [number, number]): Verdict {
  if (value >= great[0] && value <= great[1]) return "great";
  if (value >= ok[0] && value <= ok[1]) return "ok";
  return "poor";
}

function MetricRow({
  label,
  value,
  explain,
  verdict,
  locale,
}: {
  label: string;
  value: ReactNode;
  explain?: string;
  verdict?: Verdict;
  locale: Locale;
}) {
  return (
    <div className="border-b border-line py-2.5 last:border-b-0">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[12px] text-ink-muted">{label}</span>
        <span className="flex shrink-0 items-center gap-2">
          <span className="font-mono text-[13px] text-ink tabular-nums">{value}</span>
          {verdict ? (
            <Badge tone={VERDICT_TONE[verdict]}>{ui(`metrics.verdict.${verdict}` as UiKey, locale)}</Badge>
          ) : null}
        </span>
      </div>
      {explain ? <p className="mt-1 text-[11px] leading-relaxed text-ink-faint">{explain}</p> : null}
    </div>
  );
}

export function MetricsReport({ metrics }: { metrics: StitchMetrics | null }) {
  const { locale } = useLocale();

  if (!metrics) {
    return <p className="text-[13px] text-ink-faint">{ui("metrics.none", locale)}</p>;
  }

  const stages = Object.entries(metrics.stage_seconds)
    .filter(([, seconds]) => seconds > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  const slowest = stages[0]?.[1] ?? 1;

  const label = (key: StringKey) => t(key, locale);

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-line bg-surface-2/40 px-3 py-1">
        <MetricRow
          locale={locale}
          label={label("metrics.keyframes")}
          value={`${metrics.n_keyframes} / ${metrics.n_frames_extracted}`}
          verdict={insideBand(metrics.n_keyframes, [24, 60], [16, 90])}
          explain={ui("metrics.explain.keyframes", locale)}
        />
        <MetricRow
          locale={locale}
          label={label("metrics.reprojection")}
          value={`${metrics.mean_reprojection_error_px.toFixed(2)} px`}
          verdict={lowerIsBetter(metrics.mean_reprojection_error_px, 1.5, 3)}
          explain={ui("metrics.explain.reprojection", locale)}
        />
        <MetricRow
          locale={locale}
          label={label("metrics.loop")}
          value={`${metrics.loop_closure_residual_deg.toFixed(2)}°`}
          verdict={lowerIsBetter(metrics.loop_closure_residual_deg, 1, 3)}
          explain={ui("metrics.explain.loop", locale)}
        />
        <MetricRow
          locale={locale}
          label={label("metrics.seam")}
          value={metrics.seam_gradient_energy.toFixed(3)}
          verdict={lowerIsBetter(metrics.seam_gradient_energy, 0.12, 0.3)}
          explain={ui("metrics.explain.seam", locale)}
        />
        <MetricRow
          locale={locale}
          label={ui("metrics.exposure", locale)}
          value={metrics.exposure_spread.toFixed(3)}
          verdict={lowerIsBetter(metrics.exposure_spread, 0.15, 0.35)}
          explain={ui("metrics.explain.exposure", locale)}
        />
        <MetricRow
          locale={locale}
          label={label("metrics.hfov")}
          value={`${metrics.hfov_deg.toFixed(1)}°`}
          explain={ui("metrics.explain.hfov", locale)}
        />
        {/* The phone, and its lens against the solve's own answer. Worth a row of its own
            because the two are different claims: the lens is what the hardware is, the
            field of view above is what the imagery measured, and a hand-held sweep can only
            push the measurement one way — narrower. A gap here is the parallax, quantified. */}
        {metrics.device && metrics.device.source !== "none" ? (
          <MetricRow
            locale={locale}
            label={ui("metrics.device", locale)}
            value={deviceValue(metrics.device)}
            explain={deviceExplain(metrics.device, metrics.hfov_deg, locale)}
          />
        ) : null}
        <MetricRow
          locale={locale}
          label={ui("metrics.pairs", locale)}
          value={`${metrics.n_pairs} · ${metrics.n_loop_closures} ${ui("metrics.loops", locale).toLocaleLowerCase(locale)}`}
          explain={ui("metrics.explain.pairs", locale)}
        />
        <MetricRow
          locale={locale}
          label={ui("metrics.pattern", locale)}
          value={ui(`metrics.pattern.${metrics.sweep_pattern}` as UiKey, locale)}
        />
      </div>

      {stages.length > 0 ? (
        <div>
          <h4 className="mb-2 text-[12px] font-semibold text-ink">{ui("metrics.timing", locale)}</h4>
          <ul className="space-y-1.5">
            {stages.map(([key, seconds]) => (
              <li key={key} className="flex items-center gap-2.5">
                <span className="w-28 shrink-0 truncate text-[11px] text-ink-muted">
                  {(JOB_STAGES as readonly string[]).includes(key) ? stageLabel(key as JobStage, locale) : key}
                </span>
                <span className="h-1 flex-1 overflow-hidden rounded-full bg-surface-3">
                  <span
                    className="block h-full rounded-full bg-bio/60"
                    style={{ width: `${Math.max(4, (seconds / slowest) * 100)}%` }}
                  />
                </span>
                <span className="w-14 shrink-0 text-right font-mono text-[11px] text-ink-faint tabular-nums">
                  {formatDuration(seconds)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {metrics.warnings.length > 0 ? (
        <div>
          <h4 className="mb-2 flex items-center gap-2 text-[12px] font-semibold text-amber">
            <TriangleAlert className="size-3.5" aria-hidden />
            {ui("metrics.warnings", locale)}
          </h4>
          <ul className="space-y-1.5">
            {metrics.warnings.map((warning) => (
              <li
                key={warning}
                className="rounded-sm border border-amber/20 bg-amber/5 px-2.5 py-1.5 text-[11px] leading-relaxed text-amber"
              >
                {warning}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}


/** "vivo Y22s · 51.7°" — the phone and the field its lens actually has. */
function deviceValue(device: DeviceInfo): string {
  const name =
    device.source === "generic"
      ? "?"
      : [device.brand, device.model].filter(Boolean).join(" ").replace(/^Apple iPhone/, "iPhone");
  const lens = device.hfov_deg === null ? "" : ` · ${device.hfov_deg.toFixed(1)}°`;
  return `${name}${lens}`;
}

/**
 * Says what the identification was worth and what the disagreement means.
 *
 * Deliberately not a verdict badge: a solve that lands wider or narrower than the lens is
 * not a failure, it is a measurement of how far the capture was from a turn on the spot.
 */
function deviceExplain(device: DeviceInfo, solvedHfov: number, locale: Locale): string {
  const tr = locale === "tr";
  const bits: string[] = [];
  if (device.source === "ios-fingerprint" && device.candidates.length > 1) {
    bits.push(
      tr
        ? `Ekran ölçüsü ${device.candidates.length} modele uyuyor; objektif değeri bu ailenin ortası.`
        : `The screen matches ${device.candidates.length} models; the lens figure is that family's midpoint.`,
    );
  } else if (device.source === "generic") {
    bits.push(
      tr
        ? "Telefon tanınamadı; sektörün standart ana kamerası varsayıldı."
        : "The phone could not be identified; the industry-standard main camera is assumed.",
    );
  } else if (device.source === "brand") {
    bits.push(tr ? "Model tanınamadı; markanın ortanca objektifi." : "Model unknown; the brand's median lens.");
  }
  if (device.hfov_deg !== null && device.hfov_deg > 0) {
    const drift = ((solvedHfov - device.hfov_deg) / device.hfov_deg) * 100;
    if (Math.abs(drift) >= 4) {
      bits.push(
        tr
          ? `Çözüm ${Math.abs(drift).toFixed(0)}% ${drift < 0 ? "daha dar" : "daha geniş"} bir alan ölçtü — telefon dönerken yer de değiştirdiğinde beklenen budur.`
          : `The solve measured a field ${Math.abs(drift).toFixed(0)}% ${drift < 0 ? "narrower" : "wider"}, which is what a capture that moved as well as turned does.`,
      );
    } else {
      bits.push(
        tr
          ? "Çözüm objektifin kendi değeriyle uyuşuyor: çekim neredeyse yerinde dönmüş."
          : "The solve agrees with the lens, so the capture turned very nearly on the spot.",
      );
    }
  }
  if (!device.ois) {
    bits.push(
      tr
        ? "Optik sabitleme yok: yavaş çevirmek ve daha çok ışık, netliği doğrudan artırır."
        : "No optical stabilisation: a slower sweep and more light are what buy sharpness here.",
    );
  }
  if (device.lidar) {
    bits.push(
      tr
        ? "Bu modelde LiDAR var, ama tarayıcı ona erişemez; ölçüm kamera yüksekliğinden gelir."
        : "This model has LiDAR, but a browser cannot read it; measurements come from the camera height.",
    );
  }
  return bits.join(" ");
}
