/**
 * The four numbers a buyer looks for first — computed by the engine, with no model in the loop.
 *
 * Gross yield, amortisman, price per m² and an example mortgage payment, each with the line it
 * rests on ("beklenen kira 60.000 TL/ay ile", "%50 peşinat · 120 ay · aylık %2,85 faiz"), then
 * the net yield and the purchase costs beneath. The small print is part of each tile rather
 * than a footnote: a number on a card a buyer screenshots has to carry its own basis.
 *
 * The same component is the visitor's card in the player and the agent's preview of it in the
 * dashboard, so the agent sees exactly what the visitor will.
 */

import { t } from "@/lib/i18n";
import type { AdvisorCard, Locale } from "@/lib/types";
import { cn } from "@/lib/utils";
import { fill, formatCompact, formatPct, formatRate, formatTl } from "./format";

interface Tile {
  label: string;
  value: string;
  note: string;
  muted?: boolean;
}

function tiles(card: AdvisorCard, locale: Locale): { primary: Tile[]; secondary: Tile[] } {
  const income = card.income;
  const rent = card.rent;
  const rentNote = rent
    ? fill(t(rent.status === "actual" ? "player.invest.kpi.rentActual" : "player.invest.kpi.rentExpected", locale), {
        rent: formatTl(rent.monthly_tl, locale),
      })
    : t("player.invest.kpi.noRent", locale);
  const years = t("player.invest.kpi.years", locale);
  const loan = card.mortgage;

  const primary: Tile[] = [
    {
      label: t("player.invest.kpi.gross", locale),
      value: income ? formatPct(income.gross_pct, locale) : "—",
      note: rentNote,
      muted: !income,
    },
    {
      label: t("player.invest.kpi.payback", locale),
      value: income ? `${formatCompact(income.gross_payback_years, locale, 1)} ${years}` : "—",
      note: income ? t("player.invest.kpi.paybackNote", locale) : t("player.invest.kpi.noRent", locale),
      muted: !income,
    },
    {
      label: t("player.invest.kpi.perM2", locale),
      value: card.per_m2 ? formatTl(card.per_m2.tl, locale) : "—",
      note: card.per_m2
        ? fill(t(card.per_m2.basis === "net" ? "player.invest.kpi.areaNet" : "player.invest.kpi.areaGross", locale), {
            area: formatCompact(card.per_m2.area_m2, locale, 1),
          })
        : t("player.invest.kpi.noArea", locale),
      muted: !card.per_m2,
    },
    {
      label: t("player.invest.kpi.mortgage", locale),
      value: `${formatTl(loan.monthly_payment_tl, locale)}${t("player.invest.kpi.perMonth", locale)}`,
      note: fill(t("player.invest.kpi.mortgageNote", locale), {
        down: formatRate(loan.down_payment_pct, locale),
        term: loan.term_months,
        rate: formatRate(loan.monthly_rate_pct, locale),
      }),
    },
  ];

  const secondary: Tile[] = [];
  if (income) {
    secondary.push({
      label: t("player.invest.kpi.net", locale),
      value: formatPct(income.net_pct, locale),
      note: fill(t("player.invest.kpi.netNote", locale), {
        vacancy: formatCompact(income.assumptions.vacancy_months, locale, 1),
      }),
    });
  }
  secondary.push({
    label: t("player.invest.kpi.costs", locale),
    value: formatTl(card.purchase.total_costs_tl, locale),
    note: t("player.invest.kpi.costsNote", locale),
  });
  return { primary, secondary };
}

export interface KpiCardProps {
  card: AdvisorCard;
  locale: Locale;
  className?: string;
}

export function KpiCard({ card, locale, className }: KpiCardProps) {
  const { primary, secondary } = tiles(card, locale);
  const source = card.mortgage.rate_source;
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-baseline justify-between gap-2 text-[11px] text-ink-faint">
        <span className="truncate">{card.location ?? ""}</span>
        <span className="shrink-0 font-medium tabular-nums text-ink-muted">{formatTl(card.price_tl, locale)}</span>
      </div>
      <dl className="grid grid-cols-2 gap-1.5">
        {primary.map((tile) => (
          <div key={tile.label} className="rounded-[var(--radius-sm)] border border-line bg-white/[0.03] px-2.5 py-2">
            <dt className="text-[10.5px] font-medium uppercase tracking-wide text-ink-faint">{tile.label}</dt>
            <dd className={cn("mt-0.5 text-[15px] font-semibold tabular-nums tracking-tight", tile.muted ? "text-ink-faint" : "text-ink")}>
              {tile.value}
            </dd>
            <dd className="mt-0.5 text-[10.5px] leading-snug text-ink-faint">{tile.note}</dd>
          </div>
        ))}
      </dl>
      <dl className="space-y-1 rounded-[var(--radius-sm)] border border-line px-2.5 py-2 text-[11.5px]">
        {secondary.map((tile) => (
          <div key={tile.label}>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-muted">{tile.label}</dt>
              <dd className="font-medium tabular-nums text-ink">{tile.value}</dd>
            </div>
            <dd className="text-[10.5px] leading-snug text-ink-faint">{tile.note}</dd>
          </div>
        ))}
      </dl>
      {source ? (
        <p className="text-[10.5px] leading-snug text-ink-faint">
          {t("player.invest.kpi.rateSource", locale)}:{" "}
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="underline decoration-line-strong underline-offset-2 hover:text-ink-muted"
          >
            {source.title}
          </a>{" "}
          ({source.date})
        </p>
      ) : null}
    </div>
  );
}
