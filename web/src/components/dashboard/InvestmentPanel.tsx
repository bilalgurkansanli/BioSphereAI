"use client";

/**
 * The agent's side of the investment assistant: the figures it works from, the switch that
 * offers it to visitors, what visitors will see, and what it has cost.
 *
 * ── THE AGENT'S FIGURES ARE THE AUTHORITY ─────────────────────────────────────────────
 * Price, rent, aidat, the areas and the location are what every calculation in the chat is
 * built on — the engine fills them into its calculators itself rather than trusting a model to
 * copy them. So this form is the feature's real input, and each field says what happens when
 * it is left empty (emlak vergisi becomes an upper-bound estimate, commission the legal
 * ceiling) instead of quietly treating empty as zero.
 *
 * ── THE OPT-IN IS PER PROPERTY, AND SAYS WHAT LEAVES ──────────────────────────────────
 * Switching it on sends this listing's figures (never the street address, never the internal
 * title) and visitors' questions to a hosted model. The switch's own description says so; the
 * agent decides per listing.
 *
 * ── WHY IS IT NOT SHOWING? ────────────────────────────────────────────────────────────
 * The panel answers that before it is asked: the engine returns every blocker — not opted in,
 * no price, not published, no API key, today's budget spent — and each is listed in words with
 * what fixes it. The preview is the card exactly as a visitor sees it, from the same component.
 *
 * ── WHAT IT COSTS ────────────────────────────────────────────────────────────────────
 * Today, the last 7 and 30 days, questions asked and the average per question, from the
 * engine's per-call usage table — which holds tokens and dollars and nothing a visitor typed.
 */

import { AlertTriangle, Check, ExternalLink, TrendingUp } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { Switch } from "@/components/ui/Switch";
import { Textarea } from "@/components/ui/Textarea";
import { KpiCard } from "@/components/invest/KpiCard";
import { fill, formatNumber, formatUsd, parseTypedNumber } from "@/components/invest/format";
import { BioSphereError, api } from "@/lib/api";
import { useApi } from "@/lib/hooks/useApi";
import { t, type StringKey } from "@/lib/i18n";
import type { InvestmentInputs, InvestmentState, Locale, PropertyDetail } from "@/lib/types";

type NumberKind = "money" | "decimal" | "integer";
type NumberKey =
  | "asking_price_tl"
  | "monthly_rent_tl"
  | "aidat_monthly_tl"
  | "emlak_vergisi_yearly_tl"
  | "dask_yearly_tl"
  | "gross_m2"
  | "net_m2"
  | "building_age_years"
  | "floor"
  | "total_floors"
  | "commission_pct"
  | "daily_budget_usd";
type TextKey = "il" | "ilce" | "mahalle" | "room_layout" | "heating" | "notes";

const NUMBERS: Record<NumberKey, NumberKind> = {
  asking_price_tl: "money",
  monthly_rent_tl: "money",
  aidat_monthly_tl: "money",
  emlak_vergisi_yearly_tl: "money",
  dask_yearly_tl: "money",
  gross_m2: "decimal",
  net_m2: "decimal",
  building_age_years: "integer",
  floor: "integer",
  total_floors: "integer",
  commission_pct: "decimal",
  daily_budget_usd: "decimal",
};
const TEXTS: TextKey[] = ["il", "ilce", "mahalle", "room_layout", "heating", "notes"];

/** The form's state: what the agent typed, as typed. Numbers are parsed only on save. */
interface Draft {
  enabled: boolean;
  numbers: Record<NumberKey, string>;
  texts: Record<TextKey, string>;
  rent_status: string;
  energy_class: string;
  sale_kind: string;
  in_site: string;
}

function draftOf(inputs: InvestmentInputs, locale: Locale): Draft {
  const numbers = {} as Record<NumberKey, string>;
  for (const key of Object.keys(NUMBERS) as NumberKey[]) {
    const value = inputs[key];
    numbers[key] =
      value === null || value === undefined
        ? ""
        : NUMBERS[key] === "money"
          ? formatNumber(value, locale, 0)
          : String(value).replace(".", locale === "tr" ? "," : ".");
  }
  const texts = {} as Record<TextKey, string>;
  for (const key of TEXTS) texts[key] = inputs[key] ?? "";
  return {
    enabled: inputs.enabled,
    numbers,
    texts,
    rent_status: inputs.rent_status ?? "",
    energy_class: inputs.energy_class ?? "",
    sale_kind: inputs.sale_kind ?? "",
    in_site: inputs.in_site === true ? "yes" : inputs.in_site === false ? "no" : "",
  };
}

function inputsOf(draft: Draft): { inputs: InvestmentInputs; invalid: NumberKey[] } {
  const invalid: NumberKey[] = [];
  const inputs: InvestmentInputs = { enabled: draft.enabled };
  for (const key of Object.keys(NUMBERS) as NumberKey[]) {
    const value = parseTypedNumber(draft.numbers[key], NUMBERS[key]);
    if (value !== null && Number.isNaN(value)) invalid.push(key);
    else inputs[key] = value;
  }
  for (const key of TEXTS) inputs[key] = draft.texts[key].trim() || null;
  inputs.rent_status = (draft.rent_status || null) as InvestmentInputs["rent_status"];
  inputs.energy_class = (draft.energy_class || null) as InvestmentInputs["energy_class"];
  inputs.sale_kind = (draft.sale_kind || null) as InvestmentInputs["sale_kind"];
  inputs.in_site = draft.in_site === "yes" ? true : draft.in_site === "no" ? false : null;
  return { inputs, invalid };
}

export interface InvestmentPanelProps {
  property: PropertyDetail;
  locale: Locale;
}

export function InvestmentPanel({ property, locale }: InvestmentPanelProps) {
  const { data: state, initial, setData } = useApi<InvestmentState>(
    () => api.investment.get(property.id),
    [property.id],
  );
  // Derived from the server until the first keystroke, as `ListingPanel` does: copying the
  // loaded document into state with an effect would fight the agent's typing.
  const [edited, setEdited] = useState<Draft | null>(null);
  const [saving, setSaving] = useState(false);
  const [invalid, setInvalid] = useState<NumberKey[]>([]);

  const draft = useMemo(
    () => edited ?? (state ? draftOf(state.inputs, locale) : null),
    [edited, state, locale],
  );

  if (initial || !draft) {
    return (
      <section className="glass mt-6 rounded-lg p-5">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="mt-4 h-40 w-full" />
      </section>
    );
  }

  const change = (next: Partial<Draft>) => setEdited({ ...draft, ...next });
  const setNumber = (key: NumberKey, value: string) => change({ numbers: { ...draft.numbers, [key]: value } });
  const setText = (key: TextKey, value: string) => change({ texts: { ...draft.texts, [key]: value } });

  const save = async () => {
    const { inputs, invalid: bad } = inputsOf(draft);
    setInvalid(bad);
    if (bad.length) return;
    setSaving(true);
    try {
      const next = await api.investment.update(property.id, inputs);
      setData(next);
      setEdited(null);
      toast.success(t("invest.panel.saved", locale));
    } catch (cause) {
      toast.error(cause instanceof BioSphereError ? cause.localised(locale) : t("invest.panel.failed", locale));
    } finally {
      setSaving(false);
    }
  };

  const numberField = (key: NumberKey, label: StringKey, hint?: string) => (
    <Input
      key={key}
      label={t(label, locale)}
      inputMode={NUMBERS[key] === "integer" ? "numeric" : "decimal"}
      value={draft.numbers[key]}
      hint={hint}
      error={invalid.includes(key) ? (locale === "tr" ? "Geçerli bir sayı girin." : "Enter a valid number.") : undefined}
      onChange={(event) => setNumber(key, event.target.value)}
    />
  );
  const textField = (key: TextKey, label: StringKey, maxLength: number) => (
    <Input
      key={key}
      label={t(label, locale)}
      value={draft.texts[key]}
      maxLength={maxLength}
      onChange={(event) => setText(key, event.target.value)}
    />
  );

  const visible = state?.visible_to_visitors ?? false;
  const usage = state?.usage;
  const budget = state?.budget;

  return (
    <section aria-label={t("invest.panel.title", locale)} className="glass mt-6 rounded-lg p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-ink">
            <TrendingUp className="size-4 text-bio" aria-hidden />
            {t("invest.panel.title", locale)}
          </h2>
          <p className="mt-1 max-w-2xl text-[12.5px] leading-relaxed text-ink-muted">{t("invest.panel.body", locale)}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Badge tone={visible ? "bio" : "neutral"} dot>
            {t(visible ? "invest.panel.visible" : "invest.panel.hidden", locale)}
          </Badge>
          {state?.status.available ? (
            <span className="text-[11px] text-ink-faint">{state.status.model}</span>
          ) : null}
        </div>
      </div>

      {state && state.blockers.length > 0 && state.inputs.enabled ? (
        <div className="mb-4 rounded-md border border-amber/40 bg-amber/5 p-3">
          <ul className="space-y-1 text-[12.5px] leading-relaxed text-ink-muted">
            {state.blockers.map((code) => (
              <li key={code} className="flex gap-1.5">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-amber" aria-hidden />
                {t(`invest.blocker.${code}` as StringKey, locale)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 space-y-5">
          <Switch
            checked={draft.enabled}
            onChange={(enabled) => change({ enabled })}
            label={t("invest.panel.enable", locale)}
            description={t("invest.panel.enable.desc", locale)}
          />

          <fieldset className="space-y-3">
            <legend className="mb-2 text-[12px] font-semibold tracking-tight text-ink">
              {t("invest.panel.section.money", locale)}
            </legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {numberField("asking_price_tl", "invest.field.price")}
              {numberField("monthly_rent_tl", "invest.field.rent")}
              <Select
                label={t("invest.field.rentStatus", locale)}
                value={draft.rent_status}
                onChange={(event) => change({ rent_status: event.target.value })}
                options={[
                  { value: "", label: t("invest.field.unknown", locale) },
                  { value: "expected", label: t("invest.field.rentStatus.expected", locale) },
                  { value: "actual", label: t("invest.field.rentStatus.actual", locale) },
                ]}
              />
              {numberField("aidat_monthly_tl", "invest.field.aidat")}
            </div>
          </fieldset>

          <fieldset className="space-y-3">
            <legend className="mb-2 text-[12px] font-semibold tracking-tight text-ink">
              {t("invest.panel.section.place", locale)}
            </legend>
            <div className="grid gap-3 sm:grid-cols-3">
              {textField("il", "invest.field.il", 40)}
              {textField("ilce", "invest.field.ilce", 60)}
              {textField("mahalle", "invest.field.mahalle", 80)}
            </div>
            <div className="grid gap-3 sm:grid-cols-4">
              {numberField("gross_m2", "invest.field.gross")}
              {numberField("net_m2", "invest.field.net")}
              {textField("room_layout", "invest.field.rooms", 12)}
              {numberField("building_age_years", "invest.field.age")}
              {numberField("floor", "invest.field.floor")}
              {numberField("total_floors", "invest.field.totalFloors")}
              {textField("heating", "invest.field.heating", 40)}
              <Select
                label={t("invest.field.energy", locale)}
                value={draft.energy_class}
                onChange={(event) => change({ energy_class: event.target.value })}
                options={[
                  { value: "", label: t("invest.field.unknown", locale) },
                  ...["A", "B", "C", "D", "E", "F", "G"].map((value) => ({ value, label: value })),
                ]}
              />
              <Select
                label={t("invest.field.saleKind", locale)}
                value={draft.sale_kind}
                onChange={(event) => change({ sale_kind: event.target.value })}
                options={[
                  { value: "", label: t("invest.field.unknown", locale) },
                  { value: "second_hand", label: t("invest.field.saleKind.second_hand", locale) },
                  { value: "new_build", label: t("invest.field.saleKind.new_build", locale) },
                ]}
              />
              <Select
                label={t("invest.field.site", locale)}
                value={draft.in_site}
                onChange={(event) => change({ in_site: event.target.value })}
                options={[
                  { value: "", label: t("invest.field.unknown", locale) },
                  { value: "yes", label: t("invest.field.yes", locale) },
                  { value: "no", label: t("invest.field.no", locale) },
                ]}
              />
            </div>
          </fieldset>

          <fieldset className="space-y-3">
            <legend className="mb-2 text-[12px] font-semibold tracking-tight text-ink">
              {t("invest.panel.section.costs", locale)}
            </legend>
            <div className="grid gap-3 sm:grid-cols-2">
              {numberField("emlak_vergisi_yearly_tl", "invest.field.emlak", t("invest.field.emlak.hint", locale))}
              {numberField("dask_yearly_tl", "invest.field.dask")}
              {numberField("commission_pct", "invest.field.commission", t("invest.field.commission.hint", locale))}
              {numberField(
                "daily_budget_usd",
                "invest.field.budget",
                budget
                  ? fill(t("invest.field.budget.hint", locale), {
                      d: formatNumber(budget.default_usd, locale, 0),
                      max: formatNumber(budget.operator_max_usd, locale, 0),
                    })
                  : undefined,
              )}
            </div>
            <Textarea
              label={t("invest.field.notes", locale)}
              rows={3}
              maxLength={600}
              placeholder={t("invest.field.notes.placeholder", locale)}
              hint={t("invest.field.notes.hint", locale)}
              value={draft.texts.notes}
              onChange={(event) => setText("notes", event.target.value)}
            />
          </fieldset>

          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" loading={saving} disabled={!edited} onClick={() => void save()}>
              <Check className="size-4" aria-hidden />
              {t("invest.panel.save", locale)}
            </Button>
            {property.published ? (
              <Link
                href={`/embed/${property.public_token}`}
                target="_blank"
                className="inline-flex items-center gap-1.5 text-[12.5px] text-ink-muted hover:text-ink"
              >
                <ExternalLink className="size-3.5" aria-hidden />
                {t("invest.panel.openEmbed", locale)}
              </Link>
            ) : null}
          </div>
        </div>

        <aside className="min-w-0 space-y-4">
          <div className="rounded-md border border-line bg-surface-2/40 p-3.5">
            <p className="mb-2.5 text-[11.5px] font-medium text-ink-muted">{t("invest.panel.preview", locale)}</p>
            {state?.card ? (
              <KpiCard card={state.card} locale={locale} />
            ) : (
              <p className="text-[12px] text-ink-faint">{t("invest.panel.previewEmpty", locale)}</p>
            )}
          </div>

          {usage && budget ? (
            <div className="rounded-md border border-line bg-surface-2/40 p-3.5">
              <p className="mb-2 text-[11.5px] font-medium text-ink-muted">{t("invest.usage.title", locale)}</p>
              <dl className="space-y-1 text-[12px]">
                {(
                  [
                    ["invest.usage.today", formatUsd(usage.today_usd, locale)],
                    ["invest.usage.week", formatUsd(usage.last_7_days_usd, locale)],
                    ["invest.usage.month", formatUsd(usage.last_30_days_usd, locale)],
                    ["invest.usage.questions", formatNumber(usage.questions_30d, locale)],
                    [
                      "invest.usage.perQuestion",
                      usage.avg_cost_per_question_usd !== null ? formatUsd(usage.avg_cost_per_question_usd, locale) : "—",
                    ],
                    ["invest.usage.searches", formatNumber(usage.web_searches_30d, locale)],
                  ] as [StringKey, string][]
                ).map(([label, value]) => (
                  <div key={label} className="flex justify-between gap-3">
                    <dt className="text-ink-faint">{t(label, locale)}</dt>
                    <dd className="tabular-nums text-ink">{value}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
                {fill(t("invest.usage.cap", locale), {
                  spent: formatUsd(budget.spent_today_usd, locale),
                  cap: formatUsd(budget.daily_budget_usd, locale),
                })}
                {usage.by_model.length
                  ? ` · ${usage.by_model.map((m) => `${m.model} ${formatUsd(m.cost_usd, locale)}`).join(", ")}`
                  : ""}
              </p>
              {usage.questions_30d === 0 ? (
                <p className="mt-1 text-[11px] text-ink-faint">{t("invest.usage.none", locale)}</p>
              ) : null}
            </div>
          ) : null}

          {state ? (
            <details className="rounded-md border border-line bg-surface-2/40 p-3.5 text-[12px]">
              <summary className="cursor-pointer text-[11.5px] font-medium text-ink-muted">
                {t("invest.rates.title", locale)}
              </summary>
              <p className="mt-1.5 text-[11px] text-ink-faint">
                {fill(t("invest.rates.asOf", locale), { date: state.rates_as_of })}
              </p>
              <dl className="mt-2 space-y-2">
                {state.rates.map((row) => (
                  <div key={row.key}>
                    <dt className="text-ink-faint">{row.label}</dt>
                    <dd className="text-ink">{row.value}</dd>
                    {row.source ? (
                      <dd className="text-[11px] text-ink-faint">
                        <a
                          href={row.source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline decoration-line-strong underline-offset-2 hover:text-ink-muted"
                        >
                          {row.source.title}
                        </a>{" "}
                        ({row.source.date})
                        {state.stale_rates.includes(row.key) ? (
                          <span className="ml-1 text-amber">· {t("invest.rates.stale", locale)}</span>
                        ) : null}
                      </dd>
                    ) : null}
                  </div>
                ))}
              </dl>
            </details>
          ) : null}
        </aside>
      </div>
    </section>
  );
}
