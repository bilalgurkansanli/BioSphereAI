/**
 * Numbers as a Turkish buyer reads them — and as an English one does.
 *
 * The engine sends the card's figures as raw numbers and the client formats them, so the same
 * card reads "7.850.000 TL · %4,89 · 20,4 yıl" in Turkish and "7,850,000 TL · 4.89% · 20.4 years"
 * in English. Turkish puts the percent sign FIRST, which `Intl.NumberFormat`'s percent style does
 * on its own for `tr-TR` — but it also multiplies by a hundred and rounds its own way, so the
 * sign is placed here and the digits come from a plain decimal formatter.
 */

import type { Locale } from "@/lib/types";

function numberFormat(locale: Locale, digits: number): Intl.NumberFormat {
  return new Intl.NumberFormat(locale === "tr" ? "tr-TR" : "en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** `7.850.000` / `7,850,000`. */
export function formatNumber(value: number, locale: Locale, digits = 0): string {
  return numberFormat(locale, digits).format(value);
}

/** Drops trailing zeros: `2` not `2,00`, `2,85` stays. */
export function formatCompact(value: number, locale: Locale, maxDigits = 2): string {
  return new Intl.NumberFormat(locale === "tr" ? "tr-TR" : "en-GB", { maximumFractionDigits: maxDigits }).format(value);
}

export function formatTl(value: number, locale: Locale): string {
  return `${formatNumber(Math.round(value), locale, 0)} TL`;
}

/** `%4,89` in Turkish, `4.89%` in English. */
export function formatPct(value: number, locale: Locale, digits = 2): string {
  const digitsText = formatNumber(Math.abs(value), locale, digits);
  const sign = value < 0 ? "-" : "";
  return locale === "tr" ? `${sign}%${digitsText}` : `${sign}${digitsText}%`;
}

/** Rates as quoted: `%2,85`, `%50`. */
export function formatRate(value: number, locale: Locale): string {
  const digitsText = formatCompact(Math.abs(value), locale, 2);
  const sign = value < 0 ? "-" : "";
  return locale === "tr" ? `${sign}%${digitsText}` : `${sign}${digitsText}%`;
}

export function formatUsd(value: number, locale: Locale): string {
  const digits = value > 0 && value < 1 ? 3 : 2;
  return `$${formatNumber(value, locale, digits)}`;
}

/** Fills `{name}` placeholders in a dictionary string. Unknown names are left as written. */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    name in values ? String(values[name]) : match,
  );
}

/**
 * A number typed by an agent, read generously: `14.750.000`, `14750000`, `14 750 000` and
 * `14,75` are all what they look like. `kind: "money"` reads dots and spaces as thousands
 * separators (a Turkish price never has decimals worth keeping); `"decimal"` reads a comma as
 * the decimal point. Empty input is `null` — "not entered" — never zero.
 */
export function parseTypedNumber(text: string, kind: "money" | "decimal" | "integer"): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  let normalised: string;
  if (kind === "money") {
    normalised = trimmed.replace(/[\s.]/g, "").replace(",", ".");
  } else if (kind === "integer") {
    normalised = trimmed.replace(/[\s.,]/g, "");
  } else {
    normalised = trimmed.replace(/\s/g, "").replace(",", ".");
  }
  if (!/^-?\d+(\.\d+)?$/.test(normalised)) return Number.NaN;
  const value = Number(normalised);
  return Number.isFinite(value) ? value : Number.NaN;
}
