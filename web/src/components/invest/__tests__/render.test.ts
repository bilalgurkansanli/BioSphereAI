import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { AdvisorCard } from "@/lib/types";
import { KpiCard } from "../KpiCard";
import { MarkdownLite } from "../MarkdownLite";

/** The card the engine returned for the bench listing (Caferağa 3+1, 14.750.000 TL). */
const CARD: AdvisorCard = {
  price_tl: 14_750_000,
  location: "Caferağa, Kadıköy, İstanbul",
  missing: [],
  purchase: { total_costs_tl: 658_181, total_outlay_tl: 15_408_181, costs_pct_of_price: 4.46, commission_pct: 2, commission_basis: "varsayılan" },
  rent: { monthly_tl: 60_000, status: "expected" },
  income: {
    gross_pct: 4.88,
    net_pct: 3.82,
    net_annual_income_tl: 588_500,
    gross_payback_years: 20.5,
    net_payback_years: 26.2,
    assumptions: {
      vacancy_months: 1,
      maintenance_pct_of_rent: 5,
      emlak_vergisi_tl: 29_500,
      emlak_vergisi_basis: "tahmin",
      dask_tl: 2_500,
      dask_basis: "varsayılan",
      aidat_monthly_tl: 3_500,
    },
  },
  per_m2: { tl: 125_000, basis: "net", area_m2: 118 },
  mortgage: {
    monthly_payment_tl: 217_656,
    loan_tl: 7_375_000,
    down_payment_pct: 50,
    down_payment_tl: 7_375_000,
    term_months: 120,
    monthly_rate_pct: 2.85,
    ltv_cap_pct: 50,
    rate_source: { title: "Konut kredisi faizleri — Aydınlık", url: "https://www.aydinlik.com.tr/x", date: "2026-09-17" },
    rent_covers_payment_pct: 27.6,
  },
};

describe("the figures card", () => {
  it("shows the four headline figures with what they rest on, in Turkish", () => {
    const html = renderToStaticMarkup(createElement(KpiCard, { card: CARD, locale: "tr" }));
    for (const text of ["%4,88", "20,5 yıl", "125.000 TL", "217.656 TL/ay", "%3,82", "658.181 TL", "net 118 m²"]) {
      expect(html).toContain(text);
    }
    expect(html).toContain("beklenen kira 60.000 TL/ay ile");
    expect(html).toContain("%50 peşinat · 120 ay · aylık %2,85 faiz");
  });

  it("says what is missing instead of showing a zero", () => {
    const html = renderToStaticMarkup(
      createElement(KpiCard, { card: { ...CARD, rent: null, income: null, per_m2: null, missing: ["rent", "area"] }, locale: "tr" }),
    );
    expect(html).toContain("kira bilgisi girilmemiş");
    expect(html).toContain("alan girilmemiş");
    expect(html).not.toContain("%0");
  });

  it("reads in English with English number formats", () => {
    const html = renderToStaticMarkup(createElement(KpiCard, { card: CARD, locale: "en" }));
    expect(html).toContain("4.88%");
    expect(html).toContain("14,750,000 TL");
  });
});

describe("the answer renderer", () => {
  it("renders links that open elsewhere and never renders markup from the text", () => {
    const html = renderToStaticMarkup(
      createElement(MarkdownLite, {
        text: "**Piyasa verileri**\n- [Emlakjet](https://www.emlakjet.com/a): 114.792 TL/m²\n- <script>alert(1)</script>",
      }),
    );
    expect(html).toContain('href="https://www.emlakjet.com/a"');
    expect(html).toContain('rel="noopener noreferrer nofollow"');
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>");
  });
});
