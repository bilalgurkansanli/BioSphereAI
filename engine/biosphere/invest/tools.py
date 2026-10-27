"""The calculators as client tools the model calls — strict schemas in, engine numbers out.

Three rules shape these definitions, and each was a choice between two reasonable designs:

**The agent's figures are filled in here, not passed by the model.** Every tool reads the
asking price, the rent, the aidat and the areas from the conversation's snapshot of the
property, and the model passes a ``scenario_*`` value only when it means a different figure.
The alternative — the model copying 7.850.000 into every call — is one transposed digit away
from a yield that is wrong in a way no reader would notice.

**Strict schemas, and therefore lean ones.** ``strict: true`` makes the API guarantee the
arguments match the schema, which is what lets a tool trust a number is a number. The price is
a budget: at most 24 optional parameters and 16 union-typed ones across ALL strict tools in a
request, and no ``minimum``/``maximum`` at all. So optional parameters are spent only where
"absent" has a meaning (use the agent's figure), there are no nullable unions, and the bounds
live in the pydantic models below, which validate every call before it runs. A call that fails
validation is answered with an ``is_error`` result in plain words, so the model can correct it.

**Sources travel with the numbers they justify.** A benchmark price per m², a market rent or a
mortgage rate the model found on the web goes into the tool WITH its URL and date, and the
result echoes them — so the answer, the panel's source chips and the bench's checker all see
the same provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from biosphere.invest import calc
from biosphere.invest.inputs import (
    InvestmentInputs,
    commission_pct,
    dask_yearly,
    owner_costs,
    resolve_rent,
)
from biosphere.invest.rates import CostRates

__all__ = ["ToolContext", "ToolOutcome", "definitions", "run", "TOOL_NAMES", "summary_rows"]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool may read: the property as the conversation knows it, and the rates."""

    inputs: InvestmentInputs
    rates: CostRates


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """A tool's answer: the JSON the model reads, and whether it is an error."""

    payload: dict[str, Any]
    is_error: bool = False

    def content(self) -> str:
        # Compact and key-sorted: tool results become part of the cached conversation prefix,
        # so the same result must serialise to the same bytes every time.
        return json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# ======================================================================================
# Validated inputs (the bounds strict mode cannot express)
# ======================================================================================


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Scenario(_In):
    scenario_price_tl: float | None = Field(default=None, gt=10_000, lt=10_000_000_000)


class _RentScenario(_Scenario):
    scenario_monthly_rent_tl: float | None = Field(default=None, gt=100, lt=50_000_000)
    scenario_rent_source: str | None = Field(default=None, max_length=400)


class RentalYieldIn(_RentScenario):
    vacancy_months_per_year: float | None = Field(default=None, ge=0, le=12)


class PaybackIn(_RentScenario):
    annual_rent_growth_pct: float | None = Field(default=None, gt=-100, le=500)


class PurchaseCostsIn(_Scenario):
    with_mortgage: bool
    buyer_pays_seller_tapu_share: bool
    loan_amount_tl: float | None = Field(default=None, gt=0, lt=10_000_000_000)


class MortgageIn(_Scenario):
    down_payment_pct: float = Field(ge=0, le=100)
    #: No upper bound HERE: :func:`calc.mortgage` refuses a rate of 20 or more with a message
    #: that names the likely mistake (an annual rate passed as monthly), which a model can act
    #: on; pydantic's "should be less than 20" it cannot.
    monthly_rate_pct: float = Field(ge=0)
    term_months: int = Field(ge=1, le=480)
    rate_source: str = Field(min_length=1, max_length=400)


class _SourceRef(_In):
    label: str = Field(max_length=200)
    url: str = Field(max_length=600)
    date: str = Field(max_length=40)


class ProjectionIn(_RentScenario):
    years: int = Field(ge=1, le=30)
    annual_price_growth_pct: float = Field(gt=-100, le=500)
    annual_rent_growth_pct: float = Field(gt=-100, le=500)
    annual_inflation_pct: float = Field(gt=-100, le=500)
    assumption_sources: list[_SourceRef] = Field(max_length=8)


class _BenchmarkIn(_In):
    label: str = Field(max_length=200)
    tl_per_m2: float = Field(gt=100, lt=100_000_000)
    area_basis: Literal["net", "brut", "bilinmiyor"]
    source_url: str = Field(max_length=600)
    source_title: str = Field(max_length=300)
    source_date: str = Field(max_length=40)


class PricePerM2In(_Scenario):
    benchmarks: list[_BenchmarkIn] = Field(max_length=8)


# ======================================================================================
# Definitions sent to the API
# ======================================================================================

_SCENARIO_PRICE = {
    "type": "number",
    "description": (
        "A price in TL other than the asking price — e.g. a negotiated price the visitor asks "
        "about. Omit it to use the agent's asking price."
    ),
}
_SCENARIO_RENT = {
    "type": "number",
    "description": (
        "A monthly rent in TL other than the agent's: a market rent you found on the web (then "
        "also give scenario_rent_source) or the visitor's own assumption. Omit it to use the "
        "agent's rent."
    ),
}
_SCENARIO_RENT_SOURCE = {
    "type": "string",
    "description": "Where scenario_monthly_rent_tl comes from: 'page title — URL (date)' of the source, or 'ziyaretçi varsayımı'.",
}


def _strict(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_DEFINITIONS: list[dict[str, Any]] = [
    _strict(
        "rental_yield",
        "Gross and net rental yield of THIS property, with the owner's yearly costs and net rental "
        "income. Call it whenever you state a yield or a net income that PRECOMPUTED does not "
        "already give — a different rent, a different price, a different vacancy. The agent's "
        "price, rent, aidat, emlak vergisi and DASK are filled in automatically.",
        {
            "scenario_price_tl": _SCENARIO_PRICE,
            "scenario_monthly_rent_tl": _SCENARIO_RENT,
            "scenario_rent_source": _SCENARIO_RENT_SOURCE,
            "vacancy_months_per_year": {
                "type": "number",
                "description": "Months a year the flat stands empty (0-12). Omit it to use the default in COST PARAMETERS.",
            },
        },
        [],
    ),
    _strict(
        "payback_period",
        "Years the rent takes to pay back the purchase (amortisman): gross (price / yearly rent — "
        "the figure listings quote), net ((price + purchase costs) / yearly net income) and, if "
        "you pass annual_rent_growth_pct, net with the rent rising every year (nominal TL). Call "
        "it for any payback figure PRECOMPUTED does not already give.",
        {
            "scenario_price_tl": _SCENARIO_PRICE,
            "scenario_monthly_rent_tl": _SCENARIO_RENT,
            "scenario_rent_source": _SCENARIO_RENT_SOURCE,
            "annual_rent_growth_pct": {
                "type": "number",
                "description": "Yearly rent growth in percent (e.g. 30 for 30 %). Omit it for the no-growth payback only.",
            },
        },
        [],
    ),
    _strict(
        "purchase_costs",
        "Everything the buyer pays on top of the price in Turkey, item by item: tapu harcı, "
        "döner sermaye, the agent's commission with KDV, DASK and — with a mortgage — appraisal, "
        "mortgage-deed fee and the bank's allocation fee. Call it for any purchase-cost figure.",
        {
            "with_mortgage": {"type": "boolean", "description": "True if the visitor buys with a housing loan."},
            "buyer_pays_seller_tapu_share": {
                "type": "boolean",
                "description": (
                    "Normally false: tapu harcı is 2 % + 2 % and each side pays its own. True only "
                    "if the visitor says the buyer pays both shares."
                ),
            },
            "scenario_price_tl": _SCENARIO_PRICE,
            "loan_amount_tl": {
                "type": "number",
                "description": "The loan amount in TL, if known from mortgage_payment, to include the allocation fee.",
            },
        },
        ["with_mortgage", "buyer_pays_seller_tapu_share"],
    ),
    _strict(
        "mortgage_payment",
        "Monthly payment, total repayment and total interest of a Turkish housing loan (annuity) "
        "for this property, capped at BDDK's loan-to-value limit; also how much of the payment "
        "the rent covers. Call it for any instalment figure. The rate must come from a source "
        "you cite (a bank page you found, dated) or the visitor, or be the example rate in "
        "COST PARAMETERS.",
        {
            "down_payment_pct": {"type": "number", "description": "Down payment as a percentage of the price (e.g. 30)."},
            "monthly_rate_pct": {
                "type": "number",
                "description": "MONTHLY interest rate in percent, as Turkish banks quote it (e.g. 2.85 — not the annual rate).",
            },
            "term_months": {"type": "integer", "description": "Loan term in months (e.g. 120)."},
            "rate_source": {
                "type": "string",
                "description": "Where the rate comes from: 'page title — URL (date)', 'ziyaretçi varsayımı', or 'örnek oran (COST PARAMETERS)'.",
            },
            "scenario_price_tl": _SCENARIO_PRICE,
        },
        ["down_payment_pct", "monthly_rate_pct", "term_months", "rate_source"],
    ),
    _strict(
        "investment_projection",
        "Buy now, let it, sell after N years: property value, cumulative net rent, total and "
        "yearly return in nominal TL and in real (inflation-adjusted) terms. Call it for any "
        "multi-year return figure. Inflation should come from a source you cite (e.g. TCMB "
        "inflation expectations) listed in assumption_sources, or from the visitor. Without a "
        "sourced forecast for price or rent growth, pass the inflation figure for both — the "
        "neutral case — and say so; never invent a growth rate.",
        {
            "years": {"type": "integer", "description": "Holding period in years (1-30)."},
            "annual_price_growth_pct": {"type": "number", "description": "Yearly property price growth in percent."},
            "annual_rent_growth_pct": {"type": "number", "description": "Yearly rent growth in percent."},
            "annual_inflation_pct": {"type": "number", "description": "Yearly consumer inflation in percent, for the real return."},
            "assumption_sources": {
                "type": "array",
                "description": "Sources of the growth and inflation figures; an empty list if they are the visitor's assumptions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "What the source supports, e.g. 'TÜFE beklentisi'."},
                        "url": {"type": "string"},
                        "date": {"type": "string", "description": "Publication or data date."},
                    },
                    "required": ["label", "url", "date"],
                    "additionalProperties": False,
                },
            },
            "scenario_price_tl": _SCENARIO_PRICE,
            "scenario_monthly_rent_tl": _SCENARIO_RENT,
            "scenario_rent_source": _SCENARIO_RENT_SOURCE,
        },
        ["years", "annual_price_growth_pct", "annual_rent_growth_pct", "annual_inflation_pct", "assumption_sources"],
    ),
    _strict(
        "price_per_m2_comparison",
        "This property's price per m² (net and gross, from the agent's areas) and its difference "
        "in percent from market benchmarks you found. Each benchmark goes in WITH its source URL "
        "and date and the area basis the source uses, exactly as published: if a source gives a "
        "range, pass its low and its high end as two benchmarks — never a midpoint you worked "
        "out. Call it for any price-per-m² comparison.",
        {
            "benchmarks": {
                "type": "array",
                "description": "Published average prices per m² for comparable flats; an empty list for the property's own figure only.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "What the figure is, e.g. 'Kadıköy ortalama satılık konut m² fiyatı'."},
                        "tl_per_m2": {"type": "number"},
                        "area_basis": {
                            "type": "string",
                            "enum": ["net", "brut", "bilinmiyor"],
                            "description": "Whether the source divides by net or gross m², or 'bilinmiyor'.",
                        },
                        "source_url": {"type": "string"},
                        "source_title": {"type": "string"},
                        "source_date": {"type": "string", "description": "Publication or data date of the figure."},
                    },
                    "required": ["label", "tl_per_m2", "area_basis", "source_url", "source_title", "source_date"],
                    "additionalProperties": False,
                },
            },
            "scenario_price_tl": _SCENARIO_PRICE,
        },
        ["benchmarks"],
    ),
]

TOOL_NAMES: frozenset[str] = frozenset(d["name"] for d in _DEFINITIONS)


def definitions(*, eager_input_streaming: bool) -> list[dict[str, Any]]:
    """The tool list, in a fixed order with fixed bytes.

    Fixed because tools render FIRST in the prompt: any change to this list between two
    requests of one conversation invalidates the whole prompt cache and, on the newest models,
    every thinking block after it. ``eager_input_streaming`` is per deployment, never per
    request, for the same reason.
    """
    out = []
    for definition in _DEFINITIONS:
        item = dict(definition)
        if eager_input_streaming:
            item["eager_input_streaming"] = True
        out.append(item)
    return out


# ======================================================================================
# Running a call
# ======================================================================================


def _error(message: str) -> ToolOutcome:
    return ToolOutcome({"error": message}, is_error=True)


def _used(value: float, basis: str, note: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {"value": value, "basis": basis}
    if note:
        entry["note"] = note
    return entry


def _price(ctx: ToolContext, scenario: float | None) -> tuple[float, dict[str, Any]] | None:
    if scenario is not None:
        return float(scenario), _used(calc.money(scenario), "senaryo")
    if ctx.inputs.asking_price_tl:
        return float(ctx.inputs.asking_price_tl), _used(calc.money(ctx.inputs.asking_price_tl), "ilan")
    return None


_NO_PRICE = "The agent entered no asking price; pass scenario_price_tl or say the price is unknown."
_NO_RENT = (
    "No rent figure: the agent entered none. Search the typical rent of comparable flats in the "
    "area and pass scenario_monthly_rent_tl with scenario_rent_source, or ask the visitor."
)


def _yield_parts(
    ctx: ToolContext, price: float, rent: float, vacancy: float | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The yield calculation every rent-based tool shares, plus what it assumed."""
    commission = commission_pct(ctx.inputs, ctx.rates)
    dask = dask_yearly(ctx.inputs, ctx.rates)
    costs = calc.purchase_costs(price, ctx.rates, commission_pct=commission.value, dask_yearly=dask.value)
    own = owner_costs(ctx.inputs, ctx.rates, price)
    vacancy_used = _used(vacancy, "senaryo") if vacancy is not None else _used(
        own["vacancy_months"].value, "varsayılan", own["vacancy_months"].note
    )
    yld = calc.rental_yield(
        price,
        rent,
        acquisition_costs=costs["total_costs_tl"],
        aidat_monthly=ctx.inputs.aidat_monthly_tl,
        vacancy_months=float(vacancy_used["value"]),
        maintenance_pct_of_rent=own["maintenance_pct"].value,
        emlak_vergisi_yearly=own["emlak_vergisi"].value,
        dask_yearly=own["dask"].value,
    )
    used = {
        "aidat_monthly": _used(calc.money(ctx.inputs.aidat_monthly_tl), "ilan")
        if ctx.inputs.aidat_monthly_tl is not None
        else _used(0, "girilmedi", "aidat girilmedi; boş ayların aidatı sıfır sayıldı"),
        "emlak_vergisi_yearly": _used(calc.money(own["emlak_vergisi"].value), own["emlak_vergisi"].basis, own["emlak_vergisi"].note),
        "dask_yearly": _used(calc.money(own["dask"].value), own["dask"].basis, own["dask"].note),
        "maintenance_pct_of_rent": _used(own["maintenance_pct"].value, "varsayılan", own["maintenance_pct"].note),
        "vacancy_months": vacancy_used,
        "commission_pct": _used(commission.value, commission.basis, commission.note),
    }
    return yld, costs, used


def _rent(ctx: ToolContext, args: _RentScenario) -> tuple[float, dict[str, Any]] | None:
    rent = resolve_rent(ctx.inputs, args.scenario_monthly_rent_tl, args.scenario_rent_source)
    if rent is None:
        return None
    return rent.value, _used(calc.money(rent.value), rent.basis, rent.note)


def _rental_yield(ctx: ToolContext, args: RentalYieldIn) -> ToolOutcome:
    price = _price(ctx, args.scenario_price_tl)
    if price is None:
        return _error(_NO_PRICE)
    rent = _rent(ctx, args)
    if rent is None:
        return _error(_NO_RENT)
    yld, _, used = _yield_parts(ctx, price[0], rent[0], args.vacancy_months_per_year)
    return ToolOutcome({"inputs_used": {"price": price[1], "monthly_rent": rent[1], **used}, "result": yld})


def _payback(ctx: ToolContext, args: PaybackIn) -> ToolOutcome:
    price = _price(ctx, args.scenario_price_tl)
    if price is None:
        return _error(_NO_PRICE)
    rent = _rent(ctx, args)
    if rent is None:
        return _error(_NO_RENT)
    yld, _, used = _yield_parts(ctx, price[0], rent[0], None)
    back = calc.payback(
        price[0],
        rent[0],
        net_annual_income=yld["net_annual_income_tl"],
        total_investment=yld["total_investment_tl"],
        annual_rent_growth_pct=args.annual_rent_growth_pct,
    )
    back["net_annual_income_tl"] = yld["net_annual_income_tl"]
    back["total_investment_tl"] = yld["total_investment_tl"]
    return ToolOutcome({"inputs_used": {"price": price[1], "monthly_rent": rent[1], **used}, "result": back})


def _purchase_costs(ctx: ToolContext, args: PurchaseCostsIn) -> ToolOutcome:
    price = _price(ctx, args.scenario_price_tl)
    if price is None:
        return _error(_NO_PRICE)
    commission = commission_pct(ctx.inputs, ctx.rates)
    dask = dask_yearly(ctx.inputs, ctx.rates)
    costs = calc.purchase_costs(
        price[0],
        ctx.rates,
        commission_pct=commission.value,
        dask_yearly=dask.value,
        buyer_pays_seller_tapu=args.buyer_pays_seller_tapu_share,
        with_mortgage=args.with_mortgage,
        loan_amount=args.loan_amount_tl,
    )
    return ToolOutcome(
        {
            "inputs_used": {
                "price": price[1],
                "commission_pct": _used(commission.value, commission.basis, commission.note),
                "dask_yearly": _used(calc.money(dask.value), dask.basis, dask.note),
            },
            "result": costs,
            "rates_used": _rate_refs(ctx.rates, "tapu_harci", "doner_sermaye", "commission", "ekspertiz"),
        }
    )


def _mortgage(ctx: ToolContext, args: MortgageIn) -> ToolOutcome:
    price = _price(ctx, args.scenario_price_tl)
    if price is None:
        return _error(_NO_PRICE)
    rent = resolve_rent(ctx.inputs, None, None)
    loan = calc.mortgage(
        price[0],
        ctx.rates,
        down_payment_pct=args.down_payment_pct,
        monthly_rate_pct=args.monthly_rate_pct,
        term_months=args.term_months,
        monthly_rent=rent.value if rent else None,
    )
    used: dict[str, Any] = {"price": price[1], "rate_source": args.rate_source.strip()}
    if rent:
        used["monthly_rent"] = _used(calc.money(rent.value), rent.basis, rent.note)
    return ToolOutcome({"inputs_used": used, "result": loan, "rates_used": _rate_refs(ctx.rates, "ltv")})


def _projection(ctx: ToolContext, args: ProjectionIn) -> ToolOutcome:
    price = _price(ctx, args.scenario_price_tl)
    if price is None:
        return _error(_NO_PRICE)
    rent = _rent(ctx, args)
    if rent is None:
        return _error(_NO_RENT)
    yld, costs, used = _yield_parts(ctx, price[0], rent[0], None)
    commission = commission_pct(ctx.inputs, ctx.rates).value
    exit_pct = ctx.rates.tapu_harci_seller_pct + commission * (1.0 + ctx.rates.kdv_pct / 100.0)
    proj = calc.projection(
        price[0],
        years=args.years,
        acquisition_costs=costs["total_costs_tl"],
        net_income_year1=yld["net_annual_income_tl"],
        price_growth_pct=args.annual_price_growth_pct,
        rent_growth_pct=args.annual_rent_growth_pct,
        inflation_pct=args.annual_inflation_pct,
        exit_costs_pct=exit_pct,
    )
    return ToolOutcome(
        {
            "inputs_used": {
                "price": price[1],
                "monthly_rent": rent[1],
                "net_income_year1_tl": yld["net_annual_income_tl"],
                **used,
            },
            "assumption_sources": [s.model_dump() for s in args.assumption_sources],
            "result": proj,
        }
    )


def _price_per_m2(ctx: ToolContext, args: PricePerM2In) -> ToolOutcome:
    price = _price(ctx, args.scenario_price_tl)
    if price is None:
        return _error(_NO_PRICE)
    if not ctx.inputs.net_m2 and not ctx.inputs.gross_m2:
        return _error(
            "The agent entered neither net nor gross m², so there is no price per m² for this "
            "property. Do not use the tour's measured room areas as the flat's size."
        )
    benches = [
        calc.Benchmark(b.label, b.tl_per_m2, b.area_basis, b.source_url, b.source_title, b.source_date)
        for b in args.benchmarks
    ]
    per = calc.price_per_m2(price[0], net_m2=ctx.inputs.net_m2, gross_m2=ctx.inputs.gross_m2, benchmarks=benches)
    return ToolOutcome({"inputs_used": {"price": price[1]}, "result": per})


def _rate_refs(rates: CostRates, *keys: str) -> dict[str, Any]:
    return {k: {"title": s.title, "url": s.url, "date": s.date} for k in keys if (s := rates.sources.get(k))}


_HANDLERS: dict[str, tuple[type[_In], Any]] = {
    "rental_yield": (RentalYieldIn, _rental_yield),
    "payback_period": (PaybackIn, _payback),
    "purchase_costs": (PurchaseCostsIn, _purchase_costs),
    "mortgage_payment": (MortgageIn, _mortgage),
    "investment_projection": (ProjectionIn, _projection),
    "price_per_m2_comparison": (PricePerM2In, _price_per_m2),
}


def run(name: str, arguments: object, ctx: ToolContext) -> ToolOutcome:
    """Validate and run one call. Never raises: every failure is an ``is_error`` result.

    The arguments are validated here even though the tools are strict, for two reasons: the
    schemas cannot carry bounds (a rate of 285 % is schema-valid), and with eager input
    streaming the API no longer validates a tool's input at all — a truncated or malformed
    one arrives as a partial object, and this is the check that keeps it from running.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return _error(f"Unknown tool {name!r}. Available: {', '.join(sorted(_HANDLERS))}.")
    model, fn = handler
    if not isinstance(arguments, dict):
        return _error("The tool input must be a JSON object.")
    try:
        args = model.model_validate(arguments)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ())) or 'input'}: {e.get('msg', 'invalid')}"
            for e in exc.errors()[:6]
        )
        return _error(f"Invalid input — {problems}. Correct the values and call again.")
    try:
        return fn(ctx, args)
    except ValueError as exc:
        return _error(str(exc))


# ======================================================================================
# What the panel shows of a call
# ======================================================================================

_TITLES = {
    "rental_yield": "Kira getirisi",
    "payback_period": "Amortisman",
    "purchase_costs": "Alım masrafları",
    "mortgage_payment": "Kredi taksidi",
    "investment_projection": "Getiri projeksiyonu",
    "price_per_m2_comparison": "m² fiyatı karşılaştırması",
}


def summary_rows(name: str, outcome: ToolOutcome) -> dict[str, Any]:
    """A short labelled list of the headline figures of one call, for the panel's "Hesaplama"
    disclosure. The answer text cites the same figures; this is the receipt beside it."""
    title = _TITLES.get(name, name)
    if outcome.is_error:
        return {"tool": name, "title": title, "rows": [], "error": True}
    result = outcome.payload.get("result", {})
    display = result.get("display", {}) if isinstance(result, dict) else {}
    rows: list[list[str]] = []
    if name == "rental_yield":
        rows = [
            ["Brüt getiri", display.get("gross_yield", "")],
            ["Net getiri", display.get("net_yield", "")],
            ["Yıllık net kira geliri", display.get("net_annual_income", "")],
        ]
    elif name == "payback_period":
        rows = [["Brüt amortisman", display.get("gross_payback", "")], ["Net amortisman", display.get("net_payback", "")]]
        if "net_payback_with_growth" in display:
            rows.append(["Kira artışıyla (nominal)", display["net_payback_with_growth"]])
    elif name == "purchase_costs":
        rows = [[i["label"], calc.fmt_tl(i["amount_tl"])] for i in result.get("items", [])]
        rows.append(["Toplam masraf", display.get("total_costs", "")])
    elif name == "mortgage_payment":
        rows = [
            ["Aylık taksit", display.get("monthly_payment", "")],
            ["Kredi tutarı", display.get("loan", "")],
            ["Toplam faiz", display.get("total_interest", "")],
        ]
        if "rent_covers_payment" in display:
            rows.append(["Kiranın taksidi karşılama oranı", display["rent_covers_payment"]])
    elif name == "investment_projection":
        rows = [
            ["Nominal yıllık getiri", display.get("nominal_annual_return", "")],
            ["Reel yıllık getiri", display.get("real_annual_return", "")],
            ["Dönem sonu değer", display.get("final_value", "")],
        ]
    elif name == "price_per_m2_comparison":
        if "per_net_m2" in display:
            rows.append(["Bu evin m² fiyatı (net)", display["per_net_m2"]])
        if "per_gross_m2" in display:
            rows.append(["Bu evin m² fiyatı (brüt)", display["per_gross_m2"]])
        for comp in result.get("comparisons", []):
            rows.append([comp["label"], f"{comp['display']['benchmark']} → fark {comp['display']['difference']}"])
    return {"tool": name, "title": title, "rows": [r for r in rows if r[1]]}
