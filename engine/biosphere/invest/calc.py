"""Every number the investment assistant states that is not quoted from somewhere.

Pure functions over plain numbers: no property, no database, no model. They are the reason the
assistant can be trusted with a yield at all — a language model asked for 12 × 32.000 / 7.850.000
will usually say 4,89 % and occasionally say 4,98 %, in the same confident sentence — so the
model is given these as tools and told it may not do arithmetic of its own, not even a sum.

Conventions, each chosen so that a figure here can be checked with a pocket calculator:

* **Money is in Turkish lira, rounded to the whole lira** at the boundary (half up), never in
  between: intermediate values keep full precision, so the rounding happens once.
* **Percentages are percentages** (4.89, not 0.0489), rounded to two decimals, and growth rates
  are passed the same way (a 30 % rent rise is ``30.0``).
* **Gross yield is the Turkish convention**: twelve months of rent over the price. **Amortisman**
  ("brüt") is its reciprocal in years — the figure listings quote. The net figures subtract
  what the owner actually pays and divide by what the buyer actually spends, including the
  purchase costs; both are returned, labelled, because the market quotes the first and an
  investor needs the second.
* **Every assumption the function made is in its result**, by name. A result that silently used
  a default vacancy is a result nobody can argue with, which is the wrong property for a number
  somebody may buy a flat on.

The ``display`` strings are Turkish-formatted ("7.850.000 TL", "%4,89", "20,4 yıl") because the
model must quote figures exactly as computed and the cheapest way to stop it reformatting a
number — and rounding it differently on the way — is to hand it the string.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from biosphere.invest.rates import CostRates

__all__ = [
    "Benchmark",
    "money",
    "round_to",
    "fmt_number",
    "fmt_tl",
    "fmt_pct",
    "fmt_years",
    "fmt_compact",
    "fmt_pct_compact",
    "purchase_costs",
    "rental_yield",
    "payback",
    "mortgage",
    "projection",
    "price_per_m2",
]


# ======================================================================================
# Rounding and formatting
# ======================================================================================


def round_to(value: float, digits: int) -> float:
    """Half-up rounding to ``digits`` decimals.

    Not :func:`round`, which rounds half to EVEN: ``round(2.675, 2)`` is 2.67 in binary
    floating point and ``round(0.125, 2)`` is 0.12 by design. A buyer checking a yield by hand
    rounds half up, and so does every bank statement, so this does too — through ``repr`` so
    that the decimal the float prints as is the one that gets rounded.
    """
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def money(value: float) -> int:
    """Whole lira, half up."""
    return int(Decimal(repr(float(value))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def fmt_number(value: float, digits: int = 0) -> str:
    """Turkish digits: ``.`` groups thousands, ``,`` separates decimals."""
    text = f"{round_to(value, digits):,.{digits}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_tl(value: float) -> str:
    return f"{fmt_number(money(value), 0)} TL"


def fmt_pct(value: float, digits: int = 2) -> str:
    """``%4,89``. The sign goes before the percent sign, as Turkish writes it: ``-%1,25``."""
    if value < 0:
        return f"-%{fmt_number(abs(value), digits)}"
    return f"%{fmt_number(value, digits)}"


def fmt_years(value: float) -> str:
    return f"{fmt_number(value, 1)} yıl"


def fmt_compact(value: float, digits: int = 2) -> str:
    """A number without trailing decimal zeros: ``2`` rather than ``2,00``, ``0,5`` not ``0,50``."""
    text = fmt_number(value, digits)
    if "," in text:
        text = text.rstrip("0").rstrip(",")
    return text


def fmt_pct_compact(value: float, digits: int = 2) -> str:
    """``%2`` and ``%0,5`` — for rates, which are quoted without trailing zeros."""
    if value < 0:
        return f"-%{fmt_compact(abs(value), digits)}"
    return f"%{fmt_compact(value, digits)}"


def _positive(name: str, value: float) -> float:
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def _non_negative(name: str, value: float) -> float:
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be zero or more, got {value!r}")
    return float(value)


# ======================================================================================
# Buying
# ======================================================================================


def purchase_costs(
    price: float,
    rates: CostRates,
    *,
    commission_pct: float,
    dask_yearly: float,
    buyer_pays_seller_tapu: bool = False,
    with_mortgage: bool = False,
    loan_amount: float | None = None,
) -> dict[str, Any]:
    """What the buyer pays on top of the price, item by item.

    Tapu harcı is levied on the declared price, which the law requires to be the real one;
    the buyer's share is ``tapu_harci_buyer_pct`` and the seller's share is the seller's
    unless the parties agree otherwise (``buyer_pays_seller_tapu``). Commission is the
    agent's buyer-side fee plus KDV. DASK is the first year's premium, due before the deed
    changes hands. With a mortgage the bank adds an appraisal, a döner sermaye for the
    mortgage deed (konut-kredisi ipotekleri harçtan muaftır, the fee is not) and an
    allocation fee of at most binde 5 of the loan — the last only when the loan is known.

    Not included, and said so in ``excluded``: KDV on a new build bought from a developer
    (usually inside the price), moving and furnishing, and the notary for a power of attorney.
    """
    price = _positive("price", price)
    commission_pct = _non_negative("commission_pct", commission_pct)
    dask_yearly = _non_negative("dask_yearly", dask_yearly)

    tapu_pct = rates.tapu_harci_buyer_pct + (rates.tapu_harci_seller_pct if buyer_pays_seller_tapu else 0.0)
    items: list[dict[str, Any]] = [
        {
            "key": "tapu_harci",
            "label": "Tapu harcı" + (" (alıcı + satıcı payı)" if buyer_pays_seller_tapu else " (alıcı payı)"),
            "amount_tl": money(price * tapu_pct / 100.0),
            "basis": f"satış bedelinin {fmt_pct_compact(tapu_pct)} oranında",
        },
        {
            "key": "doner_sermaye",
            "label": "Tapu döner sermaye ücreti",
            "amount_tl": money(rates.doner_sermaye_tl),
            "basis": "2026 göstergesi × ilçe katsayısı; ilçeye göre değişir",
        },
    ]
    commission = price * commission_pct / 100.0
    if commission > 0:
        items.append(
            {
                "key": "komisyon",
                "label": "Emlakçı komisyonu",
                "amount_tl": money(commission),
                "basis": f"satış bedelinin {fmt_pct_compact(commission_pct)} oranında",
            }
        )
        items.append(
            {
                "key": "komisyon_kdv",
                "label": "Komisyon KDV'si",
                "amount_tl": money(commission * rates.kdv_pct / 100.0),
                "basis": f"komisyonun {fmt_pct_compact(rates.kdv_pct)} oranında",
            }
        )
    items.append(
        {
            "key": "dask",
            "label": "DASK (ilk yıl)",
            "amount_tl": money(dask_yearly),
            "basis": "zorunlu deprem sigortası",
        }
    )
    if with_mortgage:
        items.append(
            {
                "key": "ekspertiz",
                "label": "Ekspertiz (değerleme) ücreti",
                "amount_tl": money(rates.ekspertiz_tl),
                "basis": "2026 ortalaması; bankaya göre değişir",
            }
        )
        items.append(
            {
                "key": "ipotek_doner_sermaye",
                "label": "İpotek tesisi döner sermaye ücreti",
                "amount_tl": money(rates.doner_sermaye_tl),
                "basis": "konut kredisi ipoteği harçtan muaf; döner sermaye alınır",
            }
        )
        if loan_amount is not None and loan_amount > 0:
            items.append(
                {
                    "key": "kredi_tahsis",
                    "label": "Kredi tahsis ücreti (azami)",
                    "amount_tl": money(loan_amount * rates.kredi_tahsis_pct / 100.0),
                    "basis": f"en fazla kredi tutarının {fmt_pct_compact(rates.kredi_tahsis_pct)} oranında",
                }
            )

    # Summed from the ROUNDED items, so the total is exactly the sum a reader adds up from the
    # list — never a lira off because the unrounded parts carried a fraction.
    total = sum(int(item["amount_tl"]) for item in items)
    return {
        "price_tl": money(price),
        "items": items,
        "total_costs_tl": total,
        "total_outlay_tl": money(price) + total,
        "costs_pct_of_price": round_to(total / price * 100.0, 2),
        "display": {
            "total_costs": fmt_tl(total),
            "total_outlay": fmt_tl(money(price) + total),
            "costs_pct_of_price": fmt_pct(total / price * 100.0),
        },
        "excluded": [
            "Yeni konutta müteahhit satışındaki KDV (genellikle fiyata dahildir)",
            "Taşınma, eşya ve vekâletname noterliği gibi kişisel masraflar",
        ],
    }


# ======================================================================================
# Owning: yield
# ======================================================================================


def rental_yield(
    price: float,
    monthly_rent: float,
    *,
    acquisition_costs: float,
    aidat_monthly: float | None,
    vacancy_months: float,
    maintenance_pct_of_rent: float,
    emlak_vergisi_yearly: float,
    dask_yearly: float,
) -> dict[str, Any]:
    """Gross and net yield, and the net annual income both of the other tools build on.

    The Turkish norm decides who pays what: a tenant pays the aidat (the building's running
    costs) while living there, so the owner pays it only for the months the flat stands empty;
    emlak vergisi and DASK are the owner's whatever happens; upkeep is a share of the rent.
    Income tax on rent is NOT deducted — it depends on the owner's other income and the year's
    exemption — and the result says so rather than pretending the figure is after tax.
    """
    price = _positive("price", price)
    monthly_rent = _positive("monthly_rent", monthly_rent)
    acquisition_costs = _non_negative("acquisition_costs", acquisition_costs)
    vacancy_months = _non_negative("vacancy_months", vacancy_months)
    if vacancy_months > 12:
        raise ValueError(f"vacancy_months cannot exceed 12, got {vacancy_months!r}")
    maintenance_pct_of_rent = _non_negative("maintenance_pct_of_rent", maintenance_pct_of_rent)
    emlak_vergisi_yearly = _non_negative("emlak_vergisi_yearly", emlak_vergisi_yearly)
    dask_yearly = _non_negative("dask_yearly", dask_yearly)
    aidat = _non_negative("aidat_monthly", aidat_monthly) if aidat_monthly is not None else 0.0

    gross_annual_rent = 12.0 * monthly_rent
    collected_rent = (12.0 - vacancy_months) * monthly_rent
    vacancy_aidat = vacancy_months * aidat
    maintenance = gross_annual_rent * maintenance_pct_of_rent / 100.0
    owner_costs = emlak_vergisi_yearly + dask_yearly + maintenance + vacancy_aidat
    net_income = collected_rent - owner_costs
    total_investment = price + acquisition_costs

    gross_yield = gross_annual_rent / price * 100.0
    net_yield = net_income / total_investment * 100.0
    return {
        "price_tl": money(price),
        "monthly_rent_tl": money(monthly_rent),
        "gross_annual_rent_tl": money(gross_annual_rent),
        "gross_yield_pct": round_to(gross_yield, 2),
        "collected_rent_tl": money(collected_rent),
        "owner_costs": {
            "emlak_vergisi_tl": money(emlak_vergisi_yearly),
            "dask_tl": money(dask_yearly),
            "bakim_tl": money(maintenance),
            "bos_kalma_aidati_tl": money(vacancy_aidat),
            "toplam_tl": money(owner_costs),
        },
        "net_annual_income_tl": money(net_income),
        "acquisition_costs_tl": money(acquisition_costs),
        "total_investment_tl": money(total_investment),
        "net_yield_pct": round_to(net_yield, 2),
        "display": {
            "gross_yield": fmt_pct(gross_yield),
            "net_yield": fmt_pct(net_yield),
            "gross_annual_rent": fmt_tl(gross_annual_rent),
            "net_annual_income": fmt_tl(net_income),
            "owner_costs": fmt_tl(owner_costs),
            "total_investment": fmt_tl(total_investment),
        },
        "definitions": {
            "gross_yield": "12 aylık kira / satış fiyatı",
            "net_yield": "(tahsil edilen kira − sahibin giderleri) / (fiyat + alım masrafları)",
        },
        "excluded": ["Kira gelirinin gelir vergisi (sahibin diğer gelirine ve istisnaya bağlı)"],
    }


def payback(
    price: float,
    monthly_rent: float,
    *,
    net_annual_income: float,
    total_investment: float,
    annual_rent_growth_pct: float | None = None,
) -> dict[str, Any]:
    """How many years of rent buy the flat back.

    Three answers, because the question has three honest readings:

    * **brüt** — price over a year's rent, the "amortisman süresi" listings quote;
    * **net** — what was spent (price and purchase costs) over a year's net income;
    * **net, with rent growth** — the same, with the rent (and, proportionally, the owner's
      costs) rising by ``annual_rent_growth_pct`` a year. Solved in closed form from the
      geometric series ``I·((1+g)^t − 1)/g = T``, so ``t = ln(1 + T·g/I) / ln(1+g)``. It is a
      NOMINAL figure: rent rising with inflation pays back nominal lira faster without the
      investment being any better, and the result says so.
    """
    price = _positive("price", price)
    monthly_rent = _positive("monthly_rent", monthly_rent)
    total_investment = _positive("total_investment", total_investment)

    gross_years = price / (12.0 * monthly_rent)
    net_years = total_investment / net_annual_income if net_annual_income > 0 else None

    growth_years: float | None = None
    if annual_rent_growth_pct is not None:
        if not math.isfinite(annual_rent_growth_pct) or annual_rent_growth_pct <= -100:
            raise ValueError(f"annual_rent_growth_pct must be above -100, got {annual_rent_growth_pct!r}")
        g = annual_rent_growth_pct / 100.0
        if net_annual_income > 0:
            if abs(g) < 1e-12:
                growth_years = total_investment / net_annual_income
            else:
                inner = 1.0 + total_investment * g / net_annual_income
                growth_years = math.log(inner) / math.log1p(g) if inner > 0 else None

    result: dict[str, Any] = {
        "gross_payback_years": round_to(gross_years, 1),
        "net_payback_years": round_to(net_years, 1) if net_years is not None else None,
        "net_payback_with_growth_years": round_to(growth_years, 1) if growth_years is not None else None,
        "annual_rent_growth_pct": annual_rent_growth_pct,
        "display": {
            "gross_payback": fmt_years(gross_years),
            "net_payback": fmt_years(net_years) if net_years is not None else "hesaplanamaz (net gelir sıfır ya da eksi)",
        },
        "definitions": {
            "gross_payback": "satış fiyatı / yıllık brüt kira",
            "net_payback": "(fiyat + alım masrafları) / yıllık net kira geliri",
        },
    }
    if annual_rent_growth_pct is not None:
        result["display"]["net_payback_with_growth"] = (
            fmt_years(growth_years) if growth_years is not None else "hesaplanamaz"
        )
        result["definitions"]["net_payback_with_growth"] = (
            "kira ve giderler her yıl aynı oranda artarsa, birikimli net kiranın toplam harcamayı "
            "karşıladığı süre (nominal TL ile)"
        )
    return result


# ======================================================================================
# Financing
# ======================================================================================


def mortgage(
    price: float,
    rates: CostRates,
    *,
    down_payment_pct: float,
    monthly_rate_pct: float,
    term_months: int,
    monthly_rent: float | None = None,
) -> dict[str, Any]:
    """An annuity loan for the part of the price the buyer does not pay in cash.

    The payment is the standard annuity ``L·r / (1 − (1+r)^−n)`` with ``r`` the MONTHLY rate,
    because that is how Turkish banks quote housing loans ("aylık %2,85"). Housing loans are
    exempt from KKDF and BSMV, so the quoted rate is the rate.

    The loan is capped at BDDK's loan-to-value limit for the price band. The cap used is the
    A/B energy-class column — the most a flat of this value can borrow in any class — and
    when it binds, the down payment is raised to meet it and the result says so, together
    with the fact that lower energy classes are capped lower still.
    """
    price = _positive("price", price)
    if not math.isfinite(down_payment_pct) or not 0 <= down_payment_pct <= 100:
        raise ValueError(f"down_payment_pct must be between 0 and 100, got {down_payment_pct!r}")
    if not math.isfinite(monthly_rate_pct) or not 0 <= monthly_rate_pct < 20:
        raise ValueError(
            f"monthly_rate_pct must be a MONTHLY rate between 0 and 20 (e.g. 2.85), got {monthly_rate_pct!r}"
        )
    term_months = int(term_months)
    if not 1 <= term_months <= 480:
        raise ValueError(f"term_months must be between 1 and 480, got {term_months!r}")

    cap = rates.ltv_cap_pct(price)
    requested_ltv = 100.0 - down_payment_pct
    capped = requested_ltv > cap + 1e-9
    ltv = cap if capped else requested_ltv
    loan = price * ltv / 100.0
    down_payment = price - loan

    r = monthly_rate_pct / 100.0
    if loan <= 0:
        payment = 0.0
    elif r == 0:
        payment = loan / term_months
    else:
        payment = loan * r / (1.0 - (1.0 + r) ** (-term_months))
    total_paid = payment * term_months
    interest = total_paid - loan
    annual_equivalent = ((1.0 + r) ** 12 - 1.0) * 100.0

    result: dict[str, Any] = {
        "price_tl": money(price),
        "down_payment_pct": round_to(100.0 - ltv, 2),
        "down_payment_tl": money(down_payment),
        "loan_tl": money(loan),
        "loan_to_value_pct": round_to(ltv, 2),
        "ltv_cap_pct": cap,
        "ltv_capped": capped,
        "monthly_rate_pct": monthly_rate_pct,
        "annual_compound_rate_pct": round_to(annual_equivalent, 2),
        "term_months": term_months,
        "monthly_payment_tl": money(payment),
        "total_paid_tl": money(total_paid),
        "total_interest_tl": money(interest),
        "max_allocation_fee_tl": money(loan * rates.kredi_tahsis_pct / 100.0),
        "display": {
            "monthly_payment": fmt_tl(payment),
            "loan": fmt_tl(loan),
            "down_payment": fmt_tl(down_payment),
            "total_paid": fmt_tl(total_paid),
            "total_interest": fmt_tl(interest),
            "monthly_rate": fmt_pct(monthly_rate_pct),
            "annual_compound_rate": fmt_pct(annual_equivalent),
        },
        "notes": ["Konut kredilerinde KKDF ve BSMV alınmaz; aylık oran bankanın tek faiz oranıdır."],
    }
    if capped:
        result["notes"].append(
            f"İstenen kredi oranı BDDK sınırını aşıyor: {fmt_number(price / 1e6, 1)} milyon TL "
            f"değerindeki konut için en fazla {fmt_pct(cap, 0)} kredi kullanılabilir (A/B enerji "
            "sınıfı); peşinat buna göre yükseltildi."
        )
    result["notes"].append(
        f"BDDK sınırı {fmt_pct(cap, 0)}, A ve B enerji sınıfı içindir; daha düşük sınıflarda oran daha düşüktür."
    )
    if monthly_rent is not None and monthly_rent > 0 and payment > 0:
        coverage = monthly_rent / payment * 100.0
        result["rent_covers_payment_pct"] = round_to(coverage, 1)
        result["display"]["rent_covers_payment"] = fmt_pct(coverage, 1)
        result["monthly_gap_tl"] = money(payment - monthly_rent)
        result["display"]["monthly_gap"] = fmt_tl(payment - monthly_rent)
    return result


# ======================================================================================
# Holding: a projection
# ======================================================================================


def projection(
    price: float,
    *,
    years: int,
    acquisition_costs: float,
    net_income_year1: float,
    price_growth_pct: float,
    rent_growth_pct: float,
    inflation_pct: float,
    exit_costs_pct: float,
) -> dict[str, Any]:
    """Buy today, rent it out, sell after ``years`` — in nominal lira and in today's lira.

    The model is deliberately simple and every simplification is named in the result: the net
    rent of year ``k`` is year one's grown by ``rent_growth_pct`` (owner costs grow with it),
    rent is not reinvested, the flat sells at the end for the price grown by
    ``price_growth_pct``, minus the seller's costs (``exit_costs_pct``: the seller's tapu
    harcı and commission with KDV). Real figures deflate by ``inflation_pct``. No tax.

    In Turkey the nominal and the real answer can point in opposite directions — 30 % a year
    on paper is a loss against 35 % inflation — which is why both are always returned.
    """
    price = _positive("price", price)
    years = int(years)
    if not 1 <= years <= 30:
        raise ValueError(f"years must be between 1 and 30, got {years!r}")
    acquisition_costs = _non_negative("acquisition_costs", acquisition_costs)
    for name, value in (
        ("price_growth_pct", price_growth_pct),
        ("rent_growth_pct", rent_growth_pct),
        ("inflation_pct", inflation_pct),
    ):
        if value is None or not math.isfinite(value) or value <= -100 or value > 500:
            raise ValueError(f"{name} must be a yearly percentage above -100 and at most 500, got {value!r}")
    exit_costs_pct = _non_negative("exit_costs_pct", exit_costs_pct)

    gp, gr, inf = price_growth_pct / 100.0, rent_growth_pct / 100.0, inflation_pct / 100.0
    outlay = price + acquisition_costs
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for k in range(1, years + 1):
        income = net_income_year1 * (1.0 + gr) ** (k - 1)
        cumulative += income
        value = price * (1.0 + gp) ** k
        rows.append(
            {
                "year": k,
                "value_tl": money(value),
                "net_rent_tl": money(income),
                "cumulative_net_rent_tl": money(cumulative),
            }
        )

    final_value = price * (1.0 + gp) ** years
    exit_costs = final_value * exit_costs_pct / 100.0
    wealth = final_value - exit_costs + cumulative
    multiple = wealth / outlay
    nominal_total = (multiple - 1.0) * 100.0
    nominal_annual = (multiple ** (1.0 / years) - 1.0) * 100.0 if multiple > 0 else -100.0
    deflator = (1.0 + inf) ** years
    real_total = (multiple / deflator - 1.0) * 100.0
    real_annual = ((1.0 + nominal_annual / 100.0) / (1.0 + inf) - 1.0) * 100.0

    return {
        "years": years,
        "initial_outlay_tl": money(outlay),
        "final_value_tl": money(final_value),
        "exit_costs_tl": money(exit_costs),
        "cumulative_net_rent_tl": money(cumulative),
        "final_wealth_tl": money(wealth),
        "nominal_total_return_pct": round_to(nominal_total, 2),
        "nominal_annual_return_pct": round_to(nominal_annual, 2),
        "real_total_return_pct": round_to(real_total, 2),
        "real_annual_return_pct": round_to(real_annual, 2),
        "final_value_in_todays_tl": money(final_value / deflator),
        "assumptions": {
            "price_growth_pct": price_growth_pct,
            "rent_growth_pct": rent_growth_pct,
            "inflation_pct": inflation_pct,
            "exit_costs_pct": round_to(exit_costs_pct, 2),
        },
        "rows": rows,
        "display": {
            "final_value": fmt_tl(final_value),
            "cumulative_net_rent": fmt_tl(cumulative),
            "final_wealth": fmt_tl(wealth),
            "nominal_total_return": fmt_pct(nominal_total),
            "nominal_annual_return": fmt_pct(nominal_annual),
            "real_total_return": fmt_pct(real_total),
            "real_annual_return": fmt_pct(real_annual),
            "final_value_in_todays_tl": fmt_tl(final_value / deflator),
        },
        "simplifications": [
            "Kira geliri yeniden yatırılmıyor; giderler kirayla aynı oranda artıyor",
            "Satışta satıcı tapu harcı ve komisyon (KDV dahil) düşülüyor; vergi yok",
            "5 yıldan önce satışta değer artış kazancı vergisi doğabilir; hesaba katılmadı",
        ],
    }


# ======================================================================================
# Price per square metre
# ======================================================================================


@dataclass(frozen=True, slots=True)
class Benchmark:
    """A market price per m² somebody published, with where and when."""

    label: str
    tl_per_m2: float
    #: ``net``, ``brut`` or ``bilinmiyor`` — which area the published figure divides by.
    area_basis: str
    source_url: str
    source_title: str
    source_date: str


def price_per_m2(
    price: float,
    *,
    net_m2: float | None,
    gross_m2: float | None,
    benchmarks: list[Benchmark],
) -> dict[str, Any]:
    """The flat's price per m², and how far it sits from each published benchmark.

    A benchmark is compared on its OWN basis: a gross-m² market figure against the price per
    gross m², a net one against net. Comparing a gross benchmark with a net price per m²
    overstates the flat by the ratio of the two areas (often 15-25 %), which is the commonest
    way a "this flat is expensive" claim is wrong. When the basis is unknown, or the flat has
    only the other area, the comparison is still made — and flagged.
    """
    price = _positive("price", price)
    net = float(net_m2) if net_m2 and net_m2 > 0 else None
    gross = float(gross_m2) if gross_m2 and gross_m2 > 0 else None
    if net is None and gross is None:
        raise ValueError("the property has no m² figure: the agent entered neither net nor gross area")

    per_net = price / net if net else None
    per_gross = price / gross if gross else None
    compared: list[dict[str, Any]] = []
    for bench in benchmarks:
        value = _positive("benchmark tl_per_m2", bench.tl_per_m2)
        basis = bench.area_basis if bench.area_basis in ("net", "brut") else "bilinmiyor"
        flag: str | None = None
        if basis == "net":
            own, used = (per_net, "net") if per_net is not None else (per_gross, "brut")
            if per_net is None:
                flag = "Kaynak net m² veriyor; evin yalnızca brüt alanı girildiği için brüt m² fiyatıyla karşılaştırıldı."
        elif basis == "brut":
            own, used = (per_gross, "brut") if per_gross is not None else (per_net, "net")
            if per_gross is None:
                flag = "Kaynak brüt m² veriyor; evin yalnızca net alanı girildiği için net m² fiyatıyla karşılaştırıldı (ev olduğundan pahalı görünür)."
        else:
            own, used = (per_gross, "brut") if per_gross is not None else (per_net, "net")
            flag = "Kaynağın net mi brüt mü m² kullandığı belli değil; karşılaştırma yaklaşıktır."
        assert own is not None
        difference = (own / value - 1.0) * 100.0
        compared.append(
            {
                "label": bench.label,
                "benchmark_tl_per_m2": money(value),
                "benchmark_basis": basis,
                "property_tl_per_m2": money(own),
                "property_basis": used,
                "difference_pct": round_to(difference, 1),
                "display": {
                    "benchmark": f"{fmt_number(money(value))} TL/m²",
                    "property": f"{fmt_number(money(own))} TL/m² ({'net' if used == 'net' else 'brüt'})",
                    "difference": fmt_pct(difference, 1),
                },
                "source": {"url": bench.source_url, "title": bench.source_title, "date": bench.source_date},
                "flag": flag,
            }
        )

    display: dict[str, str] = {}
    if per_net is not None:
        display["per_net_m2"] = f"{fmt_number(money(per_net))} TL/m² (net)"
    if per_gross is not None:
        display["per_gross_m2"] = f"{fmt_number(money(per_gross))} TL/m² (brüt)"
    return {
        "price_tl": money(price),
        "net_m2": net,
        "gross_m2": gross,
        "tl_per_net_m2": money(per_net) if per_net is not None else None,
        "tl_per_gross_m2": money(per_gross) if per_gross is not None else None,
        "comparisons": compared,
        "display": display,
    }
