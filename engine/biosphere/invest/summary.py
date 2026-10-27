"""The card a visitor sees the moment the panel opens: four numbers, no model.

Gross yield, amortisman, price per m² and an example mortgage payment are arithmetic on the
agent's own figures, so they cost nothing to show and they are shown first — before the chat,
and whether or not a model is configured at all. A deployment without an API key, a property
whose daily budget is spent, a visitor who never types a word: all of them still get the card.

The same values travel into the chat's FACTS block as "precomputed", so the commonest question
("kira getirisi ne kadar?") is answered from the same engine output the card shows, without a
tool round trip and without the two ever disagreeing.
"""

from __future__ import annotations

from typing import Any

from biosphere.invest import calc
from biosphere.invest.inputs import InvestmentInputs, commission_pct, dask_yearly, owner_costs, resolve_rent
from biosphere.invest.rates import CostRates

__all__ = ["card"]


def card(inputs: InvestmentInputs, rates: CostRates) -> dict[str, Any] | None:
    """The deterministic summary, or ``None`` when there is no price to compute it from.

    Every figure is accompanied by what it rests on (the rent's status, the area's basis, the
    loan's terms and the rate's source), because a number on a card a buyer screenshots has to
    carry its own small print.
    """
    price = inputs.asking_price_tl
    if not price:
        return None

    out: dict[str, Any] = {
        "price_tl": calc.money(price),
        "location": inputs.location_label(),
        "missing": [],
    }

    commission = commission_pct(inputs, rates)
    dask = dask_yearly(inputs, rates)
    costs = calc.purchase_costs(price, rates, commission_pct=commission.value, dask_yearly=dask.value)
    out["purchase"] = {
        "total_costs_tl": costs["total_costs_tl"],
        "total_outlay_tl": costs["total_outlay_tl"],
        "costs_pct_of_price": costs["costs_pct_of_price"],
        "commission_pct": commission.value,
        "commission_basis": commission.basis,
    }

    rent = resolve_rent(inputs, None, None)
    if rent is None:
        out["rent"] = None
        out["income"] = None
        out["missing"].append("rent")
    else:
        own = owner_costs(inputs, rates, price)
        yld = calc.rental_yield(
            price,
            rent.value,
            acquisition_costs=costs["total_costs_tl"],
            aidat_monthly=inputs.aidat_monthly_tl,
            vacancy_months=own["vacancy_months"].value,
            maintenance_pct_of_rent=own["maintenance_pct"].value,
            emlak_vergisi_yearly=own["emlak_vergisi"].value,
            dask_yearly=own["dask"].value,
        )
        back = calc.payback(
            price,
            rent.value,
            net_annual_income=yld["net_annual_income_tl"],
            total_investment=yld["total_investment_tl"],
        )
        out["rent"] = {
            "monthly_tl": calc.money(rent.value),
            "status": inputs.rent_status or "expected",
        }
        out["income"] = {
            "gross_pct": yld["gross_yield_pct"],
            "net_pct": yld["net_yield_pct"],
            "net_annual_income_tl": yld["net_annual_income_tl"],
            "gross_payback_years": back["gross_payback_years"],
            "net_payback_years": back["net_payback_years"],
            "assumptions": {
                "vacancy_months": own["vacancy_months"].value,
                "maintenance_pct_of_rent": own["maintenance_pct"].value,
                "emlak_vergisi_tl": calc.money(own["emlak_vergisi"].value),
                "emlak_vergisi_basis": own["emlak_vergisi"].basis,
                "dask_tl": calc.money(own["dask"].value),
                "dask_basis": own["dask"].basis,
                "aidat_monthly_tl": calc.money(inputs.aidat_monthly_tl) if inputs.aidat_monthly_tl is not None else None,
            },
        }

    area_basis, area = ("net", inputs.net_m2) if inputs.net_m2 else ("brut", inputs.gross_m2)
    if area:
        per = calc.price_per_m2(price, net_m2=inputs.net_m2, gross_m2=inputs.gross_m2, benchmarks=[])
        out["per_m2"] = {
            "tl": per["tl_per_net_m2"] if area_basis == "net" else per["tl_per_gross_m2"],
            "basis": area_basis,
            "area_m2": area,
        }
    else:
        out["per_m2"] = None
        out["missing"].append("area")

    cap = rates.ltv_cap_pct(price)
    down = max(rates.mortgage_down_payment_pct, 100.0 - cap)
    loan = calc.mortgage(
        price,
        rates,
        down_payment_pct=down,
        monthly_rate_pct=rates.mortgage_monthly_rate_pct,
        term_months=rates.mortgage_term_months,
        monthly_rent=rent.value if rent else None,
    )
    source = rates.sources.get("mortgage_rate")
    out["mortgage"] = {
        "monthly_payment_tl": loan["monthly_payment_tl"],
        "loan_tl": loan["loan_tl"],
        "down_payment_pct": loan["down_payment_pct"],
        "down_payment_tl": loan["down_payment_tl"],
        "term_months": loan["term_months"],
        "monthly_rate_pct": rates.mortgage_monthly_rate_pct,
        "ltv_cap_pct": cap,
        "rate_source": {"title": source.title, "url": source.url, "date": source.date} if source else None,
        "rent_covers_payment_pct": loan.get("rent_covers_payment_pct"),
    }
    return out
