"""What the agent tells the assistant about the property, and how the calculators read it.

The agent's figures are the AUTHORITATIVE source for this flat: its asking price, its rent, its
aidat. They are kept on the property (``Property.investment``, one JSON document) and every
calculator reads them from there rather than from the model — the model passes a figure only
when it means a DIFFERENT one (a negotiated price, a market rent it found). That is the
cheapest way to make sure "7.850.000" never becomes "7.580.000" on its way into a formula.

Each resolved value carries its basis (``ilan`` = the agent entered it, ``senaryo`` = the model
or the visitor supplied it, ``varsayılan`` / ``tahmin`` = a configured default or an
estimate), and the tool results repeat the basis next to the number, so the answer can say
which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from biosphere.invest.rates import CostRates

__all__ = ["InvestmentInputs", "Resolved", "resolve_rent", "owner_costs", "commission_pct", "dask_yearly"]

EnergyClass = Literal["A", "B", "C", "D", "E", "F", "G"]


class InvestmentInputs(BaseModel):
    """The agent's figures for one property. Every field is optional except the opt-in.

    Bounds are generous rather than clever: they exist to stop a slipped digit (a price of
    7.850 TL, a rent of 3.200.000 TL) from becoming a confident 4.000 % yield, not to encode
    the market. The chat additionally needs ``enabled`` and ``asking_price_tl``.
    """

    model_config = ConfigDict(extra="forbid")

    #: The agent's opt-in. Off by default: the chat sends the agent's figures and the visitor's
    #: questions to a hosted model, which is the agent's decision to make, per property.
    enabled: bool = False

    asking_price_tl: float | None = Field(default=None, gt=10_000, lt=10_000_000_000)
    monthly_rent_tl: float | None = Field(default=None, gt=100, lt=50_000_000)
    #: ``actual`` — the flat is let at this rent today; ``expected`` — the agent's estimate.
    rent_status: Literal["expected", "actual"] | None = None
    aidat_monthly_tl: float | None = Field(default=None, ge=0, lt=1_000_000)
    emlak_vergisi_yearly_tl: float | None = Field(default=None, ge=0, lt=10_000_000)
    dask_yearly_tl: float | None = Field(default=None, ge=0, lt=1_000_000)

    il: str | None = Field(default=None, max_length=40)
    ilce: str | None = Field(default=None, max_length=60)
    mahalle: str | None = Field(default=None, max_length=80)

    gross_m2: float | None = Field(default=None, gt=5, lt=100_000)
    net_m2: float | None = Field(default=None, gt=5, lt=100_000)
    room_layout: str | None = Field(default=None, max_length=12)
    building_age_years: int | None = Field(default=None, ge=0, le=300)
    floor: int | None = Field(default=None, ge=-10, le=200)
    total_floors: int | None = Field(default=None, ge=1, le=200)
    heating: str | None = Field(default=None, max_length=40)
    energy_class: EnergyClass | None = None
    sale_kind: Literal["second_hand", "new_build"] | None = None
    in_site: bool | None = None

    #: The buyer-side commission the agent charges, before KDV. ``None`` uses the legal
    #: ceiling from the rates table; ``0`` is a real answer ("alıcıdan komisyon alınmaz").
    commission_pct: float | None = Field(default=None, ge=0, le=4)

    #: Free text the agent wants the assistant to know ("otoparklı, metroya 5 dk yürüme").
    #: Facts the agent asserts — the prompt frames them as the agent's claims, not as
    #: instructions, and the visitor never sees this field directly.
    notes: str | None = Field(default=None, max_length=600)

    #: The agent's own daily spending cap for this property's chat, in USD. Bounded above by
    #: the operator's ``advisor_daily_budget_max_usd`` when it is applied.
    daily_budget_usd: float | None = Field(default=None, ge=0, le=1000)

    def location_label(self) -> str | None:
        parts = [p.strip() for p in (self.mahalle, self.ilce, self.il) if p and p.strip()]
        return ", ".join(parts) or None

    def chat_ready(self) -> bool:
        """The agent opted in and gave at least a price — the two things the chat needs."""
        return bool(self.enabled and self.asking_price_tl)

    @classmethod
    def from_column(cls, value: object) -> InvestmentInputs:
        """The stored JSON, tolerant of an older shape: unknown keys dropped, bad values reset.

        Never raises. A document written by a future version, or edited by hand, must not make
        the property page fail to open; the fields that still validate survive.
        """
        if not isinstance(value, dict):
            return cls()
        clean = {k: v for k, v in value.items() if k in cls.model_fields}
        try:
            return cls.model_validate(clean)
        except Exception:  # noqa: BLE001 - salvage field by field
            kept: dict[str, object] = {}
            for key, item in clean.items():
                try:
                    cls.model_validate({key: item})
                except Exception:  # noqa: BLE001
                    continue
                kept[key] = item
            return cls.model_validate(kept)


@dataclass(frozen=True, slots=True)
class Resolved:
    """A value the calculators use, and where it came from."""

    value: float
    #: ``ilan`` | ``senaryo`` | ``varsayılan`` | ``tahmin``
    basis: str
    note: str = ""


def resolve_rent(inputs: InvestmentInputs, scenario_rent: float | None, scenario_source: str | None) -> Resolved | None:
    """The rent a calculation uses: the scenario's when given, else the agent's, else none."""
    if scenario_rent is not None:
        return Resolved(float(scenario_rent), "senaryo", (scenario_source or "").strip()[:300])
    if inputs.monthly_rent_tl:
        status = "mevcut kira (ev kiracılı)" if inputs.rent_status == "actual" else "ilan sahibinin beklediği kira"
        return Resolved(float(inputs.monthly_rent_tl), "ilan", status)
    return None


def dask_yearly(inputs: InvestmentInputs, rates: CostRates) -> Resolved:
    if inputs.dask_yearly_tl is not None:
        return Resolved(float(inputs.dask_yearly_tl), "ilan")
    return Resolved(rates.dask_yearly_tl, "varsayılan", "bölge risk grubuna göre değişir")


def commission_pct(inputs: InvestmentInputs, rates: CostRates) -> Resolved:
    if inputs.commission_pct is not None:
        return Resolved(float(inputs.commission_pct), "ilan")
    return Resolved(rates.commission_pct, "varsayılan", "yasal tavan; ilan sahibi farklı oran uygulayabilir")


def owner_costs(inputs: InvestmentInputs, rates: CostRates, price: float) -> dict[str, Resolved]:
    """The owner's yearly costs, each with its basis.

    Emlak vergisi is the one that is ESTIMATED when missing, and estimated high: the rate
    applied to the price rather than to the (lower, unknown) tax value. A net yield that
    errs low is the honest direction for a figure a buyer may decide on.
    """
    if inputs.emlak_vergisi_yearly_tl is not None:
        emlak = Resolved(float(inputs.emlak_vergisi_yearly_tl), "ilan")
    else:
        pct = rates.emlak_vergisi_pct(inputs.il)
        emlak = Resolved(price * pct / 100.0, "tahmin", f"üst sınır: fiyatın binde {pct * 10:g}'si (vergi değeri bilinmiyor)")
    return {
        "emlak_vergisi": emlak,
        "dask": dask_yearly(inputs, rates),
        "maintenance_pct": Resolved(rates.maintenance_pct_of_rent, "varsayılan", "kiranın yüzdesi"),
        "vacancy_months": Resolved(rates.vacancy_months, "varsayılan", "yılda boş kalınan ay"),
    }
