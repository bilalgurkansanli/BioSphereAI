"""What it costs to buy, own and finance a flat in Turkey — as dated, sourced, configurable values.

Every one of these is a fact about Turkish law or the Turkish market on a particular day, and
every one of them changes: the döner sermaye indicator moves each January, BDDK rewrote the
loan-to-value bands on 29 January 2026, and the mortgage rate a bank quotes moves monthly. So
none of them is written into a formula. They live here, each with the page it came from and
the date it was read, and three things follow from that:

* An operator overrides any of them without a code change: ``BIOSPHERE_ADVISOR_RATES`` takes a
  JSON object whose keys are the field names below (``{"doner_sermaye_tl": 7200}``).
* The dashboard shows the table, so an agent can see what the buyer's cost estimate assumes.
* Every tool result echoes the rates it used, so the model can say "tapu harcı %2 (alıcı payı)"
  with the source, and a reader can check it.

The defaults were read on 2026-10-06. A rate whose source is older than a year is still used —
it is the best number on hand — but :func:`CostRates.stale_keys` names it, and the dashboard
shows the date beside it, so the age is never hidden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from typing import Any

from biosphere.core.logging import get_logger

__all__ = ["Source", "LtvBand", "CostRates", "METRO_PROVINCES", "load_rates", "is_metro"]

log = get_logger("biosphere.invest.rates")


@dataclass(frozen=True, slots=True)
class Source:
    """Where a number came from, and when it was read."""

    title: str
    url: str
    #: ISO date (``2026-09-17``) or month (``2026-09``) of the publication or the reading.
    date: str


@dataclass(frozen=True, slots=True)
class LtvBand:
    """One row of BDDK's loan-to-value table: homes valued up to ``up_to_tl`` may borrow
    at most ``max_ltv_pct`` of their value. ``None`` is the open-ended top band."""

    up_to_tl: float | None
    max_ltv_pct: float


#: BDDK decision no. 11364 (29 January 2026), energy classes A and B.
#:
#: THE MOST FAVOURABLE COLUMN, ON PURPOSE. The decision is tiered by value AND energy class,
#: and the published coverage gives the A/B column in full but not every other class. A/B is
#: therefore an UPPER BOUND for any flat: no class may borrow more than this. The mortgage
#: tool caps a loan at it and says, in the same breath, that classes below B are capped lower —
#: which is true, rather than a lower number for C that nobody here has read in the decision.
_LTV_A_B_2026: tuple[LtvBand, ...] = (
    LtvBand(5_000_000, 90.0),
    LtvBand(7_000_000, 80.0),
    LtvBand(10_000_000, 70.0),
    LtvBand(20_000_000, 50.0),
    LtvBand(None, 40.0),
)

#: The thirty büyükşehir provinces. Emlak vergisi on a dwelling is binde 2 of its tax value in
#: these and binde 1 elsewhere, so the province decides which rate an estimate uses. Folded to
#: ASCII lower case for matching, because agents type "Istanbul", "İstanbul" and "istanbul".
METRO_PROVINCES: frozenset[str] = frozenset(
    {
        "adana", "ankara", "antalya", "aydin", "balikesir", "bursa", "denizli", "diyarbakir",
        "erzurum", "eskisehir", "gaziantep", "hatay", "istanbul", "izmir", "kahramanmaras",
        "kayseri", "kocaeli", "konya", "malatya", "manisa", "mardin", "mersin", "mugla", "ordu",
        "sakarya", "samsun", "sanliurfa", "tekirdag", "trabzon", "van",
    }
)

_FOLD = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛI", "cgiosuaiucgiosuaiui")


def _fold(text: str) -> str:
    return text.strip().translate(_FOLD).lower()


def is_metro(il: str | None) -> bool | None:
    """True for a büyükşehir, False for another province, None when the province is unknown."""
    if not il or not il.strip():
        return None
    return _fold(il) in METRO_PROVINCES


@dataclass(frozen=True, slots=True)
class CostRates:
    """Turkey's purchase, ownership and financing parameters, as the calculators use them."""

    #: When the defaults below were read. An override carries its own date in ``sources``.
    as_of: str = "2026-10-06"

    # -- buying -----------------------------------------------------------------------------
    #: Tapu harcı, Harçlar Kanunu (492) tarife IV: 2 % from each side on the declared price.
    tapu_harci_buyer_pct: float = 2.0
    tapu_harci_seller_pct: float = 2.0
    #: Tapu döner sermaye fee for a sale. The 2026 indicator is 2.227 TL times the district's
    #: coefficient; 6.681 TL is the coefficient-3 figure the press quotes as typical. It VARIES
    #: by district, which the tool result says.
    doner_sermaye_tl: float = 6681.0
    #: Estate-agent commission charged to the buyer, before KDV. Taşınmaz Ticareti Hakkında
    #: Yönetmelik caps it at 2 % per side (4 % in total). The agent may set their own figure,
    #: including zero, on the property.
    commission_pct: float = 2.0
    #: KDV on the commission (the general rate since July 2023).
    kdv_pct: float = 20.0

    # -- owning -----------------------------------------------------------------------------
    #: Compulsory earthquake insurance, per year, when the agent has not entered one. A
    #: placeholder of the right order: the premium depends on m², construction type and the
    #: earthquake risk group, none of which this table knows. Its weight in a yield is tiny.
    dask_yearly_tl: float = 2500.0
    #: Emlak vergisi on a dwelling, as a percentage of its TAX value (binde 2 in a büyükşehir,
    #: binde 1 elsewhere). The tax value is not known here, and it sits below the market
    #: value — so the estimate applies the rate to the asking price and calls the result an
    #: UPPER BOUND, which is what it is. An agent who enters the real figure replaces it.
    emlak_vergisi_metro_pct: float = 0.2
    emlak_vergisi_other_pct: float = 0.1
    #: Upkeep the owner pays, as a share of the gross annual rent. An assumption, and labelled
    #: as one wherever it appears.
    maintenance_pct_of_rent: float = 5.0
    #: Months a year the flat stands empty between tenants, during which the owner also pays
    #: the aidat. An assumption; the model may pass another.
    vacancy_months: float = 1.0

    # -- financing ----------------------------------------------------------------------------
    #: An EXAMPLE monthly housing-loan rate for the deterministic card. Banks' 120-month
    #: quotes clustered at 2,75-2,88 % in September 2026; the card shows this rate with its
    #: source and date, and the chat finds today's rates on the web when asked.
    mortgage_monthly_rate_pct: float = 2.85
    mortgage_term_months: int = 120
    #: The card's example down payment. Raised automatically when BDDK's cap for the price
    #: band demands more (a flat over 20 M TL can borrow at most 40 %).
    mortgage_down_payment_pct: float = 50.0
    #: Appraisal fee the bank charges on a mortgage purchase (2026 average).
    ekspertiz_tl: float = 21004.0
    #: Loan allocation fee: at most binde 5 of the loan.
    kredi_tahsis_pct: float = 0.5
    ltv_bands: tuple[LtvBand, ...] = _LTV_A_B_2026

    sources: dict[str, Source] = field(
        default_factory=lambda: {
            "tapu_harci": Source(
                "Tapu devir ücreti 2026 — Milliyet Emlak",
                "https://www.milliyet.com.tr/emlak/tapu-devir-ucreti-nedir-2026-tapu-harci-masrafi-ne-kadar-oldu-6515059",
                "2026",
            ),
            "doner_sermaye": Source(
                "Tapu harcı ne kadar 2026 — döner sermaye göstergesi 2.227 TL",
                "https://kobitime.com/tapu-harci-ne-kadar-2026/",
                "2026",
            ),
            "commission": Source(
                "Taşınmaz Ticareti Hakkında Yönetmelik — komisyon tavanı",
                "https://hukukcularevi.com/tasinmaz-simsarligi-emlak-komisyonu-tbk-520/",
                "2026",
            ),
            "mortgage_rate": Source(
                "Konut kredisi faizleri — Aydınlık, 17 Eylül 2026",
                "https://www.aydinlik.com.tr/haber/konut-kredisi-faizleri-belli-oldu-iste-1-milyon-tlye-gelen-faiz-orani-510193",
                "2026-09-17",
            ),
            "ltv": Source(
                "BDDK 29 Ocak 2026 kararı (11364) — kredi/değer oranları",
                "https://www.hisse.net/haber/ev-hayali-kuran-ciftlere-buyuk-kolaylik-kredi-limitleri-ve-oranlari-tepeden-tirnaga-degisti-53869",
                "2026-01-31",
            ),
            "emlak_vergisi": Source(
                "2026 emlak vergisi oranları — konut binde 2 / binde 1",
                "https://faturaport.com/blog/on-muhasebe/2026-emlak-vergisi-oranlari-ve-bildirim-surecleri",
                "2026",
            ),
            "ekspertiz": Source(
                "Konut kredisi masrafları — ekspertiz ve tahsis ücreti",
                "https://hesap.com/hesap-defteri/konut-kredisinde-masraflar-nelerdir",
                "2026",
            ),
            "dask": Source(
                "DASK 2026 — m² birim maliyeti 9.884 TL (prim risk bölgesine göre değişir)",
                "https://www.hesapkurdu.com/konut-kredisi/rehber/dask-nedir",
                "2026-01",
            ),
        }
    )

    # -------------------------------------------------------------------------------------

    def ltv_cap_pct(self, value_tl: float) -> float:
        """The most a flat of this value may borrow under the A/B column (an upper bound)."""
        for band in self.ltv_bands:
            if band.up_to_tl is None or value_tl <= band.up_to_tl:
                return band.max_ltv_pct
        return self.ltv_bands[-1].max_ltv_pct if self.ltv_bands else 0.0

    def emlak_vergisi_pct(self, il: str | None) -> float:
        """The büyükşehir rate unless the province is known to be another one.

        Unknown province takes the HIGHER rate: the figure is presented as an upper bound,
        and an upper bound that guessed low would not be one.
        """
        return self.emlak_vergisi_other_pct if is_metro(il) is False else self.emlak_vergisi_metro_pct

    def stale_keys(self, today: date | None = None, max_age_days: int = 365) -> list[str]:
        """Sources older than ``max_age_days``. A month-only date counts from its first day."""
        today = today or date.today()
        stale: list[str] = []
        for key, source in self.sources.items():
            try:
                parts = [int(p) for p in source.date.split("-")]
                when = date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
            except (ValueError, IndexError):
                continue
            if (today - when).days > max_age_days:
                stale.append(key)
        return stale

    def as_dict(self) -> dict[str, Any]:
        """Plain JSON for the tool results and the dashboard."""
        data = asdict(self)
        data["ltv_bands"] = [asdict(b) for b in self.ltv_bands]
        data["sources"] = {k: asdict(v) for k, v in self.sources.items()}
        return data


_NUMERIC = {f.name for f in fields(CostRates) if f.type in ("float", "int")}


def load_rates(raw: str | dict[str, Any] | None = None) -> CostRates:
    """The defaults with an operator's overrides applied. Never raises.

    ``raw`` is ``settings.advisor_rates`` — a JSON object, or empty. Unknown keys and
    non-numeric values are dropped with a warning rather than failing the engine's start: a
    typo in one rate must not take the whole assistant down, and the dashboard shows the
    values actually in force, so a dropped override is visible.
    """
    base = CostRates()
    if not raw:
        return base
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        log.warning("invest.rates_unparsable")
        return base
    if not isinstance(data, dict):
        return base

    changes: dict[str, Any] = {}
    for key, value in data.items():
        if key in _NUMERIC:
            try:
                changes[key] = int(value) if key == "mortgage_term_months" else float(value)
            except (TypeError, ValueError):
                log.warning("invest.rate_ignored", key=key)
        elif key == "as_of" and isinstance(value, str):
            changes["as_of"] = value[:20]
        elif key == "ltv_bands" and isinstance(value, list):
            try:
                changes["ltv_bands"] = tuple(
                    LtvBand(None if b.get("up_to_tl") is None else float(b["up_to_tl"]), float(b["max_ltv_pct"]))
                    for b in value
                )
            except (TypeError, ValueError, KeyError, AttributeError):
                log.warning("invest.rate_ignored", key=key)
        elif key == "sources" and isinstance(value, dict):
            merged = dict(base.sources)
            for skey, sval in value.items():
                if isinstance(sval, dict) and {"title", "url", "date"} <= set(sval):
                    merged[str(skey)] = Source(str(sval["title"]), str(sval["url"]), str(sval["date"]))
            changes["sources"] = merged
        else:
            log.warning("invest.rate_ignored", key=key)
    return replace(base, **changes)
