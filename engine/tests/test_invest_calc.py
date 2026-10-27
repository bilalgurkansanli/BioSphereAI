"""The investment calculators, against numbers worked out by hand.

The reference flat is the one the bench uses: 7.850.000 TL, let at 32.000 TL a month, 2.500 TL
aidat, in Istanbul. Every expected figure below was computed on paper first (and the annuity
cross-checked by discounting its payments back to the loan), so a test failing here means the
formula changed — not that a snapshot drifted.

    purchase costs  tapu 2 % 157.000 + döner sermaye 6.681 + commission 2 % 157.000
                    + KDV 20 % 31.400 + DASK 2.500                    = 354.581 TL
    gross yield     12 × 32.000 / 7.850.000                          = 4,8917 %
    net income      11 × 32.000 − (15.700 emlak + 2.500 DASK
                    + 19.200 upkeep + 2.500 vacant-month aidat)      = 312.100 TL
    net yield       312.100 / (7.850.000 + 354.581)                  = 3,8040 %
    payback         7.850.000 / 384.000 = 20,44 y; 8.204.581 / 312.100 = 26,29 y
    with 30 % rent growth   ln(1 + 8.204.581·0,3/312.100) / ln 1,3   = 8,33 y
    mortgage        3.925.000 at 2,85 %/month over 120 months        = 115.837,47 TL/month
"""

from __future__ import annotations

import math

import pytest

from biosphere.invest import calc
from biosphere.invest.inputs import InvestmentInputs, owner_costs
from biosphere.invest.rates import CostRates, LtvBand, is_metro, load_rates

RATES = CostRates()
PRICE = 7_850_000.0
RENT = 32_000.0


# ======================================================================================
# Rounding and formatting
# ======================================================================================


def test_rounding_is_half_up_on_the_printed_decimal() -> None:
    # round() would give 2.67 and 0.12 here: binary floats and banker's rounding.
    assert calc.round_to(2.675, 2) == 2.68
    assert calc.round_to(0.125, 2) == 0.13
    assert calc.money(2.5) == 3
    assert calc.money(1_234_566.5) == 1_234_567


def test_turkish_number_formats() -> None:
    assert calc.fmt_number(1_234_567.891, 2) == "1.234.567,89"
    assert calc.fmt_tl(7_850_000) == "7.850.000 TL"
    assert calc.fmt_tl(115_837.467) == "115.837 TL"
    assert calc.fmt_pct(4.8917) == "%4,89"
    assert calc.fmt_pct(-1.25) == "-%1,25"
    assert calc.fmt_years(20.4427) == "20,4 yıl"


# ======================================================================================
# Buying
# ======================================================================================


def test_purchase_costs_item_by_item() -> None:
    result = calc.purchase_costs(PRICE, RATES, commission_pct=2.0, dask_yearly=2500.0)
    amounts = {item["key"]: item["amount_tl"] for item in result["items"]}
    assert amounts == {
        "tapu_harci": 157_000,
        "doner_sermaye": 6_681,
        "komisyon": 157_000,
        "komisyon_kdv": 31_400,
        "dask": 2_500,
    }
    assert result["total_costs_tl"] == 354_581
    assert result["total_outlay_tl"] == 8_204_581
    assert result["costs_pct_of_price"] == 4.52
    assert result["display"]["total_costs"] == "354.581 TL"


def test_zero_commission_drops_the_line_and_its_kdv() -> None:
    result = calc.purchase_costs(PRICE, RATES, commission_pct=0.0, dask_yearly=2500.0)
    keys = {item["key"] for item in result["items"]}
    assert "komisyon" not in keys and "komisyon_kdv" not in keys
    assert result["total_costs_tl"] == 157_000 + 6_681 + 2_500


def test_buyer_paying_both_tapu_shares_and_a_mortgage() -> None:
    result = calc.purchase_costs(
        PRICE,
        RATES,
        commission_pct=2.0,
        dask_yearly=2500.0,
        buyer_pays_seller_tapu=True,
        with_mortgage=True,
        loan_amount=3_925_000,
    )
    amounts = {item["key"]: item["amount_tl"] for item in result["items"]}
    assert amounts["tapu_harci"] == 314_000  # 4 % of the price
    assert amounts["ekspertiz"] == 21_004
    assert amounts["ipotek_doner_sermaye"] == 6_681
    assert amounts["kredi_tahsis"] == 19_625  # binde 5 of 3.925.000
    # The total is the sum of the rounded lines, never a lira away from what a reader adds up.
    assert result["total_costs_tl"] == sum(amounts.values())


# ======================================================================================
# Owning
# ======================================================================================


def test_rental_yield_gross_and_net() -> None:
    result = calc.rental_yield(
        PRICE,
        RENT,
        acquisition_costs=354_581,
        aidat_monthly=2500.0,
        vacancy_months=1.0,
        maintenance_pct_of_rent=5.0,
        emlak_vergisi_yearly=15_700.0,
        dask_yearly=2_500.0,
    )
    assert result["gross_annual_rent_tl"] == 384_000
    assert result["gross_yield_pct"] == 4.89
    assert result["collected_rent_tl"] == 352_000
    assert result["owner_costs"] == {
        "emlak_vergisi_tl": 15_700,
        "dask_tl": 2_500,
        "bakim_tl": 19_200,
        "bos_kalma_aidati_tl": 2_500,
        "toplam_tl": 39_900,
    }
    assert result["net_annual_income_tl"] == 312_100
    assert result["total_investment_tl"] == 8_204_581
    assert result["net_yield_pct"] == 3.80
    assert result["display"]["gross_yield"] == "%4,89"
    assert result["display"]["net_yield"] == "%3,80"


def test_vacancy_outside_a_year_is_refused() -> None:
    with pytest.raises(ValueError, match="vacancy_months"):
        calc.rental_yield(
            PRICE, RENT, acquisition_costs=0, aidat_monthly=None, vacancy_months=13,
            maintenance_pct_of_rent=5, emlak_vergisi_yearly=0, dask_yearly=0,
        )


def test_payback_gross_net_and_with_rent_growth() -> None:
    result = calc.payback(
        PRICE, RENT, net_annual_income=312_100, total_investment=8_204_581, annual_rent_growth_pct=30.0
    )
    assert result["gross_payback_years"] == 20.4
    assert result["net_payback_years"] == 26.3
    assert result["net_payback_with_growth_years"] == 8.3
    expected = math.log(1 + 8_204_581 * 0.3 / 312_100) / math.log(1.3)
    assert abs(expected - 8.326) < 1e-3


def test_payback_with_zero_growth_equals_the_net_payback() -> None:
    result = calc.payback(PRICE, RENT, net_annual_income=312_100, total_investment=8_204_581, annual_rent_growth_pct=0.0)
    assert result["net_payback_with_growth_years"] == result["net_payback_years"]


def test_payback_cannot_be_computed_from_a_loss() -> None:
    result = calc.payback(PRICE, RENT, net_annual_income=-1000, total_investment=8_000_000, annual_rent_growth_pct=10)
    assert result["net_payback_years"] is None
    assert result["net_payback_with_growth_years"] is None
    assert "hesaplanamaz" in result["display"]["net_payback"]


def test_emlak_vergisi_is_estimated_high_when_missing() -> None:
    istanbul = owner_costs(InvestmentInputs(il="İstanbul"), RATES, PRICE)
    assert istanbul["emlak_vergisi"].value == pytest.approx(15_700.0)  # binde 2
    assert istanbul["emlak_vergisi"].basis == "tahmin"
    elsewhere = owner_costs(InvestmentInputs(il="Bolu"), RATES, PRICE)
    assert elsewhere["emlak_vergisi"].value == pytest.approx(7_850.0)  # binde 1
    # Unknown province: the higher rate, because the figure is presented as an upper bound.
    unknown = owner_costs(InvestmentInputs(), RATES, PRICE)
    assert unknown["emlak_vergisi"].value == pytest.approx(15_700.0)
    entered = owner_costs(InvestmentInputs(emlak_vergisi_yearly_tl=4_200), RATES, PRICE)
    assert entered["emlak_vergisi"].value == 4_200 and entered["emlak_vergisi"].basis == "ilan"


def test_metro_provinces_match_however_they_are_typed() -> None:
    assert is_metro("İstanbul") and is_metro("istanbul") and is_metro("ISTANBUL")
    assert is_metro("Şanlıurfa") and is_metro("Eskişehir")
    assert is_metro("Bolu") is False
    assert is_metro(None) is None


# ======================================================================================
# Financing
# ======================================================================================


def test_mortgage_annuity_matches_the_hand_computation() -> None:
    result = calc.mortgage(PRICE, RATES, down_payment_pct=50, monthly_rate_pct=2.85, term_months=120, monthly_rent=RENT)
    assert result["loan_tl"] == 3_925_000
    assert result["ltv_capped"] is False
    assert result["ltv_cap_pct"] == 70.0  # 7-10 M TL band
    assert result["monthly_payment_tl"] == 115_837
    assert result["total_paid_tl"] == 13_900_496
    assert result["total_interest_tl"] == 9_975_496
    assert result["annual_compound_rate_pct"] == 40.10
    assert result["rent_covers_payment_pct"] == 27.6  # 32.000 / 115.837,47
    assert result["max_allocation_fee_tl"] == 19_625


def test_mortgage_present_value_of_payments_is_the_loan() -> None:
    result = calc.mortgage(1_000_000, RATES, down_payment_pct=20, monthly_rate_pct=3.1, term_months=60)
    r = 0.031
    payment = 800_000 * r / (1 - (1 + r) ** -60)
    present_value = sum(payment / (1 + r) ** k for k in range(1, 61))
    assert present_value == pytest.approx(800_000, rel=1e-9)
    assert result["monthly_payment_tl"] == calc.money(payment)


def test_mortgage_is_capped_at_the_bddk_band() -> None:
    # 25 M TL sits above 20 M: at most 40 % may be borrowed, so a 20 % down payment becomes 60 %.
    result = calc.mortgage(25_000_000, RATES, down_payment_pct=20, monthly_rate_pct=2.85, term_months=120)
    assert result["ltv_capped"] is True
    assert result["loan_to_value_pct"] == 40.0
    assert result["down_payment_pct"] == 60.0
    assert result["loan_tl"] == 10_000_000
    assert any("BDDK" in note for note in result["notes"])


def test_mortgage_rejects_an_annual_rate_passed_as_monthly() -> None:
    with pytest.raises(ValueError, match="MONTHLY"):
        calc.mortgage(PRICE, RATES, down_payment_pct=50, monthly_rate_pct=38.0, term_months=120)


def test_zero_interest_loan_is_a_straight_division() -> None:
    result = calc.mortgage(1_000_000, RATES, down_payment_pct=50, monthly_rate_pct=0, term_months=100)
    assert result["monthly_payment_tl"] == 5_000
    assert result["total_interest_tl"] == 0


# ======================================================================================
# Holding
# ======================================================================================


def test_projection_nominal_and_real() -> None:
    result = calc.projection(
        1_000_000,
        years=2,
        acquisition_costs=0,
        net_income_year1=50_000,
        price_growth_pct=10,
        rent_growth_pct=0,
        inflation_pct=10,
        exit_costs_pct=0,
    )
    assert result["final_value_tl"] == 1_210_000
    assert result["cumulative_net_rent_tl"] == 100_000
    assert result["final_wealth_tl"] == 1_310_000
    assert result["nominal_total_return_pct"] == 31.0
    assert result["nominal_annual_return_pct"] == pytest.approx(14.46, abs=0.005)  # √1,31 − 1
    assert result["real_total_return_pct"] == pytest.approx(8.26, abs=0.005)  # 1,31 / 1,21 − 1
    assert result["real_annual_return_pct"] == pytest.approx(4.05, abs=0.005)
    assert [row["value_tl"] for row in result["rows"]] == [1_100_000, 1_210_000]


def test_projection_subtracts_exit_costs_and_grows_rent() -> None:
    result = calc.projection(
        1_000_000,
        years=2,
        acquisition_costs=50_000,
        net_income_year1=40_000,
        price_growth_pct=0,
        rent_growth_pct=50,
        inflation_pct=0,
        exit_costs_pct=4.4,
    )
    assert result["cumulative_net_rent_tl"] == 100_000  # 40.000 + 60.000
    assert result["exit_costs_tl"] == 44_000
    assert result["final_wealth_tl"] == 1_000_000 - 44_000 + 100_000
    assert result["initial_outlay_tl"] == 1_050_000


# ======================================================================================
# Price per m²
# ======================================================================================


def test_price_per_m2_compares_each_benchmark_on_its_own_basis() -> None:
    benches = [
        calc.Benchmark("Kadıköy brüt ortalama", 55_000, "brut", "https://example.com/a", "A", "2026-09"),
        calc.Benchmark("Kadıköy net ortalama", 70_000, "net", "https://example.com/b", "B", "2026-08"),
    ]
    result = calc.price_per_m2(PRICE, net_m2=120, gross_m2=135, benchmarks=benches)
    assert result["tl_per_net_m2"] == 65_417
    assert result["tl_per_gross_m2"] == 58_148
    gross, net = result["comparisons"]
    assert gross["property_basis"] == "brut" and gross["difference_pct"] == 5.7
    assert net["property_basis"] == "net" and net["difference_pct"] == -6.5
    assert gross["source"] == {"url": "https://example.com/a", "title": "A", "date": "2026-09"}
    assert gross["flag"] is None and net["flag"] is None


def test_price_per_m2_flags_a_basis_it_had_to_substitute() -> None:
    bench = calc.Benchmark("brüt ortalama", 55_000, "brut", "https://x", "X", "2026")
    result = calc.price_per_m2(PRICE, net_m2=120, gross_m2=None, benchmarks=[bench])
    assert result["comparisons"][0]["property_basis"] == "net"
    assert "pahalı görünür" in result["comparisons"][0]["flag"]


def test_price_per_m2_needs_an_area() -> None:
    with pytest.raises(ValueError, match="neither net nor gross"):
        calc.price_per_m2(PRICE, net_m2=None, gross_m2=None, benchmarks=[])


# ======================================================================================
# Rates
# ======================================================================================


def test_ltv_bands_of_the_2026_decision() -> None:
    assert RATES.ltv_cap_pct(4_000_000) == 90.0
    assert RATES.ltv_cap_pct(5_000_000) == 90.0
    assert RATES.ltv_cap_pct(6_500_000) == 80.0
    assert RATES.ltv_cap_pct(9_999_999) == 70.0
    assert RATES.ltv_cap_pct(15_000_000) == 50.0
    assert RATES.ltv_cap_pct(30_000_000) == 40.0


def test_operator_overrides_apply_and_bad_ones_are_dropped() -> None:
    rates = load_rates(
        '{"doner_sermaye_tl": 7200, "mortgage_term_months": "180", "nonsense": 1, '
        '"commission_pct": "abc", "ltv_bands": [{"up_to_tl": null, "max_ltv_pct": 75}]}'
    )
    assert rates.doner_sermaye_tl == 7200.0
    assert rates.mortgage_term_months == 180
    assert rates.commission_pct == CostRates().commission_pct  # unparsable: default kept
    assert rates.ltv_bands == (LtvBand(None, 75.0),)
    assert load_rates("not json") == CostRates()


def test_stale_sources_are_named() -> None:
    from datetime import date

    assert RATES.stale_keys(today=date(2026, 10, 6)) == []
    assert "ltv" in RATES.stale_keys(today=date(2027, 6, 1))
