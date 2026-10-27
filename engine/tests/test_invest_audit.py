"""The bench's number checker: what it reads as a number, and what it accepts as traced."""

from __future__ import annotations

from biosphere.invest.audit import Allowed, advice_flags, audit_answer, extract_numbers


def values(text: str) -> list[float]:
    return [n.value for n in extract_numbers(text)[0]]


def test_turkish_and_english_numerals() -> None:
    assert values("Fiyat 7.850.000 TL, kira 32.000 TL") == [7_850_000, 32_000]
    assert values("brüt getiri %4,89; net 3,80%") == [4.89, 3.80]
    assert values("yaklaşık 7,85 milyon TL ve 1,2 milyar") == [7_850_000, 1_200_000_000]
    assert values("7,850,000 TL at 2.85% a month") == [7_850_000, 2.85]
    assert values("20,4 yıl") == [20.4]


def test_dates_layouts_years_links_and_list_markers_are_not_quantities() -> None:
    numbers, skipped = extract_numbers(
        "1. Eylül 2026 verisi, 17.09.2026 tarihli, 3+1 daire, "
        "[Endeksa](https://www.endeksa.com/tr/analiz/istanbul/kadikoy/2026) ve 2026-09"
    )
    assert numbers == []
    assert "3+1" in skipped and "17.09.2026" in skipped


def test_matching_is_at_the_precision_written() -> None:
    allowed = Allowed(facts=[7_850_000.0], tool=[4.8917, 20.4427])
    audit = audit_answer("Brüt getiri %4,9, amortisman 20 yıl, fiyat 7,85 milyon TL.", allowed)
    assert audit.ok and audit.traced["tool"] == 2 and audit.traced["facts"] == 1
    wrong = audit_answer("Brüt getiri %4,98.", allowed)
    assert wrong.untraced == ["%4,98"]


def test_cited_and_sourced_numbers_are_traced_separately() -> None:
    allowed = Allowed(cited=[38_000.0], sourced=[61_500.0])
    audit = audit_answer("Bölgede ortalama kira 38.000 TL; m² fiyatı 61.500 TL.", allowed)
    assert audit.traced["cited"] == 1 and audit.traced["sourced"] == 1 and audit.ok


def test_small_counts_are_trivial_but_large_unknowns_are_not() -> None:
    audit = audit_answer("2 kaynak buldum; bölgede 45.000 TL kira var.", Allowed())
    assert audit.traced["trivial"] == 1
    assert audit.untraced == ["45.000"]


def test_advice_wording_is_flagged() -> None:
    assert advice_flags("Bu daire iyi bir yatırım, kaçırmayın.")
    # The sentence the prompt prescribes for "should I buy?" declines to advise; not a flag.
    assert advice_flags("Alın ya da almayın diyemem; yatırım tavsiyesi veremem.") == []
    assert advice_flags("Almayın, bu fiyat yüksek.")
    assert advice_flags("Brüt getiri bölge ortalamasının %0,4 altında.") == []
