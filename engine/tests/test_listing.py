"""The listing copy: what it says with no model, and what it is allowed to say with one.

The interesting part of this feature is not that it produces text — it is that the text is
checked against the facts the system measured. So most of what follows is about the audit:
an invented area, a price, a claim about a metro station, all of which read perfectly well
and none of which anything here can know.

The model itself is faked. What a real one writes is not a property this suite can assert,
and the two things that must hold whatever it writes — that the room pictures reach it, and
that its answer is audited before anybody sees it — hold against a fake.
"""

# ruff: noqa: F811 - pytest fixtures are imported by name and then used as parameters
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from biosphere.ai import listing
from biosphere.ai.listing import ListingDraft, PropertyFacts, RoomFact

from .test_api import (  # noqa: F401 - fixtures
    API,
    _signed_in,
    client,
    create_property,
    ready_scene,
    settings,
    video_factory,
)


@pytest.fixture(autouse=True)
def no_model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module starts with NO writing model, whatever the machine has.

    NOT TIDINESS — this suite made real, billed calls without it. The engine reads
    ``engine/.env``, so on a developer's own machine ``llm.available()`` is true, and the two
    tests that assert the offline path went out to a hosted model instead: a hundred seconds
    slower, a few cents each run, and passing or failing depending on whose laptop it was.

    Patched at ``_key`` rather than at ``available`` so ``describe()`` agrees with it and the
    "no key" reason the panel shows is the one under test. The tests that want a model install
    a fake one over the top.
    """
    from biosphere.ai import llm

    monkeypatch.setattr(llm, "_key", lambda: "")


def facts(
    *,
    areas: tuple[float | None, ...] = (24.3, 14.1),
    ceiling: float | None = 2.7,
    address: str | None = "İstanbul",
    walk: bool = False,
) -> PropertyFacts:
    names = ["Salon", "Yatak Odası", "Mutfak"]
    rooms = tuple(
        RoomFact(
            scene_id=f"s{i}",
            name=names[i % len(names)],
            ready=True,
            has_panorama=True,
            photo_count=0,
            has_walkthrough=walk,
            area_m2=area,
            measured_fraction=0.8 if area is not None else 0.0,
        )
        for i, area in enumerate(areas)
    )
    return PropertyFacts(
        property_id="t1",
        title="Bahçeli Villa",
        address=address,
        rooms=rooms,
        camera_height_m=1.6,
        ceiling_height_m=ceiling,
        has_walkthrough=walk,
    )


# ======================================================================================
# 1. The answer with no model in it
# ======================================================================================


@pytest.mark.parametrize("locale", ["tr", "en"])
def test_offline_copy_names_the_rooms_and_survives_its_own_audit(locale: str) -> None:
    subject = facts()
    draft = listing.compose_offline(subject, locale, "professional")

    assert draft.title
    assert draft.description.count("\n\n") == 1, "two paragraphs, as the schema promises"
    for room in subject.rooms:
        assert room.name in draft.description
    # The composer only ever quotes numbers it was given, so the audit it is checked by must
    # have nothing to say about it. If this ever fails, the audit and the composer disagree
    # about what "grounded" means — and the audit is the one the model's answer goes through.
    assert listing.audit(draft, subject, locale) == []


@pytest.mark.parametrize("locale", ["tr", "en"])
def test_the_internal_name_is_never_the_headline(locale: str) -> None:
    """"7eylul — camlı kapılı, beyaz tonlarda kompakt oda" is a file name with a dash after it.

    The property's title is the agent's own label in their dashboard — a date, a code, a
    laptop — and it is not a fact about the flat. It belongs in neither path's copy.
    """
    subject = facts()
    draft = listing.compose_offline(subject, locale, "professional")
    blob = " ".join([draft.title, *draft.alt_titles, draft.description, *draft.highlights])
    assert subject.title not in blob
    assert "Bahçeli" not in blob


def test_offline_copy_states_no_area_when_no_room_was_measured() -> None:
    subject = facts(areas=(None, None), ceiling=None)
    draft = listing.compose_offline(subject, "tr", "professional")

    assert subject.total_area_m2 is None
    assert "m²" not in draft.description
    assert not any(h for h in draft.highlights if "m²" in h)


def test_the_total_is_the_measured_rooms_only() -> None:
    subject = facts(areas=(24.3, None, 14.1))
    # 38.4, not "the flat": the unmeasured room is not in it, and neither is any hallway.
    assert subject.total_area_m2 == pytest.approx(38.4)
    assert subject.measured_rooms and len(subject.measured_rooms) == 2


# ======================================================================================
# 2. The audit
# ======================================================================================


def draft_of(text: str, *, title: str = "Bahçeli Villa") -> ListingDraft:
    return ListingDraft(title=title, alt_titles=[], description=text, highlights=[])


def test_audit_accepts_a_figure_that_was_given_in_either_rounding() -> None:
    subject = facts(areas=(24.3, 14.1))  # rooms, and 38.4 as their sum
    for text in ("Salon yaklaşık 24 m².", "Salon yaklaşık 24,3 m².", "Toplam 38 m2."):
        assert listing.audit(draft_of(text), subject, "tr") == []


def test_audit_catches_an_invented_area() -> None:
    subject = facts(areas=(24.3, 14.1))
    warnings = listing.audit(draft_of("Ferah, 120 m² bir daire."), subject, "tr")
    assert any(w.startswith("ungrounded_area:") for w in warnings)


def test_audit_catches_a_price_and_an_unsupported_claim() -> None:
    subject = facts()
    warnings = listing.audit(
        draft_of("Metroya yürüme mesafesinde, 4.500.000 ₺."), subject, "tr"
    )
    assert "mentions_price" in warnings
    assert any(w == "unsupported_claim:metro" for w in warnings)


def test_audit_reads_the_title_and_the_highlights_too() -> None:
    subject = facts(areas=(24.3,))
    drafted = ListingDraft(
        title="Kadıköy'de 200 m² daire",
        alt_titles=[],
        description="Sade bir tur.",
        highlights=["Tapulu"],
    )
    warnings = listing.audit(drafted, subject, "tr")
    assert any(w.startswith("ungrounded_area:") for w in warnings)
    assert "unsupported_claim:tapu" in warnings


def test_the_digest_changes_when_the_property_does() -> None:
    before = facts().digest()
    assert facts().digest() == before, "the same facts must fingerprint the same"
    assert facts(areas=(24.3, 14.1, 9.0)).digest() != before
    assert facts(address="Ankara").digest() != before


def test_the_portal_rules_are_checked_even_when_the_agent_asked_for_them() -> None:
    """sahibinden pulls a listing whose text carries a phone number, an e-mail or a link."""
    subject = facts()
    drafted = draft_of(
        "Detaylı bilgi için 0532 111 22 33 ya da ofis@emlak.com — ilanlarımız emlakofisi.com'da."
    )
    warnings = listing.audit(drafted, subject, "tr", instructions="telefonumu da yaz")
    assert "contact_phone" in warnings
    assert "contact_email" in warnings
    assert "contact_link" in warnings


def test_ordinary_copy_does_not_trip_the_contact_rules() -> None:
    """The phone pattern must not fire on a room's measurements or on ordinary prose."""
    subject = facts()
    clean = draft_of(
        "Salon yaklaşık 24 m², tavan 2,5 m. Mutfak 2020'de yenilenmiş, zemin açık renk karo."
    )
    assert listing.audit(clean, subject, "tr") == []


def test_what_the_agent_asserts_is_not_flagged_back_at_them() -> None:
    """The audit polices the model, not the person who owns the property.

    An agent who types "asansörlü, 8. kat, 120 m²" is asserting facts about their own flat.
    Warning them about their own sentence is how a panel teaches somebody to ignore it.
    """
    subject = facts(areas=(24.3, 14.1))
    drafted = draft_of("Asansörlü binada, 120 m² kullanım alanı.")

    assert listing.audit(drafted, subject, "tr") != [], "unasked, both claims are the model's"
    assert listing.audit(drafted, subject, "tr", instructions="asansörlü, 120 m²") == []


def test_a_high_ceiling_claim_is_checked_against_the_measured_ceiling() -> None:
    """The cheap model headlined "yüksek tavan" over a measured 2.59 m. Twice."""
    ordinary = facts(ceiling=2.59)
    assert "ceiling_not_high:2.59" in listing.audit(
        draft_of("Ferah oda.", title="Yaklaşık 12,8 m² oda, yüksek tavan"), ordinary, "tr"
    )
    # Above the threshold the claim is the copy's to make, and this audit says nothing.
    tall = facts(ceiling=3.2)
    assert listing.audit(draft_of("Ferah oda.", title="Yüksek tavanlı daire"), tall, "tr") == []
    # And with no ceiling measured there is nothing to contradict.
    assert listing.audit(draft_of("Yüksek tavan."), facts(ceiling=None), "tr") == []


def test_a_call_is_priced_from_the_model_that_served_it() -> None:
    from biosphere.ai import llm

    # 10k in, 2k out on Opus 5: 10_000 * $5/M + 2_000 * $25/M.
    assert llm.price_of("claude-opus-5", 10_000, 2_000) == pytest.approx(0.10)
    # A dated snapshot prices like the alias it came from; a version suffix is not a price.
    assert llm.price_of("claude-opus-5-20260114", 10_000, 2_000) == pytest.approx(0.10)
    # And a model nobody has published a rate for here answers nothing rather than a guess.
    assert llm.price_of("some-other-model", 10_000, 2_000) is None


# ======================================================================================
# 3. The endpoint
# ======================================================================================


def test_listing_answers_without_a_model_and_says_so(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, _ = ready_scene(client, video_factory, monkeypatch)

    response = client.post(f"{API}/properties/{property['id']}/listing", json={"locale": "tr"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model_backed"] is False
    assert body["images_used"] == 0, "no model, no reason to render and bill for pictures"
    assert any(w.startswith("model_off:") for w in body["warnings"])
    assert body["title"] and body["description"]
    assert body["facts"]["room_count"] == 1
    assert body["facts"]["rooms"][0]["has_panorama"] is True


def test_listing_state_is_empty_before_anything_is_generated(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, _ = ready_scene(client, video_factory, monkeypatch)
    body = client.get(f"{API}/properties/{property['id']}/listing").json()
    assert body["latest"] is None
    assert body["available"] is False and body["reason"] in {"no_key", "no_sdk"}


def test_the_copy_is_cached_until_the_property_changes(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, scene = ready_scene(client, video_factory, monkeypatch)
    url = f"{API}/properties/{property['id']}/listing"

    first = client.post(url, json={"locale": "tr"}).json()
    assert first["cached"] is False
    assert client.post(url, json={"locale": "tr"}).json()["cached"] is True
    # Another language is another question, so it misses.
    assert client.post(url, json={"locale": "en"}).json()["cached"] is False
    # And so is "write it again", which is the whole point of the flag.
    assert client.post(url, json={"locale": "tr", "refresh": True}).json()["cached"] is False

    state = client.get(url).json()
    assert state["latest"] is not None and state["latest"]["cached"] is True

    # Renaming a room changes the facts, so the stored copy is no longer about this property.
    assert client.patch(f"{API}/scenes/{scene['id']}", json={"name": "Mutfak"}).status_code == 200
    assert client.get(url).json()["latest"] is None


def test_listing_404s_for_an_unknown_property(client: TestClient) -> None:
    assert client.post(f"{API}/properties/nope/listing", json={}).status_code == 404
    assert client.get(f"{API}/properties/nope/listing").status_code == 404


# ======================================================================================
# 4. With a model
# ======================================================================================


def fake_model(monkeypatch: pytest.MonkeyPatch, draft: ListingDraft) -> list[list[dict[str, Any]]]:
    """Answer every call with ``draft`` and record what was sent."""
    from biosphere.ai import llm

    seen: list[list[dict[str, Any]]] = []

    def parse(*, system: str, content: list[dict[str, Any]], schema: Any, **_: Any) -> llm.LlmResult:
        seen.append(content)
        return llm.LlmResult(
            value=draft, model="fake-model", input_tokens=10, output_tokens=20, elapsed_ms=1
        )

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "model_name", lambda: "fake-model")
    monkeypatch.setattr(llm, "parse", parse)
    return seen


def test_the_rooms_pictures_reach_the_model_and_its_answer_is_audited(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, scene = ready_scene(client, video_factory, monkeypatch)
    seen = fake_model(
        monkeypatch,
        ListingDraft(
            title="Işıklı salonu olan bir daire",
            alt_titles=["Ferah bir daire"],
            description="Salon güneş alıyor.\n\nToplam 300 m², metroya beş dakika.",
            highlights=["Işıklı salon"],
        ),
    )

    body = client.post(f"{API}/properties/{property['id']}/listing", json={"locale": "tr"}).json()

    assert body["model_backed"] is True and body["model"] == "fake-model"
    assert body["images_used"] == 1, "the room's panorama should have been rendered and sent"
    assert seen and any(block.get("type") == "image" for block in seen[0])
    facts_block = next(
        b["text"] for b in seen[0] if b.get("type") == "text" and b["text"].startswith("FACTS")
    )
    assert scene["name"] in facts_block

    # THE PROPERTY'S INTERNAL NAME IS NOT SENT AT ALL. Told merely not to use it, the model
    # put it in the headline and made it the subject of the opening sentence — "7eylul, beyaz
    # yüzeyleri ve sade hatlarıyla..." — because a name in the context is a name the copy can
    # reach for. The fix that holds is not giving it one.
    sent = "\n".join(b.get("text", "") for b in seen[0] if b.get("type") == "text")
    assert property["title"] not in sent

    # What the call cost travels back with the copy, not only into a log line.
    assert body["usage"]["input_tokens"] == 10 and body["usage"]["output_tokens"] == 20

    # The two things the model said that nothing here can know.
    assert any(w.startswith("ungrounded_area:300") for w in body["warnings"])
    assert "unsupported_claim:metro" in body["warnings"]
    # And the text is handed over exactly as written: the audit reports, it does not edit.
    assert "300 m²" in body["description"]


def test_a_model_that_fails_falls_back_rather_than_erroring(
    client: TestClient, video_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    property, _ = ready_scene(client, video_factory, monkeypatch)
    from biosphere.ai import llm

    def boom(**_: Any) -> Any:
        raise llm.LlmError("the writing model returned 401: invalid x-api-key")

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "parse", boom)

    response = client.post(f"{API}/properties/{property['id']}/listing", json={"locale": "tr"})
    assert response.status_code == 200, "a dead key is not a 500 for the person pressing the button"
    body = response.json()
    assert body["model_backed"] is False
    assert body["title"] and body["description"]
    assert any("401" in w for w in body["warnings"])
