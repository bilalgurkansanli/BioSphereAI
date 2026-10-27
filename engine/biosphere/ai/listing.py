"""The listing's title and description, written from the property itself.

An agent finishes a property and then has to write the advertisement: a headline and a paragraph
that describe a flat somebody else will read before deciding whether to come and see it. The
material for that is already here — the rooms and their names, the measured floor areas, the
ceiling height, whether there is a 3D walkthrough, and the rooms' own pictures — so this
module collects it, hands it to a model that can look at the pictures, and gets back copy
that is about *this* flat rather than about flats in general.

Three decisions shape everything below.

**The model is given facts, not asked for them.**
The measured areas, the room list and the ceiling height are computed here, from the same
outlines the floor plan is drawn from, and travel in the prompt as a block the model may
quote and may not extend. A vision model asked "how big is this room" will answer, fluently
and wrongly; asked "here is the room, it is 24.3 m², write about it" it has nothing to guess.

**Then the answer is checked against those facts.**
:func:`audit` reads every area figure back out of the finished text and refuses to recognise
one that was not given — and looks for the claims a picture cannot support at all: a price, a
deed, a monthly charge, a distance to a station. Anything it finds becomes a warning on the
response, next to the text, where the person about to publish it can see it. This is the same
argument as the room emptier's: a tool that invents furniture is worse than a tool that
leaves a soft wall, and a paragraph that invents a title deed is worse than a dull one. It is
not a filter, because silently rewriting somebody's copy is its own kind of lie.

**Without a model this still answers.**
No key configured, no client library, the service down: :func:`compose_offline` writes a
plain, true sentence out of the same facts and the response says ``model_backed: false``. It
is not good copy — it is a starting point in the right boxes with the right numbers in it,
which is what the field was before this feature existed.

The pictures
------------
One rectilinear view per room, aimed where the scene's own ``initial_view`` points, because
that is the direction the agent chose as the room's front. Rendering a window rather than
sending the equirect is not politeness: a 2:1 panorama shows a model a bent room with the
door stretched across a third of the frame, and it will describe the bend. Rooms with
photographs and no panorama send their first photograph, which is the same picture a visitor
sees. Rooms with neither send nothing and are still named in the facts.
"""

from __future__ import annotations

import base64
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from biosphere.core.logging import get_logger
from biosphere.core.models import Property, Scene
from biosphere.core.storage import LocalStorage

__all__ = [
    "TONES",
    "LOCALES",
    "RoomFact",
    "PropertyFacts",
    "RoomImage",
    "ListingDraft",
    "collect",
    "gather_images",
    "compose_offline",
    "audit",
    "write_listing",
]

log = get_logger("biosphere.ai.listing")

#: The voices on offer, and what each one asks the writer for.
#:
#: Four rather than a free-text style box: an agent picking from a list gets a repeatable
#: house voice across their listings, and every one of these is a sentence in the prompt that
#: somebody has read. A free instruction is still possible — ``instructions`` on the request —
#: but it is an addition to a tone, not a replacement for one.
TONES: dict[str, str] = {
    "professional": (
        "Measured and factual, the way a good estate agent writes: concrete nouns, no "
        "adjective without something in the picture behind it."
    ),
    "warm": (
        "Warm and human, addressed to somebody imagining living there. Still concrete; "
        "warmth comes from what the rooms are for, never from exclamation."
    ),
    "concise": (
        "Short and dense. Every sentence carries a fact. Prefer the shorter of two true "
        "sentences; no scene-setting opener."
    ),
    "premium": (
        "Restrained and high-end. Understated vocabulary, longer rhythm, no luxury cliches "
        "and no superlative the pictures do not earn."
    ),
}

LOCALES = ("tr", "en")

#: Panorama rungs this feature will read, smallest usable first.
#:
#: NOT ``empty_room.source_level``, which picks the *widest* rung because it is about to
#: deliver a panorama. This one is about to shrink whatever it reads to a 768-pixel window,
#: so decoding an 8K JPEG to do it would be seconds of CPU thrown away per room.
_LEVEL_PREFERENCE = ("sd", "hd", "uhd", "preview")

#: Side of the rendered room view, and its field of view.
#:
#: 768 px is about what a vision model resolves at this aspect; more is billed and then
#: downsampled. 95 degrees is a shade wider than the player's default so the answer is about
#: the room rather than about whatever the agent left in the middle of the frame.
_VIEW_PIXELS = 768
_VIEW_FOV_DEG = 95.0

#: Longest side of a room photograph sent as it is.
_PHOTO_MAX_SIDE = 1024

_JPEG_QUALITY = 82


# ======================================================================================
# Facts
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RoomFact:
    """One room, as the system actually knows it."""

    scene_id: str
    name: str
    ready: bool
    has_panorama: bool
    photo_count: int
    has_walkthrough: bool
    #: Floor area in m², from the room's measured outline. ``None`` when it was never
    #: measured — which is most rooms until somebody presses "measure" on the plan.
    area_m2: float | None
    #: Share of directions where a wall was actually found rather than interpolated. The
    #: honest headline for an area figure; below ~0.6 the number is a shape, not a survey.
    measured_fraction: float


@dataclass(frozen=True, slots=True)
class PropertyFacts:
    """Everything about one property that a sentence may be built on."""

    property_id: str
    title: str
    address: str | None
    rooms: tuple[RoomFact, ...]
    #: Height the areas are scaled by; the one assumption under every square metre here.
    camera_height_m: float
    #: Median ceiling height in metres, when the outlines fixed one.
    ceiling_height_m: float | None
    has_walkthrough: bool

    @property
    def room_count(self) -> int:
        return len(self.rooms)

    @property
    def measured_rooms(self) -> tuple[RoomFact, ...]:
        return tuple(r for r in self.rooms if r.area_m2 is not None)

    @property
    def total_area_m2(self) -> float | None:
        """Sum of the measured rooms, or ``None`` when none were measured.

        Deliberately NOT presented as the flat's size: it is the sum of the rooms that were
        measured, which is smaller than the flat whenever a room was skipped and excludes
        every corridor nobody stood in. The prompt says so, and so does the offline copy.
        """
        measured = self.measured_rooms
        if not measured:
            return None
        return round(sum(float(r.area_m2 or 0.0) for r in measured), 1)

    def digest(self) -> str:
        """Stable fingerprint of the facts, for the cache key.

        Room *order* is part of it: reordering the property changes which rooms lead the
        description, so the old copy is no longer the answer to the new question.
        """
        import hashlib

        # The property's own title is NOT in here, because it is no longer an input to the
        # copy: renaming a file must not throw away a paragraph that was never written from
        # the name. The address is, since the copy may repeat it.
        parts = [
            self.address or "",
            f"{self.camera_height_m:.3f}",
            f"{self.ceiling_height_m:.3f}" if self.ceiling_height_m else "-",
            "1" if self.has_walkthrough else "0",
        ]
        for room in self.rooms:
            parts.append(
                "|".join(
                    [
                        room.name,
                        "r" if room.ready else "-",
                        "p" if room.has_panorama else "-",
                        "w" if room.has_walkthrough else "-",
                        str(room.photo_count),
                        f"{room.area_m2:.1f}" if room.area_m2 is not None else "-",
                    ]
                )
            )
        return hashlib.blake2b("\n".join(parts).encode("utf-8"), digest_size=8).hexdigest()

    def allowed_areas(self) -> set[int]:
        """Every square-metre figure a sentence is allowed to contain, rounded.

        Both roundings of each number are admitted (24.3 may honestly be written as 24), and
        so is the total. Anything else in the finished text was invented.
        """
        allowed: set[int] = set()
        for room in self.measured_rooms:
            value = float(room.area_m2 or 0.0)
            allowed.update({int(math.floor(value)), int(math.ceil(value)), int(round(value))})
        total = self.total_area_m2
        if total is not None:
            allowed.update({int(math.floor(total)), int(math.ceil(total)), int(round(total))})
        return allowed


def _measured_fraction(outline: Any) -> float:
    try:
        sources = tuple(outline.profile.source)
    except Exception:  # noqa: BLE001 - an outline from an older cache
        return 0.0
    if not sources:
        return 0.0
    return round(sum(1 for s in sources if s != "none") / len(sources), 3)


def collect(property: Property, scenes: list[Scene], storage: LocalStorage) -> PropertyFacts:
    """Read the facts off the property. Never runs a model, never raises.

    The areas come from the cached outlines — the same files the floor plan is drawn from —
    rather than from the assembled plan, because placement is irrelevant here and a room
    nobody could place still has a floor.
    """
    from biosphere.plan import cache as plan_cache
    from biosphere.plan.types import CAMERA_HEIGHT_M

    rooms: list[RoomFact] = []
    ceilings: list[float] = []
    for scene in scenes:
        try:
            outline = plan_cache.read_outline(scene, storage)
        except Exception:  # noqa: BLE001 - a missing or corrupt outline is "not measured"
            outline = None
        area: float | None = None
        measured = 0.0
        if outline is not None and getattr(outline, "source", "none") == "model":
            try:
                area = round(float(outline.area_m2), 1)
            except Exception:  # noqa: BLE001
                area = None
            measured = _measured_fraction(outline)
            height = getattr(outline.profile, "ceiling_height", None)
            if height:
                ceilings.append(float(height) * float(outline.camera_height_m))
        walk = scene.walkthrough or {}
        rooms.append(
            RoomFact(
                scene_id=scene.id,
                name=scene.name,
                ready=scene.status == "ready",
                has_panorama=bool(scene.panorama),
                photo_count=_photo_count(scene, storage),
                has_walkthrough=str(walk.get("status") or "") == "ready",
                area_m2=area,
                measured_fraction=measured,
            )
        )

    ceiling = None
    if ceilings:
        ceilings.sort()
        ceiling = round(ceilings[len(ceilings) // 2], 2)
    return PropertyFacts(
        property_id=property.id,
        title=property.title,
        address=(property.address or "").strip() or None,
        rooms=tuple(rooms),
        camera_height_m=CAMERA_HEIGHT_M,
        ceiling_height_m=ceiling,
        has_walkthrough=any(r.has_walkthrough for r in rooms),
    )


def _photo_count(scene: Scene, storage: LocalStorage) -> int:
    """How many photographs the room has, counted on disk.

    From the directory rather than from the ``scene_photo`` table because this module is
    handed rows and no session, and the two cannot disagree about a file that is there.
    """
    directory = storage.scene_photos_dir(scene.property_id, scene.id)
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() != ".json")


# ======================================================================================
# Pictures
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RoomImage:
    """One room's picture, ready to travel."""

    scene_id: str
    room: str
    #: ``view`` (rendered from the panorama) or ``photo`` (the room's own photograph).
    kind: str
    jpeg: bytes

    def block(self) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(self.jpeg).decode("ascii"),
            },
        }


def _panorama_path(scene: Scene, storage: LocalStorage) -> Path | None:
    directory = storage.scene_dir(scene.property_id, scene.id)
    for level in _LEVEL_PREFERENCE:
        candidate = directory / f"pano_{level}.jpg"
        if candidate.is_file():
            return candidate
    return None


def _room_view(scene: Scene, storage: LocalStorage) -> bytes | None:
    """A rectilinear window onto the room, aimed where the scene opens."""
    import cv2

    from biosphere.ai import views

    path = _panorama_path(scene, storage)
    if path is None:
        return None
    pano = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if pano is None:
        return None
    view = scene.initial_view or {}
    yaw = math.degrees(float(view.get("yaw", 0.0) or 0.0))
    # Clamped rather than trusted: a pitch near a pole renders a picture of the ceiling,
    # which says nothing about the room. The player never stores one; an import might.
    pitch = max(-35.0, min(35.0, math.degrees(float(view.get("pitch", 0.0) or 0.0))))
    rot = views.rotation(yaw, pitch)
    intrinsics = views.intrinsics(_VIEW_PIXELS, _VIEW_FOV_DEG)
    window = views.render_view(pano, rot, intrinsics, _VIEW_PIXELS)
    ok, buffer = cv2.imencode(".jpg", window, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def _room_photo(scene: Scene, storage: LocalStorage) -> bytes | None:
    """The room's first photograph, shrunk to something worth sending."""
    import cv2

    directory = storage.scene_photos_dir(scene.property_id, scene.id)
    if not directory.is_dir():
        return None
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() != ".json")
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = _PHOTO_MAX_SIDE / float(max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
            )
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if ok:
            return buffer.tobytes()
    return None


def gather_images(
    scenes: list[Scene], storage: LocalStorage, *, cover_scene_id: str | None, limit: int
) -> list[RoomImage]:
    """Up to ``limit`` room pictures, the cover first and then in property order.

    Decoding is the expensive half, so the list is walked in the order the pictures will be
    used and stops at the limit, rather than rendering everything and choosing afterwards.
    """
    ordered = list(scenes)
    if cover_scene_id:
        ordered.sort(key=lambda s: 0 if s.id == cover_scene_id else 1)
    out: list[RoomImage] = []
    for scene in ordered:
        if len(out) >= max(0, int(limit)):
            break
        jpeg: bytes | None = None
        kind = "view"
        if scene.panorama:
            jpeg = _room_view(scene, storage)
        if jpeg is None:
            jpeg, kind = _room_photo(scene, storage), "photo"
        if jpeg is None:
            continue
        out.append(RoomImage(scene_id=scene.id, room=scene.name, kind=kind, jpeg=jpeg))
    return out


# ======================================================================================
# What the model is asked
# ======================================================================================


class ListingDraft(BaseModel):
    """The shape the writing model must answer in.

    EVERY FIELD IS REQUIRED, and the two lists are the reason this note exists. With defaults
    on them the schema said they were optional, and a `professional`-tone run came back with
    no highlights at all — not an error, just a field the model was told it could skip, and a
    panel with an empty column beside the copy. A default here is a suggestion the model
    reads; no default is a field the API will not let it omit.
    """

    title: str = Field(
        description="İlan başlığı, ready to paste into sahibinden.com. At most 65 characters, "
        "with the property's type and strongest concrete feature in the first words."
    )
    alt_titles: list[str] = Field(
        description="Exactly two alternative headlines, same rules as the title."
    )
    description: str = Field(
        description="İlan açıklaması: two or three short plain-text paragraphs, 90-160 words in "
        "total, separated by blank lines. No markdown, no HTML, no bullet characters."
    )
    highlights: list[str] = Field(
        description="Three to five bullet points, at most six words each, no full stops. The "
        "agent pastes these under the description as a list."
    )


_SYSTEM = """You write property-listing copy that an estate agent pastes straight into \
sahibinden.com — Turkey's listing portal — in its "İlan Başlığı" and "Açıklama" boxes. You are \
given the FACTS a 360-tour system measured about one property, and rendered views of its rooms.

TRUTH RULES. A sentence that breaks one of these can cost the agent the sale or the listing.

1. Never state anything that is not in the FACTS block, plainly visible in the pictures, or \
asserted by the agent in their own instruction. No price, no neighbourhood or transport claim, \
no building age, no floor number, no heating or fittings that are not shown, no legal or \
ownership status, no furniture that is not in the picture. If it is neither given nor visible, \
it does not go in the copy.
2. Every number comes from the FACTS block, exactly as given. Do not add areas up yourself and \
do not convert them. Areas are approximate: whenever you use one, hedge it in the language you \
are writing in ("yaklaşık 24 m²").
3. NEVER STATE THE PROPERTY'S ROOM COUNT OR ITS "3+1" DESIGNATION unless the agent gave it in \
their instruction. The FACTS block says how many rooms were photographed, which is not the same \
claim as how many rooms the flat has, and "3+1" is a claim about a layout nobody measured.
4. Write about what the pictures actually show — light, layout, what a room opens onto, the \
condition and finish of what is visible, how the rooms relate to each other. That is the part a \
photograph can support, and it is what makes the copy about this property rather than about any \
property.
5. The address is given as the agent typed it. You may repeat it; you may not infer anything \
from it — not a district's reputation, not a distance, not a view.
6. A room the FACTS block marks as not photographed may be named, but you have not seen it — do \
not describe it.
7. WRITE ABOUT THE PROPERTY, NOT ABOUT THE TOUR. The reader is a buyer looking at an \
advertisement; they are not looking at our software. So: never mention the tour, the panoramas, \
the photographs, the 3D walkthrough, the filming, the measuring, the phone or the camera \
height, and never say what is missing — that a room was not photographed, that a door was not \
shown, that an area is only part of the flat. Those lines in the FACTS block exist to keep YOU \
from overclaiming; they are not material for the copy. Write in the present tense about rooms \
that exist, not about pictures that were taken of them.
8. Hedging belongs on the number, not on the sentence. "Yaklaşık 24 m² salon" is right; \
"ölçümler yaklaşıktır ve telefonun tutuluş yüksekliğine dayanır" is a disclaimer, and \
disclaimers do not go in listing copy.

PORTAL RULES. sahibinden.com removes listings that break these.

9. No telephone number, no e-mail address, no web address, no link, no agency or company name, \
no contact detail of any kind — not in the title and not in the description. The portal shows \
the seller's own number in its own box; a number in the text gets the listing pulled.
10. No advertising slogans, no words in capitals for emphasis, no repeated punctuation, no \
keyword stuffing, no emoji, no exclamation marks, no rhetorical questions, no "hayalinizdeki \
ev" cliches, and no superlative the pictures do not earn.
11. The property has an internal name in the agent's own dashboard ("7eylul", "Laptop", a date, \
a code). YOU ARE NOT GIVEN IT AND MUST NOT INVENT ONE. The copy never opens with a name and \
never refers to the property by one; it opens with what the property is.

HOW A SAHIBINDEN HEADLINE WORKS.

12. The showcase truncates the headline hard — on some surfaces only the first few words are \
read at all — so what the property IS and its strongest concrete feature must both be in the \
opening words. Write it as ONE NATURAL NOUN PHRASE, the way an agent speaks: "Camlı bölmeli, \
yaklaşık 12,8 m² odalı daire". Not a comma-separated list of features with the type stranded at \
the front ("Daire, yaklaşık 12,8 m² oda ve yüksek tavan") — that reads like a database row. Aim \
for 50-65 characters and never exceed 65.
13. The description is plain text in two or three short paragraphs: what the property is and \
its best feature, then the rooms as a buyer would walk them, then what is worth knowing. No \
markdown, no HTML, no bullet characters — the highlights are delivered separately for the agent \
to paste as a list.
14. DO NOT END WITH A CALL TO ACTION. "Detaylı bilgi için iletişime geçebilirsiniz", "yerinde \
görmek için arayın", "kaçırmayın" — the portal already puts a contact button next to the \
listing, and a sentence asking for a call is a sentence not describing the property. End on \
the property."""


def _facts_block(facts: PropertyFacts, locale: str) -> str:
    """The FACTS the copy may be built on, as plain text.

    Plain text rather than JSON on purpose: it is read by a model that writes prose, and the
    qualifications ("measured from the panorama, approximate") have to travel *with* the
    numbers rather than sit in a schema comment nobody reads.
    """
    # The property's own title is DELIBERATELY ABSENT. It is the agent's file name for the
    # listing — "7eylul", "Laptop", a date, a code — and given it, the model used it: as the
    # headline, and as the subject of the opening sentence ("7eylul, beyaz yüzeyleri ve sade
    # hatlarıyla..."). Told not to use it, a model still leaks it, because a name in the
    # context is a name the copy can reach for. Not sending it is the fix that holds.
    lines = [
        "FACTS",
        f"- Address as the agent typed it: {facts.address or '(not given)'}",
        f"- Rooms photographed for this property: {facts.room_count}",
    ]
    for room in facts.rooms:
        bits = []
        if room.has_panorama:
            bits.append("360 panorama")
        if room.photo_count:
            bits.append(f"{room.photo_count} photograph(s)")
        if room.has_walkthrough:
            bits.append("3D walkthrough")
        if not bits:
            bits.append("not captured yet")
        area = ""
        if room.area_m2 is not None:
            confidence = "well measured" if room.measured_fraction >= 0.6 else "partly interpolated"
            area = f", floor area about {room.area_m2:g} m² ({confidence})"
        lines.append(f"  - {room.name}: {', '.join(bits)}{area}")
    total = facts.total_area_m2
    if total is not None:
        lines.append(
            f"- Sum of the measured rooms: about {total:g} m². This is NOT the flat's total "
            "floor area: rooms that were never measured, and every hallway nobody stood in, "
            "are missing from it. Do not present it as the property's size."
        )
    else:
        lines.append("- No room has been measured, so there is no area figure. Do not invent one.")
    if facts.ceiling_height_m:
        lines.append(f"- Ceiling height, measured: about {facts.ceiling_height_m:g} m")
    lines.append(
        f"- Every measurement above rests on one assumption: that the phone was held "
        f"{facts.camera_height_m:g} m above the floor. Treat the figures as approximate — and "
        "keep this line to yourself: hedge the number, never explain where it came from."
    )
    lines.append(f"- Write in: {'Turkish' if locale == 'tr' else 'English'}")
    return "\n".join(lines)


def _user_content(
    facts: PropertyFacts, images: list[RoomImage], *, locale: str, tone: str, instructions: str | None
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        label = "a view from its 360 panorama" if image.kind == "view" else "one of its photographs"
        content.append({"type": "text", "text": f"Room '{image.room}' — {label}:"})
        content.append(image.block())
    content.append({"type": "text", "text": _facts_block(facts, locale)})
    language = "Turkish" if locale == "tr" else "English"
    ask = [
        "",
        f"Tone: {TONES.get(tone, TONES['professional'])}",
        "",
        f"Write the sahibinden.com headline and description in {language}. The title is at most "
        "65 characters with the property and its strongest feature in the first words. Give two "
        "alternative titles and three to five short highlights.",
        "",
        # A SENTENCE COUNT RATHER THAN A WORD COUNT. Asked for "90 to 160 words" the cheap
        # model delivered 61, then 55 — it cannot count words it has not written yet, and the
        # instruction reads as a vague preference. Told how many sentences each paragraph
        # holds, it has something it can actually follow while writing.
        "Shape of the description — three paragraphs, one blank line between them:",
        "1. What the property is and its strongest concrete feature. Three sentences.",
        "2. The rooms as a buyer would walk them, with the finishes and what each opens onto. "
        "Four or five sentences.",
        "3. What else is worth knowing from the pictures and the facts. Two or three sentences.",
        "That comes to roughly 110-150 words. Shorter than 90 is not a listing, it is a caption.",
        "",
        "Write natural, idiomatic Turkish. Every phrase must be one a Turkish estate agent "
        "would actually write; no calques, no broken compounds.",
        "",
        # LAST, BECAUSE THE END OF THE PROMPT IS WHERE A SMALL MODEL LOOKS. Every line here is
        # a defect a real run produced, and all three came from the cheap model rather than
        # the expensive one: it wrote 61 words when asked for 90, invented "iyi hava
        # sirkülasyonu" from a photograph, and headlined "yüksek tavan" over a measured 2.59 m
        # while the paragraph underneath called the same ceiling standard. The rules above
        # already forbid all three; repeating them at the end is what makes them stick.
        "Before you answer, check your own draft against these three:",
        "- Length: the description really is 90 words or more. Count them.",
        "- Only what a photograph can carry. Air circulation, quiet, smell, a view, the "
        "neighbours, how warm a room is — none of these are in a picture. Cut the sentence.",
        "- No adjective that argues with a measured number: a 2.59 m ceiling is not a high "
        "ceiling, and nothing in the copy may contradict anything else in it.",
    ]
    if instructions and instructions.strip():
        # The agent's own instruction is an AUTHORISED SOURCE OF FACT, and that is the whole
        # reason the box exists: the floor, the aspect, the heating, "3+1" — everything a
        # panorama cannot measure and the person standing in the flat knows. Rule 1 admits it
        # explicitly, and `audit` exempts what the instruction asserts, so an agent who asks
        # for "asansörlü, 8. kat" does not get their own sentence flagged back at them.
        ask += [
            "",
            "The agent added this instruction. Anything it asserts about the property is theirs "
            "to assert: treat it as fact and use it. Follow the rest of it where it does not "
            "conflict with the rules above; where it does, the rules win:\n"
            f"{instructions.strip()}",
        ]
    content.append({"type": "text", "text": "\n".join(ask)})
    return content


# ======================================================================================
# Checking the answer
# ======================================================================================

#: Area figures in either language and either spelling of the unit.
_AREA_RE = re.compile(
    r"(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:m²|m2|metrekare|sq\s?m|square\s+met(?:re|er)s?)", re.I
)

#: Claims a picture cannot support and the facts never contain. Matched on a word boundary so
#: "kredi" does not fire inside "kredibilite".
_UNSUPPORTED: dict[str, tuple[str, ...]] = {
    "tr": (
        "tapu", "kredi", "aidat", "peşinat", "yatırım", "kira", "kiralık", "metro",
        "metrobüs", "otobüs", "dakika", "yürüme mesafesi", "okul", "hastane", "doğalgaz",
        "kombi", "asansör", "otopark", "site içinde", "deprem",
    ),
    "en": (
        "deed", "mortgage", "service charge", "deposit", "investment", "rent", "rental",
        "metro", "subway", "bus", "minutes", "walking distance", "school", "hospital",
        "earthquake", "central heating", "lift", "elevator", "parking", "gated",
    ),
}

#: Currency marks anywhere in the copy. A price is the one claim that is never derivable from
#: anything here and always consequential.
_MONEY_RE = re.compile(r"[₺$€£]|\b(?:tl|try|usd|eur|gbp)\b", re.I)

#: "Yüksek tavan" over a measured ceiling that is not high.
#:
#: THE ONE ADJECTIVE THIS AUDIT JUDGES, because it is the one with a number underneath it. The
#: cheaper model headlined "yüksek tavan" over a measured 2.59 m — twice, once with a
#: paragraph beneath it calling the same ceiling standard — and a prompt rule did not stop it.
#: Everything else in a listing is prose, and prose is the copy's job; this is a measurement
#: being contradicted, which is the audit's.
#: Matches the CLAIM ("yüksek tavan", "tavanı yüksek") and deliberately not the neutral
#: measurement phrase "tavan yüksekliği ... 2,5 m", which is how the honest sentence reads —
#: and which the offline composer writes, so a looser pattern flagged the engine's own copy.
_HIGH_CEILING_RE = re.compile(r"y[uü]ksek\s+tavan|tavan[ıi]\s+y[uü]ksek|high\s+ceiling", re.I)

#: Below this, in metres, a ceiling is ordinary. Turkish flats are built at 2.6-2.8 m, so the
#: claim only starts meaning something above it.
_HIGH_CEILING_M = 2.8

#: Contact details, which sahibinden.com does not allow in the title or the description: the
#: portal publishes the seller's own number in its own box, and a listing whose text carries a
#: phone number, an e-mail address or a link is pulled. These three are checked EVEN WHEN THE
#: AGENT ASKED FOR THEM, unlike every other warning here — an agent who types "telefonumu da
#: yaz" is asking for a listing that will not be published, and that is worth saying.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?90[\s.-]*)?(?:0[\s.-]*)?5\d{2}[\s.-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
_LINK_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|com\.tr|org\.tr|tr)\b", re.I)


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so "Metroya" matches "metro"."""
    lowered = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in lowered if not unicodedata.combining(c))


def audit(
    draft: ListingDraft,
    facts: PropertyFacts,
    locale: str,
    instructions: str | None = None,
) -> list[str]:
    """Every claim in the copy that the facts, the pictures and the agent do not carry.

    Returns warnings and does not touch the text. The agent is the one publishing it, so the
    honest move is to hand them the paragraph *and* the list of things in it nobody measured —
    not to quietly delete a sentence and leave them believing the rest was checked.

    THIS AUDITS THE MODEL, NOT THE AGENT. Whatever the agent asserted in their own instruction
    — the floor, the lift, the heating, "3+1", even a price — is theirs to assert about their
    own property, and warning them about their own sentence would train them to ignore the
    panel. So an area or a term that appears in ``instructions`` is exempt. The portal rules
    are not: a phone number gets the listing pulled whoever asked for it.
    """
    warnings: list[str] = []
    blob = "\n".join([draft.title, *draft.alt_titles, draft.description, *draft.highlights])
    asserted = _fold(instructions or "")

    allowed = facts.allowed_areas()
    for match in _AREA_RE.finditer(blob):
        value = float(match.group(1).replace(",", "."))
        if int(round(value)) in allowed or _fold(match.group(1)) in asserted:
            continue
        warnings.append(f"ungrounded_area:{match.group(0).strip()}")

    if _MONEY_RE.search(blob) and not _MONEY_RE.search(instructions or ""):
        warnings.append("mentions_price")

    folded = _fold(blob)
    for term in _UNSUPPORTED.get(locale, _UNSUPPORTED["en"]):
        needle = _fold(term)
        if needle in asserted:
            continue
        if re.search(rf"(?<!\w){re.escape(needle)}", folded):
            warnings.append(f"unsupported_claim:{term}")

    ceiling = facts.ceiling_height_m
    if ceiling and ceiling < _HIGH_CEILING_M and _HIGH_CEILING_RE.search(blob):
        warnings.append(f"ceiling_not_high:{ceiling:g}")

    # Portal rules. Not exempted by the instruction, and reported for the title as well as the
    # description because sahibinden refuses both.
    if _PHONE_RE.search(blob):
        warnings.append("contact_phone")
    if _EMAIL_RE.search(blob):
        warnings.append("contact_email")
    if _LINK_RE.search(blob):
        warnings.append("contact_link")

    if len(draft.title) > 70:
        warnings.append(f"title_too_long:{len(draft.title)}")
    words = len(draft.description.split())
    if words > 200:
        warnings.append(f"description_long:{words}")
    if not draft.description.strip():
        warnings.append("description_empty")
    return warnings


# ======================================================================================
# The answer with no model in it
# ======================================================================================


def _join(names: list[str], locale: str) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    tail = " ve " if locale == "tr" else " and "
    return ", ".join(names[:-1]) + tail + names[-1]


def compose_offline(facts: PropertyFacts, locale: str, tone: str) -> ListingDraft:
    """Copy built by arithmetic: true, dull, and available with no key and no network.

    It exists so the panel is never empty and never lies. It says what the property contains and
    what was measured, in the right boxes, and leaves the writing to the person who knows the
    flat — which is what they were doing before this feature existed.
    """
    ready = [r for r in facts.rooms if r.ready]
    names = [r.name for r in ready] or [r.name for r in facts.rooms]
    total = facts.total_area_m2
    # The property's internal name is not here either, for the same reason it is not in the
    # prompt: "7eylul — 4 oda" is a file name with a dash after it, not a headline. What is
    # left is what was actually measured, which is little but true.
    rooms_named = _join(names, locale)

    if locale == "tr":
        title = "Ölçülü ve fotoğraflı daire"
        if total is not None:
            title = f"Yaklaşık {total:g} m² ölçülen odalarıyla daire"
        first = f"Görüntülenen bölümler: {rooms_named}."
        if total is not None:
            first += (
                f" Ölçülen odaların toplam taban alanı yaklaşık {total:g} m²; ölçülmemiş "
                "bölümler ve koridorlar bu toplamın dışında."
            )
        if facts.ceiling_height_m:
            first += f" Tavan yüksekliği yaklaşık {facts.ceiling_height_m:g} m."
        second = (
            "Her bölüm 360° olarak gezilebiliyor"
            + (" ve 3B gezintide yürünebiliyor" if facts.has_walkthrough else "")
            + "; zemindeki iki nokta arasındaki mesafe ölçülebiliyor."
        )
        if facts.address:
            second += f" Adres: {facts.address}."
        highlights = []
        if total is not None:
            highlights.append(f"Ölçülen alan ~{total:g} m²")
        if facts.ceiling_height_m:
            highlights.append(f"Tavan ~{facts.ceiling_height_m:g} m")
        highlights.append("360° sanal tur")
        if facts.has_walkthrough:
            highlights.append("3B gezinti")
        alts = ["360° gezilebilen, ölçülü daire", "Ölçülü odalarıyla gezilebilir daire"]
    else:
        title = "A measured, photographed flat"
        if total is not None:
            title = f"A flat with about {total:g} m² of measured rooms"
        first = f"The rooms shown: {rooms_named}."
        if total is not None:
            first += (
                f" The measured rooms add up to about {total:g} m²; rooms that were not "
                "measured, and the hallways, are not in that figure."
            )
        if facts.ceiling_height_m:
            first += f" Ceiling height about {facts.ceiling_height_m:g} m."
        second = (
            "Every room can be explored in 360°"
            + (" and walked through in 3D" if facts.has_walkthrough else "")
            + "; any two points on the floor can be measured."
        )
        if facts.address:
            second += f" Address: {facts.address}."
        highlights = []
        if total is not None:
            highlights.append(f"About {total:g} m² measured")
        if facts.ceiling_height_m:
            highlights.append(f"Ceilings about {facts.ceiling_height_m:g} m")
        highlights.append("360° virtual tour")
        if facts.has_walkthrough:
            highlights.append("3D walkthrough")
        alts = ["A flat you can walk in 360°", "Measured rooms, explorable in 360°"]

    return ListingDraft(
        title=title[:120],
        alt_titles=alts,
        description=f"{first}\n\n{second}",
        highlights=highlights[:5],
    )


# ======================================================================================
# Entry point
# ======================================================================================


def _max_images() -> int:
    try:
        from biosphere.core.config import get_settings

        return int(get_settings().listing_max_images)
    except Exception:  # noqa: BLE001 - a broken .env must not cost the feature
        return 6


def write_listing(
    property: Property,
    scenes: list[Scene],
    *,
    storage: LocalStorage,
    locale: str = "tr",
    tone: str = "professional",
    instructions: str | None = None,
    max_images: int | None = None,
) -> Any:
    """Write one property's title and description. Returns ``schemas.ListingCopy``.

    NEVER RAISES FOR WANT OF A MODEL. A missing key, a missing client library, a refused or
    unreachable service all land in the same place: the offline composer, ``model_backed:
    false``, and a warning saying which of those it was. The caller is a button in a
    dashboard, and a button that returns a 500 because somebody's card expired is a worse
    product than one that returns a plain sentence and an explanation.
    """
    from biosphere.ai import llm
    from biosphere.api import schemas as sc

    locale = locale if locale in LOCALES else "tr"
    tone = tone if tone in TONES else "professional"
    started = time.perf_counter()

    facts = collect(property, scenes, storage)
    warnings: list[str] = []
    images: list[RoomImage] = []
    draft: ListingDraft | None = None
    model: str | None = None
    usage: Any = None

    if llm.available():
        limit = int(max_images if max_images is not None else _max_images())
        images = gather_images(scenes, storage, cover_scene_id=property.cover_scene_id, limit=limit)
        try:
            result = llm.parse(
                system=_SYSTEM,
                content=_user_content(
                    facts, images, locale=locale, tone=tone, instructions=instructions
                ),
                schema=ListingDraft,
            )
            draft = result.value  # type: ignore[assignment]
            model = result.model
            # Carried back to the panel rather than only logged. "What did that cost" is a
            # question the person pressing the button asks on the second press, not in a
            # server log a week later, and it is the number that decides whether this
            # feature runs on every listing or on the ones that matter.
            usage = sc.ListingUsage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
            )
        except llm.LlmError as exc:
            log.warning("listing.model_unavailable", property=property.id, error=str(exc)[:200])
            warnings.append(f"model_unavailable:{exc}")
    else:
        warnings.append(f"model_off:{llm.describe().get('reason')}")

    if draft is None:
        images = []
        draft = compose_offline(facts, locale, tone)
    else:
        warnings.extend(audit(draft, facts, locale, instructions))

    return sc.ListingCopy(
        title=draft.title.strip(),
        description=draft.description.strip(),
        highlights=[h.strip() for h in draft.highlights if h.strip()][:6],
        alt_titles=[t.strip() for t in draft.alt_titles if t.strip()][:3],
        locale=locale,
        tone=tone,
        model_backed=model is not None,
        model=model,
        images_used=len(images),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        cached=False,
        usage=usage,
        facts=facts_schema(facts),
        warnings=warnings,
    )


def facts_schema(facts: PropertyFacts) -> Any:
    """The facts, as the dashboard sees them. Returns ``schemas.ListingFacts``.

    Sent back with the copy rather than kept private, because the panel's most useful line is
    not the paragraph — it is "these are the numbers this was written from", which is also the
    line that tells an agent to go and measure the rooms first.
    """
    from biosphere.api import schemas as sc

    return sc.ListingFacts(
        address=facts.address,
        room_count=facts.room_count,
        measured_room_count=len(facts.measured_rooms),
        total_area_m2=facts.total_area_m2,
        ceiling_height_m=facts.ceiling_height_m,
        camera_height_m=facts.camera_height_m,
        has_walkthrough=facts.has_walkthrough,
        rooms=[
            sc.ListingRoomFact(
                scene_id=r.scene_id,
                name=r.name,
                ready=r.ready,
                has_panorama=r.has_panorama,
                photo_count=r.photo_count,
                has_walkthrough=r.has_walkthrough,
                area_m2=r.area_m2,
                measured_fraction=r.measured_fraction,
            )
            for r in facts.rooms
        ],
        digest=facts.digest(),
    )
