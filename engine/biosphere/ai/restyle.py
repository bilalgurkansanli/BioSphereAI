"""Virtual staging: the same room, furnished in a chosen style.

This is :mod:`biosphere.ai.empty_room` run backwards, and the inversion is not a figure of
speech — it is why the model that lost that argument wins this one.

``empty_room`` needs an *absence*, and :mod:`biosphere.ai.sd_inpaint` records in detail what
happens when a diffusion inpainter is asked for one: at an 18 % mask it drew a fresh made bed
into the bed-shaped hole, at 74 % it drew somebody else's flat, and guidance from 6 to 15 with
the furniture named in the negative prompt changed neither. The finding was that a generative
inpainter answers *"what belongs in this hole"* and cannot be asked for a nothing.

Staging asks exactly that question. "What belongs in this hole" **is the feature**, so the
failure mode is the mechanism and the measured objection does not transfer.

Holding the room still: two conditioners, and why one was not enough
---------------------------------------------------------------------
Everything hard about this feature is the word *same* in "the same room". A style prompt is a
strong instruction and a photograph is a weak one; asked for a rustic living room, the model
will happily deliver a beautiful rustic living room that is **somewhere else**.

Measured on a real bedroom window, in the order the faults were found:

1. **Depth alone, naively normalised — the room moved.** The doorway relocated, the walls
   leant, the proportions changed. The cause was not the ControlNet: it was the depth map.
   Depth Anything's raw output was stretched min-to-max over a frame containing an open
   doorway (very far) and the corner of a bed a foot from the lens (very near), so every
   surface that actually defines the room landed within twenty grey levels of the others. See
   :func:`depth_map` — clipping to percentiles instead fixed the doorway's position in three
   of five styles.

2. **Depth alone, correctly normalised — the room still drifted under a strong prompt.**
   ``rustik`` rebuilt the walls in stone; ``klasik`` put a landscape view through a window the
   flat does not have. Depth constrains *where surfaces are*. It says nothing about the
   straight lines that make a room read as that room — the ceiling join, the door frame, the
   line where the wall meets the floor — because those are edges, not distances.

3. **Both conditioners, and the room STILL drifted — because the photograph was not in the
   starting latent at all.** ``strength`` was left at the pipeline's default of 1.0, which
   means the masked region begins as pure noise: the two ControlNets were the only thing the
   room had to say, and a style prompt outvoted them. Dropping it to :data:`STRENGTH` was the
   single largest improvement in this module's history and it changed no other line. The
   sweep is recorded on that constant.

So there are two conditioners, and they carry different information:

``depth``   where every surface sits. Generated furniture stands ON the floor, at the scale
            the floor implies, and behind the things in front of it.
``canny``   the architectural lines, and ONLY those. Edges are computed on the photograph and
            then masked down to the pixels the segmenter calls building fabric, so the door
            frame and the ceiling join are locked while the old sofa's silhouette is not.
            Locking that silhouette too would be the opposite of the feature: the new
            furniture would be traced over the shape of the old.

That last masking step is the whole reason this module needs a segmenter for more than
protection, and it is why :class:`RoomAnalysis` exists — the depth map, the edges and the
protection mask are all properties of the ROOM, computed once and reused by every style.

What may change, and what may not — this is a product decision with legal weight
---------------------------------------------------------------------------------
This picture goes in front of a buyer, so the line is drawn where misrepresentation starts
rather than where the model happens to be good:

*May change* — furniture, textiles, rugs, lighting fixtures that stand rather than fit, wall
colour and finish, floor covering. These are a proposal about how the space could be dressed,
and every one of them is something a buyer can change after they move in.

*May not change* — the window openings and everything visible through them, the doors, the
structural columns and stairs, and the plumbed or wired fittings (sink, toilet, bath, shower,
radiator, hob, fitted counter). Those are facts about the property. The view from the window
in particular is a thing people pay money for; an invented one is a lie, not a style.

The protection is enforced by COMPOSITING, not by asking the model nicely. Those pixels come
back from the original photograph after the pass, feathered at the boundary, so no prompt or
sampler behaviour can put a different view out of the window.

The residual risk, stated rather than hidden: a curtain that covers a window entirely is
labelled ``curtain`` by the segmenter and is therefore restyled, and nothing in the depth map
says whether glass or wall stands behind it. In practice the model draws new curtains — there
is no window in the geometry to draw — but an operator who cannot accept that case at all
passes ``window_dressing=True`` and gets the old curtains back with the room restyled around
them.

Determinism, because a style rail is a set of toggles
------------------------------------------------------
The same reasoning as ``sd_inpaint``: a user clicks Modern, looks at Rustic, clicks Modern
again, and must get the same room back. The seed is derived from the photograph, the mask and
the style id (:func:`_seed_for`), never from a clock or a counter, and the noise is drawn on
the CPU so the seed means the same thing on every card.

Sizing, and why it is not a square
-----------------------------------
``sd_inpaint`` runs at a 512 square because that is where its checkpoint was trained, and it
records what asking for 1024 costs: the same room drawn twice. The constraint here is the same
one but the input is not square — a room photograph is 4:3 or 16:9, and squashing it to a
square bends every vertical in the room before the model ever sees it.

So the working size preserves the aspect ratio and matches the *area* of the training square
(:data:`WORK_PIXELS`), which is what actually governs how many latent tokens self-attention has
to relate. A 4:3 photograph therefore runs at about 592x448 rather than 512x512 — the same
token budget, the same room shape.

Costs, measured
----------------
See ``scripts/bench_restyle.py``, which is where the numbers in this module come from and
which writes the results out to be looked at, because "does this look like a room somebody
lives in, and is it still this room" is not a number.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from biosphere.core.logging import get_logger

__all__ = [
    "Style",
    "STYLES",
    "STYLE_IDS",
    "style_by_id",
    "available",
    "is_loaded",
    "reset",
    "describe",
    "depth_map",
    "structure_edges",
    "analyse_room",
    "read_room",
    "RoomMasks",
    "feathered_alpha",
    "RestyleMode",
    "REPLACE_STRENGTH",
    "restyle_photo",
    "RoomAnalysis",
    "RestyleOutput",
    "WORK_PIXELS",
    "DEPTH_CONTROLNET",
    "CANNY_CONTROLNET",
    "DEPTH_MODEL",
    "CONTROL_SCALES",
    "STRENGTH",
]

log = get_logger("biosphere.ai.restyle")


# ======================================================================================
# The styles
# ======================================================================================


@dataclass(frozen=True, slots=True)
class Style:
    """One decorating look, split by WHO decides what.

    The first version of this dataclass carried a single ``prompt`` naming both the style and
    the furniture — "low grey sofa, walnut sideboard, matte black floor lamp". That is the
    right shape for furnishing an empty room and the wrong shape for restyling a furnished
    one, because the room already decides what furniture it contains. Asked for a sofa in a
    bedroom, the model draws a sofa into the bed's place.

    So the style now owns only :attr:`palette` — materials, colours, finishes — and the NOUNS
    come from the segmenter, which can see that this room has a bed and a wardrobe in it.
    :attr:`staging` is the fallback list, used only when there is nothing in the room to
    restyle and the question really is "what furniture belongs in an empty space".
    """

    id: str
    label_tr: str
    label_en: str
    #: Materials, colours and finishes. No nouns for furniture — see the class docstring.
    palette: str
    #: Furniture to invent when the room is empty. Only ever used in ``stage`` mode.
    staging: str
    #: Added to the shared negative. Mostly used to push a style away from its neighbour —
    #: without it Scandinavian and Modern converge on the same pale grey room.
    negative_extra: str = ""


#: What every style asks for on top of its own description.
#:
#: SHORT BECAUSE IT HAS TO BE, AND THAT IS A HARD LIMIT RATHER THAN A STYLE CHOICE. SD 1.5's
#: text encoder is CLIP ViT-L/14 with 77 positions, and diffusers does not error on a longer
#: prompt — it TRUNCATES and logs. The first version of this module ran at 81-93 prompt tokens
#: and 124-127 negative tokens, so roughly half of every negative prompt, including every
#: anti-duplication term in it, was silently discarded on every pass. ``test_restyle.py``
#: asserts the budget now; anything added here has to be paid for by removing something.
_PROMPT_SUFFIX = "interior photo, natural daylight, realistic shadows, photorealistic"

#: Prefixed to the object list so the model reads it as furniture to draw rather than as a
#: caption of what is already there.
_REPLACE_LEAD = "new"

#: What no style wants.
#:
#: This task's own failure list rather than a generic quality incantation. "empty unfurnished
#: room" is there because an inpainter handed a bare room will happily give it back; the
#: duplication terms are the failure ``sd_inpaint`` measured at 1024 px, which conditioning
#: suppresses but does not forbid; "floating furniture" and "wrong scale" are what a model
#: without depth produces and stay named even with it.
#:
#: THE WINDOW TERMS ARE THE MOST IMPORTANT LINE HERE AND THEY ARE NOT ENOUGH ON THEIR OWN.
#: Measured: on a crop with a large blank white wall and no window in it, three of five styles
#: invented one — because "interior photo, natural daylight" plus a big bright rectangle is
#: what a window looks like to a model trained on estate-agent photographs, and neither
#: conditioner can forbid it (a window is as flat as the wall it is in, and a white wall has no
#: edges to contradict). Naming it in the negative reduces it. What actually bounds it is the
#: protection composite, which is why a REAL window is never at risk: it is masked out of the
#: pass and pasted back afterwards. An invented one on a blank wall remains a known limit of
#: this feature and is the reason the UI labels every render as AI-generated.
_NEGATIVE_BASE = (
    "empty unfurnished room, new window, extra window, extra door, "
    "repeated furniture, floating furniture, wrong scale, warped perspective, bent walls, "
    "clutter, text, watermark, person, "
    "cgi, 3d render, cartoon, lowres, blurry, distorted"
)


STYLES: tuple[Style, ...] = (
    Style(
        id="modern",
        label_tr="Modern",
        label_en="Modern",
        palette=(
            "modern minimalist design, low clean lines, grey fabric, walnut wood, "
            "matte black metal, warm white and charcoal palette, matte finishes, uncluttered"
        ),
        staging="grey sofa, walnut sideboard, low coffee table, large neutral rug, black floor lamp",
        negative_extra="ornate carving, floral, gilded, antique, exposed brick, log cabin, stone wall",
    ),
    Style(
        id="iskandinav",
        label_tr="İskandinav",
        label_en="Scandinavian",
        palette=(
            "scandinavian design, pale oak wood, light linen fabric, knitted wool, "
            "white beige and pale wood palette, soft rounded shapes, airy and bright"
        ),
        staging="light linen sofa, pale oak table, jute rug, tall houseplant, paper pendant lamp",
        negative_extra="dark heavy furniture, ornate carving, gilded, exposed brick, stone wall, moody dark",
    ),
    Style(
        id="rustik",
        label_tr="Rustik",
        label_en="Rustic",
        palette=(
            "rustic farmhouse design, reclaimed solid wood, aged patina, woven textiles, "
            "terracotta and olive palette, handmade texture, warm earth tones"
        ),
        staging="reclaimed wood table, linen sofa with woollen blankets, woven basket, iron lamp",
        negative_extra="glossy plastic, chrome, neon, high tech, stone wall, log cabin, timber ceiling",
    ),
    Style(
        id="klasik",
        label_tr="Klasik",
        label_en="Classic",
        palette=(
            "classic elegant design, tufted upholstery, carved dark wood, polished brass, "
            "warm cream gold and deep brown palette, refined traditional detailing"
        ),
        staging="tufted beige sofa, carved wood console, persian rug, brass table lamp, framed painting",
        negative_extra="industrial, exposed concrete, neon, plastic, gaudy, gold walls, throne room",
    ),
    Style(
        id="loft",
        label_tr="Loft",
        label_en="Industrial loft",
        # NO "RUST" AND NO EDISON BULB. Both were in the first palette and on the reference
        # bedroom photograph together they drew a blanket glowing orange as if lit from inside:
        # "rust" reads to the model as a colour to apply and the bulb as a light to switch on.
        # Leather, steel and wood carry the style without either; the glow is named below.
        palette=(
            "industrial loft design, dark brown leather, black steel frame, reclaimed wood, "
            "charcoal grey and warm brown palette, matte raw materials"
        ),
        staging="brown leather sofa, black steel shelving, wood and metal table, black metal floor lamp",
        negative_extra="glowing, orange light, neon, pastel, floral, gilded, ornate carving, exposed brick wall",
    ),
)

STYLE_IDS: tuple[str, ...] = tuple(s.id for s in STYLES)

_BY_ID: dict[str, Style] = {s.id: s for s in STYLES}


def style_by_id(style_id: str) -> Style | None:
    return _BY_ID.get(str(style_id).strip().lower())


# ======================================================================================
# What the segmenter's vocabulary means to this feature
# ======================================================================================

#: ADE20K classes matched BY NAME rather than by index.
#:
#: The ids are stable in the dataset but this module reads them out of whichever checkpoint is
#: loaded, and a mismatch would be silent and severe: an off-by-a-few list would protect
#: "sconce" while restyling "windowpane", and the first anyone would know is a listing photo
#: with an invented view out of the window. Names are also the only form in which this list can
#: be reviewed by somebody who is not holding the dataset's index table.
_PROTECTED_NAMES: frozenset[str] = frozenset(
    {
        # -- the opening itself, and the glass in it -----------------------------------
        "windowpane",
        "window",
        "glass",
        "door",
        "screen door",
        "double door",
        # -- structure a buyer is buying ----------------------------------------------
        "column",
        "pillar",
        "stairs",
        "stairway",
        "step",
        "railing",
        "bannister",
        "fireplace",
        "escalator",
        # -- plumbed, wired or fitted: an estate agent's restyle keeps these ------------
        "sink",
        "toilet",
        "bathtub",
        "shower",
        "radiator",
        "stove",
        "oven",
        "dishwasher",
        "washer",
        "refrigerator",
        "microwave",
        "kitchen island",
        "countertop",
        "counter",
        "counter top",
        # -- the world outside the window ---------------------------------------------
        "sky",
        "building",
        "house",
        "skyscraper",
        "tree",
        "palm",
        "plant",  # only ever reaches here through a window; indoor pots read as "pot"
        "grass",
        "field",
        "hill",
        "mountain",
        "rock",
        "earth",
        "sand",
        "water",
        "sea",
        "river",
        "lake",
        "waterfall",
        "road",
        "sidewalk",
        "path",
        "bridge",
        "tower",
        "car",
        "truck",
        "bus",
        "van",
        "boat",
        "ship",
        "land",
        "awning",
        "fence",
    }
)

#: Curtains, blinds and shutters. PROTECTED BY DEFAULT, which is a correction rather than a
#: preference: the first version left them editable on the reasoning that window dressing is a
#: decorating choice, and the first real 360 restyle put a WARDROBE across a curtained window.
#: A curtain is where a window is, so anything drawn over one is drawn over glass — the exact
#: misrepresentation the protection rules exist to prevent. ``window_dressing=False`` opts back
#: out for a room whose curtains genuinely need restyling and whose windows are visible anyway.
_WINDOW_DRESSING_NAMES: frozenset[str] = frozenset({"curtain", "blind", "shutter", "awning"})

#: The room's own surfaces.
#:
#: PROTECTED IN ``replace`` MODE, which is the default. The wall a bed stands against is not a
#: decorating choice, it is the property; and leaving it editable is what let a wardrobe be
#: drawn across a window. In ``stage`` mode a band of wall above the floor becomes editable
#: because furniture has to be put somewhere — see :func:`_furnishable_volume`.
#:
#: Their EDGES condition the pass in both modes: the ceiling join and the wall/floor line are
#: the lines that make a room read as that room.
_SURFACE_NAMES: frozenset[str] = frozenset({"wall", "floor", "ceiling", "flooring"})

#: Floor classes alone. ``stage`` mode grows the furnishable volume upward from these.
_FLOOR_NAMES: frozenset[str] = frozenset({"floor", "flooring", "rug", "carpet"})

#: Below this solid share of the frame the room counts as EMPTY and the pass switches from
#: replacing furnishings to inventing them. Six per cent is about one piece of furniture seen
#: across a room; under it there is nothing to restyle and refusing would be a worse answer
#: than staging.
_FURNISHED_FRACTION = 0.06

#: How far the furnishable volume rises above the floor in ``stage`` mode, as a fraction of
#: frame height. Furniture stands on the floor and has a height; a band this deep covers a
#: wardrobe seen across a room and stops well short of the ceiling — which is the point, since
#: an editable upper wall is where the model invents a window that the property does not have.
_STAGE_RISE = 0.38

#: ADE20K names that read badly in a prompt. Everything else is used verbatim.
_OBJECT_RENAMES: dict[str, str] = {
    "television receiver": "television",
    "chest of drawers": "chest of drawers",
    "swivel chair": "office chair",
    "coffee table": "coffee table",
    "pool table": "table",
    "crt screen": "screen",
    "ashcan": "bin",
    "plaything": "toy",
    "apparel": "clothes",
    "case": "display cabinet",
    "buffet": "sideboard",
}

#: At most this many nouns reach the prompt. CLIP has 77 tokens and the palette needs most of
#: them; past half a dozen objects the model stops placing any of them well anyway.
_MAX_OBJECTS = 6

#: The segmenter runs at this many pixels on the long side, whatever size the pass itself uses.
#:
#: NOT A QUALITY COMPROMISE, AND ``furniture_seg`` MEASURED IT: "a segmenter is deciding what a
#: thing IS, which does not improve with pixels the way redrawing it does, and Mask2Former costs
#: 0.83 s per window at this size against three times that at 1024." On the 360 path that is the
#: dominant cost — sixteen windows, most of the wall clock — and paying it at the render
#: resolution buys a mask boundary a few pixels crisper on a mask that is deliberately grown and
#: despeckled afterwards anyway.
_SEGMENT_PIXELS = 640

#: A furnishing under this share of the frame is not worth naming — it is a vase, and naming it
#: spends a token that a bed needs.
_MIN_OBJECT_FRACTION = 0.004

#: How far the protected region is grown before it is trusted.
#:
#: A semantic mask is accurate to a few pixels and wrong at exactly the place it matters — the
#: rim of a window frame, where a one-pixel error puts generated wall over real glass. Growing
#: it costs a thin ring of original photograph around each protected thing, which is the right
#: side to be wrong on.
_PROTECT_GROW = 0.006

#: Feather width of the composite boundary, as a fraction of the image's long side. A hard
#: edge between a real window and a generated wall reads as a cut-out; this makes it a join.
_PROTECT_FEATHER = 0.004


# ======================================================================================
# Models
# ======================================================================================

#: Structure conditioning. ``f1p`` is the corrected depth ControlNet for SD 1.5 — the
#: preceding ``control_v11p_sd15_depth`` was trained on a depth normalisation that does not
#: match what any modern estimator produces.
DEPTH_CONTROLNET = "lllyasviel/control_v11f1p_sd15_depth"

#: Line conditioning. See the module docstring for what this carries that depth does not.
CANNY_CONTROLNET = "lllyasviel/control_v11p_sd15_canny"

#: Monocular depth. Small rather than Large deliberately: this map is a *constraint on layout*
#: at latent resolution, not a product, and the Large model's extra fidelity lands entirely in
#: detail the 8x VAE downscale throws away. Small is 95 MB against 1.3 GB and runs in
#: milliseconds beside a multi-second diffusion pass.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

#: Percentiles the depth map is stretched between. See :func:`depth_map` for the measurement
#: that put them here rather than at 0 and 100.
_DEPTH_CLIP_LOW = 2.0
_DEPTH_CLIP_HIGH = 98.0

#: Canny thresholds, as PERCENTILES OF THIS IMAGE'S OWN GRADIENT rather than as fixed levels.
#:
#: Fixed thresholds were the first attempt and they produced an empty conditioner: at 76/178 on
#: a white-walled bedroom the edge map came back covering 0.37 % of the frame, with no line at
#: all where the wall met the ceiling — because that junction is white against off-white and
#: its gradient is a handful of levels. A conditioner that is blank is not a weak conditioner,
#: it is an absent one, and the pass reverted to depth alone with the fault invisible.
#:
#: The strongest few per cent of an interior's gradients ARE its architecture: door frames,
#: skirting, the ceiling join, window reveals. Asking for a percentile finds them in a dim
#: room and in a bright one, and it keeps the map's density roughly constant, which is what
#: makes a single conditioning scale meaningful across photographs.
_CANNY_HIGH_PERCENTILE = 96.0
#: Hysteresis floor, as a fraction of the high threshold. Canny follows a weak edge only when
#: it touches a strong one, so this is what lets a door frame stay whole where the light falls
#: off along it.
_CANNY_LOW_RATIO = 0.4
#: Target share of the frame the edge map should cover. Not enforced — reported, so
#: ``bench_restyle`` can show when a photograph produced nothing to hold on to.
_CANNY_TARGET_FRACTION = 0.02

#: Latent-token budget, expressed as an area so a non-square photograph keeps its shape.
#: 512x512 is the training resolution of every checkpoint ``sd_inpaint`` will find.
WORK_PIXELS = 512 * 512

#: Latent stride: both sides handed to the VAE must divide by this.
_STRIDE = 8

#: Sampling. UniPC converges in far fewer steps than the checkpoint's default PNDM at equal
#: quality, which matters when five styles are five passes.
#: Twenty rather than twenty-eight. UniPC is converged well before this on an inpainting
#: checkpoint, and eight steps is nearly a third of the wall clock on a feature whose 360 path
#: is sixteen passes. Swept with ``bench_restyle.py --steps``.
_STEPS = 20

#: Higher than ``sd_inpaint``'s 6.0, and for the opposite reason. That module wants the sampler
#: to stay near the photograph because it is asking for a continuation; this one is asking for
#: something that is not in the photograph at all, and under-guided it returns a politely
#: redecorated version of the empty room.
_GUIDANCE = 7.5

#: How hard each conditioner is imposed, in the order the pipeline is built: depth, then edges.
#:
#: Depth carries more of the burden because it constrains every pixel, while the edge map is
#: sparse by construction — it is a handful of architectural lines on black. Pushing the edges
#: higher starts to show them: the model traces the line rather than putting a wall behind it.
CONTROL_SCALES: tuple[float, float] = (0.75, 0.55)

#: How much of the original photograph survives into the starting latent.
#:
#: THIS IS THE KNOB THAT DECIDES WHETHER IT IS THE SAME ROOM, and leaving it at the pipeline's
#: default of 1.0 was the largest single fault in the first working version. At 1.0 the masked
#: region starts from PURE NOISE: the photograph contributes nothing at all except through the
#: conditioners, so the pass is text-to-image with a hint, and a strong style prompt is free to
#: build a different flat around the depth silhouette. That is exactly what was measured —
#: ``rustik`` rebuilt the walls in stone, ``klasik`` put a landscape through a window the
#: property does not have.
#:
#: Below 1.0 the denoiser starts from the ACTUAL ROOM with noise added, so wall positions, the
#: light falling through the real window and the room's own colour distribution are all in the
#: signal from the first step.
#:
#: SWEPT ON THE REAL BEDROOM, ``bench_restyle.py --strength 0.55 0.7 0.85 1.0``:
#:
#:   1.00   a different flat. Rustic came back as a stone-walled cabin with a lit fireplace;
#:          modern rearranged the room. Unusable, and this was the shipped default until the
#:          sweep was run.
#:   0.85   the room holds — doorway, cornice, wall angles all in place — with the strongest
#:          styling. Rustic begins cladding the wall in board, which is a *finish* and so
#:          inside what this feature is allowed to change, but it is the edge of it.
#:   0.70   the room holds and the style reads, with the original's clutter still faintly
#:          present. The safe end.
#:   0.55   the old room with a colour cast. The laptop and the bag on the bed are still
#:          legible; nobody would call this a restyle.
#:
#: 0.80 sits just inside 0.85's styling with more of 0.70's margin, and it is cheaper as well:
#: strength scales the number of denoising steps actually run, so a pass costs 6.3 s here
#: against 8.1 s at 1.0 (RTX 5050 laptop, 512-px latent, 28 nominal steps).
#:
#: THAT SWEEP WAS RUN WHEN THE WHOLE ROOM WAS EDITABLE, and it is why the number is cautious:
#: every point of strength was a point of licence to move the walls. In ``replace`` mode the
#: walls are not editable at all, so the licence does not exist and the caution costs styling
#: for nothing — see :data:`REPLACE_STRENGTH`.
STRENGTH = 0.80

#: Strength when only the furnishings are editable.
#:
#: Higher, and safely so: the model cannot move a wall it is not allowed to draw on. What it
#: can do is make the new bed look like the old bed, which is what a low strength buys — the
#: starting latent is the old furniture with noise on it — and it is the one thing this mode
#: must not do. At 0.95 the object in the hole is genuinely new; the mask keeps it in the same
#: place and at the same size, which is what "the same room" means here.
REPLACE_STRENGTH = 0.95

_lock = threading.Lock()
_state: dict[str, Any] = {
    "checked": False,
    "pipe": None,
    "device": None,
    "ref": None,
    "depth": None,
    "labels": None,
}


def _cuda_or_cpu() -> str:
    """The depth model is small enough to be worth running on a CPU; the diffusion pass is not."""
    from biosphere.ai import sd_inpaint

    return sd_inpaint.cuda_device() or "cpu"


# --------------------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------------------


def _load_depth() -> tuple[Any, Any, str] | None:
    """``(model, processor, device)`` for the depth estimator, or ``None``."""
    if _state["depth"] is not None:
        return _state["depth"]
    device = _cuda_or_cpu()
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except Exception as exc:  # noqa: BLE001 - transformers is an optional extra
        log.info("restyle.depth_unavailable", reason=f"transformers missing: {exc}")
        return None
    try:
        from biosphere.ai import sd_inpaint

        local_only = not sd_inpaint.downloads_allowed()
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, local_files_only=local_only)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, local_files_only=local_only)
        model.eval().to(device)
    except Exception as exc:  # noqa: BLE001 - a missing checkpoint is not an error
        log.info("restyle.depth_unavailable", model=DEPTH_MODEL, error=str(exc)[:160])
        return None
    _state["depth"] = (model, processor, device)
    log.info("restyle.depth_ready", model=DEPTH_MODEL, device=device)
    return _state["depth"]


def depth_map(image: np.ndarray) -> np.ndarray | None:
    """Inverse depth of one photograph as uint8, near = bright. ``None`` with no backend.

    Near-bright is the convention the depth ControlNet was trained on, and it is also what
    Depth Anything emits, so the two agree without a flip. Getting that backwards does not
    raise: it produces a room whose far wall is treated as the nearest surface, and the
    furniture is generated inside out.

    THE NORMALISATION IS THE WHOLE VALUE OF THIS FUNCTION, AND A PLAIN MIN-MAX DESTROYS IT.
    Measured on a real bedroom window: an open doorway (very far) and the corner of a bed a
    foot from the lens (very near) between them occupied the full range, and the wall, ceiling
    and floor — every surface whose recession is what actually defines the room — landed inside
    twenty grey levels of each other. Handed that, the ControlNet has almost nothing to hold
    and the model redecorated a DIFFERENT ROOM: the doorway moved, the walls leant, the
    proportions changed. The failure looked like weak conditioning and was really a flat map.

    Clipping to :data:`_DEPTH_CLIP_LOW` / :data:`_DEPTH_CLIP_HIGH` spends the range on the room
    instead of on its two extremes. The outliers saturate, which is exactly right: an open
    doorway is "as far as it gets" and a foreground corner is "as near as it gets", and neither
    needs gradation to be obeyed.
    """
    loaded = _load_depth()
    if loaded is None:
        return None
    model, processor, device = loaded

    import torch

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        prediction = model(**inputs).predicted_depth
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)
    prediction = torch.nn.functional.interpolate(
        prediction, size=(height, width), mode="bicubic", align_corners=False
    )
    raw = prediction[0, 0].float().cpu().numpy()
    low, high = (float(v) for v in np.percentile(raw, (_DEPTH_CLIP_LOW, _DEPTH_CLIP_HIGH)))
    if high - low < 1e-6:
        return np.zeros((height, width), np.uint8)
    return np.clip((raw - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Architectural lines
# --------------------------------------------------------------------------------------


def structure_edges(image: np.ndarray, structure: np.ndarray | None = None) -> np.ndarray:
    """Canny edges of the room's ARCHITECTURE only, as a uint8 line image.

    ``structure`` is a mask of the pixels whose edges may be kept — walls, floor, ceiling,
    window frames, doors, fittings. Pass ``None`` and every edge is kept, which is the right
    fallback with no segmenter and the wrong answer with one.

    WHY THE MASK MATTERS MORE THAN THE THRESHOLDS. Canny run on the raw photograph traces the
    old sofa as faithfully as it traces the door frame, and a ControlNet handed that draws the
    new sofa *along the old one's outline*. The result is a room in the right style with the
    previous owner's furniture layout embossed into it — a failure that looks like a bad model
    and is really a bad conditioning image.

    The mask is eroded before it is applied because a furnishing's silhouette lies exactly on
    the boundary between it and the wall behind it, so that boundary belongs to both and
    keeping it would keep the silhouette.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A little blur first: Canny on a phone photograph at full resolution finds sensor noise
    # and JPEG blocking, and every one of those becomes a line the model must honour.
    grey = cv2.GaussianBlur(grey, (0, 0), max(1.0, max(image.shape[:2]) / 900.0))

    # Thresholds from this image's own gradient distribution. See the constants above for the
    # measurement that replaced the fixed pair.
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    high = float(np.percentile(magnitude, _CANNY_HIGH_PERCENTILE))
    high = min(max(high, 12.0), 255.0)
    edges = cv2.Canny(grey, int(high * _CANNY_LOW_RATIO), int(high))

    if structure is not None and structure.any():
        shrink = max(3, int(round(max(image.shape[:2]) * 0.008)) | 1)
        inner = cv2.erode(structure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink)))
        edges = cv2.bitwise_and(edges, inner)
    return edges


# --------------------------------------------------------------------------------------
# The diffusion pipeline
# --------------------------------------------------------------------------------------


def _load() -> tuple[Any, str] | None:
    """Load once per process. Returns ``(pipeline, device)`` or ``None``."""
    if _state["checked"]:
        pipe = _state["pipe"]
        return (pipe, _state["device"]) if pipe is not None else None

    with _lock:
        if _state["checked"]:
            pipe = _state["pipe"]
            return (pipe, _state["device"]) if pipe is not None else None
        _state["checked"] = True

        from biosphere.ai import sd_inpaint

        device = sd_inpaint.cuda_device()
        if device is None:
            log.info("restyle.unavailable", reason="no usable cuda device; cpu is refused by design")
            return None

        try:
            import torch
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetInpaintPipeline,
                UniPCMultistepScheduler,
            )
        except Exception as exc:  # noqa: BLE001 - diffusers is an optional extra
            log.info("restyle.unavailable", reason=f"diffusers missing: {exc}")
            return None

        local_only = not sd_inpaint.downloads_allowed()
        nets: list[Any] = []
        for reference in (DEPTH_CONTROLNET, CANNY_CONTROLNET):
            net = None
            for variant in ("fp16", None):
                try:
                    net = ControlNetModel.from_pretrained(
                        reference,
                        torch_dtype=torch.float16,
                        variant=variant,
                        use_safetensors=True,
                        local_files_only=local_only,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    log.info(
                        "restyle.controlnet_unavailable",
                        model=reference,
                        variant=variant,
                        error=str(exc)[:160],
                    )
            if net is None:
                log.info("restyle.unavailable", reason=f"missing controlnet: {reference}")
                return None
            nets.append(net)

        # The SAME inpainting checkpoint ``sd_inpaint`` uses, found the same way. Two pipelines
        # over one set of weights is the point: a machine that can empty a room can furnish one
        # without a second four-gigabyte download.
        for name, reference, is_dir in sd_inpaint.checkpoint_candidates():
            pipe = None
            for variant, safe in (("fp16", True), (None, True), ("fp16", False), (None, False)):
                try:
                    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        reference,
                        controlnet=nets,
                        torch_dtype=torch.float16,
                        variant=variant,
                        use_safetensors=safe,
                        # Off for the same reason as in ``sd_inpaint``: the checker returns a
                        # black image on a false positive, and a blanked room composited into a
                        # listing is a worse failure than anything it could catch here.
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=local_only and not is_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.info(
                        "restyle.checkpoint_unavailable",
                        checkpoint=name,
                        variant=variant,
                        safetensors=safe,
                        error=str(exc)[:160],
                    )
                    continue
                if not safe:
                    log.warning(
                        "restyle.pickled_weights",
                        checkpoint=name,
                        reference=reference,
                        detail="no safetensors for this checkpoint; loading executed a pickle",
                    )
                break
            if pipe is None:
                continue

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            for target, method in (
                (pipe, "enable_attention_slicing"),
                (getattr(pipe, "vae", None), "enable_slicing"),
            ):
                call = getattr(target, method, None) if target is not None else None
                if callable(call):
                    try:
                        call()
                    except Exception as exc:  # noqa: BLE001 - an optimisation, never a requirement
                        log.info("restyle.slicing_unavailable", call=method, error=str(exc)[:120])

            _state.update({"pipe": pipe, "device": device, "ref": reference})
            log.info("restyle.ready", checkpoint=name, reference=reference, device=device)
            return pipe, device

        log.info("restyle.unavailable", reason="no inpainting checkpoint found on this machine")
        return None


def available() -> bool:
    """True when a call to :func:`restyle_photo` will use the network. LOADS IT."""
    return _load() is not None and _load_depth() is not None


def is_loaded() -> bool:
    """Whether the pipeline is already resident. Never loads it."""
    return _state["pipe"] is not None


def describe() -> dict[str, Any]:
    """What this process would do, without loading anything to find out."""
    return {
        "loaded": is_loaded(),
        "checkpoint": _state["ref"],
        "controlnets": [DEPTH_CONTROLNET, CANNY_CONTROLNET],
        "depth_model": DEPTH_MODEL,
        "styles": list(STYLE_IDS),
    }


def reset() -> None:
    """Drop the loaded pipelines. For tests, and for freeing VRAM in a long-lived process."""
    with _lock:
        _state.update(
            {"checked": False, "pipe": None, "device": None, "ref": None, "depth": None, "labels": None}
        )
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ======================================================================================
# Reading the room
# ======================================================================================


def _label_table() -> dict[int, str] | None:
    """The segmenter's own ``id -> name`` map. ``None`` when it cannot be read."""
    cached = _state.get("labels")
    if cached is not None:
        return cached
    try:
        from transformers import AutoConfig

        from biosphere.ai import furniture_seg, sd_inpaint

        config = AutoConfig.from_pretrained(
            furniture_seg.MASK2FORMER_MODEL,
            local_files_only=not sd_inpaint.downloads_allowed(),
        )
        cached = {int(k): str(v).strip().lower() for k, v in dict(config.id2label).items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("restyle.label_table_unavailable", error=str(exc)[:160])
        return None
    _state["labels"] = cached
    return cached


def _ids_named(table: dict[int, str], wanted: frozenset[str]) -> set[int]:
    """Class ids whose ADE20K name lists any of ``wanted`` as a synonym.

    ADE20K names are comma-separated synonym lists ("windowpane, window"), so the match is per
    synonym; comparing whole strings would find almost nothing.
    """
    return {
        class_id
        for class_id, name in table.items()
        if any(part.strip() in wanted for part in name.split(","))
    }


def _mask_of(labels: np.ndarray, ids: set[int]) -> np.ndarray:
    if not ids:
        return np.zeros(labels.shape, np.uint8)
    size = int(max(int(labels.max()), max(ids))) + 2
    keep = np.zeros(size, bool)
    for class_id in ids:
        keep[class_id] = True
    return keep[np.clip(labels, 0, size - 1)].astype(np.uint8) * 255


def feathered_alpha(protect: np.ndarray, sigma: float) -> np.ndarray:
    """Composite weight of the ORIGINAL photograph: exactly 1 inside ``protect``, ramping out.

    A plain blur of the mask is the obvious construction and it is wrong in a way that matters.
    Blurring spreads the boundary BOTH ways, so the pixels just inside a window come back at
    0.99-something and the composite returns them a grey level or two off the photograph. The
    product promise here is "your window is still your window", and a promise that holds to
    within a rounding error is a different, weaker promise — one no test can state cleanly and
    no reviewer can check.

    So the ramp is taken from the blur and then the mask is put back over it at full strength:
    the feather lives entirely on the EDITABLE side of the boundary, which is the side that
    needed softening anyway. Inside the mask the weight is exactly 1.0, and
    ``styled * 0 + original * 1`` is the original, bit for bit.
    """
    hard = protect.astype(np.float32) / 255.0
    ramp = cv2.GaussianBlur(hard, (0, 0), sigma)
    return np.clip(np.maximum(ramp, hard), 0.0, 1.0)


#: What the pass is being asked to do, decided from the room rather than by the caller.
RestyleMode = Literal["replace", "stage"]


@dataclass(slots=True)
class RoomAnalysis:
    """Everything about the ROOM that every style shares.

    Computed once and handed to each pass. Recomputing it per style would pay for a
    segmentation and a depth estimate five times over for five identical answers — and worse,
    it would let two styles of the same photograph disagree about where the window is.
    """

    depth: np.ndarray
    edges: np.ndarray
    #: 255 where the model may draw.
    editable: np.ndarray
    #: Feathered 0..1 weight of the original photograph in the final composite.
    protect_alpha: np.ndarray
    protected_fraction: float
    #: False when no segmenter answered: nothing was protected and every edge was kept.
    segmented: bool
    #: ``replace`` when there were furnishings to restyle, ``stage`` when the room was empty
    #: enough that the question became what furniture belongs in it.
    mode: RestyleMode = "stage"
    #: What the segmenter found standing in the room, largest first — the nouns the prompt is
    #: built from in ``replace`` mode. Empty in ``stage`` mode and without a segmenter.
    objects: tuple[str, ...] = ()


def _segment_at_working_size(image: np.ndarray) -> np.ndarray | None:
    """ADE20K labels for ``image``, computed at :data:`_SEGMENT_PIXELS` and returned full size.

    NEAREST on the way back up: these are class ids, and interpolating between "bed" and "wall"
    invents whatever class happens to sit between their two integers.
    """
    from biosphere.ai import backend

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= _SEGMENT_PIXELS:
        return backend.segment_photo(image)

    scale = _SEGMENT_PIXELS / float(longest)
    small = cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    labels = backend.segment_photo(small)
    if labels is None:
        return None
    return cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)


def _furnishable_volume(floor: np.ndarray, height: int) -> np.ndarray:
    """The floor, plus the band of wall a piece of furniture standing on it would cover.

    ``stage`` mode needs SOMEWHERE to put furniture, and the obvious answer — "the whole wall
    and floor" — is the one that produced the fault this mode was rewritten around: an editable
    upper wall is a large bright rectangle, and what a model trained on estate-agent
    photographs draws in one of those is a window. Growing the floor upward instead says the
    same thing a room does: furniture stands on the floor and has a height.

    A plain vertical dilation is enough because the growth downward from a floor region lands
    on more floor (or off the frame), so only the upward half changes anything.
    """
    rise = max(3, int(round(height * _STAGE_RISE)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rise))
    return cv2.dilate(floor, kernel)


def _inventory(labels: np.ndarray, table: dict[int, str], furnishings: set[int]) -> tuple[str, ...]:
    """The furnishing classes actually present, largest first, as prompt-ready nouns."""
    found: list[tuple[float, str]] = []
    total = float(labels.size)
    for class_id in furnishings:
        share = float((labels == class_id).sum()) / total
        if share < _MIN_OBJECT_FRACTION:
            continue
        name = table.get(class_id, "").split(",")[0].strip()
        if not name:
            continue
        found.append((share, _OBJECT_RENAMES.get(name, name)))
    found.sort(reverse=True)
    # Dedupe while keeping the area order: two ADE20K classes can rename onto one noun.
    seen: set[str] = set()
    out: list[str] = []
    for _, name in found:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out[:_MAX_OBJECTS])


@dataclass(slots=True)
class RoomMasks:
    """Which pixels are the room, and which are things standing in it.

    The half of :class:`RoomAnalysis` that does not need a GPU-side depth estimate, split out
    because :mod:`biosphere.ai.empty_photo` wants exactly this and nothing else. Restyling
    redraws :attr:`editable`; emptying erases it. One definition of "a furnishing" behind both
    is what keeps the two features from disagreeing about what a curtain is.
    """

    #: 255 where the model may work — the furnishings in ``replace`` mode, the furnishable
    #: volume in ``stage`` mode.
    editable: np.ndarray
    #: The building fabric, for the edge conditioner.
    structure: np.ndarray
    mode: RestyleMode
    objects: tuple[str, ...]
    segmented: bool
    #: Per-pixel ADE20K ids and the protected (held) pixels, for a caller that grows the mask —
    #: ``empty_photo`` does, over what the segmenter called room but is plainly not. ``None``
    #: without a segmenter.
    labels: np.ndarray | None = None
    held: np.ndarray | None = None
    surfaces: dict[str, set[int]] | None = None


def read_room(
    image: np.ndarray,
    *,
    window_dressing: bool = True,
    extra_protect: np.ndarray | None = None,
) -> RoomMasks:
    """Segment one photograph into room and contents. Never fails; degrades and says so.

    THE MODE IS READ OFF THE ROOM, NOT PASSED IN. A furnished room and an empty one are two
    different questions and a caller has no way to know which one it is holding:

    ``replace``  the segmenter found furnishings. Only those pixels are editable; the wall,
                 floor, ceiling, windows, doors, curtains and fittings are all held.
    ``stage``    the room is empty (under :data:`_FURNISHED_FRACTION`). There is nothing to
                 replace, so the editable region is :func:`_furnishable_volume` — the floor and
                 the band of wall above it.

    With NO SEGMENTER at all this reports ``segmented=False`` and calls the whole frame
    editable, which is the only honest answer: nothing is known to be protected.
    """
    height, width = image.shape[:2]
    held = (
        np.zeros((height, width), np.uint8)
        if extra_protect is None
        else (np.asarray(extra_protect) > 127).astype(np.uint8) * 255
    )

    labels = _segment_at_working_size(image)
    table = _label_table() if labels is not None else None
    if labels is None or table is None:
        return RoomMasks(
            editable=255 - held,
            structure=np.zeros((height, width), np.uint8),
            mode="stage",
            objects=(),
            segmented=False,
        )

    keep_names = _PROTECTED_NAMES | (_WINDOW_DRESSING_NAMES if window_dressing else frozenset())
    kept_ids = _ids_named(table, keep_names)
    surface_ids = _ids_named(table, _SURFACE_NAMES)
    # Everything the segmenter knows that is neither the room nor a protected fitting is, by
    # construction, something standing in the room. An exclusion list rather than an inclusion
    # list, for the reason `furniture_seg` gives: an inclusion list silently treats every object
    # the vocabulary has that nobody thought of as part of the building.
    furnishing_ids = {c for c in table if c not in kept_ids and c not in surface_ids}

    furnishings = _mask_of(labels, furnishing_ids)
    grow = max(3, int(round(max(height, width) * _PROTECT_GROW)) | 1)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
    opening = max(3, int(round(max(height, width) * 0.004)) | 1)
    despeckle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening, opening))

    # Architecture, for the edge conditioner: the room's own surfaces plus everything held.
    # Deliberately the union — a window frame is not a "surface" but its outline is the single
    # most important line in the picture.
    structure = _mask_of(labels, kept_ids | surface_ids)

    if float((furnishings > 0).mean()) >= _FURNISHED_FRACTION:
        mode: RestyleMode = "replace"
        # Grown, then despeckled. Grown because a semantic boundary is a few pixels loose and a
        # ring of the OLD sofa left around the new one is the most visible way this can fail;
        # despeckled because a per-pixel segmenter scatters single pixels across a wall and each
        # one would become a lone hallucinated speck.
        editable = cv2.morphologyEx(cv2.dilate(furnishings, ellipse), cv2.MORPH_OPEN, despeckle)
        objects = _inventory(labels, table, furnishing_ids)
    else:
        mode = "stage"
        floor = _mask_of(labels, _ids_named(table, _FLOOR_NAMES))
        editable = _furnishable_volume(floor, height)
        # Never over a window, a door or a fitting, whatever the volume says.
        editable = cv2.bitwise_and(editable, 255 - cv2.dilate(_mask_of(labels, kept_ids), ellipse))
        objects = ()

    # The caller's own hold, applied last: it is a statement about this pass rather than about
    # the room, so it must not change what counts as a furnishing.
    editable = cv2.bitwise_and(editable, 255 - held)
    return RoomMasks(
        editable=editable,
        structure=structure,
        mode=mode,
        objects=objects,
        segmented=True,
        labels=labels,
        held=_mask_of(labels, kept_ids),
        surfaces={
            "wall": _ids_named(table, frozenset({"wall"})),
            "floor": _ids_named(table, frozenset({"floor", "flooring"})),
        },
    )


def analyse_room(
    image: np.ndarray,
    *,
    window_dressing: bool = True,
    extra_protect: np.ndarray | None = None,
) -> RoomAnalysis | None:
    """Everything a restyle pass needs: :func:`read_room` plus the structure conditioning.

    ``None`` when there is no DEPTH backend, and only then. A missing segmenter is survivable
    and reported (``segmented=False``); a missing depth map is not, because without it the
    model is not restyling a room, it is inventing one — see the module docstring.

    ``extra_protect`` is a caller-supplied mask of pixels to hold back on top of whatever the
    segmenter finds. :mod:`biosphere.ai.restyle_pano` is its reason for existing: when a sphere
    is restyled a window at a time, everything a NEIGHBOURING window already furnished has to
    stay put and act as context, and "context" and "protected" are the same instruction here.
    """
    height, width = image.shape[:2]
    depth = depth_map(image)
    if depth is None:
        log.warning("restyle.no_depth", detail="structure conditioning is not optional here")
        return None

    masks = read_room(image, window_dressing=window_dressing, extra_protect=extra_protect)
    # Edges come from the room's own fabric, NOT from what this pass happens to be holding: a
    # neighbouring window's finished furniture is held back from being redrawn, but its outline
    # is still part of what this window has to continue, and hiding it would let the two
    # disagree at the seam.
    edges = structure_edges(image, masks.structure if masks.segmented else None)

    protect = 255 - masks.editable
    alpha = feathered_alpha(protect, max(1.0, max(height, width) * _PROTECT_FEATHER))

    log.info(
        "restyle.analysed",
        mode=masks.mode,
        editable=round(float((masks.editable > 127).mean()), 4),
        objects=list(masks.objects),
    )
    return RoomAnalysis(
        depth=depth,
        edges=edges,
        editable=masks.editable,
        protect_alpha=alpha,
        protected_fraction=float((protect > 0).mean()),
        segmented=masks.segmented,
        mode=masks.mode,
        objects=masks.objects,
    )


# ======================================================================================
# Inference
# ======================================================================================


@dataclass(slots=True)
class RestyleOutput:
    """One styled room, plus everything needed to explain or reproduce it."""

    image: np.ndarray
    style_id: str
    seed: int
    elapsed_ms: int
    protected_fraction: float
    work_size: tuple[int, int]
    steps: int
    guidance: float
    control_scales: tuple[float, float]
    strength: float
    #: True when the diffusion pass ran. There is no classical fallback for this feature —
    #: nothing in OpenCV invents a sofa — so ``False`` never ships; it exists so the API
    #: response carries the same honest field every other AI route does.
    model_backed: bool = True


#: Distance between the seeds of two takes of one style: a large prime, so no take's seed is
#: another photograph's default.
_VARIATION_STRIDE = 104_729


def _seed_for(image: np.ndarray, mask: np.ndarray, style_id: str) -> int:
    """A seed that is a function of the question, so the answer is a function of it too."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(np.ascontiguousarray(image).tobytes())
    digest.update(np.ascontiguousarray(mask > 127).tobytes())
    digest.update(style_id.encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") & 0x7FFF_FFFF


def _work_size(width: int, height: int, budget: int = WORK_PIXELS) -> tuple[int, int]:
    """Stride-aligned dimensions with this aspect ratio and about ``budget`` pixels."""
    scale = (budget / float(width * height)) ** 0.5
    work_w = max(_STRIDE * 8, int(round(width * scale / _STRIDE)) * _STRIDE)
    work_h = max(_STRIDE * 8, int(round(height * scale / _STRIDE)) * _STRIDE)
    return work_w, work_h


def restyle_photo(
    image: np.ndarray,
    style: Style | str,
    *,
    analysis: RoomAnalysis | None = None,
    window_dressing: bool = True,
    steps: int = _STEPS,
    guidance: float = _GUIDANCE,
    control_scales: tuple[float, float] = CONTROL_SCALES,
    #: ``None`` picks the value for the mode the room turns out to be in — see
    #: :data:`STRENGTH` and :data:`REPLACE_STRENGTH`.
    strength: float | None = None,
    work_pixels: int = WORK_PIXELS,
    seed: int | None = None,
    #: A different take of the same style: each number is its own deterministic seed, so
    #: "another one" can be asked for and the same answer comes back for the same number.
    variation: int = 0,
) -> RestyleOutput | None:
    """Furnish one ordinary photograph in ``style``. ``None`` when no backend can answer.

    ``analysis`` is accepted so a caller rendering every style of the same photograph pays for
    the depth estimate, the segmentation and the edge map ONCE — they are properties of the
    room, and the style does not change any of them.
    """
    resolved = style if isinstance(style, Style) else style_by_id(style)
    if resolved is None:
        raise ValueError(f"unknown style: {style!r}; expected one of {', '.join(STYLE_IDS)}")

    loaded = _load()
    if loaded is None:
        return None
    pipe, _device = loaded

    import torch
    from PIL import Image

    started = time.perf_counter()
    height, width = image.shape[:2]

    room = analysis if analysis is not None else analyse_room(image, window_dressing=window_dressing)
    if room is None:
        return None
    if strength is None:
        strength = REPLACE_STRENGTH if room.mode == "replace" else STRENGTH
    if room.editable.shape[:2] != (height, width):
        # Caught here rather than left to numpy. `analysis` is a public parameter and the
        # natural misuse is to reuse a room measured on one image against another — which
        # would otherwise surface as a broadcast error inside the composite, three functions
        # away from the mistake and with nothing in the message naming it.
        raise ValueError(
            f"analysis was measured on {room.editable.shape[1]}x{room.editable.shape[0]} "
            f"but the image is {width}x{height}; a RoomAnalysis belongs to one photograph"
        )

    work_w, work_h = _work_size(width, height, work_pixels)
    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(room.editable, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    small_depth = cv2.resize(room.depth, (work_w, work_h), interpolation=cv2.INTER_AREA)
    # NEAREST for the lines. Averaging an edge map on the way down turns one-pixel lines into
    # grey smears, and a ControlNet reads a grey smear as a weak, wide, blurred edge — which is
    # exactly the soft doubled architecture it is here to prevent.
    small_edges = cv2.resize(room.edges, (work_w, work_h), interpolation=cv2.INTER_NEAREST)

    if not (small_mask > 127).any():
        # Every pixel protected. Nothing to restyle, and handing the photograph back unchanged
        # is the truthful answer to that.
        return RestyleOutput(
            image=image.copy(),
            style_id=resolved.id,
            seed=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            protected_fraction=room.protected_fraction,
            work_size=(work_w, work_h),
            steps=steps,
            guidance=guidance,
            control_scales=tuple(control_scales),
            strength=strength,
        )

    chosen_seed = (
        (_seed_for(small, small_mask, resolved.id) + _VARIATION_STRIDE * int(variation)) & 0x7FFF_FFFF
        if seed is None
        else int(seed)
    )
    # A CPU generator on a CUDA pipeline, for the reason ``sd_inpaint`` records: seeded on the
    # device, the same style comes back different on a different card, and this package is
    # explicitly built to move its model work to whichever machine has the good one.
    generator = torch.Generator(device="cpu").manual_seed(chosen_seed)

    # The nouns come from the room in `replace` mode and from the style in `stage` mode. That
    # split is the whole point of the redesign: asked for "a grey sofa" in a bedroom, the model
    # draws a sofa into the bed's place, because the prompt was the only thing telling it what
    # the hole was for.
    if room.mode == "replace" and room.objects:
        subject = f"{_REPLACE_LEAD} {', '.join(room.objects)}"
    else:
        subject = resolved.staging
    prompt = f"{subject}, {resolved.palette}, {_PROMPT_SUFFIX}"
    negative = f"{_NEGATIVE_BASE}, {resolved.negative_extra}" if resolved.negative_extra else _NEGATIVE_BASE

    try:
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
                mask_image=Image.fromarray(small_mask),
                control_image=[
                    Image.fromarray(cv2.cvtColor(small_depth, cv2.COLOR_GRAY2RGB)),
                    Image.fromarray(cv2.cvtColor(small_edges, cv2.COLOR_GRAY2RGB)),
                ],
                width=work_w,
                height=work_h,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=float(strength),
                controlnet_conditioning_scale=[float(s) for s in control_scales],
                generator=generator,
            )
    except Exception as exc:  # noqa: BLE001 - OOM on a small card is a real possibility
        log.warning(
            "restyle.inference_failed",
            style=resolved.id,
            error=str(exc)[:200],
            size=f"{work_h}x{work_w}",
        )
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return None

    styled = cv2.cvtColor(np.asarray(output.images[0], dtype=np.uint8), cv2.COLOR_RGB2BGR)
    del output
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    if (styled.shape[1], styled.shape[0]) != (width, height):
        styled = cv2.resize(styled, (width, height), interpolation=cv2.INTER_CUBIC)

    # The protection, enforced. Everything the product promised would not change comes back
    # from the original photograph here, feathered so the join is a join rather than a cut.
    if room.protected_fraction > 0.0:
        alpha = room.protect_alpha[..., None]
        # rint, not a bare cast: ``astype(np.uint8)`` TRUNCATES, so every blended pixel in the
        # feather comes back up to one level dark and a wall that should have joined seamlessly
        # carries a faint outline of the mask.
        styled = np.rint(
            np.clip(styled.astype(np.float32) * (1.0 - alpha) + image.astype(np.float32) * alpha, 0, 255)
        ).astype(np.uint8)

    elapsed = int((time.perf_counter() - started) * 1000)
    log.info(
        "restyle.done",
        style=resolved.id,
        mode=room.mode,
        seed=chosen_seed,
        ms=elapsed,
        protected=round(room.protected_fraction, 4),
        strength=strength,
        size=f"{work_h}x{work_w}",
    )
    return RestyleOutput(
        image=styled,
        style_id=resolved.id,
        seed=chosen_seed,
        elapsed_ms=elapsed,
        protected_fraction=room.protected_fraction,
        work_size=(work_w, work_h),
        steps=steps,
        guidance=guidance,
        control_scales=tuple(control_scales),
        strength=strength,
    )
