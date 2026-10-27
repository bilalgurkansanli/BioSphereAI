"""Stable Diffusion inpainting for room emptying. AVAILABLE, MEASURED, AND NOT THE DEFAULT.

Why it was built, and why it did not win
-----------------------------------------
The reasoning: a furnished room is not a small hole — the mask approaches half the sphere —
and with that little photograph left to continue, a reconstruction inpainter falls back on the
smooth field implied by whatever pixels remain. That is precisely the complaint this feature
exists to answer, in the words a user gave it: it looks like the furniture was **painted
over** rather than removed. A diffusion inpainter is not continuing a photograph, it is
*drawing a room*, so it should not have that failure.

It does not. It has a worse one, and only a real capture shows it — the numbers are under
"What it actually produces" below. The trade is softness against invention, and for a picture
an estate agent puts in front of a buyer, softness is the one to take: a soft wall is a defect,
an invented wardrobe is a misrepresentation.

Kept anyway, and for reasons rather than sentiment. It is genuinely good over small holes. It
is the right shape of seam, so the next model to try drops in here without touching
``room_fill`` or ``empty_room``. And the measurements are worth more written down than deleted,
because they say what to test — and on what — before believing any inpainter about this task.

Windows only, and why this module refuses the canvas
-----------------------------------------------------
:func:`inpaint` DECLINES ``cyclic=True`` and returns ``None`` for it. That is not a gap, it is
the contract.

``cyclic=True`` means the caller is handing over the equirectangular canvas itself — 2:1, four
thousand pixels wide. A latent diffusion model is trained at one size (512 square for every
inpainting checkpoint here) and asked for anything far from it, its attention loses global
coherence and it draws the same window three times along a wall. LaMa's FFC layers have no
such training size and handle the whole canvas natively, which is why ``room_fill`` leaves
them the canvas-wide cleanup pass: the rim fades, the poles, and whatever no window reached.
A few percent of the sphere, where softness costs nothing and duplication would cost a lot.

Through a 90-degree window the same room is an ordinary square photograph at the size the
model was trained on. That is where this module runs, and it is the only place it should.

Determinism, which a toggle needs and diffusion does not have
--------------------------------------------------------------
"Mobilyaları kaldır" is a toggle, and ``test_running_it_twice_gives_the_same_answer`` states
the reason plainly: a toggle that empties the room differently on the second press is a toggle
nobody trusts. Sampling is stochastic, so the seed is derived from the content of the window
and its mask (:func:`_seed_for`) rather than from a clock or a counter, and the noise is drawn
on the CPU so the seed means the same thing on every card. The same window asked the same
question gets the same room back — in this process, in the next one, and on the worker across
the desk. A caller that wants variety changes the strength, which changes the mask, which
changes the seed.

What it costs, measured
------------------------
MEASURED ON THIS MACHINE with ``scripts/bench_inpaint.py``. Re-run it rather than trusting
these when the card changes; the spread between two cards here is a factor of thirty.

  RTX 5050 Laptop (8 GB, sm_120, torch 2.11 cu128), 1024 px window, 30 steps, fp16,
  SD 1.5 inpainting:

      hole 5 %    6.7 s      (first call; includes the warm-up)
      hole 15 %   5.7 s
      hole 30 %   5.6 s
      hole 45 %   5.7 s

Flat in the hole size, because a diffusion pass denoises the whole latent whatever the mask
says — which is the shape of cost this feature wants, since the windows that matter are the
crowded ones. Twenty-four windows is a room, so about two and a half minutes of card time,
inside ``backend.INPAINT_TIMEOUT_S`` per call with room to spare.

For contrast, the number that stalled this feature: 185 s per window on a 4 GB GTX 1650, at
3.8 GB — a card that could hold the model and not much else. That is a factor of 32, and it is
the whole difference between a feature and a note in a commit message.

What it actually produces, and why it is NOT the default
---------------------------------------------------------
THE SYNTHETIC BENCH FLATTERED THIS MODEL AND A REAL ROOM DID NOT. On the bench's window at a
45 % hole the fill is exactly what was wanted: an unbroken skirting board, floorboards
converging on the right vanishing point, a wall meeting a floor along one straight line. That
result is real and it is also unrepresentative, because the bench's furniture is untextured
blocks in a room no checkpoint has a prior about.

MEASURED ON A REAL CAPTURE — a cluttered bedroom, mask from Mask2Former, windows taken
straight out of ``room_fill``:

    hole 11 %   clean. The clutter on a shelf is gone and wall stands where it was.
    hole 18 %   DRAWS A NEW BED. Not a failure to remove the old one — a fresh, made bed
                with a pillow and a duvet, in the same place.
    hole 74 %   draws somebody else's flat: a window with buildings behind it, a radiator,
                a vase of flowers. The flowers arrive despite `plant` in the negative prompt.

Guidance 6, 9, 12 and 15 were swept, with the negative prompt extended to name `bed`,
`mattress`, `pillow` and `duvet` explicitly. NONE of them changed the outcome, and that is the
finding rather than a tuning problem: this is a *generative* inpainter. It has no notion of
"take this away"; it answers "what belongs in this hole", and the honest answer for a
bed-shaped hole in a bedroom is a bed. A text prompt can argue a model out of a noun. It
cannot ask one for an absence.

So the default fill is the classical per-surface path, which has no vocabulary for objects and
therefore cannot invent one. On the same capture it empties the room properly in 53 s; its
failure is a soft wall and some drift across the floor, which is a defect a viewer forgives and
an invented wardrobe is not. This module stays because it is the right shape of seam and it is
genuinely good over small holes — set ``BIOSPHERE_AI_INPAINTER=sd`` to use it, and look at the
result before shipping it to anyone.

Reproduce all of the above with ``scripts/bench_inpaint.py``, and prefer a real capture to the
synthetic room when judging a fill: this whole correction is what the synthetic room hid.

CPU IS REFUSED RATHER THAN OFFERED. The same call on a CPU is minutes per window, and there
are up to twenty-four windows in a room; ``backend.INPAINT_TIMEOUT_S`` is 300 seconds and the
job queue is not a place to hide an hour. On a machine without a usable CUDA device this
module reports itself unavailable and the caller falls back to LaMa, which is a worse room
delivered promptly rather than a better one delivered never.

Licences, since this product ships closed and they are not all the same
------------------------------------------------------------------------
  diffusers (code)              Apache-2.0        fine
  SD 2 inpainting (weights)     OpenRAIL++-M      commercial use permitted, with use
                                                  restrictions attached to the licence
  SD 1.5 inpainting (weights)   CreativeML OpenRAIL-M   same shape, older text

Both are materially freer than ``big-lama``'s CC BY-NC-SA 4.0, which is non-commercial as
published. Making the diffusion path the default therefore *improves* the licence position of
the shipped feature rather than complicating it. Recorded in ``pyproject.toml`` beside the
rest.

Offline by default
------------------
Weights are found, not fetched, unless ``settings.ai_allow_downloads`` says otherwise — that
setting exists for this and this is the module that honours it. A missing checkpoint is not an
error: :func:`available` answers ``False``, the caller uses LaMa or the classical filler, and
the response's ``model_backed`` says what happened.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biosphere.core.logging import get_logger

__all__ = [
    "available",
    "is_loaded",
    "inpaint",
    "model_ref",
    "reset",
    "checkpoint_candidates",
    "cuda_device",
    "downloads_allowed",
    "PROMPT",
    "NEGATIVE_PROMPT",
    "WORK_SIZE",
]

log = get_logger("biosphere.ai.sd_inpaint")

#: Checkpoints we know how to use, in preference order. Tried as a directory under
#: ``settings.ai_models_dir`` first and as a Hugging Face reference second, so an operator can
#: vendor the weights beside the media root without touching a cache directory.
#:
#: SD 2 IS FIRST AND IS NOT WHAT MOST MACHINES WILL GET. Its OpenRAIL++-M is the later and
#: clearer of the two licences and its inpainting UNet leaves fewer haloes around a large mask,
#: so it is the one to want — but ``stabilityai/stable-diffusion-2-inpainting`` is GATED:
#: anonymous requests get 401, and reaching it needs a Hugging Face account that has accepted
#: the terms plus a token in the environment. An operator who wants it either logs in on the
#: machine or drops the folder under ``ai_models_dir``, where the gate does not apply.
#:
#: Without that, the fall-through lands on 1.5, which is open. That is a working feature rather
#: than a broken one, and the order is what keeps "better if you have it" from becoming
#: "nothing unless you do".
#:
#: THE 1.5 REFERENCE IS THE ``stable-diffusion-v1-5`` ORG AND NOT ONE OF THE COMMUNITY MIRRORS,
#: for a reason that is not tidiness. Runway withdrew ``runwayml/stable-diffusion-inpainting``
#: and the mirrors that filled the gap mostly carry the original ``.bin`` files — pickles,
#: which ``torch.load`` deserialises by executing them. Loading one is running whatever the
#: uploader put in it, on the machine holding the media library. This org publishes
#: ``safetensors``, which cannot execute anything, and an ``fp16`` variant, which halves a
#: download that is otherwise four gigabytes to produce bit-identical inference.
_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("sd2-inpainting", "stabilityai/stable-diffusion-2-inpainting"),
    ("sd15-inpainting", "stable-diffusion-v1-5/stable-diffusion-inpainting"),
)

#: The size the network actually runs at, both sides.
#:
#: 512 IS THE TRAINING SIZE OF EVERY CHECKPOINT ABOVE AND IS NOT A BUDGET DECISION. Asked for
#: 1024 the same model draws a plausible room twice — two skirting boards at two heights, a
#: window repeated along a wall — because self-attention at that resolution no longer relates
#: one side of the image to the other. The window arrives at 1024 (``room_fill.VIEW_SIZE``)
#: and leaves at 1024; only the network's own pass happens at 512.
WORK_SIZE = 512

#: Latent stride: both sides handed to the VAE must divide by this.
_STRIDE = 8

#: Sampling.
#:
#: Thirty steps is the usual place a 1.5-family inpainting checkpoint stops changing, and
#: ``bench_inpaint.py --steps`` sweeps it if that needs settling on a particular checkpoint.
#:
#: Guidance is REASONED RATHER THAN MEASURED and is the first knob to reach for if the fill
#: looks wrong. It sits below the 7.5 default because this prompt describes an *absence*, and
#: the harder a sampler is pushed toward a text description the more it delivers the idea of
#: the thing rather than the continuation of this room — showroom-white walls that do not match
#: the three-quarters of the room still standing around them. Too low and the furniture starts
#: coming back.
_STEPS = 30
_GUIDANCE = 6.0

#: What we are asking for. Concrete surfaces, not an abstraction: "empty room" alone invites
#: the model to redecorate, while naming the wall, the floor and the join between them asks it
#: to continue the room it can already see.
PROMPT = (
    "empty unfurnished room, bare painted wall, plain continuous floor, "
    "skirting board along the wall, even natural daylight, "
    "architectural interior photograph, sharp, realistic"
)

#: What we are asking it not to do. The furniture terms matter most: without them the model
#: reads the surrounding room as context for *more room contents* and puts a chair back where
#: the chair was.
NEGATIVE_PROMPT = (
    "furniture, sofa, couch, armchair, chair, table, bed, wardrobe, cabinet, shelf, "
    "rug, carpet pattern, houseplant, lamp, television, picture frame, clutter, boxes, "
    "person, people, hands, text, watermark, signature, "
    "blurry, smeared, distorted, warped, cartoon, painting, lowres"
)

_lock = threading.Lock()
_state: dict[str, Any] = {"checked": False, "pipe": None, "device": None, "ref": None}


# ======================================================================================
# Finding the weights
# ======================================================================================


def _models_dir() -> Path | None:
    try:
        from biosphere.core.config import settings

        return Path(settings.ai_models_dir) if settings.ai_models_dir else None
    except Exception:  # noqa: BLE001 - the engine must import without the API package
        return None


def _downloads_allowed() -> bool:
    try:
        from biosphere.core.config import settings

        return bool(settings.ai_allow_downloads)
    except Exception:  # noqa: BLE001
        return False


def _candidates() -> list[tuple[str, str, bool]]:
    """``(name, reference, is_local_dir)`` in the order they should be tried."""
    out: list[tuple[str, str, bool]] = []
    directory = _models_dir()
    if directory is not None:
        for name, repo in _CHECKPOINTS:
            for folder in (name, repo.split("/")[-1]):
                path = directory / folder
                if (path / "model_index.json").is_file():
                    out.append((name, str(path), True))
                    break
    for name, repo in _CHECKPOINTS:
        out.append((name, repo, False))
    return out


def model_ref() -> str | None:
    """Which checkpoint is loaded, as a path or a repository id. Never loads one."""
    return _state["ref"]


# --------------------------------------------------------------------------------------
# Shared with :mod:`biosphere.ai.restyle`
# --------------------------------------------------------------------------------------
#
# There are now two diffusion users in this package, running the SAME inpainting checkpoint —
# one asking for an absence, the other for a furnishing. Where those weights live, whether this
# machine may fetch them, and whether it has a card worth using are facts about the MACHINE
# rather than about either question, so they are exported rather than discovered twice.


def checkpoint_candidates() -> list[tuple[str, str, bool]]:
    """``(name, reference, is_local_dir)`` in the order they should be tried."""
    return _candidates()


def cuda_device() -> str | None:
    """``"cuda"`` when a usable, permitted device is present, else ``None`` — which is what
    turns every diffusion path in this package off. See the CPU note in the module docstring."""
    return _cuda_device()


def downloads_allowed() -> bool:
    """Whether weights may be fetched at runtime (``settings.ai_allow_downloads``)."""
    return _downloads_allowed()


# ======================================================================================
# Loading
# ======================================================================================


def _cuda_device() -> str | None:
    """``"cuda"`` when a usable device is present AND permitted, else ``None``.

    ``None`` is the answer that turns this module off. See the CPU note in the module
    docstring: a CPU diffusion pass is not a slower answer, it is a timeout.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is an optional extra
        return None
    try:
        from biosphere.core.config import settings

        if not bool(settings.enable_gpu):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:  # noqa: BLE001 - a broken driver must not break the endpoint
        return None
    return None


def _enable_slicing(pipe: Any) -> None:
    """Trade a little speed for a lot of headroom, on whichever diffusers this is.

    Eight gigabytes holds this comfortably at 512; slicing is here for the four- and six-
    gigabyte cards the feature is meant to reach, where it is the difference between a slower
    pass and an out-of-memory one.

    Both switches have moved during diffusers' 0.3x series — ``enable_vae_slicing`` on the
    pipeline became ``vae.enable_slicing`` — and a wrong guess is an ``AttributeError`` at load
    time that takes the whole feature down. So each is tried where it has lived and its absence
    is survivable: the model still runs, it just holds more.
    """
    for target, names in ((pipe, ("enable_attention_slicing",)), (getattr(pipe, "vae", None), ("enable_slicing",))):
        if target is None:
            continue
        for name in names:
            method = getattr(target, name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:  # noqa: BLE001 - an optimisation, never a requirement
                    log.info("sd_inpaint.slicing_unavailable", call=name, error=str(exc)[:120])
                break


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

        device = _cuda_device()
        if device is None:
            log.info("sd_inpaint.unavailable", reason="no usable cuda device; cpu is refused by design")
            return None

        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
        except Exception as exc:  # noqa: BLE001 - diffusers is an optional extra
            log.info("sd_inpaint.unavailable", reason=f"diffusers missing: {exc}")
            return None

        local_only = not _downloads_allowed()
        for name, reference, is_dir in _candidates():
            pipe = None
            # Ordered by what we would rather have, and both axes earn their place:
            #
            #   safetensors   a `.bin` is a pickle, and `torch.load` deserialises a pickle by
            #                 EXECUTING it. Preferring safetensors means the ordinary path
            #                 never runs code out of a downloaded file. Pickles are still
            #                 accepted last, because an operator vendoring an older checkpoint
            #                 into `ai_models_dir` has chosen what they are loading — and that
            #                 attempt says so in the log at warning level.
            #   fp16 variant  the model runs in fp16 either way, so the fp32 files cost twice
            #                 the download and twice the disk for bit-identical inference.
            for variant, safe in (("fp16", True), (None, True), ("fp16", False), (None, False)):
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        reference,
                        torch_dtype=torch.float16,
                        variant=variant,
                        use_safetensors=safe,
                        # OFF DELIBERATELY, AND THE REASON IS PRODUCT RATHER THAN POLICY. The
                        # checker returns a black image on a false positive, and the input here
                        # is a wall with a sofa cut out of it — an interior surface, never a
                        # person. A blanked window would composite as a black patch in the
                        # middle of a room, which is a worse failure than anything it could
                        # catch. It also costs half a gigabyte of VRAM on a card that has
                        # eight.
                        safety_checker=None,
                        requires_safety_checker=False,
                        local_files_only=local_only and not is_dir,
                    )
                except Exception as exc:  # noqa: BLE001 - a missing checkpoint is not an error
                    log.info(
                        "sd_inpaint.checkpoint_unavailable",
                        checkpoint=name,
                        variant=variant,
                        safetensors=safe,
                        local=is_dir,
                        error=str(exc)[:160],
                    )
                    continue
                if not safe:
                    log.warning(
                        "sd_inpaint.pickled_weights",
                        checkpoint=name,
                        reference=reference,
                        detail="no safetensors for this checkpoint; loading executed a pickle",
                    )
                break
            if pipe is None:
                continue

            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            _enable_slicing(pipe)

            _state.update({"pipe": pipe, "device": device, "ref": reference})
            log.info("sd_inpaint.ready", checkpoint=name, reference=reference, device=device)
            return pipe, device

        log.info("sd_inpaint.unavailable", reason="no inpainting checkpoint found on this machine")
        return None


def available() -> bool:
    """True when a call to :func:`inpaint` will actually use the network. LOADS IT."""
    return _load() is not None


def is_loaded() -> bool:
    """Whether the pipeline is already resident. Never loads it."""
    return _state["pipe"] is not None


def reset() -> None:
    """Drop the loaded pipeline. For tests, and for freeing VRAM in a long-lived process."""
    with _lock:
        _state.update({"checked": False, "pipe": None, "device": None, "ref": None})
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ======================================================================================
# Inference
# ======================================================================================


def _seed_for(image: np.ndarray, hole: np.ndarray) -> int:
    """A seed that is a function of the question, so the answer is a function of it too.

    Hashing the arrays rather than counting calls is what makes this reproducible across
    processes and across machines: the engine and the worker seed identically, so moving the
    GPU work to another box does not change the room.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(np.ascontiguousarray(image).tobytes())
    digest.update(np.ascontiguousarray(hole > 127).tobytes())
    return int.from_bytes(digest.digest(), "big") & 0x7FFF_FFFF


def _round_to_stride(value: int) -> int:
    return max(_STRIDE, int(round(value / _STRIDE)) * _STRIDE)


def inpaint(
    image: np.ndarray, hole: np.ndarray, *, cyclic: bool = True
) -> tuple[np.ndarray, bool] | None:
    """Redraw ``image`` without the masked part. ``None`` when this backend cannot answer.

    Signature and return shape match :func:`biosphere.ai.lama.inpaint` exactly, because
    ``backend`` chooses between them at the call site and ``room_fill`` must not know which
    one it got.

    ``cyclic=True`` is REFUSED — see the module docstring. The caller falls back to LaMa for
    the canvas-wide pass, which is the pass LaMa is better at anyway.

    Only the hole is meaningful in the result: a diffusion pass rewrites every pixel it is
    given, so the caller composites and the panorama outside the mask stays byte-identical.
    """
    if cyclic:
        return None

    loaded = _load()
    if loaded is None:
        return None
    pipe, device = loaded

    import torch
    from PIL import Image

    height, width = image.shape[:2]
    if not np.any(hole > 127):
        return image.copy(), True

    # Square at the training size. The window is already square; anything else is resized to
    # a stride-aligned box with the same aspect, which keeps a caller that is not `room_fill`
    # from silently getting a stretched room.
    if width == height:
        work_w = work_h = WORK_SIZE
    else:
        scale = WORK_SIZE / float(max(width, height))
        work_w = _round_to_stride(int(round(width * scale)))
        work_h = _round_to_stride(int(round(height * scale)))

    small = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
    small_hole = cv2.resize(
        (hole > 127).astype(np.uint8) * 255, (work_w, work_h), interpolation=cv2.INTER_NEAREST
    )
    if not small_hole.any():
        # The hole survived the full-resolution test and vanished in the downscale: a sliver a
        # few pixels wide. There is nothing here for a diffusion model to reason about and the
        # caller's compositing would paste a rewritten wall over a real one.
        return image.copy(), True

    # A CPU GENERATOR, DELIBERATELY, ON A CUDA PIPELINE. Seeding the device generator makes the
    # starting latents a function of the card's own RNG, so the same room comes back different
    # on a different GPU — and this feature is explicitly built to move its model work to
    # whichever machine has the good card. Sampled on the CPU and copied across, the latents
    # are identical everywhere; what is left is kernel-level float ordering, which moves a
    # pixel by a grey level rather than redrawing the wall.
    generator = torch.Generator(device="cpu").manual_seed(_seed_for(small, small_hole))
    try:
        with torch.inference_mode():
            output = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
                mask_image=Image.fromarray(small_hole),
                width=work_w,
                height=work_h,
                num_inference_steps=_STEPS,
                guidance_scale=_GUIDANCE,
                generator=generator,
            )
    except Exception as exc:  # noqa: BLE001 - OOM on a small card is a real possibility
        log.warning(
            "sd_inpaint.inference_failed", error=str(exc)[:200], size=f"{work_h}x{work_w}"
        )
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return None

    filled = cv2.cvtColor(np.asarray(output.images[0], dtype=np.uint8), cv2.COLOR_RGB2BGR)
    del output
    # Kept even though this runs once per WINDOW rather than once per canvas as `lama` does,
    # so a room pays it two dozen times. It is worth roughly a tenth of a second against a
    # five-second pass, and what it buys is the four- and six-gigabyte cards: there the
    # allocator's retained blocks are the difference between the next window and an OOM.
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    if (filled.shape[1], filled.shape[0]) != (width, height):
        # CUBIC on the way back up, as in `lama`. The structure this pass exists for survives
        # a 2x enlargement; the grain does not, and `empty_room._reinject_grain` puts the
        # room's own grain back on top afterwards, which is where grain should come from.
        filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)
    return filled, True
