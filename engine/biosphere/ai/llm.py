"""The one model in this engine that is not on this machine.

Everything else under :mod:`biosphere.ai` is weights on a disk — ours to load, ours to run,
free per call and silent about what it saw. This is a hosted text-and-vision model reached
over the network, and the three differences that follow are the whole reason it gets its own
seam instead of joining :mod:`biosphere.ai.backend`:

*It is billed per call.* So nothing calls it speculatively: the caller decides, the answer is
cached beside the property, and a health probe never touches it.

*It leaves the building.* One request carries the room names, the measured areas and a
handful of room pictures to Anthropic. That is a decision an operator makes by putting a key
in the configuration, and it is off — with no fallback to a "free" hosted tier and no reading
of some other service's key — until they do.

*It answers in prose, which cannot be validated by looking at it.* A stitched panorama is
either aligned or visibly not; a paragraph about a flat reads exactly as well whether or not
the flat has the balcony it mentions. So this module hands back a **parsed object against a
schema the caller declares**, never free text, and the caller checks the claims in it against
the facts it supplied. :mod:`biosphere.ai.listing` is where that checking lives.

What this module is not
-----------------------
It is not a chat client, and it holds no conversation: one system prompt, one user turn, one
structured answer. Nothing here retries a refusal onto a second model either — a listing
paragraph is not a request any model declines for policy reasons, and if one ever does, the
caller already has a path that needs no model at all. That is a better answer than a
silent second model of unknown price.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from biosphere.core.logging import get_logger

__all__ = ["LlmError", "LlmResult", "available", "describe", "parse", "model_name"]

log = get_logger("biosphere.ai.llm")


class LlmError(RuntimeError):
    """The call could not be made, or came back unusable.

    Deliberately one class. Every caller does the same thing with it — fall back to whatever
    it can do without a model and report why — so splitting "no key", "network" and "refused"
    into three exceptions would buy three ``except`` clauses that all say the same thing. The
    message carries the distinction for the log and for the user.
    """


#: USD per million tokens, ``(input, output)``, for the models this feature is run on.
#:
#: A TABLE RATHER THAN A LOOKUP because there is no price endpoint to ask, and a wrong number
#: here is worse than none: it would be shown to somebody deciding whether the feature is
#: worth running. So an unpriced model answers ``None`` and the dashboard says "bilinmiyor"
#: instead of guessing from a neighbouring model's rate. Check against
#: https://www.anthropic.com/pricing when adding a row.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def price_of(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """What one call cost, in USD, or ``None`` for a model with no published rate here.

    Matched on a prefix so a dated snapshot (``claude-opus-5-20260114``) prices like the alias
    it came from — the API answers with whichever it likes, and a version suffix is not a
    different price.
    """
    for name, (per_in, per_out) in PRICES.items():
        if model.startswith(name):
            return round((input_tokens * per_in + output_tokens * per_out) / 1_000_000.0, 6)
    return None


@dataclass(frozen=True, slots=True)
class LlmResult:
    """A parsed answer plus what it cost to get."""

    value: BaseModel
    model: str
    input_tokens: int
    output_tokens: int
    elapsed_ms: int
    #: USD for this call, or ``None`` when the model is not in :data:`PRICES`.
    cost_usd: float | None = None


def _settings() -> Any:
    from biosphere.core.config import get_settings

    return get_settings()


def _key() -> str:
    try:
        return str(_settings().listing_key or "")
    except Exception:  # noqa: BLE001 - the engine must run with a broken .env
        return ""


def model_name() -> str:
    try:
        return str(_settings().listing_model or "claude-haiku-4-5")
    except Exception:  # noqa: BLE001
        return "claude-haiku-4-5"


def _sdk_present() -> bool:
    """Whether the client library is installed, WITHOUT importing it.

    ``importlib.util.find_spec`` reads the metadata; importing ``anthropic`` pulls its whole
    dependency tree, and this question is asked on a page load.
    """
    from importlib.util import find_spec

    try:
        return find_spec("anthropic") is not None
    except Exception:  # noqa: BLE001 - a broken installation is "no"
        return False


def available() -> bool:
    """True when a call would be attempted. Cheap enough for a page load."""
    return bool(_key()) and _sdk_present()


def describe() -> dict[str, Any]:
    """What this deployment can do, and why not when it cannot.

    ``reason`` is for the dashboard, which otherwise shows an empty panel and a disabled
    button with nothing to act on: the two ways this feature is off — no key, no library —
    have different fixes and only one of them is a code change.
    """
    key, sdk = bool(_key()), _sdk_present()
    if key and sdk:
        return {"available": True, "model": model_name(), "reason": None}
    return {
        "available": False,
        "model": model_name(),
        "reason": "no_key" if not key else "no_sdk",
    }


def parse(
    *,
    system: str,
    content: list[dict[str, Any]],
    schema: type[BaseModel],
    max_tokens: int = 8000,
    timeout_s: float | None = None,
) -> LlmResult:
    """One request, answered as an instance of ``schema``. Raises :class:`LlmError`.

    ``content`` is a Messages API content list — text blocks and base64 image blocks — built
    by the caller, because what is worth sending is a product question and this module has no
    opinion about it.
    """
    if not _key():
        raise LlmError("no API key is configured for the writing model")
    try:
        import anthropic
    except Exception as exc:  # noqa: BLE001 - an optional extra
        raise LlmError("the anthropic client library is not installed") from exc

    settings = _settings()
    timeout = float(timeout_s if timeout_s is not None else getattr(settings, "listing_timeout_s", 180.0))
    client = anthropic.Anthropic(api_key=_key(), timeout=timeout)
    started = time.perf_counter()
    try:
        response = client.messages.parse(
            model=model_name(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
    except anthropic.APIStatusError as exc:
        # The status is the useful half: 401 is a wrong key, 429 is a rate limit, 5xx is
        # theirs. The body may contain the prompt, so only the message goes to the log.
        raise LlmError(f"the writing model returned {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LlmError("the writing model could not be reached") from exc
    except Exception as exc:  # noqa: BLE001 - an SDK version we do not know
        raise LlmError(f"the writing model call failed: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    # A refusal is an HTTP 200 with no content. Checking `stop_reason` before reading the
    # answer is the difference between a fallback and an AttributeError.
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise LlmError(f"the writing model declined ({getattr(details, 'category', 'unknown')})")
    value = getattr(response, "parsed_output", None)
    if not isinstance(value, schema):
        raise LlmError("the writing model answered in a shape that did not parse")

    usage = getattr(response, "usage", None)
    served_by = str(getattr(response, "model", model_name()))
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    result = LlmResult(
        value=value,
        model=served_by,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_ms=elapsed_ms,
        cost_usd=price_of(served_by, input_tokens, output_tokens),
    )
    log.info(
        "llm.parsed",
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        elapsed_ms=result.elapsed_ms,
    )
    return result
