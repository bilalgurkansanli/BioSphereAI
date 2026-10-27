"""What one call to the hosted model cost, from its own usage report.

Separate from :data:`biosphere.ai.llm.PRICES` on purpose. That table prices one prompt-and-answer
with two numbers; a chat turn here is a different shape of bill: cache writes and cache reads
priced apart, web searches billed per search, and — with server-side fallback — attempts by two
different models inside one response. Folding that into the listing table would change what
the listing panel reports, so this package keeps its own, read from
https://platform.claude.com/docs/en/about-claude/pricing on 2026-10-06.

THE LONGEST PREFIX WINS. ``claude-opus-5-5`` starts with ``claude-opus-5``, and a first-match scan
of a dict in insertion order prices Opus 5.5 at Opus 5's rates — 25 % high on every token. The
same trap exists between ``claude-sonnet-5-5`` and ``claude-sonnet-5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Rates", "PRICES", "WEB_SEARCH_USD", "rates_for", "CallCost", "cost_of"]


@dataclass(frozen=True, slots=True)
class Rates:
    """USD per million tokens."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


#: Models a turn may be served by: the three the assistant runs on, and the server-side
#: fallback targets Opus 5.5 and Sonnet 5.5 publish (``allowed_fallback_models``).
PRICES: dict[str, Rates] = {
    "claude-opus-5-5": Rates(4.00, 5.00, 8.00, 0.20, 20.00),
    "claude-opus-5": Rates(5.00, 6.25, 10.00, 0.50, 25.00),
    "claude-opus-4-8": Rates(5.00, 6.25, 10.00, 0.50, 25.00),
    "claude-sonnet-5-5": Rates(2.00, 2.50, 4.00, 0.20, 10.00),
    "claude-sonnet-5": Rates(2.00, 2.50, 4.00, 0.20, 10.00),
    "claude-haiku-4-5": Rates(1.00, 1.25, 2.00, 0.10, 5.00),
}

#: $10 per 1,000 searches; a search that errors is not billed.
WEB_SEARCH_USD = 0.01

#: Used for a model this table does not know, so that the BUDGET still counts it. Opus 5's
#: rates are the highest in the table: an unknown model is assumed expensive, never free.
_CEILING = PRICES["claude-opus-5"]


def rates_for(model: str) -> Rates | None:
    """The rates of the longest matching prefix, or ``None`` for an unknown model."""
    best: tuple[int, Rates] | None = None
    for name, rates in PRICES.items():
        if model.startswith(name) and (best is None or len(name) > best[0]):
            best = (len(name), rates)
    return best[1] if best else None


@dataclass(frozen=True, slots=True)
class CallCost:
    """Tokens and dollars for one API call (all attempts in it)."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    web_search_requests: int
    cost_usd: float
    #: False when a model in the call had no published rate and the ceiling was used.
    priced: bool
    #: True when a server-side fallback served (part of) the call.
    fallback: bool


def _get(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tokens(entry: Any) -> tuple[int, int, int, int, int]:
    """(input, output, cache_read, cache_write_5m, cache_write_1h) of one usage object."""
    inp = int(_get(entry, "input_tokens", 0) or 0)
    out = int(_get(entry, "output_tokens", 0) or 0)
    read = int(_get(entry, "cache_read_input_tokens", 0) or 0)
    write = int(_get(entry, "cache_creation_input_tokens", 0) or 0)
    breakdown = _get(entry, "cache_creation", None)
    w5 = int(_get(breakdown, "ephemeral_5m_input_tokens", 0) or 0) if breakdown else 0
    w1 = int(_get(breakdown, "ephemeral_1h_input_tokens", 0) or 0) if breakdown else 0
    if breakdown is None or (w5 + w1) != write:
        # No breakdown, or one that does not add up: everything this code writes is 5-minute.
        w5, w1 = write, 0
    return inp, out, read, w5, w1


def cost_of(model: str, usage: Any) -> CallCost:
    """Price one response's ``usage``.

    ``usage.iterations`` is the per-attempt record when it is present — server-side fallback
    reports the declined attempt and the serving one there, each under its own model, and the
    top-level counts then describe ONLY the serving attempt. Summing the iterations, each at its
    own model's rates, is therefore the bill; the top level is used when there are none.
    Compaction entries are skipped: this assistant never asks for compaction.

    A pre-output refusal in an unbilled category is reported in ``iterations`` with its tokens
    and counted here anyway. That over-states a rare refusal's cost by its input tokens — the
    safe direction for a number a budget is enforced on.
    """
    entries = _get(usage, "iterations", None) or []
    parts: list[tuple[str, tuple[int, int, int, int, int]]] = []
    fallback = False
    for entry in entries:
        kind = str(_get(entry, "type", "message"))
        if kind not in ("message", "fallback_message"):
            continue
        fallback = fallback or kind == "fallback_message"
        parts.append((str(_get(entry, "model", model) or model), _tokens(entry)))
    if not parts:
        parts = [(model, _tokens(usage))]

    total = 0.0
    priced = True
    sums = [0, 0, 0, 0]
    for served_by, (inp, out, read, w5, w1) in parts:
        rates = rates_for(served_by)
        if rates is None:
            rates, priced = _CEILING, False
        total += (
            inp * rates.input
            + out * rates.output
            + read * rates.cache_read
            + w5 * rates.cache_write_5m
            + w1 * rates.cache_write_1h
        ) / 1_000_000.0
        sums[0] += inp
        sums[1] += out
        sums[2] += read
        sums[3] += w5 + w1

    searches = int(_get(_get(usage, "server_tool_use", None), "web_search_requests", 0) or 0)
    total += searches * WEB_SEARCH_USD
    return CallCost(
        model=parts[-1][0],
        input_tokens=sums[0],
        output_tokens=sums[1],
        cache_read_tokens=sums[2],
        cache_write_tokens=sums[3],
        web_search_requests=searches,
        cost_usd=round(total, 6),
        priced=priced,
        fallback=fallback,
    )
