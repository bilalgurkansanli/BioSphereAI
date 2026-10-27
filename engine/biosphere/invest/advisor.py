"""One visitor question, answered: the streaming tool loop against the Messages API.

The shape of a turn is fixed by what an investment answer needs. The model may search the web
for market figures (a server tool: Anthropic runs the search and hands the results back inside
the same response), may call the calculators (client tools: this module runs them and sends the
results back), and then writes. That is one to three API calls — "rounds" — per question,
streamed to the browser as they happen, so a visitor sees "piyasa verileri aranıyor…" while the
search runs rather than a frozen panel for twenty seconds.

What the API requires of a loop like this, and where each requirement is met:

* **The history is append-only.** Every assistant turn goes back exactly as it came — thinking
  blocks with their signatures, search results, citations — because the newest models bind a
  thinking block to the exact conversation before it and reject an edited one. This module only
  ever appends, and appends a question's messages to the stored conversation only once the
  question has been answered completely (:meth:`~biosphere.invest.store.Conversation.commit`).
  A question that fails half-way leaves the stored history exactly as it was.
* **The prefix never changes within a conversation.** Model, system blocks and the tool list are
  frozen when the conversation starts. The last round of a question switches tools off with
  ``tool_choice`` — a request parameter, not part of the prefix — rather than by removing them.
* **A refusal is a 200.** ``stop_reason == "refusal"`` is checked before anything is read, the
  partial answer is withdrawn (``discard``), and with server-side fallback switched on the API
  usually never returns one: it re-runs the declined call on Anthropic's recommended model.
* **``pause_turn`` resumes by re-sending, never by adding a "continue".**
* **Forced tool use is gone on Opus 5.5 / Sonnet 5.5.** The model is steered to the calculators by
  the prompt and the tool descriptions; ``strict`` keeps the arguments schema-valid.
* **Eager input streaming means the client validates.** Tool inputs are validated by pydantic in
  :mod:`biosphere.invest.tools`; an unparseable one raises ``ValueError`` from the stream, which
  ends the question with an error rather than running a half-read tool.

Every call is priced from its own usage report and written down before the next one starts, so
the daily budget counts money already spent, including a question the visitor walked away from.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from biosphere.core.logging import get_logger
from biosphere.invest import pricing, prompts
from biosphere.invest import store as store_mod
from biosphere.invest import tools as tools_mod
from biosphere.invest.store import Conversation

__all__ = [
    "AdvisorConfig",
    "ModelProfile",
    "profile",
    "config",
    "describe",
    "build_request",
    "run_turn",
    "client_factory",
    "MESSAGES",
]

log = get_logger("biosphere.invest.advisor")


# ======================================================================================
# Configuration
# ======================================================================================


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """What one model family accepts. Sending a parameter a model rejects is a 400 per call."""

    #: ``output_config.effort`` — every current model except Haiku 4.5.
    effort: bool
    #: ``web_search_20260209`` (dynamic filtering) on the 5.x models; Haiku 4.5 has only the
    #: basic ``web_search_20250305``.
    web_search_type: str
    #: Server-side ``fallbacks: "default"`` — Opus 5.5 and Sonnet 5.5, Claude API only.
    fallbacks: bool


def profile(model: str) -> ModelProfile:
    if model.startswith("claude-haiku-4-5"):
        return ModelProfile(effort=False, web_search_type="web_search_20250305", fallbacks=False)
    if model.startswith(("claude-opus-5-5", "claude-sonnet-5-5")):
        return ModelProfile(effort=True, web_search_type="web_search_20260209", fallbacks=True)
    return ModelProfile(effort=True, web_search_type="web_search_20260209", fallbacks=False)


@dataclass(frozen=True, slots=True)
class AdvisorConfig:
    """The settings a turn runs under, read once per request."""

    enabled: bool
    key: str
    model: str
    effort: str
    max_output_tokens: int
    max_rounds: int
    max_searches_per_call: int
    web_search: bool
    allowed_domains: tuple[str, ...]
    fallbacks: bool
    max_turns: int
    max_message_chars: int
    max_context_tokens: int
    timeout_s: float
    #: ``auto`` (the newest version the model takes), ``dynamic`` (``web_search_20260209``) or
    #: ``basic`` (``web_search_20250305``). See :func:`_web_search_tool` for the trade.
    web_search_version: str = "auto"
    eager_input_streaming: bool = True


def config(settings: Any | None = None, **overrides: Any) -> AdvisorConfig:
    """The advisor's configuration from :class:`~biosphere.core.config.Settings`.

    ``overrides`` exist for the bench, which runs the same questions through several models
    and effort levels in one process without touching the environment.
    """
    if settings is None:
        from biosphere.core.config import get_settings

        settings = get_settings()
    effort = str(getattr(settings, "advisor_effort", "medium") or "medium").lower()
    values = {
        "enabled": bool(getattr(settings, "advisor_enabled", True)),
        "key": str(getattr(settings, "advisor_key", "") or ""),
        "model": str(getattr(settings, "advisor_model", "claude-sonnet-5-5") or "claude-sonnet-5-5"),
        "effort": effort if effort in ("low", "medium", "high", "xhigh", "max") else "medium",
        "max_output_tokens": max(1024, int(getattr(settings, "advisor_max_output_tokens", 8000))),
        "max_rounds": max(2, int(getattr(settings, "advisor_max_rounds", 4))),
        "max_searches_per_call": max(1, int(getattr(settings, "advisor_max_searches_per_call", 3))),
        "web_search": bool(getattr(settings, "advisor_web_search", True)),
        "allowed_domains": tuple(str(d).strip() for d in (getattr(settings, "advisor_web_allowed_domains", None) or []) if str(d).strip()),
        "fallbacks": bool(getattr(settings, "advisor_fallbacks", True)),
        "max_turns": max(1, int(getattr(settings, "advisor_max_turns", 12))),
        "max_message_chars": max(40, int(getattr(settings, "advisor_max_message_chars", 600))),
        "max_context_tokens": max(8000, int(getattr(settings, "advisor_max_context_tokens", 80000))),
        "timeout_s": max(10.0, float(getattr(settings, "advisor_timeout_s", 120.0))),
        "web_search_version": str(getattr(settings, "advisor_web_search_version", "auto") or "auto").lower(),
    }
    values.update(overrides)
    return AdvisorConfig(**values)


def _sdk_present() -> bool:
    from importlib.util import find_spec

    try:
        return find_spec("anthropic") is not None
    except Exception:  # noqa: BLE001 - a broken installation is "no"
        return False


def describe(cfg: AdvisorConfig | None = None) -> dict[str, Any]:
    """Whether a question would be answered here, and why not. Cheap: no network, no import.

    The three ways the chat is off have three different fixes — an operator switch, a key, a
    package — so the dashboard is told which one it is rather than "unavailable".
    """
    cfg = cfg or config()
    reason: str | None = None
    if not cfg.enabled:
        reason = "disabled"
    elif not cfg.key:
        reason = "no_key"
    elif not _sdk_present():
        reason = "no_sdk"
    return {"available": reason is None, "reason": reason, "model": cfg.model, "web_search": cfg.web_search}


# ======================================================================================
# The client
# ======================================================================================

_clients: dict[tuple[int, str, float], Any] = {}
_warmed = False


def warm_up() -> None:
    """Import the client library in the background, once.

    Measured on this laptop: ``import anthropic`` takes 4.4 s cold, and the first question
    after an engine restart took 47 s end to end under load against 8 s for the model call
    itself — the rest was this import, paid inside the visitor's first request. The player
    asks for the investment card when a listing opens, long before anybody types a question,
    so that request starts the import on a daemon thread and the question finds it done.
    """
    global _warmed
    if _warmed:
        return
    _warmed = True

    def _import() -> None:
        try:
            import anthropic  # noqa: F401
        except Exception:  # noqa: BLE001 - describe() reports a missing SDK on its own
            pass

    import threading

    threading.Thread(target=_import, name="advisor-warm-up", daemon=True).start()


def _default_client(cfg: AdvisorConfig) -> Any:
    """One ``AsyncAnthropic`` per event loop.

    Per LOOP because the client's connection pool belongs to the loop it was first used on; a
    process that runs several loops (the bench does, with ``asyncio.run`` per configuration)
    would otherwise hand a closed loop's pool to a new one.
    """
    import anthropic

    loop_id = id(asyncio.get_running_loop())
    key = (loop_id, cfg.key[-6:], cfg.timeout_s)
    client = _clients.get(key)
    if client is None:
        client = anthropic.AsyncAnthropic(api_key=cfg.key, timeout=cfg.timeout_s, max_retries=2)
        _clients[key] = client
    return client


#: Swapped by the tests for a scripted fake, so the suite never makes a billed call.
client_factory: Callable[[AdvisorConfig], Any] = _default_client


# ======================================================================================
# The request
# ======================================================================================


def _web_search_tool(cfg: AdvisorConfig, prof: ModelProfile) -> dict[str, Any]:
    kind = prof.web_search_type
    if cfg.web_search_version == "basic":
        kind = "web_search_20250305"
    elif cfg.web_search_version == "dynamic" and prof.web_search_type == "web_search_20260209":
        kind = "web_search_20260209"
    tool: dict[str, Any] = {
        "type": kind,
        "name": "web_search",
        "max_uses": cfg.max_searches_per_call,
        # Country only, no city. The tool list is the first thing in the prompt, so a
        # per-property city here would split the cache of the (much longer) rules behind it
        # by property; the district goes into the query text instead, where it matters more.
        "user_location": {"type": "approximate", "country": "TR", "timezone": "Europe/Istanbul"},
    }
    if cfg.allowed_domains:
        tool["allowed_domains"] = list(cfg.allowed_domains)
    return tool


def build_request(
    conversation: Conversation,
    messages: list[dict[str, Any]],
    cfg: AdvisorConfig,
    *,
    tools_enabled: bool = True,
) -> dict[str, Any]:
    """The keyword arguments of one ``client.beta.messages.stream`` call.

    Three cache breakpoints, at the three stability boundaries: the end of the rules (shared by
    every conversation on the installation), the end of the property's facts (shared by that
    property's conversations), and the conversation's tail (top-level automatic caching, which
    moves forward as the chat grows).
    """
    prof = profile(conversation.model)
    tools = tools_mod.definitions(eager_input_streaming=cfg.eager_input_streaming)
    if cfg.web_search:
        tools.append(_web_search_tool(cfg, prof))
    request: dict[str, Any] = {
        "model": conversation.model,
        "max_tokens": cfg.max_output_tokens,
        "system": [
            {"type": "text", "text": prompts.SYSTEM, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": conversation.facts, "cache_control": {"type": "ephemeral"}},
        ],
        "tools": tools,
        "messages": messages,
        "cache_control": {"type": "ephemeral"},
    }
    if prof.effort:
        request["output_config"] = {"effort": cfg.effort}
    if not tools_enabled:
        request["tool_choice"] = {"type": "none"}
    if prof.fallbacks and cfg.fallbacks:
        # The "default" form: Anthropic routes each refusal category to the model it
        # recommends, so there is no list of fallback models here to go stale.
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"
    return request


# ======================================================================================
# What the visitor is told
# ======================================================================================

#: The panel's sentences for the outcomes that are not an answer. Engine-side because the
#: engine decides which one applies; both languages, because the visitor's is not known here.
MESSAGES: dict[str, tuple[str, str]] = {
    "refusal": (
        "Bu soruya yanıt veremiyorum. Bu evin getirisi, bölgedeki fiyatlar ya da alım maliyeti hakkında sorabilirsiniz.",
        "I can't answer that one. Ask about this flat's yield, prices in the area or the cost of buying.",
    ),
    "truncated": (
        "Yanıt uzunluk sınırında kesildi; daha dar bir soru sorarsanız tamamını yazabilirim.",
        "The answer was cut off at its length limit; a narrower question will get a complete one.",
    ),
    "upstream_busy": (
        "Asistan şu anda yoğun. Birkaç saniye sonra tekrar deneyin.",
        "The assistant is busy right now. Try again in a few seconds.",
    ),
    "unavailable": (
        "Asistan şu anda yanıt veremiyor. Yukarıdaki hesaplamalar ilan bilgileriyle güncel.",
        "The assistant cannot answer right now. The figures above are current with the listing.",
    ),
    "network": (
        "Asistana ulaşılamadı. Bağlantı sorunu olabilir; tekrar deneyin.",
        "The assistant could not be reached. Try again.",
    ),
    "timeout": (
        "Yanıt çok uzun sürdü ve durduruldu. Soruyu tekrar sorabilirsiniz.",
        "The answer took too long and was stopped. You can ask again.",
    ),
}


def message(code: str, locale: str) -> str:
    tr, en = MESSAGES.get(code, MESSAGES["unavailable"])
    return en if locale == "en" else tr


# ======================================================================================
# The turn
# ======================================================================================


def _blocks(content: list[Any]) -> list[dict[str, Any]]:
    """The response's content as plain JSON, exactly as received (unset fields stay unset)."""
    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            out.append(block)
        elif hasattr(block, "to_dict"):
            out.append(block.to_dict(mode="json"))
        else:  # pragma: no cover - an SDK that changed shape
            out.append(dict(block))
    return out


def echo_content(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The assistant turn as it must be sent back, after a server-side fallback.

    The API's rules for echoing a turn that crossed a ``fallback`` block: keep the block where it
    is; before the LAST one, drop thinking, redacted thinking, connector text and client
    ``tool_use`` blocks, and drop a ``server_tool_use`` whose result is missing. Everything else,
    and everything after the boundary, goes back as received. A turn without a fallback block is
    returned untouched.
    """
    last = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    if last < 0:
        return blocks
    answered = {b.get("tool_use_id") for b in blocks if str(b.get("type", "")).endswith("tool_result")}
    out: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        kind = block.get("type")
        if index < last:
            if kind in ("thinking", "redacted_thinking", "connector_text", "tool_use"):
                continue
            if kind == "server_tool_use" and block.get("id") not in answered:
                continue
        out.append(block)
    return out


@dataclass
class TurnStats:
    """What one question cost and how long it took — for the bench and the logs."""

    calls: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    first_text_ms: int | None = None
    elapsed_ms: int = 0
    models: list[str] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)
    fallback: bool = False
    priced: bool = True
    tools: list[str] = field(default_factory=list)
    outcome: str = "ok"


def _context_tokens(usage: Any) -> int:
    """Prompt size of a call: what the next call will have to read again."""
    get = (lambda n: int(getattr(usage, n, 0) or 0)) if usage is not None else (lambda n: 0)
    return get("input_tokens") + get("cache_read_input_tokens") + get("cache_creation_input_tokens")


async def _record(conversation: Conversation, turn: int, round_: int, cost: pricing.CallCost, latency_ms: int, stop: str) -> None:
    from fastapi.concurrency import run_in_threadpool

    await run_in_threadpool(
        store_mod.record_call,
        property_id=conversation.property_id,
        conversation_id=conversation.id,
        turn=turn,
        round_=round_,
        cost=cost,
        latency_ms=latency_ms,
        stop_reason=stop,
    )


async def run_turn(
    conversation: Conversation,
    text: str,
    cfg: AdvisorConfig,
    *,
    locale: str = "tr",
    stats: TurnStats | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Answer one question, yielding the events the panel renders.

    Events: ``round`` (a call begins), ``status`` (``thinking`` / ``searching`` + ``query`` /
    ``computing`` + ``tool`` / ``writing``), ``text`` (a ``delta`` of the answer), ``retract``
    (withdraw the text since the last ``round`` — see below), ``cite`` (a source the answer
    cites), ``calc`` (the headline figures of one calculator call), ``discard`` (withdraw ALL the
    text so far — a refusal), ``notice`` and ``error`` (``code`` + ``message``), then ``done``.

    TEXT WRITTEN BEFORE A TOOL CALL IS WITHDRAWN. A model that narrates — "Hemen araştırıyorum.",
    "kalem kalem dökümü için aracı çağırıyorum" — writes it before it has a single result, and
    the measured transcripts showed it glued to the front of the real answer ("araştırıyorum.Şimdi
    bulduğum..."). The 5.5 models mostly put such notes in thinking blocks; Haiku writes them as
    text. So the text of a call that ends in a tool call is streamed (the visitor sees progress)
    and then retracted; it stays in the stored history, which must not be edited.

    The conversation is committed only when the answer is complete. Cancelling the iteration
    (the visitor closed the panel) stops the call in flight and still writes down its cost.
    """
    import anthropic

    stats = stats if stats is not None else TurnStats()
    client = client_factory(cfg)
    ctx = tools_mod.ToolContext(inputs=conversation.inputs, rates=conversation.rates)
    turn = conversation.turns + 1
    pending: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    started = time.perf_counter()
    kept_text = False
    separate = False
    results: dict[str, tuple[str, str | None]] = {}
    cited: set[str] = set()
    context_tokens = 0
    finished = False

    for round_ in range(1, cfg.max_rounds + 1):
        last_round = round_ == cfg.max_rounds
        request = build_request(conversation, conversation.messages + pending, cfg, tools_enabled=not last_round)
        call_started = time.perf_counter()
        stream_ref: Any = None
        streamed_chars = 0
        completed = False
        final: Any = None
        round_text = False
        yield {"type": "round", "n": round_}
        yield {"type": "status", "phase": "thinking"}
        try:
            async with client.beta.messages.stream(**request) as stream:
                stream_ref = stream
                async for event in stream:
                    if time.perf_counter() - call_started > cfg.timeout_s:
                        raise TimeoutError("advisor call exceeded its deadline")
                    kind = getattr(event, "type", "")
                    if kind == "content_block_start":
                        block = event.content_block
                        btype = getattr(block, "type", "")
                        if btype == "text":
                            if separate and kept_text and not round_text:
                                yield {"type": "text", "delta": "\n\n"}
                            separate = False
                            yield {"type": "status", "phase": "writing"}
                        elif btype in ("server_tool_use", "tool_use"):
                            # Text before a search or a calculation is narration, written before
                            # any result existed — and when the search runs inside the same call
                            # as the answer (a server tool), it would otherwise stay glued to the
                            # front of it. Withdrawn the moment the tool starts.
                            if round_text:
                                yield {"type": "retract"}
                                round_text = False
                            if btype == "server_tool_use":
                                yield {"type": "status", "phase": "searching"}
                            else:
                                yield {"type": "status", "phase": "computing", "tool": getattr(block, "name", "")}
                        elif btype == "web_search_tool_result":
                            content = getattr(block, "content", None)
                            if isinstance(content, list):
                                for item in content:
                                    url = getattr(item, "url", None)
                                    if url:
                                        results[url] = (getattr(item, "title", "") or "", getattr(item, "page_age", None))
                    elif kind == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta" and delta.text:
                            if stats.first_text_ms is None:
                                stats.first_text_ms = int((time.perf_counter() - started) * 1000)
                            round_text = True
                            streamed_chars += len(delta.text)
                            yield {"type": "text", "delta": delta.text}
                        elif dtype == "citations_delta":
                            citation = delta.citation
                            url = getattr(citation, "url", None)
                            if url and url not in cited:
                                cited.add(url)
                                title, age = results.get(url, (getattr(citation, "title", "") or "", None))
                                yield {"type": "cite", "url": url, "title": title or getattr(citation, "title", "") or url, "date": age}
                    elif kind == "content_block_stop":
                        block = getattr(event, "content_block", None)
                        if getattr(block, "type", "") == "server_tool_use" and getattr(block, "name", "") == "web_search":
                            query = (getattr(block, "input", None) or {}).get("query") if isinstance(getattr(block, "input", None), dict) else None
                            if query:
                                yield {"type": "status", "phase": "searching", "query": str(query)[:120]}
                final = await stream.get_final_message()
            completed = True
        except ValueError as exc:
            # Unparseable tool JSON under eager input streaming. There is no tool_use id to
            # answer, so the question ends here; the history is untouched.
            log.warning("advisor.tool_json_unparseable", error=str(exc)[:200])
            stats.outcome = "error"
            yield {"type": "error", "code": "unavailable", "message": message("unavailable", locale)}
            return
        except TimeoutError:
            stats.outcome = "timeout"
            yield {"type": "error", "code": "timeout", "message": message("timeout", locale)}
            return
        except anthropic.RateLimitError as exc:
            log.warning("advisor.rate_limited", status=getattr(exc, "status_code", None))
            stats.outcome = "error"
            yield {"type": "error", "code": "upstream_busy", "message": message("upstream_busy", locale)}
            return
        except anthropic.APIStatusError as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            # The message can name the failing block; the body may hold the prompt, so only the
            # message is logged.
            log.error("advisor.api_error", status=status, error=str(getattr(exc, "message", exc))[:400])
            stats.outcome = "error"
            code = "upstream_busy" if status >= 500 or status == 529 else "unavailable"
            yield {"type": "error", "code": code, "message": message(code, locale)}
            return
        except anthropic.APIConnectionError:
            stats.outcome = "error"
            yield {"type": "error", "code": "network", "message": message("network", locale)}
            return
        finally:
            if not completed and stream_ref is not None:
                # Cut off: the visitor left, or the call failed mid-stream. The tokens were still
                # generated; count them, conservatively, so the budget sees them. Shielded,
                # because Starlette cancels a disconnected response's task scope and anyio's
                # cancellation is level-triggered: any unshielded await here would be cancelled
                # before the row was written.
                import anyio

                with anyio.CancelScope(shield=True):
                    await _record_partial(conversation, turn, round_, stream_ref, streamed_chars, call_started, stats)

        latency_ms = int((time.perf_counter() - call_started) * 1000)
        cost = pricing.cost_of(str(getattr(final, "model", conversation.model)), getattr(final, "usage", None))
        stop = str(getattr(final, "stop_reason", "") or "")
        stats.calls += 1
        stats.cost_usd += cost.cost_usd
        stats.input_tokens += cost.input_tokens
        stats.output_tokens += cost.output_tokens
        stats.cache_read_tokens += cost.cache_read_tokens
        stats.cache_write_tokens += cost.cache_write_tokens
        stats.web_searches += cost.web_search_requests
        stats.models.append(cost.model)
        stats.stop_reasons.append(stop)
        stats.fallback = stats.fallback or cost.fallback
        stats.priced = stats.priced and cost.priced
        context_tokens = max(context_tokens, _context_tokens(getattr(final, "usage", None)))
        await _record(conversation, turn, round_, cost, latency_ms, stop)

        if stop == "refusal":
            details = getattr(final, "stop_details", None)
            log.info("advisor.refusal", category=getattr(details, "category", None), property=conversation.property_id)
            stats.outcome = "refusal"
            yield {"type": "discard"}
            yield {"type": "notice", "code": "refusal", "message": message("refusal", locale)}
            return

        content = echo_content(_blocks(getattr(final, "content", []) or []))
        pending.append({"role": "assistant", "content": content})

        if stop == "pause_turn":
            # A long server-side search loop paused itself. Re-send as is; the API resumes it.
            kept_text = kept_text or round_text
            continue

        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if stop == "tool_use" and tool_uses:
            if last_round:
                break  # tools were off; a tool call here cannot be answered
            if round_text:
                yield {"type": "retract"}
            outputs: list[dict[str, Any]] = []
            for use in tool_uses:
                name = str(use.get("name", ""))
                outcome = tools_mod.run(name, use.get("input"), ctx)
                stats.tools.append(name + ("!" if outcome.is_error else ""))
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": use.get("id"), "content": outcome.content()}
                if outcome.is_error:
                    block["is_error"] = True
                outputs.append(block)
                yield {"type": "calc", **tools_mod.summary_rows(name, outcome)}
            # All results of a round in ONE user message: splitting them teaches the model to
            # stop calling tools in parallel.
            pending.append({"role": "user", "content": outputs})
            separate = True
            continue

        kept_text = kept_text or round_text
        if stop == "max_tokens":
            if tool_uses:
                stats.outcome = "error"
                yield {"type": "error", "code": "truncated", "message": message("truncated", locale)}
                return
            yield {"type": "notice", "code": "truncated", "message": message("truncated", locale)}
        finished = True
        break

    stats.elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not finished:
        stats.outcome = "error"
        yield {"type": "error", "code": "unavailable", "message": message("unavailable", locale)}
        return

    conversation.commit(pending, context_tokens)
    log.info(
        "advisor.answered",
        property=conversation.property_id,
        turn=turn,
        calls=stats.calls,
        cost_usd=round(stats.cost_usd, 5),
        searches=stats.web_searches,
        tools=",".join(stats.tools),
        elapsed_ms=stats.elapsed_ms,
        first_text_ms=stats.first_text_ms,
        fallback=stats.fallback,
    )
    yield {
        "type": "done",
        "turns_left": max(0, cfg.max_turns - conversation.turns),
        "full": conversation.context_tokens >= cfg.max_context_tokens,
    }


async def _record_partial(
    conversation: Conversation,
    turn: int,
    round_: int,
    stream: Any,
    streamed_chars: int,
    call_started: float,
    stats: TurnStats,
) -> None:
    """Price a call that never finished, from what the stream had reported so far.

    Output tokens arrive in the final ``message_delta``, which a cut-off stream never sees, so
    they are ESTIMATED: at least a third of the characters streamed, and at least 500 tokens for
    the thinking that preceded them. Over-counting a cut-off call by a cent is the right error
    for a figure a budget is enforced on.
    """
    try:
        snapshot = stream.current_message_snapshot
    except AssertionError:
        return
    except Exception:  # noqa: BLE001
        return
    usage = getattr(snapshot, "usage", None)
    estimated_out = max(int(getattr(usage, "output_tokens", 0) or 0), math.ceil(streamed_chars / 3), 500)
    model = str(getattr(snapshot, "model", conversation.model) or conversation.model)
    rates = pricing.rates_for(model) or pricing.PRICES["claude-opus-5"]
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    searches = int(getattr(getattr(usage, "server_tool_use", None), "web_search_requests", 0) or 0)
    cost_usd = (
        inp * rates.input + estimated_out * rates.output + read * rates.cache_read + write * rates.cache_write_5m
    ) / 1_000_000.0 + searches * pricing.WEB_SEARCH_USD
    cost = pricing.CallCost(
        model=model,
        input_tokens=inp,
        output_tokens=estimated_out,
        cache_read_tokens=read,
        cache_write_tokens=write,
        web_search_requests=searches,
        cost_usd=round(cost_usd, 6),
        priced=pricing.rates_for(model) is not None,
        fallback=False,
    )
    stats.cost_usd += cost.cost_usd
    stats.outcome = "aborted"
    try:
        await _record(conversation, turn, round_, cost, int((time.perf_counter() - call_started) * 1000), "aborted")
    except Exception:  # noqa: BLE001 - shutting down; the row is best effort
        log.warning("advisor.partial_usage_not_recorded", property=conversation.property_id)
