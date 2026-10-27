"""The investment assistant's routes: two public ones a visitor reaches through the embed token,
and two private ones the agent's dashboard uses.

``GET  /public/properties/{token}/advisor``        the card and whether to show the chat
``POST /public/properties/{token}/advisor/chat``   one question, answered as a server-sent stream
``GET  /properties/{id}/investment``                the agent's figures, the card, usage, rates
``PUT  /properties/{id}/investment``                save the agent's figures and the opt-in

Why the chat is a POST that answers with ``text/event-stream``
-----------------------------------------------------------
``EventSource`` can only GET, and a question does not belong in a URL (it lands in access logs
and browser history). So the browser POSTs with ``fetch`` and reads the body as a stream.

The body is JSON SENT AS ``text/plain``. That is deliberate: the public prefix answers any origin
— the player is framed on third-party listing portals — and the permissive CORS policy there
allows GET/HEAD only, because nothing else under ``/public`` writes. A ``text/plain`` POST with
no custom headers and no credentials is a CORS "simple request": the browser sends it without a
preflight, and the response's ``Access-Control-Allow-Origin: *`` lets the page read the stream.
Widening the public CORS policy to POST for every route under it would have been the other way,
and a wider door than this one route needs. The body is parsed here with the same pydantic model
either way.

Who may ask
-----------
The property must be PUBLISHED (the token resolves through ``public._property_by_token``, which
answers 404 otherwise), its agent must have opted in, and it must have a price. Then, in order:
the deployment can answer at all, the question is not too long, the visitor and the property are
inside their rate windows, today's spend is under the property's and the installation's caps,
and the conversation still exists and has room. Each refusal is a JSON error with a Turkish
message the panel shows as it is — and none of them costs a model call.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, SessionDep, require_property, scenes_of_property
from biosphere.api.routes.public import _property_by_token
from biosphere.core.config import get_settings
from biosphere.core.logging import get_logger
from biosphere.core.models import ApiException, Property, utcnow
from biosphere.invest import advisor, prompts, summary
from biosphere.invest import store as store_mod
from biosphere.invest.inputs import InvestmentInputs
from biosphere.invest.ratelimit import Limiter, client_digest
from biosphere.invest.rates import CostRates, load_rates

log = get_logger("biosphere.api.advisor")

public_router = APIRouter(prefix="/public/properties", tags=["advisor"])
private_router = APIRouter(prefix="/properties", tags=["advisor"])

#: A question is a few hundred characters; this is the ceiling on the raw body before parsing.
_MAX_BODY_BYTES = 16 * 1024

#: Seconds of silence after which the stream sends an SSE comment, so a proxy that closes idle
#: connections does not close one whose model is thinking.
_HEARTBEAT_S = 12.0


# ======================================================================================
# Process-wide state (memory only)
# ======================================================================================


class _State:
    """The conversation store, the limiter and the concurrency gate, built from settings on
    first use. One per process; :func:`reset` exists for the tests."""

    def __init__(self) -> None:
        settings = get_settings()
        self.conversations = store_mod.ConversationStore(
            ttl_s=float(settings.advisor_conversation_ttl_minutes) * 60.0,
            max_items=int(settings.advisor_max_conversations),
        )
        self.limiter = Limiter(
            ip_per_minute=int(settings.advisor_rate_ip_per_minute),
            ip_per_hour=int(settings.advisor_rate_ip_per_hour),
            property_per_hour=int(settings.advisor_rate_property_per_hour),
        )
        self.max_concurrent = max(1, int(settings.advisor_max_concurrent))
        self.active = 0


_state: _State | None = None


def state() -> _State:
    global _state
    if _state is None:
        _state = _State()
    return _state


def reset() -> None:
    """Forget every conversation and every count. For tests and for an operator's restart."""
    global _state
    _state = None


def current_rates() -> CostRates:
    return load_rates(get_settings().advisor_rates)


# ======================================================================================
# Shared helpers
# ======================================================================================


def _inputs(property: Property) -> InvestmentInputs:
    return InvestmentInputs.from_column(property.investment)


def _budget_cap(inputs: InvestmentInputs) -> float:
    settings = get_settings()
    wanted = inputs.daily_budget_usd if inputs.daily_budget_usd is not None else settings.advisor_daily_budget_usd
    return max(0.0, min(float(wanted), float(settings.advisor_daily_budget_max_usd)))


def _over_budget(property_id: str, inputs: InvestmentInputs) -> bool:
    """Today's spend has reached the property's cap or the installation's."""
    settings = get_settings()
    if store_mod.spent_today(property_id) >= _budget_cap(inputs):
        return True
    return store_mod.spent_today(None) >= float(settings.advisor_global_daily_budget_usd)


def _card(inputs: InvestmentInputs, rates: CostRates) -> sc.AdvisorCard | None:
    data = summary.card(inputs, rates)
    return sc.AdvisorCard.model_validate(data) if data else None


def _error(code: str, en: str, tr: str, status: int, *, retry_after: float | None = None) -> ApiException:
    exc = ApiException(code, en, tr, status_code=status)
    if retry_after is not None:
        exc.hint = f"retry_after={max(1, math.ceil(retry_after))}"
    return exc


# ======================================================================================
# Public: the card
# ======================================================================================


@public_router.get(
    "/{public_token}/advisor",
    response_model=sc.AdvisorPublicState,
    summary="Investment card and chat availability",
)
def advisor_state(public_token: str, session: SessionDep, response: Response, locale: str = "tr") -> sc.AdvisorPublicState:
    """What the player asks on mount. Calls no model; reads one row and the usage sums."""
    property = _property_by_token(session, public_token)
    inputs = _inputs(property)
    response.headers["Cache-Control"] = "no-store"
    if not inputs.chat_ready():
        return sc.AdvisorPublicState(enabled=False)

    settings = get_settings()
    status = advisor.describe()
    reason = status["reason"]
    available = bool(status["available"])
    if available and _over_budget(property.id, inputs):
        available, reason = False, "budget"
    if available:
        # The visitor has just opened the listing; a question may follow. Load the client
        # library now, off the request, rather than inside that question.
        advisor.warm_up()
    return sc.AdvisorPublicState(
        enabled=True,
        chat_available=available,
        reason=reason,
        card=_card(inputs, current_rates()),
        suggestions=prompts.suggestions(inputs, "en" if locale == "en" else "tr") if available else [],
        max_message_chars=int(settings.advisor_max_message_chars),
        max_turns=int(settings.advisor_max_turns),
    )


# ======================================================================================
# Public: the chat
# ======================================================================================


def _client_address(request: Request) -> str:
    if get_settings().advisor_trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


async def _with_heartbeat(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[bytes]:
    """The events as SSE frames, with a comment line whenever the model has been silent.

    The pending ``__anext__`` is waited on, never cancelled, when the heartbeat fires: cancelling
    it would cancel the model call mid-thought.
    """
    iterator = events.__aiter__()
    pending: asyncio.Task[dict[str, Any]] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=_HEARTBEAT_S)
            if not done:
                yield b": ping\n\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            finally:
                pending = None
            yield _sse(event)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except BaseException:  # noqa: BLE001 - the stream is over either way
                pass
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except BaseException:  # noqa: BLE001
                pass


@public_router.post(
    "/{public_token}/advisor/chat",
    summary="Ask the investment assistant (server-sent events)",
    response_class=StreamingResponse,
)
async def advisor_chat(public_token: str, request: Request) -> StreamingResponse:
    settings = get_settings()
    raw = await request.body()
    if len(raw) > _MAX_BODY_BYTES:
        raise _error("payload_too_large", "The question is too long.", "Soru çok uzun.", 413)
    try:
        payload = sc.AdvisorChatRequest.model_validate_json(raw or b"{}")
    except ValidationError as exc:
        raise _error(
            "validation_error",
            "The question could not be read.",
            "Soru okunamadı.",
            422,
        ) from exc
    locale = payload.locale
    text = payload.message.strip()

    def _load() -> tuple[Property, InvestmentInputs]:
        from sqlmodel import Session

        from biosphere.core.db import get_engine

        with Session(get_engine()) as session:
            prop = _property_by_token(session, public_token)
            session.expunge(prop)
            return prop, _inputs(prop)

    property, inputs = await run_in_threadpool(_load)
    if not inputs.chat_ready():
        # The same 404 an unknown token gets: a listing without the assistant does not have
        # a chat endpoint, as far as anyone outside can tell.
        raise _error("advisor_not_found", "This listing has no investment assistant.", "Bu ilanda yatırım asistanı yok.", 404)

    cfg = advisor.config(settings)
    status = advisor.describe(cfg)
    if not status["available"]:
        raise _error(
            "advisor_unavailable",
            "The assistant is not available on this server.",
            "Yatırım asistanı bu sunucuda şu anda kapalı.",
            503,
        )
    if not text:
        raise _error("validation_error", "The question is empty.", "Soru boş.", 422)
    if len(text) > cfg.max_message_chars:
        raise _error(
            "message_too_long",
            f"Questions are limited to {cfg.max_message_chars} characters.",
            f"Soru en fazla {cfg.max_message_chars} karakter olabilir.",
            422,
        )

    shared = state()
    ip_key = "ip:" + client_digest(_client_address(request))
    blocked = shared.limiter.check(ip_key=ip_key, property_id=property.id)
    if blocked is not None:
        scope, wait = blocked
        log.info("advisor.rate_limited", scope=scope, property=property.id)
        raise _error(
            "rate_limited",
            "Too many questions. Wait a moment and ask again.",
            "Çok sık soru gönderildi. Biraz bekleyip tekrar sorun.",
            429,
            retry_after=wait,
        )

    if await run_in_threadpool(_over_budget, property.id, inputs):
        raise _error(
            "budget_exhausted",
            "The assistant has reached today's limit for this listing. The figures above stay available.",
            "Yatırım asistanı bu ilan için bugünkü kullanım sınırına ulaştı. Yukarıdaki hesaplamalar geçerli; yarın tekrar sorabilirsiniz.",
            429,
        )

    conversation: store_mod.Conversation | None = None
    if payload.conversation_id:
        conversation = shared.conversations.get(payload.conversation_id, public_token=public_token)
        if conversation is None:
            raise _error(
                "conversation_expired",
                "This conversation has ended. A new one will start.",
                "Bu sohbetin süresi doldu; yeni bir sohbet başlıyor.",
                409,
            )
        if conversation.turns >= cfg.max_turns:
            raise _error(
                "conversation_limit",
                "This conversation has reached its question limit. Start a new one.",
                "Bu sohbet soru sınırına ulaştı. Yeni bir sohbet başlatın.",
                409,
            )
        if conversation.context_tokens >= cfg.max_context_tokens:
            raise _error(
                "conversation_full",
                "This conversation has grown too long. Start a new one.",
                "Bu sohbet çok uzadı. Yeni bir sohbet başlatın.",
                409,
            )
        if conversation.lock.locked():
            raise _error("turn_in_progress", "An answer is still being written.", "Önceki yanıt henüz yazılıyor.", 409)

    if shared.active >= shared.max_concurrent:
        raise _error("busy", "The assistant is busy. Try again shortly.", "Asistan şu anda yoğun. Birazdan tekrar deneyin.", 503)

    if conversation is None:
        conversation = await run_in_threadpool(_new_conversation, property, inputs, cfg.model, public_token)

    shared.limiter.record(ip_key=ip_key, property_id=property.id)

    async def events() -> AsyncIterator[dict[str, Any]]:
        shared.active += 1
        try:
            async with conversation.lock:
                yield {
                    "type": "meta",
                    "conversation_id": conversation.id,
                    "turn": conversation.turns + 1,
                    "max_turns": cfg.max_turns,
                }
                try:
                    async for event in advisor.run_turn(conversation, text, cfg, locale=locale):
                        yield event
                except Exception as exc:  # noqa: BLE001 - the stream is open; say so in it
                    # Headers are long sent, so an exception here cannot become a 500 — it
                    # would only cut the stream, which the panel can only call a network
                    # failure. An error event it can show in the visitor's language.
                    log.error("advisor.turn_failed", property=conversation.property_id, error=str(exc)[:300])
                    yield {"type": "error", "code": "unavailable", "message": advisor.message("unavailable", locale)}
        finally:
            shared.active -= 1

    return StreamingResponse(
        _with_heartbeat(events()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # nginx buffers a proxied body by default, which turns a stream into one late blob.
            "X-Accel-Buffering": "no",
        },
    )


def _new_conversation(property: Property, inputs: InvestmentInputs, model: str, public_token: str) -> store_mod.Conversation:
    """Freeze what the model will be told for the life of this conversation.

    The tour's measured facts are read from the cached outlines on disk (no model, no GPU); a
    failure there costs the block its measurements, never the conversation.
    """
    from sqlmodel import Session

    from biosphere.ai.listing import collect
    from biosphere.core.db import get_engine
    from biosphere.core.storage import get_storage

    rates = current_rates()
    tour = None
    try:
        with Session(get_engine()) as session:
            scenes = scenes_of_property(session, property.id)
            for scene in scenes:
                session.expunge(scene)
        tour = collect(property, scenes, get_storage())
    except Exception as exc:  # noqa: BLE001 - measurements are optional context
        log.warning("advisor.tour_facts_unavailable", property=property.id, error=str(exc)[:160])
    facts = prompts.facts_block(inputs, rates, summary.card(inputs, rates), tour, store_mod.istanbul_today())
    return state().conversations.create(
        property_id=property.id,
        public_token=public_token,
        model=model,
        facts=facts,
        inputs=inputs,
        rates=rates,
    )


# ======================================================================================
# Private: the agent's panel
# ======================================================================================

_RATE_LABELS: tuple[tuple[str, str, str], ...] = (
    ("tapu_harci", "Tapu harcı", "tapu_harci"),
    ("doner_sermaye", "Tapu döner sermaye ücreti", "doner_sermaye"),
    ("commission", "Emlakçı komisyonu (alıcı, yasal tavan)", "commission"),
    ("mortgage_rate", "Örnek konut kredisi faizi", "mortgage_rate"),
    ("ltv", "BDDK kredi/değer sınırı (A/B enerji sınıfı)", "ltv"),
    ("emlak_vergisi", "Emlak vergisi oranı (konut)", "emlak_vergisi"),
    ("ekspertiz", "Ekspertiz ve kredi tahsis ücreti", "ekspertiz"),
    ("dask", "DASK (girilmezse)", "dask"),
    ("defaults", "Varsayımlar", ""),
)


def _rate_rows(rates: CostRates) -> list[sc.AdvisorRateRow]:
    from biosphere.invest import calc

    values = {
        "tapu_harci": f"alıcı {calc.fmt_pct(rates.tapu_harci_buyer_pct, 1)} + satıcı {calc.fmt_pct(rates.tapu_harci_seller_pct, 1)}",
        "doner_sermaye": f"{calc.fmt_tl(rates.doner_sermaye_tl)} (ilçe katsayısına göre değişir)",
        "commission": f"{calc.fmt_pct(rates.commission_pct, 1)} + {calc.fmt_pct(rates.kdv_pct, 0)} KDV",
        "mortgage_rate": f"aylık {calc.fmt_pct(rates.mortgage_monthly_rate_pct)}, {rates.mortgage_term_months} ay, "
        f"örnek peşinat {calc.fmt_pct(rates.mortgage_down_payment_pct, 0)}",
        "ltv": ", ".join(
            (f"≤{calc.fmt_number(b.up_to_tl / 1e6, 0)} M TL" if b.up_to_tl else "üstü") + f" {calc.fmt_pct(b.max_ltv_pct, 0)}"
            for b in rates.ltv_bands
        ),
        "emlak_vergisi": f"büyükşehir {calc.fmt_pct(rates.emlak_vergisi_metro_pct, 1)}, diğer {calc.fmt_pct(rates.emlak_vergisi_other_pct, 1)} "
        "(vergi değeri üzerinden; girilmezse fiyat üzerinden üst sınır)",
        "ekspertiz": f"{calc.fmt_tl(rates.ekspertiz_tl)}; tahsis en fazla kredinin {calc.fmt_pct(rates.kredi_tahsis_pct, 1)}'i",
        "dask": f"{calc.fmt_tl(rates.dask_yearly_tl)}/yıl",
        "defaults": f"yılda {calc.fmt_number(rates.vacancy_months, 1)} ay boşluk; bakım kiranın {calc.fmt_pct(rates.maintenance_pct_of_rent, 0)}'i",
    }
    rows = []
    for key, label, source_key in _RATE_LABELS:
        source = rates.sources.get(source_key) if source_key else None
        rows.append(
            sc.AdvisorRateRow(
                key=key,
                label=label,
                value=values[key],
                source=sc.AdvisorSourceRef(title=source.title, url=source.url, date=source.date) if source else None,
            )
        )
    return rows


def _investment_state(property: Property) -> sc.InvestmentState:
    settings = get_settings()
    inputs = _inputs(property)
    rates = current_rates()
    status = advisor.describe()
    cap = _budget_cap(inputs)
    spent = store_mod.spent_today(property.id)
    spent_all = store_mod.spent_today(None)

    blockers: list[str] = []
    if not inputs.enabled:
        blockers.append("not_enabled")
    if not inputs.asking_price_tl:
        blockers.append("no_price")
    if not property.published:
        blockers.append("not_published")
    if status["reason"]:
        blockers.append(str(status["reason"]))
    if spent >= cap or spent_all >= float(settings.advisor_global_daily_budget_usd):
        blockers.append("budget")

    return sc.InvestmentState(
        inputs=inputs,
        card=_card(inputs, rates),
        status=sc.AdvisorStatus(**status),
        visible_to_visitors=not blockers,
        blockers=blockers,
        usage=sc.AdvisorUsageSummary.model_validate(store_mod.usage_summary(property.id)),
        budget=sc.AdvisorBudget(
            daily_budget_usd=round(cap, 2),
            default_usd=float(settings.advisor_daily_budget_usd),
            operator_max_usd=float(settings.advisor_daily_budget_max_usd),
            spent_today_usd=round(spent, 4),
            global_daily_budget_usd=float(settings.advisor_global_daily_budget_usd),
            global_spent_today_usd=round(spent_all, 4),
        ),
        rates=_rate_rows(rates),
        rates_as_of=rates.as_of,
        stale_rates=rates.stale_keys(),
    )


@private_router.get(
    "/{property_id}/investment",
    response_model=sc.InvestmentState,
    summary="The property's investment-assistant settings, card and usage",
)
def get_investment(property_id: str, session: SessionDep, user: CurrentUser) -> sc.InvestmentState:
    """Reads one row and sums the usage table. Calls no model."""
    return _investment_state(require_property(session, property_id, user))


@private_router.put(
    "/{property_id}/investment",
    response_model=sc.InvestmentState,
    summary="Save the agent's figures and the opt-in",
)
def put_investment(
    property_id: str, payload: sc.InvestmentInputs, session: SessionDep, user: CurrentUser
) -> sc.InvestmentState:
    """Replace the whole document.

    Written whole, like every JSON column here (SQLAlchemy does not see a mutation inside one),
    and validated by the same model the calculators read, so nothing the engine cannot compute
    with is ever stored. Conversations already open keep the figures they started with; the
    next one sees these.
    """
    property = require_property(session, property_id, user)
    property.investment = payload.model_dump(mode="json", exclude_none=True)
    property.updated_at = utcnow()
    session.add(property)
    session.commit()
    session.refresh(property)
    log.info("advisor.inputs_saved", property=property.id, enabled=payload.enabled)
    return _investment_state(property)
