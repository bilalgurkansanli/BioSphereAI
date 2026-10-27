"""Where a conversation lives while it lasts, and where its bill lives after.

Two stores with opposite lifetimes, kept apart on purpose.

**Conversations live in memory, and only there.** A visitor's questions are not the agent's
data and not ours: they exist to be answered. So the history a conversation needs — the full
assistant content blocks, thinking signatures included, which the API requires back unchanged —
is held in this process for an hour of inactivity at most, bounded in number and size, and is
gone on restart. The privacy policy's promise that nothing about a visitor is kept stays true
without an exception for the chat. The cost is that an engine restart ends open chats; the
panel says so and starts a new one.

**The bill lives in the database.** One :class:`~biosphere.core.models.AdvisorUsage` row per
billed call — tokens, dollars, latency, no content — because the daily budget must survive a
restart (otherwise restarting the engine would reset the cap) and the agent's dashboard sums
it.

The history is APPEND-ONLY by construction: :meth:`Conversation.commit` adds a finished
question's messages at the end and nothing ever rewrites one. That is a correctness property,
not tidiness: the newest models bind every thinking block to the exact conversation before it,
and an edited history is rejected outright on accounts created after 31 August 2026.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func
from sqlmodel import Session, select

from biosphere.core.db import get_engine
from biosphere.core.logging import get_logger
from biosphere.core.models import AdvisorUsage, ensure_utc
from biosphere.invest.inputs import InvestmentInputs
from biosphere.invest.pricing import CallCost
from biosphere.invest.rates import CostRates

__all__ = [
    "Conversation",
    "ConversationStore",
    "ISTANBUL",
    "istanbul_today",
    "day_start_utc",
    "record_call",
    "spent_today",
    "usage_summary",
]

log = get_logger("biosphere.invest.store")

#: Turkey has kept UTC+3 all year since 2016. A fixed offset rather than ``zoneinfo``: the IANA
#: database is not shipped with Python on Windows, and the one zone this needs never shifts.
ISTANBUL = timezone(timedelta(hours=3), "Europe/Istanbul")


def istanbul_today(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(ISTANBUL).date()


def day_start_utc(day: date) -> datetime:
    """Midnight in Istanbul on ``day``, as a UTC instant — where a budget day begins."""
    return datetime(day.year, day.month, day.day, tzinfo=ISTANBUL).astimezone(UTC)


# ======================================================================================
# Conversations (memory)
# ======================================================================================


def _approx_bytes(messages: list[dict[str, Any]]) -> int:
    import json

    try:
        return len(json.dumps(messages, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return 0


@dataclass
class Conversation:
    """One visitor's chat with one property's assistant."""

    id: str
    property_id: str
    public_token: str
    #: Frozen at the start: the model that answers, the facts it was told, the figures and the
    #: rates its tools read. A conversation never changes any of these mid-way, because each is
    #: part of the prefix the earlier answers' thinking is bound to.
    model: str
    facts: str
    inputs: InvestmentInputs
    rates: CostRates
    created_at: float = field(default_factory=time.time)
    last_at: float = field(default_factory=time.time)
    turns: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    #: Prompt size of the latest call — the cost of the NEXT question grows with it.
    context_tokens: int = 0
    approx_bytes: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def commit(self, new_messages: list[dict[str, Any]], context_tokens: int) -> None:
        """Append one finished question and its answer. The only way the history grows."""
        self.messages.extend(new_messages)
        self.turns += 1
        self.context_tokens = max(self.context_tokens, int(context_tokens))
        self.approx_bytes = _approx_bytes(self.messages)
        self.last_at = time.time()


class ConversationStore:
    """An LRU of live conversations with an idle timeout. Thread-safe, in-process."""

    def __init__(self, *, ttl_s: float, max_items: int) -> None:
        self.ttl_s = float(ttl_s)
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        expired = [cid for cid, c in self._items.items() if now - c.last_at > self.ttl_s and not c.lock.locked()]
        for cid in expired:
            self._items.pop(cid, None)
        while len(self._items) > self.max_items:
            oldest = next(iter(self._items))
            if self._items[oldest].lock.locked():
                break
            self._items.pop(oldest, None)

    def create(
        self,
        *,
        property_id: str,
        public_token: str,
        model: str,
        facts: str,
        inputs: InvestmentInputs,
        rates: CostRates,
    ) -> Conversation:
        conversation = Conversation(
            id=secrets.token_urlsafe(18),
            property_id=property_id,
            public_token=public_token,
            model=model,
            facts=facts,
            inputs=inputs,
            rates=rates,
        )
        with self._lock:
            self._sweep(time.time())
            self._items[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: str, *, public_token: str) -> Conversation | None:
        """The live conversation, or ``None`` when it expired, never existed, or belongs to
        another property's link — the same answer for all three, so an id cannot be probed."""
        now = time.time()
        with self._lock:
            self._sweep(now)
            conversation = self._items.get(conversation_id)
            if conversation is None or conversation.public_token != public_token:
                return None
            self._items.move_to_end(conversation_id)
            conversation.last_at = now
            return conversation

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# ======================================================================================
# The bill (database)
# ======================================================================================


def record_call(
    *,
    property_id: str | None,
    conversation_id: str,
    turn: int,
    round_: int,
    cost: CallCost,
    latency_ms: int,
    stop_reason: str,
) -> None:
    """Write one usage row. Never raises: a failed write is logged, not shown to a visitor.

    Called from a worker thread (``run_in_threadpool``): SQLite is quick, but not quick enough
    to belong on the event loop that is streaming somebody's answer.
    """
    try:
        with Session(get_engine()) as session:
            session.add(
                AdvisorUsage(
                    property_id=property_id,
                    conversation_id=conversation_id,
                    turn=turn,
                    round=round_,
                    model=cost.model[:64],
                    input_tokens=cost.input_tokens,
                    output_tokens=cost.output_tokens,
                    cache_read_tokens=cost.cache_read_tokens,
                    cache_write_tokens=cost.cache_write_tokens,
                    web_search_requests=cost.web_search_requests,
                    cost_usd=float(cost.cost_usd),
                    latency_ms=int(latency_ms),
                    stop_reason=(stop_reason or "")[:32],
                )
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 - accounting must not take an answer down
        log.error("advisor.usage_write_failed", error=str(exc)[:200])


def spent_today(property_id: str | None) -> float:
    """USD spent today (Istanbul day) by one property, or by everything when ``None``."""
    since = day_start_utc(istanbul_today())
    with Session(get_engine()) as session:
        query = select(func.coalesce(func.sum(AdvisorUsage.cost_usd), 0.0)).where(AdvisorUsage.created_at >= since)
        if property_id is not None:
            query = query.where(AdvisorUsage.property_id == property_id)
        return float(session.exec(query).one() or 0.0)


def usage_summary(property_id: str, days: int = 30) -> dict[str, Any]:
    """The dashboard's view of one property's spending over ``days`` Istanbul days.

    Computed in Python over the window's rows rather than in SQL: a month of a busy property is
    a few thousand rows at most, and the per-day bucketing has to happen in Istanbul time,
    which SQLite's date functions do not know.
    """
    today = istanbul_today()
    start_day = today - timedelta(days=days - 1)
    since = day_start_utc(start_day)
    with Session(get_engine()) as session:
        rows = list(
            session.exec(
                select(AdvisorUsage).where(AdvisorUsage.property_id == property_id, AdvisorUsage.created_at >= since)
            ).all()
        )
        total_all = float(
            session.exec(
                select(func.coalesce(func.sum(AdvisorUsage.cost_usd), 0.0)).where(AdvisorUsage.property_id == property_id)
            ).one()
            or 0.0
        )

    daily: dict[date, dict[str, Any]] = {
        start_day + timedelta(days=i): {"cost_usd": 0.0, "questions": set(), "calls": 0} for i in range(days)
    }
    by_model: dict[str, dict[str, Any]] = {}
    conversations: set[str] = set()
    questions: set[tuple[str, int]] = set()
    searches = 0
    latency_total = 0
    for row in rows:
        when = (ensure_utc(row.created_at) or datetime.now(UTC)).astimezone(ISTANBUL).date()
        bucket = daily.get(when)
        if bucket is not None:
            bucket["cost_usd"] += row.cost_usd
            bucket["calls"] += 1
            bucket["questions"].add((row.conversation_id, row.turn))
        model = by_model.setdefault(row.model, {"model": row.model, "calls": 0, "cost_usd": 0.0})
        model["calls"] += 1
        model["cost_usd"] += row.cost_usd
        conversations.add(row.conversation_id)
        questions.add((row.conversation_id, row.turn))
        searches += row.web_search_requests
        latency_total += row.latency_ms

    def window(n: int) -> float:
        first = today - timedelta(days=n - 1)
        return round(sum(b["cost_usd"] for d, b in daily.items() if d >= first), 4)

    cost_30 = sum(row.cost_usd for row in rows)
    return {
        "today_usd": window(1),
        "last_7_days_usd": window(7),
        "last_30_days_usd": round(cost_30, 4),
        "all_time_usd": round(total_all, 4),
        "conversations_30d": len(conversations),
        "questions_30d": len(questions),
        "calls_30d": len(rows),
        "web_searches_30d": searches,
        "avg_cost_per_question_usd": round(cost_30 / len(questions), 4) if questions else None,
        "avg_call_latency_ms": int(latency_total / len(rows)) if rows else None,
        "by_model": sorted(
            ({**m, "cost_usd": round(m["cost_usd"], 4)} for m in by_model.values()),
            key=lambda m: -m["cost_usd"],
        ),
        "daily": [
            {"date": d.isoformat(), "cost_usd": round(b["cost_usd"], 4), "questions": len(b["questions"])}
            for d, b in sorted(daily.items())
        ],
    }
