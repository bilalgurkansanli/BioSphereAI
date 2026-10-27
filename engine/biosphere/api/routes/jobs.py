"""Job status, cancellation, and the server-sent-events progress stream."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import orjson
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from biosphere.api import schemas as sc
from biosphere.api.deps import CurrentUser, JobsDep, SessionDep, require_job
from biosphere.core.jobs import TERMINAL_STATUSES, is_terminal_event

router = APIRouter(prefix="/jobs", tags=["jobs"])

#: Comment frames keep intermediaries from tearing down an idle connection. Nginx's default
#: proxy_read_timeout is 60s and a slow bundle-adjust stage can easily be quieter than that.
HEARTBEAT_SECONDS = 15.0

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Connection": "keep-alive",
    # Tells nginx not to buffer the stream; without it progress arrives in one lump at the end.
    "X-Accel-Buffering": "no",
}


def _frame(event_id: int, payload: dict[str, Any]) -> str:
    data = orjson.dumps(payload).decode("utf-8")
    return f"id: {event_id}\nevent: message\ndata: {data}\n\n"


@router.get("/{job_id}", response_model=sc.Job, summary="Job status")
def get_job(job_id: str, session: SessionDep, user: CurrentUser) -> sc.Job:
    return sc.job_to_schema(require_job(session, job_id, user))


@router.post("/{job_id}/cancel", response_model=sc.Job, summary="Cancel a job")
def cancel_job(job_id: str, session: SessionDep, jobs: JobsDep, user: CurrentUser) -> sc.Job:
    """Cooperative cancellation: the pipeline stops at its next progress checkpoint."""
    require_job(session, job_id, user)
    session.commit()  # let the worker's session see a consistent row
    return sc.job_to_schema(jobs.cancel(job_id))


@router.get(
    "/{job_id}/events",
    summary="Progress stream (text/event-stream)",
    response_class=StreamingResponse,
)
async def job_events(job_id: str, session: SessionDep, jobs: JobsDep, user: CurrentUser) -> StreamingResponse:
    """Stream :class:`JobEvent` frames until the job reaches a terminal state.

    The connection closes after ``done``/``error`` rather than idling, because EventSource
    reconnects automatically and a client that has its answer should not hold a socket open.
    """
    job = require_job(session, job_id, user)
    replay_only = job.status in TERMINAL_STATUSES and not jobs.has_history(job_id)
    session.commit()

    async def stream() -> AsyncIterator[str]:
        # `retry` tells EventSource how long to wait before reconnecting after a drop.
        yield "retry: 3000\n\n"

        if replay_only:
            final = jobs.terminal_event(job_id)
            if final is not None:
                yield _frame(1, final)
            return

        events = jobs.subscribe(job_id).__aiter__()
        pending: asyncio.Task[dict[str, Any]] | None = None
        counter = 0
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(events.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_SECONDS)
                if not done:
                    # The task stays pending across iterations; re-creating it would drop
                    # whatever arrived in the meantime.
                    yield ": ping\n\n"
                    continue
                task, pending = pending, None
                try:
                    event = task.result()
                except StopAsyncIteration:
                    break
                counter += 1
                yield _frame(counter, event)
                if is_terminal_event(event):
                    break
        finally:
            if pending is not None:
                pending.cancel()
            await events.aclose()

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)
