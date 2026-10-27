"""Structured logging for the API and the stitch worker.

Why structlog rather than plain ``logging``: a single panorama job produces log lines from
three different places — the request handler that accepted the upload, the worker thread
that runs the pipeline, and the SSE endpoint that streams progress back. Correlating them
after the fact is only possible if every line carries the same ``request_id`` / ``job_id``.
``structlog.contextvars`` gives us that binding for free across ``await`` points *and*
across the thread boundary (we re-bind explicitly inside the worker), and the JSON renderer
makes the output greppable in production while the console renderer stays readable in dev.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import orjson
import structlog

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_context",
    "clear_context",
]

_configured = False


def _orjson_dumps(obj: Any, /, **_kwargs: Any) -> str:
    """orjson returns bytes; structlog's JSONRenderer wants ``str``."""
    return orjson.dumps(obj, default=str).decode("utf-8")


def configure_logging(level: str | int = "INFO", *, json_logs: bool | None = None) -> None:
    """Install the structlog pipeline. Idempotent, so tests and the CLI can call it freely.

    ``json_logs`` defaults to "JSON when not attached to a TTY", which is exactly the
    behaviour you want: human output in a terminal, machine output under a process manager.
    """
    global _configured
    if _configured:
        return

    if json_logs is None:
        json_logs = not sys.stderr.isatty()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    # uvicorn installs its own noisy access logger; ours already logs every request.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer(serializer=_orjson_dumps))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "biosphere") -> Any:
    """Return a bound logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind_context(**values: Any) -> None:
    """Bind key/values onto the current context (request or worker task)."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Drop everything bound to the current context."""
    structlog.contextvars.clear_contextvars()
