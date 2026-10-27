"""How often a visitor, and a listing, may ask — counted in memory, keyed by nothing personal.

The chat is the one public route here that spends money per request, so it is the one that
needs limiting, and on three axes: a visitor hammering one listing (per-IP), a listing being
hammered by many visitors or a script (per-property), and the installation as a whole (the
daily USD cap, enforced elsewhere against the usage table).

The IP limits key on a SALTED DIGEST of the address, and the salt is random per process. The
address is needed for a minute or an hour to count requests; it is not needed afterwards, and
the privacy policy says the product keeps none. A digest with a salt that dies with the process
cannot be turned back into an address or joined with any other log, and a restart forgets the
counts, which is acceptable for a limit measured in minutes.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import deque

__all__ = ["SlidingWindow", "Limiter", "client_digest"]

_SALT = secrets.token_bytes(16)


def client_digest(address: str) -> str:
    return hashlib.blake2b(address.encode("utf-8", "replace"), key=_SALT, digest_size=12).hexdigest()


class SlidingWindow:
    """At most ``limit`` events per ``window_s`` seconds, per key. A log of timestamps."""

    def __init__(self, limit: int, window_s: float, *, max_keys: int = 50_000) -> None:
        self.limit = int(limit)
        self.window_s = float(window_s)
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds until ``key`` may act again; ``0`` when it may act now. Does not record."""
        if self.limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 0.0
            while events and now - events[0] >= self.window_s:
                events.popleft()
            if len(events) < self.limit:
                return 0.0
            return max(0.0, self.window_s - (now - events[0]))

    def record(self, key: str, now: float | None = None) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            if key not in self._events and len(self._events) >= self.max_keys:
                # Bounded memory under a flood of distinct addresses: drop the keys whose
                # newest event is oldest. A key dropped here has not acted for a while.
                stale = sorted(self._events, key=lambda k: self._events[k][-1] if self._events[k] else 0.0)
                for old in stale[: max(1, len(stale) // 10)]:
                    self._events.pop(old, None)
            self._events.setdefault(key, deque()).append(now)


class Limiter:
    """The chat's three windows, checked together and recorded together."""

    def __init__(self, *, ip_per_minute: int, ip_per_hour: int, property_per_hour: int) -> None:
        self.ip_minute = SlidingWindow(ip_per_minute, 60.0)
        self.ip_hour = SlidingWindow(ip_per_hour, 3600.0)
        self.property_hour = SlidingWindow(property_per_hour, 3600.0)

    def check(self, *, ip_key: str, property_id: str) -> tuple[str, float] | None:
        """``(scope, retry_after_s)`` for the first window that is full, else ``None``."""
        for scope, window, key in (
            ("ip_minute", self.ip_minute, ip_key),
            ("ip_hour", self.ip_hour, ip_key),
            ("property_hour", self.property_hour, property_id),
        ):
            wait = window.retry_after(key)
            if wait > 0:
                return scope, wait
        return None

    def record(self, *, ip_key: str, property_id: str) -> None:
        self.ip_minute.record(ip_key)
        self.ip_hour.record(ip_key)
        self.property_hour.record(property_id)
