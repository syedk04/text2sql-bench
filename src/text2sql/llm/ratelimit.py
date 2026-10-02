"""Staying inside a free tier: request pacing and a daily token budget.

Both take their clock (and the bucket its sleep function) as parameters so tests
can drive time by hand instead of waiting.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

from text2sql.llm.types import BudgetExceeded


class TokenBucket:
    """Classic token bucket: ``rate_per_minute`` requests on average, bursts up
    to ``capacity``. ``acquire`` blocks (via ``sleep``) until a token is free."""

    def __init__(
        self,
        rate_per_minute: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate_per_s = rate_per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_minute))
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_s)

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def acquire(self) -> float:
        """Take one token; returns how long we waited (seconds)."""
        waited = 0.0
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return waited
            delay = (1.0 - self._tokens) / self.rate_per_s
            self._sleep(delay)
            waited += delay


def utc_day() -> str:
    return datetime.now(UTC).date().isoformat()


class DailyTokenBudget:
    """Counts tokens spent per UTC day and refuses to go past ``limit``.

    With ``path`` set, the count is kept in a small JSON file so separate runs
    on the same day share one budget.
    """

    def __init__(
        self,
        limit: int,
        path: Path | None = None,
        *,
        today: Callable[[], str] = utc_day,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.path = path
        self._today = today
        self._day = today()
        self._used = 0
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if data.get("day") == self._day:
                self._used = int(data.get("used", 0))

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day = day
            self._used = 0

    @property
    def used(self) -> int:
        self._roll()
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def check(self, estimate: int = 0) -> None:
        """Raise :class:`BudgetExceeded` if spending ``estimate`` more would pass the limit."""
        if self.used + max(0, estimate) > self.limit:
            raise BudgetExceeded(
                f"daily token budget {self.limit:,} would be exceeded "
                f"(used {self._used:,}, next call ~{estimate:,}) for {self._day} UTC"
            )

    def record(self, tokens: int) -> None:
        self._roll()
        self._used += max(0, int(tokens))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"day": self._day, "used": self._used}), encoding="utf-8")
            os.replace(tmp, self.path)


MAX_RETRY_AFTER_S = 300.0
_DELAY_SECONDS = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def parse_retry_after(
    value: str | None, now: datetime | None = None, *, max_s: float = MAX_RETRY_AFTER_S
) -> float | None:
    """Parse an HTTP Retry-After header (delay-seconds or an HTTP-date).

    Returns seconds clamped to ``[0, max_s]``, or None when the header is
    missing or not one of the two valid forms. Plain digits only for the
    numeric form: no sign, exponent, underscore, inf or nan. A very long or
    far-future value is clamped rather than trusted, so a bad header can never
    park a run for days.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if _DELAY_SECONDS.fullmatch(value):
        return min(float(value), max_s)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    try:
        delay = (when - now).total_seconds()
    except OverflowError:
        return max_s
    return min(max(0.0, delay), max_s)
