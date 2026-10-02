"""Staying inside a free tier: request pacing and a daily token budget.

Both take their clock (and the bucket its sleep function) as parameters so tests
can drive time by hand instead of waiting.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

from text2sql.llm.types import BudgetExceeded


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


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
        if not _finite_number(rate_per_minute) or rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be a finite positive number")
        if capacity is not None and (not _finite_number(capacity) or capacity < 1):
            raise ValueError("capacity must be a finite number >= 1")
        self.rate_per_s = rate_per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_minute))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = self.capacity
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_s)

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def acquire(self) -> float:
        """Take one token; returns how long we waited (seconds).

        Thread-safe: the refill-check-take step happens under a lock, and the
        sleep happens outside it so other threads are not blocked meanwhile.
        """
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                delay = (1.0 - self._tokens) / self.rate_per_s
            self._sleep(delay)
            waited += delay


def utc_day() -> str:
    return datetime.now(UTC).date().isoformat()


class BudgetStateError(BudgetExceeded):
    """The shared budget file is unreadable or malformed.

    This fails safe: it is a kind of :class:`BudgetExceeded`, so callers stop
    spending instead of guessing how much of today's quota is left. Fix or
    delete the file to carry on.
    """


class _FileLock:
    """Tiny cross-process lock: an exclusively created ``.lock`` file.

    A lock older than ``stale_after`` seconds is assumed to belong to a crashed
    process and is removed.
    """

    def __init__(self, path: Path, timeout: float = 10.0, stale_after: float = 60.0) -> None:
        self.path = path
        self.timeout = timeout
        self.stale_after = stale_after

    def __enter__(self) -> _FileLock:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (FileExistsError, PermissionError):
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_after:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise BudgetStateError(
                        f"could not lock {self.path} within {self.timeout}s"
                    ) from None
                time.sleep(0.005)
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return self

    def __exit__(self, *exc: object) -> None:
        for attempt in range(50):
            try:
                self.path.unlink(missing_ok=True)
                return
            except PermissionError:
                time.sleep(0.005 * (attempt + 1))


class DailyTokenBudget:
    """Counts tokens spent per UTC day and refuses to go past ``limit``.

    With ``path`` set, the count lives in a small JSON file (``{"day", "used"}``)
    shared by every run on the machine. Each check and record re-reads the file
    under a thread lock plus a lock file, so two clients, in one process or
    several, never lose each other's spend. A malformed file raises
    :class:`BudgetStateError` rather than being silently reset.
    """

    def __init__(
        self,
        limit: int,
        path: Path | None = None,
        *,
        today: Callable[[], str] = utc_day,
    ) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        self.limit = limit
        self.path = path
        self._today = today
        self._lock = threading.Lock()
        self._mem_day = today()
        self._mem_used = 0

    # -- storage ---------------------------------------------------------------

    def _read(self, day: str) -> int:
        if self.path is None:
            if self._mem_day != day:
                self._mem_day, self._mem_used = day, 0
            return self._mem_used
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise BudgetStateError(f"cannot read budget file {self.path}: {exc}") from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BudgetStateError(f"budget file {self.path} is not valid JSON") from exc
        used = data.get("used") if isinstance(data, dict) else None
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("day"), str)
            or isinstance(used, bool)
            or not isinstance(used, int)
            or used < 0
        ):
            raise BudgetStateError(
                f"budget file {self.path} should hold {{'day': str, 'used': int >= 0}}; "
                "fix or delete it"
            )
        return used if data["day"] == day else 0

    def _write(self, day: str, used: int) -> None:
        if self.path is None:
            self._mem_day, self._mem_used = day, used
            return
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps({"day": day, "used": used}), encoding="utf-8")
        for attempt in range(50):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:  # Windows: a reader has the file open
                time.sleep(0.005 * (attempt + 1))
        tmp.unlink(missing_ok=True)
        raise BudgetStateError(f"could not update budget file {self.path}")

    def _locked(self) -> _FileLock | nullcontext[None]:
        if self.path is None:
            return nullcontext()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return _FileLock(self.path.with_name(self.path.name + ".lock"))

    # -- API -------------------------------------------------------------------

    @property
    def used(self) -> int:
        with self._lock, self._locked():
            return self._read(self._today())

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def check(self, estimate: int = 0) -> None:
        """Raise :class:`BudgetExceeded` if spending ``estimate`` more would pass the limit."""
        day = self._today()
        with self._lock, self._locked():
            used = self._read(day)
        if used + max(0, estimate) > self.limit:
            raise BudgetExceeded(
                f"daily token budget {self.limit:,} would be exceeded "
                f"(used {used:,}, next call ~{estimate:,}) for {day} UTC"
            )

    def record(self, tokens: int) -> None:
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            raise TypeError("tokens must be an int")
        day = self._today()
        with self._lock, self._locked():
            self._write(day, self._read(day) + max(0, tokens))


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
