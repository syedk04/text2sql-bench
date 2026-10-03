"""The one place model calls go through.

Flow for each request:

1. Look the request up in the disk cache. A hit returns immediately: no rate
   limiter token, no budget, no provider call; it is still logged.
2. On a miss, for each attempt: wait for the rate limiter, check the daily token
   budget, call the provider.
3. ``RateLimitError`` / ``TransientError`` are retried up to ``max_attempts``
   times, sleeping for the provider's Retry-After when given, otherwise
   full-jitter exponential backoff (``uniform(0, min(60, 2 * 2**attempt))``).
4. A success is written to the cache, its tokens are charged to the budget, and
   the call is logged. Failures are logged but never cached.

Clock, sleep and random source are injectable so tests run instantly.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Any

from text2sql.llm.base import Provider
from text2sql.llm.cache import DiskCache, cache_key
from text2sql.llm.calllog import CallLogger
from text2sql.llm.ratelimit import DailyTokenBudget, TokenBucket
from text2sql.llm.requestlog import RequestLogger
from text2sql.llm.types import (
    BudgetExceeded,
    CompletionRequest,
    CompletionResponse,
    RateLimitError,
    TransientError,
)

MAX_BACKOFF_S = 60.0
BASE_BACKOFF_S = 2.0


def estimate_tokens(request: CompletionRequest) -> int:
    """Rough upper-ish estimate used only for the budget pre-check."""
    return request.prompt_chars() // 4 + (request.max_tokens or 0)


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        *,
        cache: DiskCache | None = None,
        bucket: TokenBucket | None = None,
        budget: DailyTokenBudget | None = None,
        logger: CallLogger | None = None,
        request_logger: RequestLogger | None = None,
        max_attempts: int = 6,
        max_retry_after_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.provider = provider
        self.cache = cache
        self.bucket = bucket
        self.budget = budget
        self.logger = logger
        self.request_logger = request_logger
        self.max_attempts = max_attempts
        if not math.isfinite(max_retry_after_s) or max_retry_after_s < 0:
            raise ValueError("max_retry_after_s must be a finite number >= 0")
        self.max_retry_after_s = float(max_retry_after_s)
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """Delay before retry number ``attempt`` (0-based).

        A provider-supplied ``retry_after`` is used when it is a sane number,
        capped at ``max_retry_after_s``; otherwise full-jitter backoff applies.
        """
        if (
            isinstance(retry_after, int | float)
            and math.isfinite(retry_after)
            and retry_after >= 0
        ):
            return min(float(retry_after), self.max_retry_after_s)
        return self._rng.uniform(0.0, min(MAX_BACKOFF_S, BASE_BACKOFF_S * 2**attempt))

    def _log(self, **fields: object) -> None:
        if self.logger is not None:
            self.logger.log(**fields)

    def complete(
        self,
        request: CompletionRequest,
        *,
        run_id: str | None = None,
        question_id: int | None = None,
    ) -> CompletionResponse:
        key = cache_key(request, provider=self.provider.name)
        start = self._clock()
        base = dict(
            run_id=run_id,
            question_id=question_id,
            provider=self.provider.name,
            model=request.model,
            key=key,
        )

        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self._log(
                    **base,
                    cache_hit=True,
                    attempts=0,
                    status="ok",
                    prompt_tokens=cached.prompt_tokens,
                    completion_tokens=cached.completion_tokens,
                    latency_s=self._clock() - start,
                )
                return cached

        attempts = 0
        last_error: Exception | None = None
        while attempts < self.max_attempts:
            if self.bucket is not None:
                self.bucket.acquire()
            if self.budget is not None:
                try:
                    self.budget.check(estimate_tokens(request))
                except BudgetExceeded as exc:
                    self._log(
                        **base,
                        cache_hit=False,
                        attempts=attempts,
                        status="budget_exceeded",
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_s=self._clock() - start,
                        error=str(exc),
                    )
                    raise
            attempts += 1
            attempt_start = self._clock()

            def record(
                outcome: str, _attempt: int = attempts, _start: float = attempt_start, **kw: Any
            ) -> None:
                if self.request_logger is not None:
                    self.request_logger.log(
                        key=key,
                        run_id=run_id,
                        question_id=question_id,
                        provider=self.provider.name,
                        attempt=_attempt,
                        request=request,
                        outcome=outcome,
                        latency_s=self._clock() - _start,
                        **kw,
                    )

            try:
                response = self.provider.complete(request)
            except (RateLimitError, TransientError) as exc:
                record(
                    "rate_limited" if isinstance(exc, RateLimitError) else "transient",
                    error=f"{type(exc).__name__}: {exc}",
                )
                last_error = exc
                if attempts >= self.max_attempts:
                    break
                self._sleep(self.backoff(attempts - 1, exc.retry_after))
                continue
            except Exception as exc:
                record("error", error=f"{type(exc).__name__}: {exc}")
                self._log(
                    **base,
                    cache_hit=False,
                    attempts=attempts,
                    status="error",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_s=self._clock() - start,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

            if not response.text.strip():
                record("empty", response=response)
            elif response.truncated:
                record("truncated", response=response)
            else:
                record("ok", response=response)
            # An empty answer is more likely a provider hiccup than a real
            # reply, and a truncated one is incomplete; both are returned but
            # not cached.
            if self.cache is not None and response.text.strip() and not response.truncated:
                self.cache.put(key, request, response)
            if self.budget is not None:
                self.budget.record(response.total_tokens)
            self._log(
                **base,
                cache_hit=False,
                attempts=attempts,
                status="ok",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_s=self._clock() - start,
            )
            return response

        assert last_error is not None
        self._log(
            **base,
            cache_hit=False,
            attempts=attempts,
            status="error",
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=self._clock() - start,
            error=f"gave up after {attempts} attempts: {type(last_error).__name__}: {last_error}",
        )
        raise last_error
