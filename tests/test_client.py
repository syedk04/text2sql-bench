import random

import pytest

from text2sql.llm.cache import DiskCache, cache_key
from text2sql.llm.calllog import CallLogger, read_log
from text2sql.llm.client import LLMClient, estimate_tokens
from text2sql.llm.fake import FakeProvider
from text2sql.llm.ratelimit import DailyTokenBudget, TokenBucket
from text2sql.llm.types import (
    BudgetExceeded,
    CompletionRequest,
    CompletionResponse,
    Message,
    RateLimitError,
    TransientError,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def _req(text="question?") -> CompletionRequest:
    return CompletionRequest("m", [Message("user", text)], temperature=0.0, max_tokens=100)


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    log_path = tmp_path / "logs" / "calls.jsonl"

    def make(script, *, budget_limit=1_000_000, rpm=600, max_attempts=6):
        provider = FakeProvider(script)
        bucket = TokenBucket(rpm, capacity=1, clock=clock, sleep=clock.sleep)
        client = LLMClient(
            provider,
            cache=DiskCache(tmp_path / "cache"),
            bucket=bucket,
            budget=DailyTokenBudget(budget_limit, today=lambda: "2026-10-02"),
            logger=CallLogger(log_path),
            max_attempts=max_attempts,
            clock=clock,
            sleep=clock.sleep,
            rng=random.Random(0),
        )
        return client, provider

    return make, clock, log_path


def test_miss_then_hit(env):
    make, clock, log_path = env
    client, provider = make(["SELECT 1"])
    first = client.complete(_req(), run_id="r", question_id=5)
    second = client.complete(_req(), run_id="r", question_id=5)
    assert first == second
    assert provider.calls == 1
    log = read_log(log_path)
    assert [(r.cache_hit, r.attempts, r.status) for r in log] == [(False, 1, "ok"), (True, 0, "ok")]
    assert log[0].key == cache_key(_req()) and log[0].question_id == 5 and log[0].run_id == "r"
    assert client.budget.used == first.total_tokens  # the hit was not charged


def test_cache_hit_takes_no_rate_limit_token(env):
    make, clock, _ = env
    client, _ = make(["SELECT 1"], rpm=1)
    client.complete(_req())
    for _ in range(5):
        client.complete(_req())
    assert clock.sleeps == []


def test_rate_limit_retry_uses_retry_after(env):
    make, clock, log_path = env
    client, provider = make([RateLimitError(retry_after=7), "SELECT 1"])
    assert client.complete(_req()).text == "SELECT 1"
    assert provider.calls == 2
    assert 7 in clock.sleeps
    assert read_log(log_path)[-1].attempts == 2


def test_backoff_is_full_jitter_and_capped(env):
    make, *_ = env
    client, _ = make(["x"])
    for attempt in range(10):
        cap = min(60.0, 2.0 * 2**attempt)
        delays = [client.backoff(attempt, None) for _ in range(200)]
        assert all(0.0 <= d <= cap for d in delays)
        assert max(delays) > cap * 0.8  # jitter spans the range
    assert client.backoff(3, 1.5) == 1.5


def test_transient_errors_retry_with_backoff(env):
    make, clock, _ = env
    client, provider = make([TransientError(), TransientError(), "ok"])
    assert client.complete(_req()).text == "ok"
    assert provider.calls == 3
    assert len(clock.sleeps) >= 2


def test_gives_up_after_max_attempts_and_caches_nothing(env, tmp_path):
    make, _, log_path = env
    client, provider = make([RateLimitError()], max_attempts=6)
    with pytest.raises(RateLimitError):
        client.complete(_req())
    assert provider.calls == 6
    assert len(client.cache) == 0
    last = read_log(log_path)[-1]
    assert (last.status, last.attempts) == ("error", 6)
    assert "gave up after 6 attempts" in last.error
    # a later success is cached normally
    client.provider = FakeProvider(["SELECT 2"])
    assert client.complete(_req()).text == "SELECT 2"
    assert len(client.cache) == 1


def test_other_errors_are_not_retried_or_cached(env):
    make, _, log_path = env
    client, provider = make([ValueError("bad request")])
    with pytest.raises(ValueError):
        client.complete(_req())
    assert provider.calls == 1
    assert len(client.cache) == 0
    assert read_log(log_path)[-1].error == "ValueError: bad request"


def test_budget_stops_before_calling(env):
    make, _, log_path = env
    client, provider = make(["SELECT 1"], budget_limit=50)
    assert estimate_tokens(_req()) > 50
    with pytest.raises(BudgetExceeded):
        client.complete(_req())
    assert provider.calls == 0
    assert read_log(log_path)[-1].status == "budget_exceeded"


def test_budget_records_real_usage(env):
    make, *_ = env
    client, _ = make([CompletionResponse("SELECT 1", 400, 30)])
    client.complete(_req())
    assert client.budget.used == 430


def test_rate_limiter_paces_cache_misses(env):
    make, clock, _ = env
    client, _ = make(["a"], rpm=60)
    for i in range(4):
        client.complete(_req(f"q{i}"))
    assert clock.now == pytest.approx(3.0)


def test_works_without_optional_parts():
    client = LLMClient(FakeProvider(["SELECT 1"]))
    assert client.complete(_req()).text == "SELECT 1"
    with pytest.raises(ValueError):
        LLMClient(FakeProvider(["x"]), max_attempts=0)
