from datetime import UTC, datetime

import pytest

from text2sql.llm.ratelimit import DailyTokenBudget, TokenBucket, parse_retry_after
from text2sql.llm.types import BudgetExceeded


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_bucket_allows_burst_then_paces():
    clock = FakeClock()
    bucket = TokenBucket(30, capacity=3, clock=clock, sleep=clock.sleep)
    assert [bucket.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    waited = bucket.acquire()
    assert waited == pytest.approx(2.0)  # 30/min = one every 2 s
    assert clock.sleeps == [pytest.approx(2.0)]


def test_bucket_long_run_rate():
    clock = FakeClock()
    bucket = TokenBucket(60, capacity=1, clock=clock, sleep=clock.sleep)
    start = clock.now
    for _ in range(121):
        bucket.acquire()
    assert clock.now - start == pytest.approx(120.0)


def test_bucket_refills_but_caps_at_capacity():
    clock = FakeClock()
    bucket = TokenBucket(60, capacity=2, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    bucket.acquire()
    clock.now += 3600
    assert bucket.available == 2.0


def test_bucket_survives_clock_going_backwards():
    clock = FakeClock()
    bucket = TokenBucket(60, capacity=1, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    clock.now -= 50
    assert bucket.acquire() == pytest.approx(1.0)


@pytest.mark.parametrize("rate, cap", [(0, None), (-1, None), (10, 0.5)])
def test_bucket_bad_config(rate, cap):
    with pytest.raises(ValueError):
        TokenBucket(rate, cap)


def test_budget_blocks_at_limit():
    budget = DailyTokenBudget(1000, today=lambda: "2026-10-02")
    budget.check(900)
    budget.record(900)
    assert budget.remaining == 100
    budget.check(100)
    with pytest.raises(BudgetExceeded, match="1,000"):
        budget.check(101)


def test_budget_resets_on_new_utc_day():
    day = ["2026-10-02"]
    budget = DailyTokenBudget(10, today=lambda: day[0])
    budget.record(10)
    with pytest.raises(BudgetExceeded):
        budget.check(1)
    day[0] = "2026-10-03"
    budget.check(10)
    assert budget.used == 0


def test_budget_persists_across_instances_same_day(tmp_path):
    path = tmp_path / "state dir" / "budget.json"
    a = DailyTokenBudget(100, path, today=lambda: "2026-10-02")
    a.record(60)
    b = DailyTokenBudget(100, path, today=lambda: "2026-10-02")
    assert b.used == 60
    c = DailyTokenBudget(100, path, today=lambda: "2026-10-03")
    assert c.used == 0
    path.write_text("{corrupt", encoding="utf-8")
    assert DailyTokenBudget(100, path, today=lambda: "2026-10-02").used == 0


def test_budget_rejects_bad_limit():
    with pytest.raises(ValueError):
        DailyTokenBudget(0)


NOW = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "header, expected",
    [
        ("7", 7.0),
        (" 1.5 ", 1.5),
        ("0", 0.0),
        ("-3", None),
        ("inf", None),
        ("", None),
        (None, None),
        ("Fri, 02 Oct 2026 12:00:30 GMT", 30.0),
        ("Fri, 02 Oct 2026 11:59:00 GMT", 0.0),  # in the past: retry now
        ("not a date", None),
    ],
)
def test_parse_retry_after(header, expected):
    assert parse_retry_after(header, now=NOW) == expected
