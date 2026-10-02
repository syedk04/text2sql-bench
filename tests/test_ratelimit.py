from datetime import UTC, datetime
from pathlib import Path

import pytest

from text2sql.llm.ratelimit import (
    BudgetStateError,
    DailyTokenBudget,
    TokenBucket,
    parse_retry_after,
)
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


@pytest.mark.parametrize(
    "rate, cap",
    [
        (0, None),
        (-1, None),
        (10, 0.5),
        (float("nan"), None),
        (float("inf"), None),
        (10, float("nan")),
        (10, float("inf")),
        (10, 0),
        (True, None),
        ("60", None),
    ],
)
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


@pytest.mark.parametrize(
    "header, expected",
    [
        ("1_000", None),
        ("1e3", None),
        ("+5", None),
        ("nan", None),
        ("99999999999999999999", 300.0),
        ("1e400", None),
        ("86400", 300.0),
        ("Sat, 01 Jan 9999 00:00:00 GMT", 300.0),
        ("Fri, 02 Oct 2026 12:00:30 +0200", 0.0),  # 10:00:30 UTC, already past
    ],
)
def test_parse_retry_after_is_strict_and_clamped(header, expected):
    assert parse_retry_after(header, now=NOW) == expected


def test_parse_retry_after_custom_cap():
    assert parse_retry_after("120", now=NOW, max_s=60) == 60
    assert parse_retry_after("Fri, 02 Oct 2026 12:10:00 GMT", now=NOW, max_s=60) == 60


def test_two_instances_share_spend(tmp_path):
    path = tmp_path / "budget.json"
    a = DailyTokenBudget(100, path, today=lambda: "2026-10-02")
    b = DailyTokenBudget(100, path, today=lambda: "2026-10-02")
    a.record(60)
    b.record(60)
    assert a.used == b.used == 120
    with pytest.raises(BudgetExceeded):
        b.check(0)
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.lock"))


@pytest.mark.parametrize(
    "content",
    [
        b"[1, 2]",
        bytes([0xFF, 0xFE]),
        b"{corrupt",
        b'{"day": "2026-10-02", "used": "lots"}',
        b'{"day": "2026-10-02", "used": null}',
        b'{"day": "2026-10-02", "used": -1000}',
        b'{"day": "2026-10-02", "used": true}',
        b'{"used": 5}',
    ],
)
def test_corrupt_budget_file_fails_safe(tmp_path, content):
    path = tmp_path / "budget.json"
    path.write_bytes(content)
    budget = DailyTokenBudget(100, path, today=lambda: "2026-10-02")
    with pytest.raises(BudgetStateError, match="budget file"):
        budget.check(1)
    with pytest.raises(BudgetStateError):
        budget.record(1)
    assert issubclass(BudgetStateError, BudgetExceeded)
    assert path.read_bytes() == content  # never silently overwritten


def test_record_rejects_non_int():
    budget = DailyTokenBudget(100)
    with pytest.raises(TypeError):
        budget.record(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DailyTokenBudget(True)  # type: ignore[arg-type]


def test_concurrent_threads_lose_nothing(tmp_path):
    import threading

    path = tmp_path / "budget.json"
    budgets = [DailyTokenBudget(10**9, path, today=lambda: "2026-10-02") for _ in range(4)]

    def work(b):
        for _ in range(50):
            b.record(1)

    threads = [threading.Thread(target=work, args=(budgets[i % 4],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert budgets[0].used == 400


def _record_in_process(path: str, n: int) -> None:
    budget = DailyTokenBudget(10**9, Path(path), today=lambda: "2026-10-02")
    for _ in range(n):
        budget.record(1)


def test_concurrent_processes_lose_nothing(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    path = tmp_path / "budget.json"
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(_record_in_process, [str(path)] * 3, [40] * 3))
    assert DailyTokenBudget(10**9, path, today=lambda: "2026-10-02").used == 120


def test_stale_lock_file_is_cleared(tmp_path):
    import os
    import time

    path = tmp_path / "budget.json"
    lock = tmp_path / "budget.json.lock"
    lock.write_text("12345")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    DailyTokenBudget(100, path, today=lambda: "2026-10-02").record(5)
    assert not lock.exists()


def test_bucket_never_over_grants_across_threads():
    import sys
    import threading

    class Stop(Exception):
        pass

    def no_sleep(_d):
        raise Stop

    bucket = TokenBucket(60, capacity=1000, clock=lambda: 0.0, sleep=no_sleep)
    granted = []
    lock = threading.Lock()

    def worker():
        n = 0
        while True:
            try:
                bucket.acquire()
            except Stop:
                break
            n += 1
        with lock:
            granted.append(n)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert sum(granted) == 1000
    assert bucket.available == 0.0
