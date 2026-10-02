import time

import pytest

from synthetic import SCHEMAS, make_db
from text2sql.sql.executor import execute

SLOW = (
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1000000000) "
    "SELECT COUNT(*) FROM n"
)
# Produces rows one by one, so the time is spent while fetching, not in execute().
SLOW_FETCH = (
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1000000000) "
    "SELECT i FROM n"
)


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "with space" / "shop.sqlite", SCHEMAS["shop"])


def test_ok_result_with_columns(db):
    res = execute(db, "SELECT name, price FROM item WHERE price > 5 ORDER BY name")
    assert res.ok and res.status == "ok"
    assert res.columns == ["name", "price"]
    assert res.rows == [("broom", 12.0), ("melon", 6.0), ("soap", 7.25)]
    assert res.truncated is False
    assert res.error is None
    assert res.elapsed_s >= 0


@pytest.mark.parametrize(
    "limit, n_rows, truncated",
    [(None, 5, False), (10, 5, False), (5, 5, False), (2, 2, True), (0, 0, True)],
)
def test_row_limit(db, limit, n_rows, truncated):
    res = execute(db, "SELECT * FROM item", row_limit=limit)
    assert res.ok
    assert len(res.rows) == n_rows
    assert res.truncated is truncated


def test_negative_row_limit_is_a_bug(db):
    with pytest.raises(ValueError):
        execute(db, "SELECT 1", row_limit=-1)


def test_row_limit_bounds_an_endless_result(db):
    res = execute(db, SLOW_FETCH, row_limit=3, timeout_s=5)
    assert res.ok and res.rows == [(1,), (2,), (3,)] and res.truncated


@pytest.mark.parametrize("sql", [SLOW, SLOW_FETCH])
def test_timeout_stops_long_queries(db, sql):
    t0 = time.monotonic()
    res = execute(db, sql, timeout_s=0.3)
    assert res.status == "timeout"
    assert res.rows == []
    assert time.monotonic() - t0 < 5


def test_timeout_on_cross_join(tmp_path):
    big = make_db(
        tmp_path / "big.sqlite",
        [
            "CREATE TABLE t (a INTEGER)",
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 2000) "
            "INSERT INTO t SELECT i FROM n",
        ],
    )
    res = execute(big, "SELECT COUNT(*) FROM t a, t b, t c", timeout_s=0.3)
    assert res.status == "timeout"


def test_shared_deadline_already_spent(db):
    res = execute(db, "SELECT 1", deadline=time.monotonic() - 1)
    assert res.status == "timeout"


def test_guard_rejects_before_touching_db(db):
    before = db.read_bytes()
    res = execute(db, "SELECT 1; DELETE FROM item")
    assert res.status == "rejected"
    assert "one statement" in res.error
    assert db.read_bytes() == before


def test_without_guard_the_connection_still_refuses_writes(db):
    before = db.read_bytes()
    res = execute(db, "DELETE FROM item", guard=False)
    assert res.status == "error"
    assert db.read_bytes() == before


def test_sql_errors_are_reported(db):
    res = execute(db, "SELECT nope FROM item")
    assert res.status == "error"
    assert "no such column" in res.error


def test_missing_database(tmp_path):
    res = execute(tmp_path / "missing.sqlite", "SELECT 1")
    assert res.status == "error"
    assert "could not open" in res.error


def test_invalid_utf8_text_is_an_error_like_the_official_script(tmp_path):
    # BIRD's scorer uses sqlite3 defaults, where undecodable TEXT raises; we do
    # the same so scores line up, and surface the error instead of hiding it.
    bad = make_db(
        tmp_path / "bad.sqlite",
        ["CREATE TABLE t (s TEXT)", "INSERT INTO t VALUES (CAST(x'fffe' AS TEXT))"],
    )
    res = execute(bad, "SELECT s FROM t")
    assert res.status == "error"
    assert "decode" in res.error.lower()


def test_query_that_finishes_late_counts_as_timeout(db, monkeypatch):
    # One slow step can run past the deadline between progress checks; the
    # result must still be a timeout, as under the official wall-clock limit.
    import text2sql.sql.executor as executor

    readings = iter([0.0, 0.0])

    class Clock:
        @staticmethod
        def monotonic():
            return next(readings, 10.0)

    monkeypatch.setattr(executor, "time", Clock)
    res = execute(db, "SELECT 1", timeout_s=2.5)
    assert res.status == "timeout"
    assert "after the 2.5s limit" in res.error


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT zeroblob(500000000)",
        "SELECT length(printf('%.*c', 90000000, 'x') || printf('%.*c', 90000000, 'x'))",
        "SELECT randomblob(200000000)",
    ],
)
def test_huge_values_are_refused(db, sql):
    res = execute(db, sql, timeout_s=10)
    assert res.status == "error"
    assert "too big" in res.error


def test_oversized_printf_yields_null_not_a_huge_string(db):
    res = execute(db, "SELECT length(printf('%.*c', 900000000, 'x'))", timeout_s=10)
    assert res.ok and res.rows == [(None,)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": float("nan")},
        {"timeout_s": True},
        {"timeout_s": "5"},
        {"deadline": float("nan")},
        {"row_limit": True},
        {"row_limit": 2.5},
        {"row_limit": "3"},
    ],
)
def test_bad_arguments_are_bugs_not_results(db, kwargs):
    with pytest.raises(ValueError):
        execute(db, "SELECT 1", **kwargs)


def test_infinite_timeout_is_allowed(db):
    assert execute(db, "SELECT 1", timeout_s=float("inf")).ok
