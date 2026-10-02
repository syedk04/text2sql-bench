import pytest

from text2sql.data.bird import Question
from text2sql.eval.ex import compare, score_one, score_sql

SLOW = (
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1000000000) "
    "SELECT COUNT(*) FROM n"
)


def _q(gold: str, db_id: str = "shop") -> Question:
    return Question(1, db_id, "q?", "", gold, "simple")


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ([(1,), (2,)], [(2,), (1,)], True),  # order ignored
        ([(1,), (1,), (2,)], [(1,), (2,)], True),  # duplicates ignored
        ([(1, 2)], [(2, 1)], False),  # column order matters
        ([(1,)], [(1.0,)], True),  # Python equality: 1 == 1.0
        ([("1",)], [(1,)], False),  # but text is not a number
        ([], [], True),
        ([(None,)], [], False),
        ([(1,)], [(1,), (2,)], False),
        ([[1, 2]], [(1, 2)], True),  # lists are compared as tuples
    ],
)
def test_compare_is_set_equality(pred, gold, expected):
    assert compare(pred, gold) is expected


def test_correct_prediction(synthetic_data_dir):
    q = _q("SELECT name FROM item WHERE price > 5")
    r = score_one(q, "SELECT name FROM item WHERE price > 5 ORDER BY price DESC")
    assert r.correct == 1
    assert (r.pred_status, r.gold_status, r.error) == ("ok", "ok", None)


def test_extra_column_is_wrong(synthetic_data_dir):
    q = _q("SELECT name FROM item WHERE price > 5")
    assert score_one(q, "SELECT name, price FROM item WHERE price > 5").correct == 0


def test_prediction_error_scores_zero_and_skips_gold(synthetic_data_dir):
    r = score_one(_q("SELECT 1"), "SELECT nope FROM item")
    assert r.correct == 0
    assert r.pred_status == "error" and r.gold_status == "not_run"
    assert "no such column" in r.error


def test_gold_error_also_scores_zero(synthetic_data_dir):
    r = score_one(_q("SELECT nope FROM item"), "SELECT 1")
    assert r.correct == 0
    assert r.gold_status == "error"
    assert r.error.startswith("gold error")


@pytest.mark.parametrize("pred", [None, "", "   ", 42])
def test_missing_or_non_string_prediction_scores_zero(synthetic_data_dir, pred):
    r = score_one(_q("SELECT COUNT(*) FROM item WHERE price > 100"), pred)
    assert r.correct == 0


def test_unsafe_prediction_is_rejected(synthetic_data_dir):
    r = score_one(_q("SELECT 1"), "SELECT 1; DROP TABLE item")
    assert r.correct == 0 and r.pred_status == "rejected"


def test_timeout_scores_zero(synthetic_data_dir):
    r = score_one(_q("SELECT 1"), SLOW, timeout_s=0.3)
    assert r.correct == 0 and r.pred_status == "timeout"


def test_one_budget_covers_both_queries(synthetic_data_dir):
    # A fast prediction with a slow gold query still runs out of the shared budget.
    r = score_one(_q(SLOW), "SELECT 1", timeout_s=0.3)
    assert r.correct == 0
    assert (r.pred_status, r.gold_status) == ("ok", "timeout")


def test_score_sql_against_explicit_path(synthetic_data_dir):
    from text2sql import config

    correct, pred, gold = score_sql(
        config.db_path("zoo"), "SELECT species FROM animal", "SELECT DISTINCT species FROM animal"
    )
    assert correct == 1 and pred.ok and gold is not None and gold.ok


def test_wrong_database_is_an_error(synthetic_data_dir):
    r = score_one(_q("SELECT COUNT(*) FROM item", db_id="zoo"), "SELECT COUNT(*) FROM item")
    assert r.correct == 0 and r.pred_status == "error"
