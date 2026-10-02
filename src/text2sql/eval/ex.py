"""Execution accuracy (EX), scored the way the official BIRD Mini-Dev script does.

The official ``evaluation_ex.py`` (bird-bench/mini_dev) works like this, and so
does this module:

* run the predicted SQL, then the gold SQL, against the question's database and
  ``fetchall()`` both;
* the answer is correct iff ``set(pred_rows) == set(gold_rows)``: row order and
  duplicate rows are ignored, column order inside a row is not;
* one 30 second budget covers both queries together;
* a timeout or any error, in the prediction or in the gold query, scores 0;
* every question stays in the denominator.

Differences, all on the safe side: queries run on a read-only connection behind
the SELECT-only guard, and a prediction the guard rejects scores 0 (the official
script would run it; on BIRD that only matters for multi-statement predictions,
which sqlite3 refuses anyway).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text2sql import config
from text2sql.data.bird import Question
from text2sql.sql.executor import ExecResult, execute

OFFICIAL_TIMEOUT_S = 30.0


def compare(pred_rows: Iterable[Sequence[Any]], gold_rows: Iterable[Sequence[Any]]) -> bool:
    """Official EX comparison: equal as sets of row tuples."""
    return set(map(tuple, pred_rows)) == set(map(tuple, gold_rows))


@dataclass(frozen=True)
class ScoreResult:
    question_id: int
    db_id: str
    difficulty: str
    correct: int  # 0 or 1, like the official script
    pred_status: str
    gold_status: str  # "not_run" when the prediction already failed
    error: str | None = None
    elapsed_s: float = 0.0


def score_sql(
    db_path: str | Path,
    pred_sql: str,
    gold_sql: str,
    *,
    timeout_s: float = OFFICIAL_TIMEOUT_S,
    guard: bool = True,
) -> tuple[int, ExecResult, ExecResult | None]:
    """Score one prediction against one gold query on one database."""
    deadline = time.monotonic() + timeout_s
    pred = execute(db_path, pred_sql, deadline=deadline, row_limit=None, guard=guard)
    if not pred.ok:
        return 0, pred, None
    gold = execute(db_path, gold_sql, deadline=deadline, row_limit=None, guard=guard)
    if not gold.ok:
        return 0, pred, gold
    return int(compare(pred.rows, gold.rows)), pred, gold


def score_one(
    question: Question,
    pred_sql: str | None,
    *,
    timeout_s: float = OFFICIAL_TIMEOUT_S,
    db_path: str | Path | None = None,
) -> ScoreResult:
    path = Path(db_path) if db_path is not None else config.db_path(question.db_id)
    start = time.monotonic()
    if not isinstance(pred_sql, str) or not pred_sql.strip():
        return ScoreResult(
            question.question_id,
            question.db_id,
            question.difficulty,
            0,
            "error",
            "not_run",
            "empty prediction",
            0.0,
        )
    correct, pred, gold = score_sql(path, pred_sql, question.gold_sql, timeout_s=timeout_s)
    if not pred.ok:
        error = f"prediction {pred.status}: {pred.error}"
    elif gold is not None and not gold.ok:
        error = f"gold {gold.status}: {gold.error}"
    else:
        error = None
    return ScoreResult(
        question_id=question.question_id,
        db_id=question.db_id,
        difficulty=question.difficulty,
        correct=correct,
        pred_status=pred.status,
        gold_status=gold.status if gold is not None else "not_run",
        error=error,
        elapsed_s=time.monotonic() - start,
    )
