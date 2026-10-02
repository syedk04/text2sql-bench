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


# --- batches ---------------------------------------------------------------------

OFFICIAL_SEPARATOR = "\t----- bird -----\t"
OFFICIAL_FALLBACK_DB = "financial"


@dataclass(frozen=True)
class OfficialPrediction:
    sql: str
    db_id: str
    well_formed: bool  # False when the official parser had to fall back


def parse_official_predictions(raw: dict[str, Any]) -> list[OfficialPrediction]:
    """Parse a BIRD prediction file the way the official ``package_sqls`` does.

    Values look like ``<sql> TAB ----- bird ----- TAB <db_id>``. Order is the file's
    key order (position, not key value, is what matches a gold line). A non-string
    value becomes ``" "``; a string without exactly one separator is used whole
    (stripped). Both fall back to db ``financial``, which the scorer ignores anyway
    because the gold side decides the database.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"prediction file must be a JSON object, got {type(raw).__name__}")
    out: list[OfficialPrediction] = []
    for value in raw.values():
        if isinstance(value, str):
            parts = value.split(OFFICIAL_SEPARATOR)
            if len(parts) == 2:
                out.append(OfficialPrediction(parts[0], parts[1], True))
            else:
                out.append(OfficialPrediction(value.strip(), OFFICIAL_FALLBACK_DB, False))
        else:
            out.append(OfficialPrediction(" ", OFFICIAL_FALLBACK_DB, False))
    return out


def load_official_predictions(path: str | Path) -> list[OfficialPrediction]:
    import json

    with open(path, encoding="utf-8") as fh:
        return parse_official_predictions(json.load(fh))


def score_many(
    questions: Sequence[Question],
    predictions: Sequence[str | None],
    *,
    timeout_s: float = OFFICIAL_TIMEOUT_S,
    progress: Any = None,
) -> list[ScoreResult]:
    """Score predictions against questions by position."""
    if len(predictions) != len(questions):
        raise ValueError(
            f"{len(predictions)} predictions for {len(questions)} questions; "
            "the official script pairs them by position, so the counts must match"
        )
    results = []
    for i, (q, sql) in enumerate(zip(questions, predictions, strict=True)):
        results.append(score_one(q, sql, timeout_s=timeout_s))
        if progress is not None:
            progress(i + 1, len(questions), results[-1])
    return results


@dataclass(frozen=True)
class Breakdown:
    n: int
    correct: int

    @property
    def ex(self) -> float:
        return 100.0 * self.correct / self.n if self.n else 0.0


def breakdown(results: Sequence[ScoreResult]) -> dict[str, Breakdown]:
    """EX per difficulty plus ``total``, in the official column order."""
    from text2sql.data.bird import DIFFICULTIES

    table: dict[str, Breakdown] = {}
    for difficulty in DIFFICULTIES:
        sub = [r for r in results if r.difficulty == difficulty]
        table[difficulty] = Breakdown(len(sub), sum(r.correct for r in sub))
    table["total"] = Breakdown(len(results), sum(r.correct for r in results))
    return table


def format_breakdown(table: dict[str, Breakdown]) -> str:
    cols = list(table)
    lines = [
        f"{'':12}" + "".join(f"{c:>14}" for c in cols),
        f"{'count':12}" + "".join(f"{table[c].n:>14}" for c in cols),
        f"{'EX':12}" + "".join(f"{table[c].ex:>14.2f}" for c in cols),
    ]
    return "\n".join(lines)
