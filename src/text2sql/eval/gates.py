"""Self-checks for the evaluator.

Gate: gold vs gold. Every gold query scored against itself must come out
correct. A failure means the evaluator (or the SELECT-only guard, or the
read-only connection) is wrong for a query the benchmark considers valid, and
the fix belongs in our code, never in the gold SQL.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from text2sql.data.bird import Question
from text2sql.eval.ex import OFFICIAL_TIMEOUT_S, ScoreResult, score_many


@dataclass
class GateReport:
    n: int
    passed: int
    failures: list[ScoreResult] = field(default_factory=list)
    slowest: list[ScoreResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n > 0 and self.passed == self.n


def gate_gold(
    questions: Sequence[Question],
    *,
    timeout_s: float = OFFICIAL_TIMEOUT_S,
    progress: Callable[[int, int, ScoreResult], None] | None = None,
) -> GateReport:
    results = score_many(
        questions, [q.gold_sql for q in questions], timeout_s=timeout_s, progress=progress
    )
    failures = [r for r in results if r.correct != 1]
    slowest = sorted(results, key=lambda r: r.elapsed_s, reverse=True)[:5]
    return GateReport(
        n=len(results), passed=len(results) - len(failures), failures=failures, slowest=slowest
    )
