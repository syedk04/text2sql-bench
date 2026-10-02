"""Turn stored runs into the README results table."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from text2sql.data.bird import DIFFICULTIES
from text2sql.eval.stats import wilson_pct
from text2sql.results.store import Run

START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"
EMPTY_TABLE = "No runs yet."

COLUMNS = (
    "Track",
    "Model",
    "Split",
    "n",
    "EX % [95% CI]",
    "Simple",
    "Moderate",
    "Challenging",
    "tokens/q",
    "calls/q",
    "latency/q",
    "cost/q",
)


class ReportError(ValueError):
    pass


@dataclass(frozen=True)
class Summary:
    run_id: str
    track: str
    model: str
    split: str
    n: int
    correct: int
    ex: float
    ci_low: float
    ci_high: float
    by_difficulty: dict[str, tuple[int, int]]  # difficulty -> (correct, n)
    tokens_per_q: float
    calls_per_q: float
    latency_per_q: float
    cost_per_q: float


def summarize(run: Run) -> Summary:
    rs = run.results
    n = len(rs)
    correct = sum(r.correct for r in rs)
    lo, hi = wilson_pct(correct, n)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / n if n else 0.0

    return Summary(
        run_id=run.meta.run_id,
        track=run.meta.track,
        model=run.meta.model,
        split=run.meta.split,
        n=n,
        correct=correct,
        ex=100.0 * correct / n if n else 0.0,
        ci_low=lo,
        ci_high=hi,
        by_difficulty={
            d: (
                sum(r.correct for r in rs if r.difficulty == d),
                sum(1 for r in rs if r.difficulty == d),
            )
            for d in DIFFICULTIES
        },
        tokens_per_q=mean([r.prompt_tokens + r.completion_tokens for r in rs]),
        calls_per_q=mean([r.calls for r in rs]),
        latency_per_q=mean([r.latency_s for r in rs]),
        cost_per_q=mean([r.cost_usd for r in rs]),
    )


def _pct(correct: int, n: int) -> str:
    return f"{100.0 * correct / n:.1f}" if n else "-"


def _row(s: Summary) -> list[str]:
    return [
        s.track,
        s.model,
        s.split,
        str(s.n),
        f"{s.ex:.1f} [{s.ci_low:.1f}, {s.ci_high:.1f}]",
        *(_pct(*s.by_difficulty[d]) for d in DIFFICULTIES),
        f"{s.tokens_per_q:,.0f}",
        f"{s.calls_per_q:.2f}",
        f"{s.latency_per_q:.1f}s",
        f"${s.cost_per_q:.4f}",
    ]


def render_markdown(runs: Sequence[Run]) -> str:
    if not runs:
        return EMPTY_TABLE
    summaries = sorted(
        (summarize(r) for r in runs), key=lambda s: (s.split, s.track, s.model, s.run_id)
    )
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" if i < 3 else "---:" for i in range(len(COLUMNS))) + "|",
    ]
    lines += ["| " + " | ".join(_row(s)) + " |" for s in summaries]
    return "\n".join(lines)


def replace_between_markers(text: str, table: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise ReportError(f"README needs {START_MARKER} followed by {END_MARKER}")
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ReportError("README has more than one results block")
    head = text[: start + len(START_MARKER)]
    tail = text[end:]
    return f"{head}\n{table}\n{tail}"


def update_readme(path: Path, runs: Sequence[Run]) -> bool:
    """Rewrite the results block; returns True if the file changed."""
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    updated = replace_between_markers(text, render_markdown(runs))
    if updated == text:
        return False
    path.write_bytes(updated.replace("\n", newline).encode("utf-8"))
    return True
