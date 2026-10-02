"""Append-only JSONL log of every model call, cached or not.

One line per ``LLMClient.complete`` call, so token use, retries and cache hit
rates can be measured from real runs (the "tokens per question" numbers in the
README come from here).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from text2sql import config

STATUSES = ("ok", "error", "budget_exceeded")


@dataclass(frozen=True)
class CallRecord:
    ts: str
    run_id: str | None
    question_id: int | None
    provider: str
    model: str
    key: str
    cache_hit: bool
    attempts: int
    status: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None = None


def default_log_path() -> Path:
    return config.log_dir() / "llm_calls.jsonl"


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class CallLogger:
    def __init__(self, path: Path | None = None, *, now: Callable[[], str] = _utc_iso) -> None:
        self.path = path or default_log_path()
        self._now = now
        self._lock = threading.Lock()

    def log(self, **fields: object) -> CallRecord:
        if fields.get("status") not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {fields.get('status')!r}")
        record = CallRecord(ts=self._now(), **fields)  # type: ignore[arg-type]
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One write call per line in append mode keeps lines whole even with
            # several processes logging to the same file.
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
        return record


def read_log(path: Path | None = None) -> list[CallRecord]:
    path = path or default_log_path()
    if not path.is_file():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(CallRecord(**json.loads(line)))
    return records
