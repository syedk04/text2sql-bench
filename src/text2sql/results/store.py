"""On-disk record of a scored run.

A run lives in ``runs/<run_id>/`` as two files:

* ``run.json``: one :class:`RunMeta` object describing what was run;
* ``results.jsonl``: one :class:`QuestionResult` per line.

Both are plain JSON so runs can be committed, diffed and re-rendered into the
README table without re-running anything.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from text2sql import config

RUN_FILE = "run.json"
RESULTS_FILE = "results.jsonl"
SCHEMA_VERSION = 1


class StoreError(ValueError):
    pass


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    track: str  # e.g. "T0"
    provider: str
    model: str
    split: str  # "dev50" or "full500"
    manifest_sha: str | None = None
    code_git_sha: str | None = None
    sqlite_version: str = field(default_factory=lambda: sqlite3.sqlite_version)
    started_at: str = field(default_factory=lambda: utc_now())
    notes: str = ""


@dataclass(frozen=True)
class QuestionResult:
    question_id: int
    db_id: str
    difficulty: str
    pred_sql: str | None
    correct: int
    status: str  # pred status from the scorer: ok / error / timeout / rejected
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class Run:
    meta: RunMeta
    results: list[QuestionResult]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_sha(repo: Path | None = None) -> str | None:
    """Current commit of the code, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def runs_dir() -> Path:
    return config.PROJECT_ROOT / "runs"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return (_is_int(v) or isinstance(v, float)) and math.isfinite(v)


# Per-field checks: (description, predicate). Fields not listed must be str.
_CHECKS: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "question_id": ("an integer", _is_int),
    "correct": ("0 or 1 (or a boolean)", lambda v: isinstance(v, int) and v in (0, 1)),
    "pred_sql": ("a string or null", lambda v: v is None or isinstance(v, str)),
    "error": ("a string or null", lambda v: v is None or isinstance(v, str)),
    "manifest_sha": ("a string or null", lambda v: v is None or isinstance(v, str)),
    "code_git_sha": ("a string or null", lambda v: v is None or isinstance(v, str)),
    "prompt_tokens": ("a non-negative integer", lambda v: _is_int(v) and v >= 0),
    "completion_tokens": ("a non-negative integer", lambda v: _is_int(v) and v >= 0),
    "calls": ("a non-negative integer", lambda v: _is_int(v) and v >= 0),
    "latency_s": ("a non-negative number", lambda v: _is_number(v) and v >= 0),
    "cost_usd": ("a non-negative number", lambda v: _is_number(v) and v >= 0),
}


def _from_dict(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise StoreError(f"{where}: expected a JSON object, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise StoreError(f"{where}: unknown fields {sorted(unknown)}")
    for key, value in data.items():
        what, ok = _CHECKS.get(key, ("a string", lambda v: isinstance(v, str)))
        if not ok(value):
            raise StoreError(f"{where}: field {key!r} must be {what}, got {value!r}")
    if "correct" in data:
        data = {**data, "correct": int(data["correct"])}
    try:
        return cls(**data)
    except TypeError as exc:
        raise StoreError(f"{where}: {exc}") from exc


def _load_json(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise StoreError(f"{where}: not valid JSON ({exc})") from exc


def write_run(run: Run, directory: Path) -> Path:
    """Write a run atomically-ish: results first, then run.json as the marker."""
    ids = [r.question_id for r in run.results]
    if len(set(ids)) != len(ids):
        raise StoreError("duplicate question_id in results")
    directory.mkdir(parents=True, exist_ok=True)
    results_tmp = directory / (RESULTS_FILE + ".tmp")
    with open(results_tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in run.results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(results_tmp, directory / RESULTS_FILE)
    meta = {"schema_version": SCHEMA_VERSION, **asdict(run.meta)}
    meta_tmp = directory / (RUN_FILE + ".tmp")
    with open(meta_tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(meta_tmp, directory / RUN_FILE)
    return directory


def read_run(directory: Path) -> Run:
    run_file = directory / RUN_FILE
    results_file = directory / RESULTS_FILE
    if not run_file.is_file() or not results_file.is_file():
        raise StoreError(f"{directory} is not a complete run (need {RUN_FILE} and {RESULTS_FILE})")
    meta_raw = _load_json(run_file.read_text(encoding="utf-8"), str(run_file))
    if not isinstance(meta_raw, dict):
        raise StoreError(f"{run_file}: expected a JSON object, got {type(meta_raw).__name__}")
    version = meta_raw.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise StoreError(f"{run_file}: unsupported schema_version {version!r}")
    meta = _from_dict(RunMeta, meta_raw, str(run_file))
    results = []
    with open(results_file, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                where = f"{results_file}:{n}"
                results.append(_from_dict(QuestionResult, _load_json(line, where), where))
    return Run(meta, results)


def list_runs(root: Path | None = None) -> list[Path]:
    root = root or runs_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / RUN_FILE).is_file())
