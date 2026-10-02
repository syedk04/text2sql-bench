"""On-disk record of a scored run.

A run lives in ``runs/<run_id>/`` as two files:

* ``run.json``: one :class:`RunMeta` object describing what was run;
* ``results.jsonl``: one :class:`QuestionResult` per line.

Both are plain JSON so runs can be committed, diffed and re-rendered into the
README table without re-running anything.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
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


def _from_dict(cls: type, data: dict[str, Any], where: str) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise StoreError(f"{where}: unknown fields {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise StoreError(f"{where}: {exc}") from exc


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
    meta_raw = json.loads(run_file.read_text(encoding="utf-8"))
    version = meta_raw.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise StoreError(f"{run_file}: unsupported schema_version {version!r}")
    meta = _from_dict(RunMeta, meta_raw, str(run_file))
    results = []
    with open(results_file, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                results.append(_from_dict(QuestionResult, json.loads(line), f"{results_file}:{n}"))
    return Run(meta, results)


def list_runs(root: Path | None = None) -> list[Path]:
    root = root or runs_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / RUN_FILE).is_file())
