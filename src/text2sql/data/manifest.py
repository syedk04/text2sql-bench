"""The fixed 50-question dev slice.

Full 500-question runs are expensive on a free model tier, so day-to-day work
uses a fixed subset. The subset is drawn deterministically: for each difficulty,
the questions of every database are shuffled with a seeded RNG and then taken
round-robin across databases (in sorted order) until the quota is met. That
spreads the slice over all 11 databases instead of letting a big one dominate.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from text2sql.data.bird import DIFFICULTIES, Question

DEFAULT_SEED = 20261002
DEFAULT_QUOTAS: dict[str, int] = {"simple": 15, "moderate": 25, "challenging": 10}
MANIFEST_VERSION = 1


class ManifestError(ValueError):
    pass


def select_ids(
    questions: Sequence[Question],
    quotas: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    unknown = set(quotas) - set(DIFFICULTIES)
    if unknown:
        raise ManifestError(f"unknown difficulty in quotas: {sorted(unknown)}")
    rng = random.Random(seed)
    db_ids = sorted({q.db_id for q in questions})
    chosen: list[int] = []
    for difficulty in DIFFICULTIES:
        quota = quotas.get(difficulty, 0)
        if quota <= 0:
            continue
        buckets: list[list[int]] = []
        for db_id in db_ids:
            bucket = sorted(
                q.question_id
                for q in questions
                if q.db_id == db_id and q.difficulty == difficulty
            )
            rng.shuffle(bucket)
            buckets.append(bucket)
        available = sum(len(b) for b in buckets)
        if available < quota:
            raise ManifestError(
                f"only {available} {difficulty} questions available, quota is {quota}"
            )
        taken = 0
        depth = 0
        while taken < quota:
            for bucket in buckets:
                if depth < len(bucket) and taken < quota:
                    chosen.append(bucket[depth])
                    taken += 1
            depth += 1
    return chosen


def build_manifest(
    questions: Sequence[Question],
    *,
    quotas: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED,
    name: str = "dev50",
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    ids = set(select_ids(questions, quotas, seed))
    entries = [
        {"question_id": q.question_id, "db_id": q.db_id, "difficulty": q.difficulty}
        for q in sorted(questions, key=lambda q: q.question_id)
        if q.question_id in ids
    ]
    return {
        "version": MANIFEST_VERSION,
        "name": name,
        "n": len(entries),
        "seed": seed,
        "quotas": {d: quotas.get(d, 0) for d in DIFFICULTIES},
        "method": "per difficulty: sorted db_ids, seeded shuffle per (difficulty, db), "
        "round-robin across dbs until quota",
        "source": source or {},
        "questions": entries,
    }


def dumps_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps_manifest(manifest))
    return path


def load_manifest(path: str | Path) -> list[int]:
    """Return the manifest's question ids (ascending)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        ids = [e["question_id"] for e in data["questions"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"malformed manifest {path}: {exc}") from exc
    if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise ManifestError(f"malformed manifest {path}: question ids must be integers")
    if len(set(ids)) != len(ids):
        raise ManifestError(f"duplicate question ids in manifest {path}")
    if data.get("n") is not None and data["n"] != len(ids):
        raise ManifestError(f"manifest {path} says n={data['n']} but lists {len(ids)}")
    return ids
