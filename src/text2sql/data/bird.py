"""BIRD Mini-Dev questions.

Questions and gold SQL come from the Hugging Face dataset ``birdsql/bird_mini_dev``
(SQLite split, 500 records), pinned to one revision. The file is a JSON array whose
records carry ``question_id``, ``db_id``, ``question``, ``evidence``, ``SQL`` and
``difficulty``. File order is significant: the official evaluator matches
predictions to gold by position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIFFICULTIES = ("simple", "moderate", "challenging")
EXPECTED_TOTAL = 500
EXPECTED_DB_COUNT = 11

_REQUIRED: dict[str, type] = {
    "question_id": int,
    "db_id": str,
    "question": str,
    "evidence": str,
    "SQL": str,
    "difficulty": str,
}


class DatasetError(ValueError):
    """The question file is not what we expect."""


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def _parse_record(index: int, record: Any) -> Question:
    if not isinstance(record, dict):
        raise DatasetError(f"record {index}: expected an object, got {type(record).__name__}")
    for field, kind in _REQUIRED.items():
        if field not in record:
            raise DatasetError(f"record {index}: missing field {field!r}")
        value = record[field]
        # bool is a subclass of int; reject it explicitly for question_id
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise DatasetError(
                f"record {index}: field {field!r} should be {kind.__name__}, "
                f"got {type(value).__name__}"
            )
    if record["difficulty"] not in DIFFICULTIES:
        raise DatasetError(f"record {index}: unknown difficulty {record['difficulty']!r}")
    if not record["db_id"].strip():
        raise DatasetError(f"record {index}: empty db_id")
    if not record["SQL"].strip():
        raise DatasetError(f"record {index}: empty gold SQL")
    return Question(
        question_id=record["question_id"],
        db_id=record["db_id"],
        question=record["question"],
        evidence=record["evidence"],
        gold_sql=record["SQL"],
        difficulty=record["difficulty"],
    )


def parse_questions(
    raw: Any, expect_total: int | None = None, *, allow_duplicate_ids: bool = False
) -> list[Question]:
    """Validate raw records. ``allow_duplicate_ids`` exists only for the legacy
    zip file, which repeats two records (ids 137 and 138); scoring there is by
    position, so the repeats are kept as they are."""
    if not isinstance(raw, list):
        raise DatasetError(f"expected a JSON array, got {type(raw).__name__}")
    questions = [_parse_record(i, rec) for i, rec in enumerate(raw)]
    seen: set[int] = set()
    for q in questions:
        if q.question_id in seen and not allow_duplicate_ids:
            raise DatasetError(f"duplicate question_id {q.question_id}")
        seen.add(q.question_id)
    if expect_total is not None and len(questions) != expect_total:
        raise DatasetError(f"expected {expect_total} questions, found {len(questions)}")
    return questions


def load_questions(
    path: str | Path, expect_total: int | None = None, *, allow_duplicate_ids: bool = False
) -> list[Question]:
    """Load and validate a question file, keeping file order."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_questions(raw, expect_total=expect_total, allow_duplicate_ids=allow_duplicate_ids)


def by_id(questions: list[Question]) -> dict[int, Question]:
    return {q.question_id: q for q in questions}


# --- download -------------------------------------------------------------------

HF_REPO = "birdsql/bird_mini_dev"
HF_REVISION = "f65faf4ae3b638c1fa6df1d3370c8d92c8366301"
HF_FILE = "data/mini_dev_sqlite-00000-of-00001.json"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}"
# sha256 of the file at HF_REVISION; a pinned revision should never change, so a
# different hash means a corrupted or wrong file.
HF_SHA256 = "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2"


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def install_questions(
    data: bytes,
    dest: Path,
    *,
    expected_sha256: str | None = HF_SHA256,
    expect_total: int | None = EXPECTED_TOTAL,
) -> Path:
    """Validate raw question-file bytes and write them (plus a .sha256 sidecar)."""
    from text2sql.net import sha256_bytes, write_atomic

    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise DatasetError(
            f"sha256 mismatch for question file: got {digest}, expected {expected_sha256}"
        )
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"question file is not valid UTF-8 JSON: {exc}") from exc
    questions = parse_questions(raw, expect_total=expect_total)
    if expect_total == EXPECTED_TOTAL:
        n_db = len({q.db_id for q in questions})
        if n_db != EXPECTED_DB_COUNT:
            raise DatasetError(f"expected {EXPECTED_DB_COUNT} databases, found {n_db}")
    write_atomic(dest, data)
    write_atomic(sidecar_path(dest), f"{digest}  {dest.name}\n".encode())
    return dest


def download_questions(dest: Path, from_file: Path | None = None) -> Path:
    """Fetch the pinned Hugging Face file (or copy a manually downloaded one)."""
    if from_file is not None:
        data = Path(from_file).read_bytes()
    else:
        from text2sql.net import fetch_bytes

        data = fetch_bytes(HF_URL)
    return install_questions(data, dest)
