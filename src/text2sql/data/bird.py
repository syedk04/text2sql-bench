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


def parse_questions(raw: Any, expect_total: int | None = None) -> list[Question]:
    if not isinstance(raw, list):
        raise DatasetError(f"expected a JSON array, got {type(raw).__name__}")
    questions = [_parse_record(i, rec) for i, rec in enumerate(raw)]
    seen: set[int] = set()
    for q in questions:
        if q.question_id in seen:
            raise DatasetError(f"duplicate question_id {q.question_id}")
        seen.add(q.question_id)
    if expect_total is not None and len(questions) != expect_total:
        raise DatasetError(f"expected {expect_total} questions, found {len(questions)}")
    return questions


def load_questions(path: str | Path, expect_total: int | None = None) -> list[Question]:
    """Load and validate a question file, keeping file order."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_questions(raw, expect_total=expect_total)


def by_id(questions: list[Question]) -> dict[int, Question]:
    return {q.question_id: q for q in questions}
