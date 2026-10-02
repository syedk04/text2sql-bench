import json
from pathlib import Path

import pytest

from text2sql.data.bird import DatasetError, Question, by_id, load_questions, parse_questions

FIXTURE = Path(__file__).parent / "fixtures" / "questions.json"


def test_loads_fixture_in_file_order():
    qs = load_questions(FIXTURE)
    assert len(qs) == 12
    assert [q.question_id for q in qs] == list(range(100, 112))
    assert {q.db_id for q in qs} == {"shop", "school", "zoo"}
    assert {q.difficulty for q in qs} == {"simple", "moderate", "challenging"}
    assert isinstance(qs[0], Question)
    assert qs[0].gold_sql == "SELECT COUNT(*) FROM item"


def test_expect_total_is_enforced():
    load_questions(FIXTURE, expect_total=12)
    with pytest.raises(DatasetError, match="expected 500"):
        load_questions(FIXTURE, expect_total=500)


def test_question_is_frozen():
    q = load_questions(FIXTURE)[0]
    with pytest.raises(AttributeError):
        q.db_id = "other"  # type: ignore[misc]


def _record(**overrides):
    rec = {
        "question_id": 1,
        "db_id": "shop",
        "question": "q?",
        "evidence": "",
        "SQL": "SELECT 1",
        "difficulty": "simple",
    }
    rec.update(overrides)
    return rec


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"not": "a list"}, "JSON array"),
        (["x"], "expected an object"),
        ([{k: v for k, v in _record().items() if k != "SQL"}], "missing field 'SQL'"),
        ([_record(question_id="1")], "question_id"),
        ([_record(question_id=True)], "question_id"),
        ([_record(difficulty="hard")], "unknown difficulty"),
        ([_record(SQL="   ")], "empty gold SQL"),
        ([_record(db_id=" ")], "empty db_id"),
        ([_record(), _record()], "duplicate question_id 1"),
    ],
)
def test_rejects_malformed_input(raw, message):
    with pytest.raises(DatasetError, match=message):
        parse_questions(raw)


def test_by_id_and_unicode(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([_record(question="Où est Zürich?")]), encoding="utf-8")
    qs = load_questions(path)
    assert by_id(qs)[1].question == "Où est Zürich?"


def test_duplicates_allowed_only_on_request():
    raw = [_record(), _record(question="same id again")]
    with pytest.raises(DatasetError, match="duplicate"):
        parse_questions(raw)
    qs = parse_questions(raw, allow_duplicate_ids=True)
    assert [q.question for q in qs] == ["q?", "same id again"]
