"""Gold vs gold on the real 500 questions (slow: several minutes)."""

import sqlite3

import pytest

from text2sql import config
from text2sql.data.bird import load_questions
from text2sql.eval.gates import gate_gold


@pytest.mark.bird
@pytest.mark.slow
def test_gold_vs_gold_all_500():
    report = gate_gold(load_questions(config.questions_path(), expect_total=500))
    failures = [(r.question_id, r.error) for r in report.failures]
    assert report.passed == 500, f"SQLite {sqlite3.sqlite_version}: {failures}"
