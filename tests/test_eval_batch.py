import json

import pytest

from text2sql import config
from text2sql.data.bird import load_questions
from text2sql.eval.ex import (
    OFFICIAL_SEPARATOR,
    breakdown,
    format_breakdown,
    load_official_predictions,
    parse_official_predictions,
    score_many,
)


def test_separator_is_tab_delimited():
    assert OFFICIAL_SEPARATOR == "\t----- bird -----\t"


def test_parse_official_predictions_matches_package_sqls():
    raw = {
        "0": "SELECT 1\t----- bird -----\tshop",
        "1": 42,
        "2": "  SELECT 2  ",
        "3": "SELECT 3\t----- bird -----\tzoo\t----- bird -----\tzoo",
        "4": None,
    }
    preds = parse_official_predictions(raw)
    assert [(p.sql, p.db_id, p.well_formed) for p in preds] == [
        ("SELECT 1", "shop", True),
        (" ", "financial", False),
        ("SELECT 2", "financial", False),
        (raw["3"].strip(), "financial", False),
        (" ", "financial", False),
    ]


def test_parse_keeps_file_order_not_key_order():
    raw = json.loads('{"1": "B\\t----- bird -----\\tx", "0": "A\\t----- bird -----\\tx"}')
    assert [p.sql for p in parse_official_predictions(raw)] == ["B", "A"]


def test_parse_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        parse_official_predictions(["SELECT 1"])  # type: ignore[arg-type]


def _write_preds(path, questions, override=None):
    override = override or {}
    data = {
        str(i): f"{override.get(i, q.gold_sql)}{OFFICIAL_SEPARATOR}{q.db_id}"
        for i, q in enumerate(questions)
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_score_many_and_breakdown(synthetic_data_dir, tmp_path):
    questions = load_questions(config.questions_path())
    preds_path = _write_preds(tmp_path / "p.json", questions, {0: "SELECT 0", 5: "SELECT nope"})
    preds = load_official_predictions(preds_path)
    seen = []
    results = score_many(
        questions, [p.sql for p in preds], progress=lambda d, t, r: seen.append((d, t))
    )
    assert seen[-1] == (12, 12)
    assert [r.correct for r in results].count(0) == 2
    table = breakdown(results)
    assert table["total"].n == 12 and table["total"].correct == 10
    assert sum(table[d].n for d in ("simple", "moderate", "challenging")) == 12
    assert table["total"].ex == pytest.approx(100 * 10 / 12)
    text = format_breakdown(table)
    assert "simple" in text and "83.33" in text


def test_score_many_requires_matching_counts(synthetic_data_dir):
    questions = load_questions(config.questions_path())
    with pytest.raises(ValueError, match="pairs them by position"):
        score_many(questions, ["SELECT 1"])


def test_cli_eval_preds(synthetic_data_dir, tmp_path, capsys):
    from text2sql import cli

    questions = load_questions(config.questions_path())
    preds = _write_preds(tmp_path / "p.json", questions, {3: "SELECT 1; SELECT 2"})
    out = tmp_path / "res.jsonl"
    assert cli.main(["eval-preds", "--preds", str(preds), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "91.67" in text
    assert "pred:rejected/gold:not_run=1" in text
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12 and json.loads(lines[3])["correct"] == 0


def test_cli_eval_preds_count_mismatch(synthetic_data_dir, tmp_path, capsys):
    from text2sql import cli

    p = tmp_path / "p.json"
    p.write_text(json.dumps({"0": "SELECT 1"}), encoding="utf-8")
    assert cli.main(["eval-preds", "--preds", str(p)]) == 1
    assert "1 predictions but 12 questions" in capsys.readouterr().out


def test_cli_eval_preds_warns_on_db_mismatch(synthetic_data_dir, tmp_path, capsys):
    from text2sql import cli

    questions = load_questions(config.questions_path())
    data = {
        str(i): f"{q.gold_sql}{OFFICIAL_SEPARATOR}{'zoo' if i == 0 else q.db_id}"
        for i, q in enumerate(questions)
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(["eval-preds", "--preds", str(p)]) == 0
    assert "differs from gold at positions [0]" in capsys.readouterr().out
