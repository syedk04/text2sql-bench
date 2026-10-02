import pytest

from text2sql import config
from text2sql.data.bird import Question, load_questions
from text2sql.eval.gates import gate_gold


def test_gold_vs_gold_on_synthetic_data(synthetic_data_dir):
    report = gate_gold(load_questions(config.questions_path()))
    assert report.ok
    assert (report.n, report.passed, report.failures) == (12, 12, [])
    assert len(report.slowest) == 5


def test_gate_reports_failures(synthetic_data_dir):
    qs = load_questions(config.questions_path())
    broken = Question(999, "shop", "q", "", "SELECT nope FROM item", "simple")
    report = gate_gold([*qs, broken])
    assert not report.ok
    assert report.passed == 12
    assert [r.question_id for r in report.failures] == [999]


def test_empty_gate_is_not_a_pass():
    assert not gate_gold([]).ok


def test_cli_gate_gold(synthetic_data_dir, capsys):
    from text2sql import cli

    assert cli.main(["gate-gold"]) == 0
    out = capsys.readouterr().out
    assert "12/12" in out and "GATE PASSED" in out


@pytest.mark.parametrize("flag", [[], ["--timeout", "5"]])
def test_cli_gate_gold_with_manifest(synthetic_data_dir, tmp_path, capsys, flag):
    import json

    from text2sql import cli

    m = tmp_path / "m.json"
    m.write_text(json.dumps({"questions": [{"question_id": 100}, {"question_id": 105}]}))
    assert cli.main(["gate-gold", "--manifest", str(m), *flag]) == 0
    assert "2/2" in capsys.readouterr().out
