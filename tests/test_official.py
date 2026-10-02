"""The official-scorer harness.

CI has no network and the official scripts are not ours to vendor, so the glue is
tested against small stand-in scripts that expose the same function names and
import the same database drivers. The real cross-check is the slow bird test at
the bottom, which fetches the pinned scripts.
"""

import json
from pathlib import Path

import pytest

from text2sql import config
from text2sql.data.bird import Question, load_questions
from text2sql.eval.ex import load_official_predictions, score_many
from text2sql.eval.official import (
    BASELINES,
    SCRIPTS,
    OfficialError,
    compare_vectors,
    fetch_baseline,
    load_official,
    run_official,
    write_gold_inputs,
)

FAKE_UTILS = '''
import json
import sqlite3
import psycopg2  # the real module imports these at top level too
import pymysql


def package_sqls(sql_path, db_root_path, mode="pred"):
    clean, dbs = [], []
    if mode == "pred":
        for _, s in json.load(open(sql_path, "r")).items():
            sql, db = s.split("\\t----- bird -----\\t")
            clean.append(sql)
    else:
        for line in open(sql_path).readlines():
            sql, db = line.strip().split("\\t")
            clean.append(sql)
            dbs.append(db_root_path + db + "/" + db + ".sqlite")
    return clean, dbs


def sort_results(rows):
    return sorted(rows, key=lambda x: x["sql_idx"])
'''

FAKE_EX = '''
from evaluation_utils import package_sqls, sort_results
import sqlite3


def result_callback(result):
    exec_result.append(result)


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0, sql_dialect="SQLite"):
    # same callback contract as the real script, but in reverse order to prove
    # that sort_results is applied
    for i in reversed(range(len(sqls))):
        pred, gold = sqls[i]
        conn = sqlite3.connect(db_places[i])
        try:
            res = int(set(conn.execute(pred).fetchall()) == set(conn.execute(gold).fetchall()))
        except Exception:
            res = 0
        conn.close()
        result_callback({"sql_idx": i, "res": res})
'''


@pytest.fixture
def fake_official(tmp_path):
    root = tmp_path / "official scripts"
    root.mkdir()
    (root / "evaluation_utils.py").write_text(FAKE_UTILS, encoding="utf-8")
    (root / "evaluation_ex.py").write_text(FAKE_EX, encoding="utf-8")
    return root


def _preds(path: Path, questions, override=None):
    override = override or {}
    data = {
        str(i): f"{override.get(i, q.gold_sql)}\t----- bird -----\t{q.db_id}"
        for i, q in enumerate(questions)
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_pins_are_full_hashes():
    assert all(len(h) == 64 for h in SCRIPTS.values())
    assert all(len(b.sha256) == 64 for b in BASELINES.values())
    assert BASELINES["gpt-4"].published_ex == 47.80
    assert BASELINES["llama-3-70b"].published_ex == 40.80


def test_write_gold_inputs(tmp_path):
    qs = [
        Question(7, "shop", "q", "", "SELECT 'é'", "moderate"),
        Question(3, "zoo", "q", "", "SELECT 2", "simple"),
    ]
    gold, diff = write_gold_inputs(qs, tmp_path / "w")
    assert gold.read_text(encoding="utf-8") == "SELECT 'é'\tshop\nSELECT 2\tzoo\n"
    rows = [json.loads(line) for line in diff.read_text().splitlines()]
    assert [r["difficulty"] for r in rows] == ["moderate", "simple"]


def test_write_gold_inputs_refuses_multiline_gold(tmp_path):
    with pytest.raises(OfficialError, match="tab or newline"):
        write_gold_inputs([Question(1, "shop", "q", "", "SELECT\n1", "simple")], tmp_path)


def test_stub_drivers_let_the_scripts_import(fake_official):
    utils, ex = load_official(fake_official, fetch=False)
    assert hasattr(utils, "package_sqls") and hasattr(ex, "run_sqls_parallel")
    assert (fake_official / "stubs" / "psycopg2.py").exists()


def test_run_official_with_stand_in_scripts(synthetic_data_dir, fake_official, tmp_path):
    questions = load_questions(config.questions_path())
    gold, _ = write_gold_inputs(questions, tmp_path / "w")
    preds = _preds(tmp_path / "p.json", questions, {2: "SELECT 0", 4: "SELECT nope"})
    official = run_official(preds, gold, root=fake_official, fetch=False)
    assert len(official) == 12
    assert [i for i, v in enumerate(official) if v == 0] == [2, 4]

    ours = score_many(questions, [p.sql for p in load_official_predictions(preds)])
    agreement = compare_vectors("fake", official, [r.correct for r in ours])
    assert agreement.ok
    assert agreement.official_ex == agreement.ours_ex == pytest.approx(100 * 10 / 12)


def test_utf8_inputs_are_read_as_utf8(synthetic_data_dir, fake_official, tmp_path):
    q = Question(1, "shop", "q", "", "SELECT 'café'", "simple")
    gold, _ = write_gold_inputs([q], tmp_path / "w")
    preds = _preds(tmp_path / "p.json", [q])
    assert run_official(preds, gold, root=fake_official, fetch=False) == [1]


def test_compare_vectors_reports_positions():
    a = compare_vectors("x", [1, 0, 1, 1], [1, 1, 1, 0])
    assert a.disagreements == [1, 3] and not a.ok
    with pytest.raises(OfficialError):
        compare_vectors("x", [1], [1, 0])


def test_unknown_baseline():
    with pytest.raises(OfficialError, match="unknown baseline"):
        fetch_baseline("gpt-5")


@pytest.mark.bird
@pytest.mark.slow
@pytest.mark.parametrize("baseline", ["gpt-4", "llama-3-70b"])
def test_gate_a_official_agreement(baseline, tmp_path):
    """Gate A: our scorer and the pinned official scorer agree on every question."""
    pytest.importorskip("func_timeout")
    questions = load_questions(config.questions_path(), expect_total=500)
    gold, _ = write_gold_inputs(questions, tmp_path / "w")
    pred_path = fetch_baseline(baseline)
    preds = load_official_predictions(pred_path)
    assert len(preds) == 500
    official = run_official(pred_path, gold)
    ours = score_many(questions, [p.sql for p in preds])
    agreement = compare_vectors(baseline, official, [r.correct for r in ours])
    assert agreement.disagreements == []


def test_missing_func_timeout_is_explained(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "func_timeout" else real(name, *a),
    )
    with pytest.raises(OfficialError, match="uv sync --group official"):
        load_official(tmp_path, fetch=True)


def test_official_run_cleans_up_wal_side_files(synthetic_data_dir, fake_official, tmp_path):
    # The official workers open databases read-write and can be shut down with
    # connections still open, which leaves empty -wal/-shm files behind.
    db = config.db_path("zoo")
    q = Question(1, "zoo", "q", "", "SELECT COUNT(*) FROM animal", "simple")
    gold, _ = write_gold_inputs([q], tmp_path / "w")
    preds = _preds(tmp_path / "p.json", [q])
    db.with_name(db.name + "-wal").write_bytes(b"")
    db.with_name(db.name + "-shm").write_bytes(bytes(32))
    assert run_official(preds, gold, root=fake_official, fetch=False) == [1]
    assert sorted(p.name for p in db.parent.iterdir()) == [db.name]


def test_remove_wal_leftovers_keeps_non_empty_wal(tmp_path):
    from text2sql.eval.official import remove_wal_leftovers

    db = tmp_path / "a.sqlite"
    db.write_bytes(b"")
    (tmp_path / "a.sqlite-wal").write_bytes(b"pending changes")
    (tmp_path / "a.sqlite-shm").write_bytes(b"x")
    assert remove_wal_leftovers([db]) == []
    assert (tmp_path / "a.sqlite-wal").exists()
    (tmp_path / "a.sqlite-wal").write_bytes(b"")
    assert len(remove_wal_leftovers([db])) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.sqlite"]
