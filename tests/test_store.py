import json
import sqlite3

import pytest

from text2sql.results.store import (
    QuestionResult,
    Run,
    RunMeta,
    StoreError,
    git_sha,
    list_runs,
    read_run,
    write_run,
)


def _run(n: int = 3) -> Run:
    meta = RunMeta(
        run_id="2026-10-02-t0-dev50",
        track="T0",
        provider="fake",
        model="fake-1",
        split="dev50",
        manifest_sha="abc",
        code_git_sha="def",
        notes="unit test",
    )
    results = [
        QuestionResult(
            question_id=100 + i,
            db_id="shop",
            difficulty=("simple", "moderate", "challenging")[i % 3],
            pred_sql="SELECT 'é'" if i == 0 else None,
            correct=i % 2,
            status="ok" if i % 2 else "error",
            error=None if i % 2 else "boom",
            prompt_tokens=10 * i,
            completion_tokens=i,
            calls=1,
            latency_s=0.5,
            cost_usd=0.0,
        )
        for i in range(n)
    ]
    return Run(meta, results)


def test_round_trip(tmp_path):
    run = _run()
    write_run(run, tmp_path / "runs" / run.meta.run_id)
    back = read_run(tmp_path / "runs" / run.meta.run_id)
    assert back.meta == run.meta
    assert back.results == run.results


def test_files_are_plain_json_with_lf(tmp_path):
    d = write_run(_run(), tmp_path / "r")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1 and meta["track"] == "T0"
    raw = (d / "results.jsonl").read_bytes()
    assert b"\r\n" not in raw
    assert len(raw.decode("utf-8").splitlines()) == 3
    assert not list(d.glob("*.tmp"))


def test_meta_defaults():
    meta = RunMeta(run_id="x", track="T0", provider="p", model="m", split="dev50")
    assert meta.sqlite_version == sqlite3.sqlite_version
    assert meta.started_at.endswith("+00:00")


def test_duplicate_ids_are_refused(tmp_path):
    run = _run()
    run.results.append(run.results[0])
    with pytest.raises(StoreError, match="duplicate"):
        write_run(run, tmp_path / "r")


def test_incomplete_or_foreign_runs_are_refused(tmp_path):
    with pytest.raises(StoreError, match="not a complete run"):
        read_run(tmp_path)
    d = write_run(_run(), tmp_path / "r")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["schema_version"] = 99
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(StoreError, match="schema_version"):
        read_run(d)


def test_unknown_result_fields_are_refused(tmp_path):
    d = write_run(_run(1), tmp_path / "r")
    row = json.loads((d / "results.jsonl").read_text(encoding="utf-8"))
    row["surprise"] = 1
    (d / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(StoreError, match="unknown fields"):
        read_run(d)


def test_list_runs(tmp_path):
    assert list_runs(tmp_path / "none") == []
    write_run(_run(), tmp_path / "b")
    write_run(_run(), tmp_path / "a")
    (tmp_path / "not-a-run").mkdir()
    assert [p.name for p in list_runs(tmp_path)] == ["a", "b"]


def test_git_sha_outside_repo(tmp_path):
    assert git_sha(tmp_path) is None


@pytest.mark.parametrize(
    "line, message",
    [
        ("[1, 2]", "expected a JSON object"),
        ("not json", "not valid JSON"),
        ('{"question_id": "abc"}', "'question_id' must be an integer"),
        ('{"question_id": 1, "db_id": 5}', "'db_id' must be a string"),
        ('{"question_id": 1, "correct": "1"}', "'correct' must be 0 or 1"),
        ('{"question_id": 1, "correct": 2}', "'correct' must be 0 or 1"),
        ('{"question_id": 1, "correct": 1.0}', "'correct' must be 0 or 1"),
        ('{"question_id": 1, "prompt_tokens": -5}', "non-negative integer"),
        ('{"question_id": 1, "latency_s": NaN}', "non-negative number"),
        ('{"question_id": 1}', "missing"),
    ],
)
def test_malformed_results_raise_store_error(tmp_path, line, message):
    d = write_run(_run(1), tmp_path / "r")
    (d / "results.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(StoreError, match=message):
        read_run(d)


def test_boolean_correct_is_accepted_as_int(tmp_path):
    d = write_run(_run(1), tmp_path / "r")
    row = json.loads((d / "results.jsonl").read_text(encoding="utf-8"))
    row["correct"] = True
    (d / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert read_run(d).results[0].correct == 1


@pytest.mark.parametrize("content", ["[]", "null", "not json"])
def test_run_json_must_be_an_object(tmp_path, content):
    d = write_run(_run(1), tmp_path / "r")
    (d / "run.json").write_text(content, encoding="utf-8")
    with pytest.raises(StoreError):
        read_run(d)


def test_meta_field_types_are_checked(tmp_path):
    d = write_run(_run(1), tmp_path / "r")
    meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
    meta["model"] = 7
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(StoreError, match="'model' must be a string"):
        read_run(d)
