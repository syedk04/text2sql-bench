import json
from collections import Counter

import pytest

from text2sql.data.bird import Question
from text2sql.data.manifest import (
    DEFAULT_QUOTAS,
    ManifestError,
    build_manifest,
    dumps_manifest,
    load_manifest,
    select_ids,
    write_manifest,
)


def _questions(n_db: int = 11, per_bucket: int = 6) -> list[Question]:
    qs = []
    qid = 1000
    for d in range(n_db):
        for difficulty in ("simple", "moderate", "challenging"):
            for _ in range(per_bucket):
                qs.append(Question(qid, f"db_{d:02d}", f"q{qid}", "", "SELECT 1", difficulty))
                qid += 1
    return qs


def test_quotas_are_met_exactly():
    qs = _questions()
    ids = select_ids(qs)
    assert len(ids) == 50 == len(set(ids))
    by_id = {q.question_id: q for q in qs}
    counts = Counter(by_id[i].difficulty for i in ids)
    assert counts == Counter(DEFAULT_QUOTAS)


def test_selection_is_deterministic_and_seed_dependent():
    qs = _questions()
    assert select_ids(qs) == select_ids(qs)
    assert select_ids(qs) == select_ids(list(reversed(qs)))  # input order does not matter
    assert set(select_ids(qs, seed=1)) != set(select_ids(qs, seed=2))


def test_selection_spreads_across_databases():
    qs = _questions()
    by_id = {q.question_id: q for q in qs}
    ids = select_ids(qs)
    for difficulty, quota in DEFAULT_QUOTAS.items():
        per_db = Counter(by_id[i].db_id for i in ids if by_id[i].difficulty == difficulty)
        # round-robin: no database gets two before every database has one
        assert max(per_db.values()) - min(per_db.values()) <= 1
        assert len(per_db) == min(quota, 11)


def test_small_buckets_are_skipped_not_failed():
    qs = _questions(n_db=3, per_bucket=2)
    qs += [Question(9000 + i, "db_00", "q", "", "SELECT 1", "simple") for i in range(10)]
    ids = select_ids(qs, quotas={"simple": 12})
    assert len(ids) == 12


def test_not_enough_questions_is_an_error():
    with pytest.raises(ManifestError, match="only 6 challenging"):
        select_ids(_questions(n_db=1), quotas={"challenging": 7})
    with pytest.raises(ManifestError, match="unknown difficulty"):
        select_ids(_questions(), quotas={"hard": 1})


def test_manifest_round_trip(tmp_path):
    qs = _questions()
    manifest = build_manifest(qs, source={"sha256": "abc"})
    assert manifest["n"] == 50
    assert [e["question_id"] for e in manifest["questions"]] == sorted(select_ids(qs))
    path = write_manifest(manifest, tmp_path / "m" / "dev50.json")
    assert load_manifest(path) == sorted(select_ids(qs))
    assert path.read_bytes() == dumps_manifest(manifest).encode("utf-8")
    assert b"\r\n" not in path.read_bytes()


def test_load_manifest_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questions": [{"id": 1}]}))
    with pytest.raises(ManifestError, match="malformed"):
        load_manifest(bad)
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps({"questions": [{"question_id": 1}, {"question_id": 1}]}))
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(dup)
    count = tmp_path / "count.json"
    count.write_text(json.dumps({"n": 2, "questions": [{"question_id": 1}]}))
    with pytest.raises(ManifestError, match="n=2"):
        load_manifest(count)


def test_cli_build_manifest(tmp_path, capsys):
    from text2sql import cli

    qfile = tmp_path / "q.json"
    qfile.write_text(
        json.dumps(
            [
                {
                    "question_id": q.question_id,
                    "db_id": q.db_id,
                    "question": q.question,
                    "evidence": q.evidence,
                    "SQL": q.gold_sql,
                    "difficulty": q.difficulty,
                }
                for q in _questions()
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "dev50.json"
    assert cli.main(["build-manifest", "--questions", str(qfile), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n"] == 50
    assert len(data["source"]["sha256"]) == 64
    assert "wrote 50" in capsys.readouterr().out


@pytest.mark.parametrize("bad_id", ["5", 5.0, 5.7, True, None])
def test_load_manifest_rejects_non_integer_ids(tmp_path, bad_id):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"questions": [{"question_id": bad_id}]}))
    with pytest.raises(ManifestError):
        load_manifest(path)
