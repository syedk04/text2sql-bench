import json
from collections import Counter

import pytest

from text2sql import config
from text2sql.data.bird import HF_REVISION, HF_SHA256, by_id, load_questions
from text2sql.data.manifest import DEFAULT_QUOTAS, build_manifest, dumps_manifest, load_manifest

MANIFEST = config.PROJECT_ROOT / "manifests" / "dev50.json"


def test_committed_manifest_shape():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = load_manifest(MANIFEST)
    assert len(ids) == 50 and ids == sorted(ids)
    assert Counter(e["difficulty"] for e in data["questions"]) == Counter(DEFAULT_QUOTAS)
    assert len({e["db_id"] for e in data["questions"]}) == 11
    assert data["source"]["hf_revision"] == HF_REVISION
    assert data["source"]["sha256"] == HF_SHA256


@pytest.mark.bird
def test_manifest_rebuilds_byte_identical_from_real_questions():
    questions = load_questions(config.questions_path(), expect_total=500)
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rebuilt = build_manifest(questions, seed=committed["seed"], source=committed["source"])
    expected = MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    assert dumps_manifest(rebuilt).encode("utf-8") == expected
    lookup = by_id(questions)
    for entry in committed["questions"]:
        q = lookup[entry["question_id"]]
        assert (q.db_id, q.difficulty) == (entry["db_id"], entry["difficulty"])
