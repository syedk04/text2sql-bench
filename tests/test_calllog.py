import json
import threading

import pytest

from text2sql import config
from text2sql.llm.calllog import CallLogger, default_log_path, read_log

FIELDS = dict(
    run_id="r1",
    question_id=7,
    provider="fake",
    model="m",
    key="a" * 64,
    cache_hit=False,
    attempts=2,
    status="ok",
    prompt_tokens=100,
    completion_tokens=20,
    latency_s=1.25,
)


def test_log_writes_one_json_line_per_call(tmp_path):
    path = tmp_path / "logs dir" / "llm_calls.jsonl"
    logger = CallLogger(path, now=lambda: "2026-10-02T12:00:00.000+00:00")
    logger.log(**FIELDS)
    logger.log(**{**FIELDS, "cache_hit": True, "attempts": 0, "error": None})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {**FIELDS, "ts": "2026-10-02T12:00:00.000+00:00", "error": None}
    assert set(first) == {
        "ts", "run_id", "question_id", "provider", "model", "key", "cache_hit",
        "attempts", "status", "prompt_tokens", "completion_tokens", "latency_s", "error",
    }
    records = read_log(path)
    assert [r.cache_hit for r in records] == [False, True]


def test_bad_status_and_unknown_fields(tmp_path):
    logger = CallLogger(tmp_path / "l.jsonl")
    with pytest.raises(ValueError, match="status"):
        logger.log(**{**FIELDS, "status": "maybe"})
    with pytest.raises(TypeError):
        logger.log(**FIELDS, surprise=1)
    assert read_log(tmp_path / "l.jsonl") == []


def test_threads_do_not_interleave_lines(tmp_path):
    path = tmp_path / "l.jsonl"
    logger = CallLogger(path)

    def work(i):
        for j in range(50):
            logger.log(**{**FIELDS, "question_id": i * 1000 + j, "error": "x" * 500})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = read_log(path)
    assert len(records) == 300
    assert len({r.question_id for r in records}) == 300


def test_default_path_follows_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    assert default_log_path() == tmp_path.resolve() / "logs" / "llm_calls.jsonl"
    assert read_log() == []
