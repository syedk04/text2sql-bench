import random

import pytest

from text2sql import config
from text2sql.llm.cache import DiskCache, cache_key
from text2sql.llm.calllog import CallLogger, read_log
from text2sql.llm.client import LLMClient
from text2sql.llm.fake import FakeProvider
from text2sql.llm.requestlog import (
    REDACTED,
    RequestLogger,
    default_request_log_path,
    read_request_log,
    scrub_secrets,
)
from text2sql.llm.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    RateLimitError,
    TransientError,
)

SECRET = "AIzaSyD-1234567890abcdefghijklmnopqrstu"


def _req(text="How many schools?"):
    return CompletionRequest(
        "m", [Message("system", "You write SQL."), Message("user", text)], 0.0, 256
    )


@pytest.mark.parametrize(
    "raw",
    [
        f"403 for url https://x.googleapis.com/v1/models/m:generate?key={SECRET}&alt=json",
        f"401 for url https://api.example.com/v1?api_key={SECRET}",
        f"headers: Authorization: Bearer {SECRET}",
        f'request failed: {{"x-goog-api-key": "{SECRET}"}}',
        f"x-api-key={SECRET}",
        "invalid key sk-proj-abcdefghijklmnopqrstuvwx",
        "invalid key gsk_abcdefghijklmnopqrstuvwxyz",
        "invalid key csk-abcdefghijklmnopqrstuvwxyz",
        SECRET,
    ],
)
def test_scrub_removes_keys(raw):
    out = scrub_secrets(raw)
    assert REDACTED in out
    for secret in (SECRET, "sk-proj-abcdef", "gsk_abcdef", "csk-abcdef"):
        assert secret not in out


def test_scrub_leaves_ordinary_text_alone():
    text = "no such column: T1.key in SELECT key FROM t WHERE token_count = 3"
    assert scrub_secrets(text) == text
    assert scrub_secrets(None) is None


@pytest.fixture
def logs(tmp_path):
    return CallLogger(tmp_path / "calls.jsonl"), RequestLogger(tmp_path / "requests.jsonl")


def _client(script, logs, cache=None, **kw):
    calls, requests = logs
    return LLMClient(
        FakeProvider(script),
        cache=cache,
        logger=calls,
        request_logger=requests,
        sleep=lambda s: None,
        rng=random.Random(0),
        **kw,
    )


def test_success_logs_full_request_and_response(logs, tmp_path):
    client = _client(["SELECT COUNT(*) FROM schools"], logs, cache=DiskCache(tmp_path / "c"))
    client.complete(_req(), run_id="r1", question_id=7)
    client.complete(_req(), run_id="r1", question_id=7)  # cache hit: not repeated
    rows = read_request_log(logs[1].path)
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "ok" and row["attempt"] == 1
    assert row["messages"] == [
        {"role": "system", "content": "You write SQL."},
        {"role": "user", "content": "How many schools?"},
    ]
    assert row["response_text"] == "SELECT COUNT(*) FROM schools"
    assert row["finish_reason"] == "stop"
    assert (row["max_tokens"], row["temperature"]) == (256, 0.0)
    assert row["key"] == cache_key(_req(), provider="fake")
    assert row["key"] == read_log(logs[0].path)[0].key  # linked to the call log
    assert (row["run_id"], row["question_id"]) == ("r1", 7)


def test_every_attempt_is_logged(logs):
    script = [RateLimitError(f"429 ?key={SECRET}"), TransientError("503"), "SELECT 1"]
    _client(script, logs).complete(_req())
    rows = read_request_log(logs[1].path)
    assert [(r["attempt"], r["outcome"]) for r in rows] == [
        (1, "rate_limited"),
        (2, "transient"),
        (3, "ok"),
    ]
    assert SECRET not in rows[0]["error"] and REDACTED in rows[0]["error"]
    assert rows[0]["response_text"] is None


def test_empty_truncated_and_failed_calls_are_logged(logs):
    cut = CompletionResponse("SELECT a FROM", 5, 50, finish_reason="length")
    client = _client(["", cut], logs)
    client.complete(_req("q1"))
    client.complete(_req("q2"))
    failing = _client([ValueError(f"bad request, Authorization: Bearer {SECRET}")], logs)
    with pytest.raises(ValueError):
        failing.complete(_req("q3"))
    rows = read_request_log(logs[1].path)
    assert [r["outcome"] for r in rows] == ["empty", "truncated", "error"]
    assert rows[1]["response_text"] == "SELECT a FROM"
    assert SECRET not in rows[2]["error"]
    # the summary call log is scrubbed too
    assert all(SECRET not in (r.error or "") for r in read_log(logs[0].path))


def test_bad_outcome_is_refused(tmp_path):
    with pytest.raises(ValueError):
        RequestLogger(tmp_path / "r.jsonl").log(
            key="k",
            run_id=None,
            question_id=None,
            provider="p",
            attempt=1,
            request=_req(),
            outcome="maybe",
        )


def test_default_path_is_next_to_call_log(monkeypatch, tmp_path):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    assert default_request_log_path() == tmp_path.resolve() / "logs" / "llm_requests.jsonl"
    assert read_request_log() == []
