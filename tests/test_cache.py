import json
import os
import threading

import pytest

from text2sql import config
from text2sql.llm.cache import DiskCache, cache_key, default_cache_dir
from text2sql.llm.types import CompletionRequest, CompletionResponse, Message


def _req(text="hi", model="m", temperature=0.0, max_tokens=None):
    return CompletionRequest(
        model, [Message("system", "s"), Message("user", text)], temperature, max_tokens
    )


def test_key_depends_on_model_messages_temperature_only():
    base = cache_key(_req())
    assert len(base) == 64
    assert cache_key(_req()) == base
    assert cache_key(_req(text="hello")) != base
    assert cache_key(_req(model="m2")) != base
    assert cache_key(_req(temperature=0.7)) != base
    assert cache_key(_req(max_tokens=50)) == base  # does not change the answer's identity


def test_key_is_stable_across_releases():
    # Changing the key format would silently invalidate every cached response.
    req = CompletionRequest("m", [Message("user", "café")], 0.0)
    assert cache_key(req) == cache_key(CompletionRequest("m", [Message("user", "café")], 0))
    assert cache_key(req) == "cd94c8162d63a3097266dc389178c38e38dba27098f6e3a996a3130383542d63"


def test_round_trip_and_layout(tmp_path):
    cache = DiskCache(tmp_path / "cache dir")
    req = _req("ünïcode")
    key = cache_key(req)
    assert cache.get(key) is None
    resp = CompletionResponse("SELECT 1", 10, 2, raw={"id": "x"})
    path = cache.put(key, req, resp)
    assert path == tmp_path / "cache dir" / key[:2] / f"{key}.json"
    assert cache.get(key) == resp
    assert cache.get(key).raw == {"id": "x"}
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["request"]["messages"][1]["content"] == "ünïcode"
    assert len(cache) == 1
    assert not list(path.parent.glob("*.tmp"))


def test_corrupt_or_foreign_entries_are_misses(tmp_path):
    cache = DiskCache(tmp_path)
    req = _req()
    key = cache_key(req)
    path = cache.put(key, req, CompletionResponse("x"))
    path.write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None
    path.write_text(json.dumps({"key": "other", "response": {"text": "x"}}), encoding="utf-8")
    assert cache.get(key) is None
    path.write_text(json.dumps({"key": key, "response": {}}), encoding="utf-8")
    assert cache.get(key) is None


@pytest.mark.parametrize("bad", ["", "../x", "A" * 64, "g" * 64, "a" * 63])
def test_bad_keys_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        DiskCache(tmp_path).path_for(bad)


def test_concurrent_writers_leave_one_valid_entry(tmp_path):
    cache = DiskCache(tmp_path)
    req = _req()
    key = cache_key(req)
    errors = []

    def write(i):
        try:
            for _ in range(20):
                cache.put(key, req, CompletionResponse(f"SELECT {i}", 1, 1))
                cache.get(key)  # readers racing writers must not crash
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    got = cache.get(key)
    assert got is not None and got.text.startswith("SELECT ")
    assert not list(tmp_path.rglob("*.tmp"))


def test_replace_gives_up_quietly_if_entry_exists(tmp_path, monkeypatch):
    cache = DiskCache(tmp_path)
    req = _req()
    key = cache_key(req)
    cache.put(key, req, CompletionResponse("first"))

    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(os, "replace", locked)
    cache.put(key, req, CompletionResponse("second"))
    monkeypatch.undo()
    assert cache.get(key).text == "first"
    assert not list(tmp_path.rglob("*.tmp"))


def test_default_dir_follows_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    assert default_cache_dir() == tmp_path.resolve() / "cache" / "llm"


def test_temp_file_name_is_short(tmp_path, monkeypatch):
    # The final path is root/xx/<64 hex>.json; the temp file next to it must not
    # be longer, or deep folders on Windows hit the 260-character path limit.
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    cache = DiskCache(tmp_path)
    req = _req()
    cache.put(cache_key(req), req, CompletionResponse("x"))
    src, dst = seen[0]
    assert len(src) <= len(dst)
    assert os.path.dirname(src) == os.path.dirname(dst)


def _entry(key, **response):
    return json.dumps({"key": key, "response": response}).encode("utf-8")


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"[]",
        b'"x"',
        b"null",
        b"42",
        b"\xff\xfe{}",
        "DEEP",
        "KEY_RESP_LIST",
        "KEY_TEXT_INT",
        "KEY_TEXT_NULL",
        "KEY_TEXT_EMPTY",
        "KEY_TOKENS_STR",
        "KEY_TOKENS_NEG",
        "KEY_TOKENS_BOOL",
        "KEY_RAW_LIST",
    ],
)
def test_any_malformed_entry_is_a_miss(tmp_path, content):
    cache = DiskCache(tmp_path)
    req = _req()
    key = cache_key(req)
    variants = {
        "KEY_RESP_LIST": json.dumps({"key": key, "response": [1]}).encode(),
        "KEY_TEXT_INT": _entry(key, text=123),
        "KEY_TEXT_NULL": _entry(key, text=None),
        "KEY_TEXT_EMPTY": _entry(key, text="  "),
        "KEY_TOKENS_STR": _entry(key, text="x", prompt_tokens="5"),
        "KEY_TOKENS_NEG": _entry(key, text="x", completion_tokens=-1),
        "KEY_TOKENS_BOOL": _entry(key, text="x", prompt_tokens=True),
        "KEY_RAW_LIST": _entry(key, text="x", raw=[1]),
        "DEEP": b"[" * 100000,
    }
    path = cache.path_for(key)
    path.parent.mkdir(parents=True)
    path.write_bytes(variants.get(content, content) if isinstance(content, str) else content)
    assert cache.get(key) is None


def test_directory_in_place_of_entry_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    key = cache_key(_req())
    cache.path_for(key).mkdir(parents=True)
    assert cache.get(key) is None


def test_negative_zero_temperature_shares_the_key():
    assert cache_key(_req(temperature=-0.0)) == cache_key(_req(temperature=0.0))
