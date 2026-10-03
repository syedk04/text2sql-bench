import json
import os
import threading

import pytest

from text2sql import config
from text2sql.llm.cache import DiskCache, cache_key, default_cache_dir
from text2sql.llm.types import CompletionRequest, CompletionResponse, Message

KEY_V2 = "47bcaba7be9f498863cff237fdab216cf1d02ce9a15595c3f5f8c5917a8a15f3"


def _key(req: CompletionRequest, provider: str = "fake") -> str:
    return cache_key(req, provider=provider)


def _req(text="hi", model="m", temperature=0.0, max_tokens=None):
    return CompletionRequest(
        model, [Message("system", "s"), Message("user", text)], temperature, max_tokens
    )


def test_key_covers_everything_that_decides_the_answer():
    base = _key(_req())
    assert len(base) == 64
    assert _key(_req()) == base
    assert _key(_req(text="hello")) != base
    assert _key(_req(model="m2")) != base
    assert _key(_req(temperature=0.7)) != base
    assert _key(_req(max_tokens=50)) != base
    assert _key(_req(), provider="other") != base
    with pytest.raises(ValueError):
        cache_key(_req(), provider="")


def test_key_is_stable_across_releases():
    # Changing the key format would silently invalidate every cached response,
    # so this pins version 2 of the format.
    req = CompletionRequest("m", [Message("user", "café")], 0.0)
    assert _key(req) == _key(CompletionRequest("m", [Message("user", "café")], 0))
    assert _key(req) == KEY_V2


def test_round_trip_and_layout(tmp_path):
    cache = DiskCache(tmp_path / "cache dir")
    req = _req("ünïcode")
    key = _key(req)
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
    key = _key(req)
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
    key = _key(req)
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
    key = _key(req)
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
    cache.put(_key(req), req, CompletionResponse("x"))
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
    key = _key(req)
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
    key = _key(_req())
    cache.path_for(key).mkdir(parents=True)
    assert cache.get(key) is None


def test_negative_zero_temperature_shares_the_key():
    assert _key(_req(temperature=-0.0)) == _key(_req(temperature=0.0))


def test_finish_reason_round_trips_and_bad_values_are_misses(tmp_path):
    cache = DiskCache(tmp_path)
    req = _req()
    key = _key(req)
    cache.put(key, req, CompletionResponse("SELECT 1", 1, 1, finish_reason="stop"))
    assert cache.get(key).finish_reason == "stop"
    path = cache.path_for(key)
    path.write_bytes(_entry(key, text="x", finish_reason=5))
    assert cache.get(key) is None
