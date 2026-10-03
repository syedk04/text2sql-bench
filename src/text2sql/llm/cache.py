"""Disk cache for model responses.

The key is a hash of exactly the inputs that decide the answer (provider, model,
messages, temperature, max_tokens), so re-running an unchanged prompt after a code change costs no
quota. Entries are written to a temp file and moved into place with
``os.replace``, so a crash or a concurrent writer never leaves half a file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from text2sql import config
from text2sql.llm.types import CompletionRequest, CompletionResponse

KEY_VERSION = 2


def cache_key(request: CompletionRequest, *, provider: str) -> str:
    """Fingerprint of everything that decides the answer. The same model name
    can mean different weights on different providers, and max_tokens can cut
    an answer short, so both are part of the key (version 2)."""
    if not provider:
        raise ValueError("provider name is required for the cache key")
    payload = {
        "v": KEY_VERSION,
        "provider": provider,
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [m.to_dict() for m in request.messages],
        # + 0.0 turns -0.0 into 0.0 so the two spellings share an entry
        "temperature": request.temperature + 0.0,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def default_cache_dir() -> Path:
    return config.cache_dir() / "llm"


class DiskCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_cache_dir()

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError(f"not a cache key: {key!r}")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CompletionResponse | None:
        """Return the cached response, or None. Anything unreadable or not in the
        expected shape is a miss (the next successful call rewrites it); this
        method never raises for a bad entry."""
        path = self.path_for(key)
        try:
            data = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError):
            return None
        if not isinstance(data, dict) or data.get("key") != key:
            return None
        resp = data.get("response")
        if not isinstance(resp, dict):
            return None
        text = resp.get("text")
        prompt_tokens = resp.get("prompt_tokens", 0)
        completion_tokens = resp.get("completion_tokens", 0)
        raw = resp.get("raw")
        finish_reason = resp.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            return None
        if not isinstance(text, str) or not text.strip():
            return None
        for n in (prompt_tokens, completion_tokens):
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                return None
        if raw is not None and not isinstance(raw, dict):
            return None
        return CompletionResponse(text, prompt_tokens, completion_tokens, raw, finish_reason)

    def put(self, key: str, request: CompletionRequest, response: CompletionResponse) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "key": key,
            "key_version": KEY_VERSION,
            "created_at": time.time(),
            "request": {
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "messages": [m.to_dict() for m in request.messages],
            },
            "response": asdict(response),
        }
        # Short temp name: the 64-character key plus pid and uuid pushed deep
        # cache folders past the 260-character Windows path limit.
        tmp = path.with_name(f"tmp-{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        self._replace(tmp, path)
        return path

    @staticmethod
    def _replace(tmp: Path, path: Path, attempts: int = 5) -> None:
        # On Windows os.replace fails with PermissionError while another process
        # has the destination open. The content for a key is interchangeable, so
        # if someone else's copy is already in place we can simply drop ours.
        for attempt in range(attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if path.exists() and attempt == attempts - 1:
                    tmp.unlink(missing_ok=True)
                    return
                time.sleep(0.05 * (attempt + 1))
        tmp.unlink(missing_ok=True)
        raise PermissionError(f"could not write cache entry {path}")

    def __len__(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(1 for _ in self.root.glob("??/*.json"))
