"""Full request/response log for every call that reaches a provider.

``llm_calls.jsonl`` (see :mod:`text2sql.llm.calllog`) has one summary line per
``LLMClient.complete`` call. This sibling file, ``llm_requests.jsonl``, has one
line per provider attempt with the full prompt and the full answer or error,
so failures can be audited later. The two are linked by ``key`` (the cache
fingerprint) plus ``run_id`` and ``question_id``. Cache hits are not repeated
here; the cache file already holds that request and answer.

Error text is passed through :func:`scrub_secrets` first, because HTTP client
errors often echo the request URL or headers, API key included.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from text2sql import config
from text2sql.llm.types import CompletionRequest, CompletionResponse

OUTCOMES = ("ok", "empty", "truncated", "rate_limited", "transient", "error")
REDACTED = "[REDACTED]"

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # query parameters such as ?key=... or &api_key=...
    (
        re.compile(r"(?i)([?&](?:key|api[_-]?key|access[_-]?token|token|secret)=)[^&\s\"'<>]+"),
        r"\1" + REDACTED,
    ),
    # Authorization: Bearer <token>
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1" + REDACTED),
    # header or JSON style: x-api-key: ..., "api_key": "...", authorization=...
    (
        re.compile(
            r"(?i)((?:x-goog-api-key|x-api-key|api[_-]?key|authorization)[\"']?\s*[:=]\s*[\"']?)"
            r"(?!bearer\s)[^\s\"',;}]+"
        ),
        r"\1" + REDACTED,
    ),
    # well-known key shapes, wherever they appear
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), REDACTED),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"\bcsk-[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), REDACTED),
)


def scrub_secrets(text: str | None) -> str | None:
    """Replace anything that looks like an API key or bearer token."""
    if text is None:
        return None
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def default_request_log_path() -> Path:
    return config.log_dir() / "llm_requests.jsonl"


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RequestLogger:
    def __init__(self, path: Path | None = None, *, now: Callable[[], str] = _utc_iso) -> None:
        self.path = path or default_request_log_path()
        self._now = now
        self._lock = threading.Lock()

    def log(
        self,
        *,
        key: str,
        run_id: str | None,
        question_id: int | None,
        provider: str,
        attempt: int,
        request: CompletionRequest,
        outcome: str,
        response: CompletionResponse | None = None,
        error: str | None = None,
        latency_s: float = 0.0,
    ) -> dict[str, Any]:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        record: dict[str, Any] = {
            "ts": self._now(),
            "key": key,
            "run_id": run_id,
            "question_id": question_id,
            "provider": provider,
            "model": request.model,
            "attempt": attempt,
            "outcome": outcome,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": [m.to_dict() for m in request.messages],
            "response_text": response.text if response is not None else None,
            "finish_reason": response.finish_reason if response is not None else None,
            "prompt_tokens": response.prompt_tokens if response is not None else 0,
            "completion_tokens": response.completion_tokens if response is not None else 0,
            "error": scrub_secrets(error),
            "latency_s": latency_s,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line)
        return record


def read_request_log(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or default_request_log_path()
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
