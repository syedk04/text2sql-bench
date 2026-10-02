"""The provider interface every model backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from text2sql.llm.types import CompletionRequest, CompletionResponse


@runtime_checkable
class Provider(Protocol):
    """One model API. Implementations translate a :class:`CompletionRequest` into a
    single HTTP call and raise :class:`RateLimitError` / :class:`TransientError`
    for retryable failures; retrying, caching and logging happen in the client."""

    name: str

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
