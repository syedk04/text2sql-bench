"""Provider-neutral request/response types and errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; expected one of {ROLES}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int | None = None

    def __init__(
        self,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if not messages:
            raise ValueError("at least one message is required")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "messages", tuple(messages))
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "max_tokens", max_tokens)

    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


@dataclass(frozen=True)
class CompletionResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """Base class for provider failures."""


class RateLimitError(LLMError):
    """HTTP 429 or equivalent. ``retry_after`` is in seconds when the provider said."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """A failure worth retrying: 5xx, connection reset, timeout."""

    def __init__(self, message: str = "transient failure", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceeded(LLMError):
    """The daily token budget is spent; stop rather than burn more quota."""
