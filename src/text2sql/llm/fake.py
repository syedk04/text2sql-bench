"""A scripted provider for tests and offline dry runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from text2sql.llm.types import CompletionRequest, CompletionResponse

Step = str | CompletionResponse | BaseException
Responder = Callable[[CompletionRequest], Step]


def rough_token_count(text: str) -> int:
    """About four characters per token; good enough for a fake."""
    return max(1, (len(text) + 3) // 4) if text else 0


@dataclass
class FakeProvider:
    """Replays a script of responses, or calls a function for each request.

    Script steps may be a string (returned as the completion text), a full
    :class:`CompletionResponse`, or an exception instance (raised). When the
    script runs out the last step repeats, unless ``strict`` is set.
    """

    script: Iterable[Step] | Responder = ("SELECT 1",)
    name: str = "fake"
    strict: bool = False
    requests: list[CompletionRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._responder: Responder | None = None
        self._steps: list[Step] = []
        if callable(self.script):
            self._responder = self.script
        else:
            self._steps = list(self.script)
            if not self._steps:
                raise ValueError("script must have at least one step")
        self._index = 0

    @property
    def calls(self) -> int:
        return len(self.requests)

    def _next_step(self, request: CompletionRequest) -> Step:
        if self._responder is not None:
            return self._responder(request)
        if self._index >= len(self._steps):
            if self.strict:
                raise AssertionError(f"fake provider script exhausted after {self._index} calls")
            return self._steps[-1]
        step = self._steps[self._index]
        self._index += 1
        return step

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        step = self._next_step(request)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, CompletionResponse):
            return step
        prompt = "\n".join(m.content for m in request.messages)
        return CompletionResponse(
            text=step,
            prompt_tokens=rough_token_count(prompt),
            completion_tokens=rough_token_count(step),
            raw={"provider": self.name},
        )
