import pytest

from text2sql.llm.base import Provider
from text2sql.llm.fake import FakeProvider, rough_token_count
from text2sql.llm.types import (
    BudgetExceeded,
    CompletionRequest,
    CompletionResponse,
    LLMError,
    Message,
    RateLimitError,
    TransientError,
)


def _req(text: str = "hi") -> CompletionRequest:
    return CompletionRequest("m", [Message("system", "be brief"), Message("user", text)])


def test_message_validation():
    assert Message("user", "x").to_dict() == {"role": "user", "content": "x"}
    with pytest.raises(ValueError):
        Message("robot", "x")
    with pytest.raises(TypeError):
        Message("user", None)  # type: ignore[arg-type]


def test_request_is_immutable_and_normalised():
    req = CompletionRequest("m", [Message("user", "abc")], temperature=0)
    assert isinstance(req.messages, tuple)
    assert req.temperature == 0.0 and isinstance(req.temperature, float)
    assert req.prompt_chars() == 3
    with pytest.raises(AttributeError):
        req.model = "other"  # type: ignore[misc]
    with pytest.raises(ValueError):
        CompletionRequest("", [Message("user", "x")])
    with pytest.raises(ValueError):
        CompletionRequest("m", [])
    assert req == CompletionRequest("m", (Message("user", "abc"),), 0.0)


def test_response_totals_and_raw_not_compared():
    a = CompletionResponse("x", 3, 4, raw={"id": 1})
    assert a.total_tokens == 7
    assert a == CompletionResponse("x", 3, 4, raw={"id": 2})


def test_errors_carry_retry_after():
    assert RateLimitError(retry_after=2.5).retry_after == 2.5
    assert TransientError().retry_after is None
    assert issubclass(BudgetExceeded, LLMError)


def test_fake_follows_script_then_repeats_last():
    fake = FakeProvider(["SELECT 1", CompletionResponse("SELECT 2", 9, 9), "SELECT 3"])
    assert isinstance(fake, Provider)
    assert fake.complete(_req()).text == "SELECT 1"
    assert fake.complete(_req()) == CompletionResponse("SELECT 2", 9, 9)
    assert fake.complete(_req()).text == "SELECT 3"
    assert fake.complete(_req()).text == "SELECT 3"
    assert fake.calls == 4


def test_fake_raises_scripted_errors():
    fake = FakeProvider([RateLimitError(retry_after=1), "ok"])
    with pytest.raises(RateLimitError):
        fake.complete(_req())
    assert fake.complete(_req()).text == "ok"


def test_fake_strict_and_callable():
    strict = FakeProvider(["a"], strict=True)
    strict.complete(_req())
    with pytest.raises(AssertionError, match="exhausted"):
        strict.complete(_req())
    echo = FakeProvider(lambda r: r.messages[-1].content.upper())
    assert echo.complete(_req("select")).text == "SELECT"
    with pytest.raises(ValueError):
        FakeProvider([])


def test_fake_token_counts():
    resp = FakeProvider(["abcdefgh"]).complete(_req("1234"))
    assert resp.completion_tokens == 2
    assert resp.prompt_tokens == rough_token_count("be brief\n1234")
    assert rough_token_count("") == 0
