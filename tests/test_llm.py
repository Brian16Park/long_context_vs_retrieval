"""Checks for call_claude's retries (lcvr/llm.py), with a fake API: no real calls, no real waiting."""
import types

import anthropic
import httpx2
import pytest

from lcvr import llm

URL = "https://api.anthropic.com/v1/messages"


def api_error(status, retry_after=None, code=None):
    """An error like the one the SDK raises for an HTTP error response."""
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx2.Response(status, headers=headers, request=httpx2.Request("POST", URL))
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    if code:
        body["error"]["details"] = {"error_code": code}
    return anthropic.APIStatusError("error", response=response, body=body)


def reply():
    usage = types.SimpleNamespace(input_tokens=10, cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0, output_tokens=3)
    return types.SimpleNamespace(usage=usage, stop_reason="end_turn",
                                 content=[types.SimpleNamespace(type="text", text="Paris")])


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    """A fake client that raises the queued errors, then answers. Waits are recorded, not slept."""
    monkeypatch.setattr(llm, "LOG_FILE", tmp_path / "calls.jsonl")
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    errors = []

    def create(**request):
        if errors:
            raise errors.pop(0)
        return reply()

    monkeypatch.setattr(llm, "client", types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    return errors, waits


def test_waits_as_long_as_a_rate_limit_asks(fake_api):
    errors, waits = fake_api
    errors += [api_error(429, retry_after=30), api_error(529)]
    answer, record = llm.call_claude("Question: ?", {"run": "test"})
    assert answer == "Paris" and record["attempt"] == 3
    assert waits == [30, 4]            # retry-after 30 s beats our 2 s; then our usual 4 s


def test_spend_cap_is_not_retried(fake_api):
    errors, waits = fake_api
    errors.append(api_error(429, code="enforced_spend_limit_reached"))
    with pytest.raises(anthropic.APIStatusError):
        llm.call_claude("Question: ?", {"run": "test"})
    assert waits == []
    assert [r["ok"] for r in llm.read_log()] == [False]       # the failed attempt is still logged


def test_gives_up_after_max_attempts(fake_api):
    errors, waits = fake_api
    errors += [api_error(529)] * 3
    with pytest.raises(anthropic.APIStatusError):
        llm.call_claude("Question: ?", {"run": "test"}, max_attempts=3)
    assert waits == [2, 4]
    assert len(llm.read_log()) == 3


def test_reading_retry_after_and_error_codes():
    assert llm.retry_after(api_error(429, retry_after=12)) == 12
    assert llm.retry_after(api_error(500)) == 0
    assert llm.retry_after(ValueError("no response at all")) == 0
    assert llm.error_code(api_error(429, code="ramp_limit_exceeded")) == "ramp_limit_exceeded"
    assert llm.error_code(api_error(500)) is None
