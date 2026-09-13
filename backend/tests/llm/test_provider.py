import httpx

from app.llm.provider import OpenRouterProvider


def _response(status_code: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(
        status_code, request=request, json=payload or {"error": "failed"}
    )


def test_openrouter_retries_transient_status(monkeypatch):
    responses = [
        _response(429),
        _response(503),
        _response(200, {"choices": [{"message": {"content": "Gotowe"}}]}),
    ]
    monkeypatch.setattr(
        "app.llm.provider.httpx.post", lambda *args, **kwargs: responses.pop(0)
    )
    monkeypatch.setattr("app.llm.provider.time.sleep", lambda _: None)

    result = OpenRouterProvider("secret", "google/gemini-2.5-flash-lite").complete(
        [{"role": "user", "content": "Test"}]
    )

    assert result == "Gotowe"
    assert responses == []


def test_openrouter_does_not_retry_credit_error(monkeypatch):
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _response(402)

    monkeypatch.setattr("app.llm.provider.httpx.post", post)

    try:
        OpenRouterProvider("secret", "model").complete(
            [{"role": "user", "content": "Test"}]
        )
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 402
    else:
        raise AssertionError("Expected provider failure")
    assert calls == 1
