import httpx

from app.llm.provider import (
    FallbackTextProvider,
    GoogleGeminiProvider,
    OpenRouterProvider,
)


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


def test_google_gemini_converts_chat_messages(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _response(
            200,
            {"candidates": [{"content": {"parts": [{"text": "Plan gotowy"}]}}]},
        )

    monkeypatch.setattr("app.llm.provider.httpx.post", post)
    result = GoogleGeminiProvider(
        "google-key", "google/gemini-2.5-flash-lite"
    ).complete(
        [
            {"role": "system", "content": "Odpowiadaj po polsku."},
            {"role": "user", "content": "Przygotuj plan."},
        ]
    )

    assert result == "Plan gotowy"
    assert captured["url"].endswith("/models/gemini-2.5-flash-lite:generateContent")
    assert captured["kwargs"]["headers"]["x-goog-api-key"] == "google-key"
    assert (
        captured["kwargs"]["json"]["systemInstruction"]["parts"][0]["text"]
        == "Odpowiadaj po polsku."
    )


def test_fallback_provider_uses_google_after_openrouter_failure():
    class Primary:
        def complete(self, messages):
            request = httpx.Request("POST", "https://openrouter.ai")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)

    class Fallback:
        def complete(self, messages):
            return "Plan z Gemini"

    assert FallbackTextProvider(Primary(), Fallback()).complete([]) == "Plan z Gemini"
