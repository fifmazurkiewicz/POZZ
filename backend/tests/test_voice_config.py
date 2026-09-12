from fastapi.testclient import TestClient

from app.main import app
from app.settings import get_settings

client = TestClient(app)


def test_voice_config_defaults_live_available():
    get_settings.cache_clear()


def test_speech_rejects_invalid_voice_id(sqlite_client: TestClient):
    response = sqlite_client.post(
        "/api/voice/speech", headers={"Authorization": "Bearer dev-token"},
        json={"text": "Dzień dobry", "voice_id": "https://evil.example/voice"},
    )
    assert response.status_code == 422


def test_speech_uses_valid_voice_override(sqlite_client: TestClient, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "elevenlabs_api_key", "test-key")
    requested_urls = []

    class FakeResponse:
        is_error = False
        content = b"mp3"

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr("app.api.routes.voice.httpx.AsyncClient", FakeClient)
    response = sqlite_client.post(
        "/api/voice/speech", headers={"Authorization": "Bearer dev-token"},
        json={"text": "Dzień dobry", "voice_id": "voice_ABC-123"},
    )
    assert response.status_code == 200
    assert requested_urls == ["https://api.elevenlabs.io/v1/text-to-speech/voice_ABC-123"]
    response = client.get("/api/voice/config")
    assert response.status_code == 200
    body = response.json()
    assert body["voice_mode"] == "speech_to_speech"
    assert body["live_available"] is True
    assert body["tts_provider"] == "elevenlabs"
    get_settings.cache_clear()
