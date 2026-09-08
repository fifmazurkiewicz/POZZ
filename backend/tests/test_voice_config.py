from fastapi.testclient import TestClient

from app.main import app
from app.settings import get_settings

client = TestClient(app)


def test_voice_config_defaults_live_available():
    get_settings.cache_clear()
    response = client.get("/api/voice/config")
    assert response.status_code == 200
    body = response.json()
    assert body["voice_mode"] == "speech_to_speech"
    assert body["live_available"] is True
    assert body["tts_provider"] == "elevenlabs"
    get_settings.cache_clear()
