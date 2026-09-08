from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    supabase_url: str = ""
    dev_auth_enabled: bool = False
    spend_cap_tz: str = "Europe/Warsaw"
    allowed_admin_emails: str = "fifmazurkiewicz@gmail.com"
    text_provider: str = "openrouter"
    text_model: str = "google/gemini-2.5-flash-lite"
    openrouter_api_key: str = ""
    voice_mode: str = "speech_to_speech"
    google_api_key: str = ""
    stt_provider: str = "groq"
    groq_api_key: str = ""
    tts_provider: str = "elevenlabs"
    tts_voice_id: str = ""
    elevenlabs_api_key: str = ""
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def live_available(self) -> bool:
        return self.voice_mode.strip().lower() == "speech_to_speech"


@lru_cache
def get_settings() -> Settings:
    return Settings()
