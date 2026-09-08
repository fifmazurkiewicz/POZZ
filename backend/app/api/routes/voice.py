from fastapi import APIRouter

from app.settings import get_settings

router = APIRouter()


@router.get("/config")
def voice_config() -> dict[str, str | bool]:
    settings = get_settings()
    mode = settings.voice_mode.strip().lower()
    return {
        "voice_mode": mode,
        "live_available": settings.live_available,
        "tts_provider": settings.tts_provider.strip().lower(),
    }
