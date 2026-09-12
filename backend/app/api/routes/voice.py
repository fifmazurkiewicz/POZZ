import re
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.auth.deps import get_approved_user
from app.db import get_db
from app.models import User
from app.patients.service import assert_under_cap
from app.settings import get_settings

router = APIRouter()
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@router.get("/config")
def voice_config() -> dict[str, str | bool]:
    settings = get_settings()
    mode = settings.voice_mode.strip().lower()
    return {
        "voice_mode": mode,
        "live_available": settings.live_available,
        "tts_provider": settings.tts_provider.strip().lower(),
    }


@router.post("/transcribe")
async def transcribe_audio(
    audio: Annotated[UploadFile, File(...)],
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    """Transcribe one short doctor turn. Audio is processed in memory only."""
    assert_under_cap(db, user)
    settings = get_settings()
    if settings.stt_provider.strip().lower() != "groq" or not settings.groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "stt_unavailable", "message": "Transkrypcja głosu nie jest skonfigurowana."},
        )
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file is too large")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            data={"model": "whisper-large-v3", "language": "pl", "response_format": "json"},
            files={"file": (audio.filename or "turn.webm", content, audio.content_type or "audio/webm")},
        )
    if response.is_error:
        raise HTTPException(status_code=502, detail="Speech transcription failed")
    text = str(response.json().get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="No speech detected")
    return {"text": text}


@router.post("/speech")
async def patient_speech(
    body: dict[str, str],
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Return a short Polish patient reply as MP3; browser TTS remains the no-key fallback."""
    assert_under_cap(db, user)
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    settings = get_settings()
    requested_voice_id = str(body.get("voice_id", "")).strip()
    if requested_voice_id and not VOICE_ID_RE.fullmatch(requested_voice_id):
        raise HTTPException(status_code=422, detail="Invalid voice_id")
    voice_id = requested_voice_id or settings.tts_voice_id
    if settings.tts_provider.strip().lower() != "elevenlabs" or not settings.elevenlabs_api_key or not voice_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "tts_unavailable", "message": "Głos pacjenta nie jest skonfigurowany."},
        )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": settings.elevenlabs_api_key, "Accept": "audio/mpeg"},
            json={"text": text[:5000], "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        )
    if response.is_error:
        raise HTTPException(status_code=502, detail="Patient speech generation failed")
    return Response(content=response.content, media_type="audio/mpeg")
