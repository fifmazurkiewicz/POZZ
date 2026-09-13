from typing import Protocol

import httpx
from fastapi import HTTPException, UploadFile, status

from app.settings import Settings

MAX_AUDIO_BYTES = 25 * 1024 * 1024


class AsyncHttpClient(Protocol):
    async def post(self, *args, **kwargs) -> httpx.Response: ...


async def transcribe_upload(
    audio: UploadFile, settings: Settings, *, client: AsyncHttpClient
) -> str:
    """Transcribe a Polish audio upload without persisting its bytes."""
    if settings.stt_provider.strip().lower() != "groq" or not settings.groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "stt_unavailable",
                "message": "Transkrypcja głosu nie jest skonfigurowana.",
            },
        )
    content = await audio.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "empty_audio", "message": "Plik audio jest pusty."},
        )
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "audio_too_large",
                "message": "Plik audio może mieć maksymalnie 25 MB.",
            },
        )
    response = await client.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        data={
            "model": "whisper-large-v3",
            "language": "pl",
            "response_format": "json",
        },
        files={
            "file": (
                audio.filename or "interview.webm",
                content,
                audio.content_type or "audio/webm",
            )
        },
    )
    if response.is_error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "stt_provider_error",
                "message": "Nie udało się przetworzyć nagrania. Spróbuj ponownie.",
            },
        )
    text = str(response.json().get("text", "")).strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "no_speech", "message": "Nie wykryto mowy w nagraniu."},
        )
    return text
