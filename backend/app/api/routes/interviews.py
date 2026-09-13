from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.auth.deps import get_approved_user
from app.db import get_db
from app.interviews.service import parse_inferred_turns
from app.llm.provider import get_text_provider
from app.models import Conversation, InterviewTranscript, Patient, User
from app.patients.service import assert_under_cap, conversation_payload
from app.prompts.simulation import create_speaker_transcript_prompt
from app.settings import get_settings
from app.voice.transcription import transcribe_upload

router = APIRouter()


@router.post("/recordings")
async def create_recorded_interview(
    audio: Annotated[UploadFile, File(...)],
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
    title: Annotated[str | None, Form(max_length=120)] = None,
) -> dict:
    """Transcribe one interview recording in memory and persist text only."""
    assert_under_cap(db, user)
    settings = get_settings()
    async with httpx.AsyncClient(timeout=120.0) as client:
        raw_text = await transcribe_upload(audio, settings, client=client)

    inferred = get_text_provider().complete(
        create_speaker_transcript_prompt(raw_text=raw_text)
    )
    turns = parse_inferred_turns(inferred, raw_text)
    fallback_title = Path(audio.filename or "Nagranie wywiadu").stem[:120]
    clean_title = title.strip() if title and title.strip() else fallback_title
    patient = Patient(
        created_by=user.id,
        scenario="Nagranie rzeczywistego wywiadu. Analizuj wyłącznie pełną transkrypcję.",
        summary=clean_title,
        is_private=True,
    )
    db.add(patient)
    db.flush()
    conversation = Conversation(
        user_id=user.id,
        patient_id=patient.id,
        kind="recorded_interview",
        title=clean_title,
        mode="doctor_asks",
    )
    db.add(conversation)
    db.flush()
    transcript = InterviewTranscript(
        conversation_id=conversation.id,
        chunk_number=1,
        transcript_json={"raw_text": raw_text, "turns": turns},
    )
    db.add(transcript)
    db.commit()
    db.refresh(conversation)
    return conversation_payload(conversation, patient) | {
        "recorded_transcript": transcript.transcript_json
    }
