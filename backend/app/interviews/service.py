import json
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Conversation, InterviewTranscript

SPEAKERS = {"doctor", "patient", "unknown"}


def parse_inferred_turns(result: str, raw_text: str) -> list[dict[str, str]]:
    """Parse model JSON defensively; raw STT remains the lossless fallback."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip(), flags=re.I)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return [{"speaker": "unknown", "text": raw_text}]
    if isinstance(data, dict):
        data = data.get("turns")
    if not isinstance(data, list):
        return [{"speaker": "unknown", "text": raw_text}]
    turns = []
    for item in data:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "unknown")).lower()
        text = str(item.get("text", "")).strip()
        if text:
            turns.append(
                {"speaker": speaker if speaker in SPEAKERS else "unknown", "text": text}
            )
    return turns or [{"speaker": "unknown", "text": raw_text}]


def recorded_transcript_payload(
    db: Session, conversation: Conversation
) -> dict[str, object] | None:
    if conversation.kind != "recorded_interview":
        return None
    row = db.scalars(
        select(InterviewTranscript)
        .where(InterviewTranscript.conversation_id == conversation.id)
        .order_by(InterviewTranscript.chunk_number)
    ).first()
    return row.transcript_json if row else None
