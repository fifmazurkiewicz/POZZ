import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.deps import get_approved_user
from app.db import get_db
from app.llm.provider import get_text_provider
from app.models import Message, User
from app.patients.service import (
    ROLE_FOR_MODE,
    assert_under_cap,
    conversation_payload,
    get_owned_conversation,
)
from app.prompts.simulation import create_simulation_prompt

router = APIRouter()


class TurnBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    mode: str | None = None


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: uuid.UUID,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    conv = get_owned_conversation(db, user, conversation_id)
    return conversation_payload(conv, conv.patient)


@router.post("/{conversation_id}/turns")
def post_turn(
    conversation_id: uuid.UUID,
    body: TurnBody,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    assert_under_cap(db, user)
    conv = get_owned_conversation(db, user, conversation_id)
    if body.mode:
        if body.mode not in ROLE_FOR_MODE:
            from fastapi import HTTPException, status

            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown mode")
        conv.mode = body.mode
    mode = conv.mode or "doctor_asks"
    role_to_play = ROLE_FOR_MODE[mode]
    history = [{"role": m.role, "content": m.content} for m in conv.messages]
    user_msg = Message(conversation_id=conv.id, role="user", content=body.content.strip())
    db.add(user_msg)
    db.flush()
    prompt = create_simulation_prompt(
        role_to_play=role_to_play,
        patient_scenario=conv.patient.scenario,
        chat_history=history,
        question=body.content.strip(),
    )
    reply = get_text_provider().complete(prompt)
    assistant = Message(conversation_id=conv.id, role="assistant", content=reply)
    db.add(assistant)
    db.commit()
    conv = get_owned_conversation(db, user, conversation_id)
    payload = conversation_payload(conv, conv.patient)
    payload["assistant"] = {"id": assistant.id, "role": "assistant", "content": reply}
    return payload
