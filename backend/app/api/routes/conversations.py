import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.auth.deps import get_approved_user
from app.db import get_db
from app.llm.provider import get_text_provider
from app.models import Conversation, Message, PatientUserState, User
from app.patients.service import (
    ROLE_FOR_MODE,
    assert_under_cap,
    conversation_payload,
    get_owned_conversation,
    get_owned_conversation_for_update,
)
from app.prompts.simulation import (
    create_case_description_prompt,
    create_evaluation_prompt,
    create_examination_prompt,
    create_reference_plan_prompt,
    create_simulation_prompt,
)

router = APIRouter()


@router.get("")
def list_conversations(
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    rows = db.scalars(
        select(Conversation)
        .options(joinedload(Conversation.patient), selectinload(Conversation.messages))
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    ).unique().all()
    return {
        "conversations": [
            conversation_payload(row, row.patient)
            | {"created_at": row.created_at.isoformat() if row.created_at else None, "ended_at": row.ended_at.isoformat() if row.ended_at else None}
            for row in rows
        ]
    }


class TurnBody(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    mode: str | None = None


class ExaminationBody(BaseModel):
    examination: str = Field(min_length=1, max_length=4000)


class FinishBody(BaseModel):
    treatment_plan: str = Field(min_length=1, max_length=12_000)


def _assert_open(conv: Conversation) -> None:
    if conv.ended_at is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "conversation_completed", "message": "Wywiad został już zakończony."},
        )


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
    conv = get_owned_conversation(db, user, conversation_id)
    _assert_open(conv)
    assert_under_cap(db, user)
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


@router.post("/{conversation_id}/examinations")
def post_examination(
    conversation_id: uuid.UUID,
    body: ExaminationBody,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    conv = get_owned_conversation_for_update(db, user, conversation_id)
    _assert_open(conv)
    assert_under_cap(db, user)
    examination = body.examination.strip()
    history = [{"role": message.role, "content": message.content} for message in conv.messages]
    result = get_text_provider().complete(
        create_examination_prompt(
            patient_scenario=conv.patient.scenario,
            chat_history=history,
            examination=examination,
        )
    ).strip()
    db.add(Message(conversation_id=conv.id, role="user", content=f"Badanie: {examination}"))
    db.add(Message(conversation_id=conv.id, role="assistant", content=f"Wynik badania: {result}"))
    db.commit()
    return conversation_payload(get_owned_conversation(db, user, conversation_id), conv.patient)


@router.post("/{conversation_id}/plan")
def generate_case_plan(
    conversation_id: uuid.UUID,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Draft a structured SOR-style case description from the doctor's own interview."""
    conv = get_owned_conversation_for_update(db, user, conversation_id)
    _assert_open(conv)
    assert_under_cap(db, user)
    history = [{"role": message.role, "content": message.content} for message in conv.messages]
    result = get_text_provider().complete(
        create_case_description_prompt(
            patient_scenario=conv.patient.scenario,
            chat_history=history,
        )
    ).strip()
    conv.interview_summary = result
    db.commit()
    return conversation_payload(get_owned_conversation(db, user, conversation_id), conv.patient)


@router.post("/{conversation_id}/finish")
def finish_conversation(
    conversation_id: uuid.UUID,
    body: FinishBody,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    conv = get_owned_conversation_for_update(db, user, conversation_id)
    if conv.ended_at is not None:
        return conversation_payload(conv, conv.patient)
    assert_under_cap(db, user)
    provider = get_text_provider()
    reference_plan = conv.patient.treatment_plan
    if not reference_plan:
        reference_plan = provider.complete(
            create_reference_plan_prompt(patient_scenario=conv.patient.scenario)
        ).strip()
    treatment_plan = body.treatment_plan.strip()
    history = [{"role": message.role, "content": message.content} for message in conv.messages]
    evaluation = provider.complete(
        create_evaluation_prompt(
            reference_plan=reference_plan,
            chat_history=history,
            treatment_plan=treatment_plan,
        )
    ).strip()
    conv.patient.treatment_plan = reference_plan
    conv.user_treatment_response = treatment_plan
    conv.diagnosis_evaluation = evaluation
    conv.ended_at = datetime.now(timezone.utc)
    if conv.kind == "simulation":
        state = db.get(PatientUserState, (user.id, conv.patient_id))
        if state is None:
            db.add(PatientUserState(user_id=user.id, patient_id=conv.patient_id, status="completed"))
        else:
            state.status = "completed"
            state.updated_at = datetime.now(timezone.utc)
    db.commit()
    return conversation_payload(get_owned_conversation(db, user, conversation_id), conv.patient)
