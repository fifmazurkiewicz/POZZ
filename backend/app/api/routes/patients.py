from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.deps import get_approved_user
from app.db import get_db
from app.models import Conversation, Patient, User
from app.patients.service import (
    assert_under_cap,
    conversation_payload,
    open_simulation,
    pick_or_generate_patient,
)

router = APIRouter()


class NextPatientBody(BaseModel):
    keywords: str | None = Field(default=None, max_length=500)


class ManualInterviewBody(BaseModel):
    scenario: str = Field(min_length=20, max_length=12_000)
    title: str | None = Field(default=None, max_length=120)


@router.post("/next")
def next_patient(
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
    body: NextPatientBody = NextPatientBody(),
) -> dict:
    assert_under_cap(db, user)
    keywords = body.keywords
    patient = pick_or_generate_patient(db, user, keywords=keywords)
    conv = open_simulation(db, user, patient)
    payload = conversation_payload(conv, patient)
    payload.pop("messages", None)
    return payload


@router.post("/manual")
def create_manual_interview(
    body: ManualInterviewBody,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Create a private, user-authored practice case for the Interview workspace."""
    title = body.title.strip() if body.title and body.title.strip() else "Ręcznie utworzony przypadek"
    patient = Patient(
        created_by=user.id,
        scenario=body.scenario.strip(),
        summary=title,
        is_private=True,
    )
    db.add(patient)
    db.flush()
    conversation = Conversation(
        user_id=user.id,
        patient_id=patient.id,
        kind="manual_interview",
        title=title,
        mode="doctor_asks",
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation_payload(conversation, patient)
