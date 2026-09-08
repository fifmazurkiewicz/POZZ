from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.deps import get_approved_user
from app.db import get_db
from app.models import User
from app.patients.service import (
    assert_under_cap,
    conversation_payload,
    open_simulation,
    pick_or_generate_patient,
)

router = APIRouter()


class NextPatientBody(BaseModel):
    keywords: str | None = Field(default=None, max_length=500)


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
