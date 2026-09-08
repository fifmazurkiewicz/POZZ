from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.llm.provider import MOCK_FIRST_TIME_PLAN, MOCK_TREATMENT_PLAN, get_text_provider
from app.models import Conversation, Patient, PatientUserState, User
from app.patients.card import parse_patient_card_from_scenario, public_card
from app.prompts.simulation import generate_patient_scenario_prompt
from app.settings import get_settings
from app.spend.service import monthly_spend_usd

MODE_DOCTOR_ASKS = "doctor_asks"
ROLE_FOR_MODE = {
    "doctor_asks": "patient",
    "patient_asks": "doctor",
    "meta_ask": "meta",
}


def assert_under_cap(db: Session, user: User) -> None:
    cap = float(user.spend_cap_usd)
    spent = monthly_spend_usd(db, user.id)
    if cap > 0 and spent >= cap:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "spend_cap_exceeded",
                "message": "Miesięczny limit wydatków został osiągnięty.",
            },
        )


def _summary_from_card(card: dict | None, scenario: str) -> str:
    if card and card.get("name"):
        age = card.get("age") or "?"
        return f"{card['name']}, {age}"
    return scenario.splitlines()[0][:80] if scenario else "Pacjent"


def generate_patient(
    db: Session,
    user: User,
    *,
    keywords: str | None = None,
    first_time: bool = False,
) -> Patient:
    provider = get_text_provider()
    messages = generate_patient_scenario_prompt(keywords=keywords, first_time_missing_basics=first_time)
    scenario = provider.complete(messages)
    card = parse_patient_card_from_scenario(scenario)
    is_first = first_time or (card.get("has_history_here") is False if card else False)
    gold = MOCK_FIRST_TIME_PLAN if is_first else MOCK_TREATMENT_PLAN
    if get_settings().openrouter_api_key:
        gold = None
    patient = Patient(
        created_by=user.id,
        scenario=scenario,
        summary=_summary_from_card(card, scenario),
        treatment_plan=gold,
        keywords=keywords,
        is_first_time=is_first,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def pick_or_generate_patient(db: Session, user: User, keywords: str | None = None) -> Patient:
    if keywords and keywords.strip():
        return generate_patient(db, user, keywords=keywords.strip())

    done_ids = select(PatientUserState.patient_id).where(
        PatientUserState.user_id == user.id,
        PatientUserState.status.in_(("completed", "skipped")),
    )
    open_ids = select(Conversation.patient_id).where(
        Conversation.user_id == user.id,
        Conversation.kind == "simulation",
        Conversation.ended_at.is_(None),
    )
    unused = db.scalars(
        select(Patient)
        .where(Patient.id.notin_(done_ids), Patient.id.notin_(open_ids))
        .order_by(Patient.created_at.asc())
    ).first()
    if unused is not None:
        return unused

    catalog_count = db.scalar(select(Patient.id)) is not None
    first_time = not catalog_count
    return generate_patient(db, user, first_time=first_time)


def open_simulation(db: Session, user: User, patient: Patient) -> Conversation:
    conv = Conversation(
        user_id=user.id,
        patient_id=patient.id,
        kind="simulation",
        title=patient.summary,
        mode=MODE_DOCTOR_ASKS,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def conversation_payload(conv: Conversation, patient: Patient) -> dict:
    card = public_card(parse_patient_card_from_scenario(patient.scenario))
    return {
        "conversation_id": str(conv.id),
        "patient_id": str(patient.id),
        "kind": conv.kind,
        "mode": conv.mode or MODE_DOCTOR_ASKS,
        "title": conv.title,
        "card": card,
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content} for m in conv.messages
        ],
    }


def get_owned_conversation(db: Session, user: User, conversation_id: uuid.UUID) -> Conversation:
    conv = db.scalars(
        select(Conversation)
        .options(joinedload(Conversation.patient), selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    ).first()
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conv
