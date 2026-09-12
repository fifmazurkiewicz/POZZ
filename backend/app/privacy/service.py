from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models import (
    Conversation,
    InterviewSuggestion,
    InterviewTranscript,
    Job,
    Message,
    Patient,
    PatientUserState,
    UsageLedger,
    User,
)


def build_user_export(db: Session, user: User) -> dict:
    conversations = db.scalars(
        select(Conversation)
        .options(joinedload(Conversation.patient), selectinload(Conversation.messages))
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.asc())
    ).unique().all()
    ledgers = db.scalars(
        select(UsageLedger)
        .where(UsageLedger.user_id == user.id)
        .order_by(UsageLedger.created_at.asc())
    ).all()
    conversation_ids = [conversation.id for conversation in conversations]
    transcripts = _group_by_conversation(
        db.scalars(
            select(InterviewTranscript)
            .where(InterviewTranscript.conversation_id.in_(conversation_ids))
            .order_by(InterviewTranscript.conversation_id, InterviewTranscript.chunk_number)
        ).all()
        if conversation_ids else []
    )
    suggestions = _group_by_conversation(
        db.scalars(
            select(InterviewSuggestion)
            .where(InterviewSuggestion.conversation_id.in_(conversation_ids))
            .order_by(InterviewSuggestion.conversation_id, InterviewSuggestion.chunk_number)
        ).all()
        if conversation_ids else []
    )
    patient_states = db.scalars(
        select(PatientUserState).where(PatientUserState.user_id == user.id)
    ).all()
    jobs = db.scalars(select(Job).where(Job.user_id == user.id).order_by(Job.created_at.asc())).all()
    return {
        "format_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "account": {
            "id": str(user.id),
            "email": user.email,
            "display_name": user.display_name,
            "created_at": _iso(user.created_at),
            "onboarding_completed_at": _iso(user.onboarding_completed_at),
        },
        "conversations": [
            {
                "id": str(conv.id),
                "kind": conv.kind,
                "title": conv.title,
                "mode": conv.mode,
                "created_at": _iso(conv.created_at),
                "ended_at": _iso(conv.ended_at),
                "user_treatment_response": conv.user_treatment_response,
                "diagnosis_evaluation": conv.diagnosis_evaluation,
                "interview_summary": conv.interview_summary,
                "extracted_info": conv.extracted_info,
                "private_case": (
                    {"scenario": conv.patient.scenario, "keywords": conv.patient.keywords}
                    if conv.patient.is_private and conv.patient.created_by == user.id
                    else None
                ),
                "messages": [
                    {
                        "id": message.id,
                        "role": message.role,
                        "content": message.content,
                        "created_at": _iso(message.created_at),
                    }
                    for message in conv.messages
                ],
                "transcripts": [
                    {"chunk_number": item.chunk_number, "transcript": item.transcript_json, "created_at": _iso(item.created_at)}
                    for item in transcripts.get(conv.id, [])
                ],
                "suggestions": [
                    {"chunk_number": item.chunk_number, "minute_number": item.minute_number, "content": item.suggestions, "created_at": _iso(item.created_at)}
                    for item in suggestions.get(conv.id, [])
                ],
            }
            for conv in conversations
        ],
        "usage": [
            {
                "action_type": item.action_type,
                "cost_usd": float(item.cost_usd),
                "provider": item.provider,
                "langfuse_trace_id": item.langfuse_trace_id,
                "created_at": _iso(item.created_at),
            }
            for item in ledgers
        ],
        "patient_state": [
            {"patient_id": str(item.patient_id), "status": item.status, "updated_at": _iso(item.updated_at)}
            for item in patient_states
        ],
        "jobs": [
            {"id": str(item.id), "kind": item.kind, "payload": item.payload, "status": item.status, "error": item.error, "created_at": _iso(item.created_at), "updated_at": _iso(item.updated_at)}
            for item in jobs
        ],
    }


def delete_user_content(db: Session, user: User) -> dict[str, int]:
    conversation_ids = list(
        db.scalars(select(Conversation.id).where(Conversation.user_id == user.id)).all()
    )
    private_patient_ids = list(
        db.scalars(
            select(Patient.id).where(Patient.created_by == user.id, Patient.is_private.is_(True))
        ).all()
    )
    deleted = {
        "messages": _delete_for_conversations(db, Message, conversation_ids),
        "transcripts": _delete_for_conversations(db, InterviewTranscript, conversation_ids),
        "suggestions": _delete_for_conversations(db, InterviewSuggestion, conversation_ids),
    }
    deleted["conversations"] = _execute_delete(
        db, delete(Conversation).where(Conversation.id.in_(conversation_ids))
    ) if conversation_ids else 0
    deleted["patient_state"] = _execute_delete(
        db, delete(PatientUserState).where(PatientUserState.user_id == user.id)
    )
    deleted["jobs"] = _execute_delete(db, delete(Job).where(Job.user_id == user.id))
    deleted["usage"] = _execute_delete(db, delete(UsageLedger).where(UsageLedger.user_id == user.id))
    deleted["private_cases"] = _execute_delete(
        db, delete(Patient).where(Patient.id.in_(private_patient_ids))
    ) if private_patient_ids else 0
    db.commit()
    return deleted


def _delete_for_conversations(db: Session, model: type, conversation_ids: list) -> int:
    if not conversation_ids:
        return 0
    return _execute_delete(db, delete(model).where(model.conversation_id.in_(conversation_ids)))


def _execute_delete(db: Session, statement) -> int:
    result = db.execute(statement)
    return int(result.rowcount or 0)


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _group_by_conversation(items: list) -> dict:
    grouped: dict = {}
    for item in items:
        grouped.setdefault(item.conversation_id, []).append(item)
    return grouped
