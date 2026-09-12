import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Conversation, InterviewSuggestion, InterviewTranscript, Job, Message, Patient


AUTH = {"Authorization": "Bearer dev-token"}


def test_export_contains_user_owned_content(sqlite_client: TestClient):
    created = sqlite_client.post("/api/patients/next", headers=AUTH, json={}).json()
    sqlite_client.post(
        f"/api/conversations/{created['conversation_id']}/turns",
        headers=AUTH,
        json={"content": "Od kiedy występują objawy?"},
    )

    response = sqlite_client.get("/api/privacy/export", headers=AUTH)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["format_version"] == 1
    assert body["account"]["email"] == "dev@localhost"
    assert len(body["conversations"]) == 1
    assert len(body["conversations"][0]["messages"]) == 2
    assert "scenario" not in body["conversations"][0]


def test_export_includes_transcripts_suggestions_and_jobs(
    sqlite_client: TestClient, db_session: Session
):
    created = sqlite_client.post("/api/patients/next", headers=AUTH, json={}).json()
    conversation_id = uuid.UUID(created["conversation_id"])
    user_id = uuid.UUID("00000000-0000-4000-8000-000000000001")
    db_session.add_all([
        InterviewTranscript(conversation_id=conversation_id, chunk_number=1, transcript_json={"transcript": [{"role": "doctor", "text": "Dzień dobry"}]}),
        InterviewSuggestion(conversation_id=conversation_id, chunk_number=1, minute_number=1, suggestions="Dopytaj o czas trwania objawów."),
        Job(user_id=user_id, kind="transcribe_interview", payload={"conversation_id": str(conversation_id)}),
    ])
    db_session.commit()

    body = sqlite_client.get("/api/privacy/export", headers=AUTH).json()

    conversation = body["conversations"][0]
    assert conversation["transcripts"][0]["chunk_number"] == 1
    assert conversation["suggestions"][0]["minute_number"] == 1
    assert body["jobs"][0]["kind"] == "transcribe_interview"


def test_delete_content_requires_typed_confirmation(sqlite_client: TestClient):
    response = sqlite_client.request(
        "DELETE",
        "/api/privacy/content",
        headers=AUTH,
        json={"confirmation": "wrong"},
    )
    assert response.status_code == 422


def test_delete_content_removes_conversations_and_private_cases(
    sqlite_client: TestClient, db_session: Session
):
    created = sqlite_client.post(
        "/api/patients/manual",
        headers=AUTH,
        json={"title": "Private", "scenario": "Fictional private medical practice case."},
    ).json()

    response = sqlite_client.request(
        "DELETE",
        "/api/privacy/content",
        headers=AUTH,
        json={"confirmation": "USUŃ MOJE DANE"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["deleted"]["conversations"] == 1
    db_session.expire_all()
    assert db_session.get(Conversation, uuid.UUID(created["conversation_id"])) is None
    assert db_session.get(Patient, uuid.UUID(created["patient_id"])) is None
    assert db_session.query(Message).count() == 0
