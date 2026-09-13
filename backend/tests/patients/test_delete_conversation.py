import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Conversation, InterviewTranscript, Patient

AUTH = {"Authorization": "Bearer dev-token"}


def test_delete_owned_conversation_and_private_patient(
    sqlite_client: TestClient, db_session: Session
):
    created = sqlite_client.post(
        "/api/patients/manual",
        headers=AUTH,
        json={
            "title": "Do usunięcia",
            "scenario": "Pacjent zgłasza ból brzucha utrzymujący się od tygodnia.",
        },
    ).json()

    response = sqlite_client.delete(
        f"/api/conversations/{created['conversation_id']}", headers=AUTH
    )

    assert response.status_code == 204
    assert db_session.get(Conversation, uuid.UUID(created["conversation_id"])) is None
    assert db_session.get(Patient, uuid.UUID(created["patient_id"])) is None


def test_delete_recorded_conversation_removes_transcript(
    sqlite_client: TestClient, db_session: Session, monkeypatch
):
    async def fake_transcribe(audio, settings, *, client):
        return "Lekarz: Co boli? Pacjent: Brzuch."

    class Provider:
        def complete(self, messages):
            return '{"turns":[{"speaker":"unknown","text":"Rozmowa"}]}'

    monkeypatch.setattr("app.api.routes.interviews.transcribe_upload", fake_transcribe)
    monkeypatch.setattr(
        "app.api.routes.interviews.get_text_provider", lambda: Provider()
    )
    created = sqlite_client.post(
        "/api/interviews/recordings",
        headers=AUTH,
        files={"audio": ("wizyta.wav", b"audio", "audio/wav")},
    ).json()

    response = sqlite_client.delete(
        f"/api/conversations/{created['conversation_id']}", headers=AUTH
    )

    assert response.status_code == 204
    assert db_session.query(InterviewTranscript).count() == 0


def test_delete_enforces_ownership(sqlite_client: TestClient):
    response = sqlite_client.delete(f"/api/conversations/{uuid.uuid4()}", headers=AUTH)

    assert response.status_code == 404
