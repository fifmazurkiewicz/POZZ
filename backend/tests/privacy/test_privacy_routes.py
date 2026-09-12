import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Conversation, Message, Patient


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

