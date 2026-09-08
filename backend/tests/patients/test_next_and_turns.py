import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.models import Conversation, Message, Patient, User


AUTH = {"Authorization": "Bearer dev-token"}


def test_next_patient_generates_mock_and_hides_gold(sqlite_client: TestClient, db_session: Session):
    response = sqlite_client.post("/api/patients/next", headers=AUTH, json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["conversation_id"]
    assert body["patient_id"]
    assert body["card"]["name"]
    assert "scenario" not in body
    assert "treatment_plan" not in body

    db_session.expire_all()
    patient = db_session.get(Patient, uuid.UUID(body["patient_id"]))
    assert patient is not None
    assert patient.treatment_plan
    assert patient.scenario
    conv = db_session.get(Conversation, uuid.UUID(body["conversation_id"]))
    assert conv is not None
    assert conv.kind == "simulation"


def test_get_conversation_omits_hidden_scenario(sqlite_client: TestClient):
    created = sqlite_client.post("/api/patients/next", headers=AUTH, json={})
    assert created.status_code == 200
    conversation_id = created.json()["conversation_id"]
    response = sqlite_client.get(f"/api/conversations/{conversation_id}", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    dumped = str(body)
    assert "scenario" not in body
    assert "treatment_plan" not in body
    assert "UKRYTY" not in dumped
    assert body["card"]["name"]
    assert body["mode"] == "doctor_asks"
    assert body["messages"] == []


def test_text_turn_persists_user_and_assistant(sqlite_client: TestClient, db_session: Session):
    created = sqlite_client.post("/api/patients/next", headers=AUTH, json={})
    conversation_id = created.json()["conversation_id"]
    response = sqlite_client.post(
        f"/api/conversations/{conversation_id}/turns",
        headers=AUTH,
        json={"content": "Co pana sprowadza?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["assistant"]["content"]
    assert "scenario" not in body
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"

    db_session.expire_all()
    rows = db_session.query(Message).filter(Message.conversation_id == uuid.UUID(conversation_id)).all()
    assert len(rows) == 2


def test_unapproved_cannot_start_patient(sqlite_client: TestClient):
    pending = User(
        id=uuid.uuid4(),
        email="wait@example.com",
        is_admin=False,
        is_approved=False,
        spend_cap_usd=10,
    )
    sqlite_client.app.dependency_overrides[get_current_user] = lambda: pending
    try:
        response = sqlite_client.post("/api/patients/next", json={})
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "account_pending_approval"
    finally:
        del sqlite_client.app.dependency_overrides[get_current_user]
