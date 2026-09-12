import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.models import Conversation, Message, PatientUserState, UsageLedger, User

AUTH = {"Authorization": "Bearer dev-token"}


def _conversation(client: TestClient) -> str:
    response = client.post("/api/patients/next", headers=AUTH, json={})
    assert response.status_code == 200, response.text
    return response.json()["conversation_id"]


def test_examination_persists_prefixed_messages_without_hidden_payload(
    sqlite_client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
):
    conversation_id = _conversation(sqlite_client)

    class Provider:
        def complete(self, messages):
            assert "TAJNY SCENARIUSZ" in messages[0]["content"]
            return "Ciśnienie 138/84 mmHg, tętno 76/min."

    monkeypatch.setattr("app.api.routes.conversations.get_text_provider", lambda: Provider())
    response = sqlite_client.post(
        f"/api/conversations/{conversation_id}/examinations", headers=AUTH,
        json={"examination": "pomiar ciśnienia"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["messages"][-2]["content"] == "Badanie: pomiar ciśnienia"
    assert body["messages"][-1]["content"].startswith("Wynik badania: Ciśnienie")
    assert "UKRYTY" not in str(body).upper()
    db_session.expire_all()
    assert db_session.query(Message).filter_by(conversation_id=uuid.UUID(conversation_id)).count() == 2


def test_finish_persists_evaluation_marks_completed_and_is_idempotent(
    sqlite_client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
):
    conversation_id = _conversation(sqlite_client)
    calls = []

    class Provider:
        def complete(self, messages):
            calls.append(messages)
            return "Ocena: plan bezpieczny, ale niepełny."

    monkeypatch.setattr("app.api.routes.conversations.get_text_provider", lambda: Provider())
    first = sqlite_client.post(
        f"/api/conversations/{conversation_id}/finish", headers=AUTH,
        json={"treatment_plan": "Leczenie objawowe i kontrola."},
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["ended_at"]
    assert body["user_treatment_response"] == "Leczenie objawowe i kontrola."
    assert body["diagnosis_evaluation"].startswith("Ocena:")
    assert "treatment_plan" not in body
    assert len(calls) == 1
    retry = sqlite_client.post(
        f"/api/conversations/{conversation_id}/finish", headers=AUTH,
        json={"treatment_plan": "Inny plan"},
    )
    assert retry.status_code == 200
    assert retry.json()["user_treatment_response"] == body["user_treatment_response"]
    assert len(calls) == 1
    db_session.expire_all()
    conv = db_session.get(Conversation, uuid.UUID(conversation_id))
    assert db_session.get(PatientUserState, (conv.user_id, conv.patient_id)).status == "completed"


def test_failed_evaluation_leaves_conversation_open(
    sqlite_client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
):
    conversation_id = _conversation(sqlite_client)

    class Provider:
        def complete(self, messages):
            raise RuntimeError("provider failed")

    monkeypatch.setattr("app.api.routes.conversations.get_text_provider", lambda: Provider())
    with pytest.raises(RuntimeError, match="provider failed"):
        sqlite_client.post(
            f"/api/conversations/{conversation_id}/finish", headers=AUTH,
            json={"treatment_plan": "Plan"},
        )
    db_session.expire_all()
    conv = db_session.get(Conversation, uuid.UUID(conversation_id))
    assert conv.ended_at is None
    assert conv.user_treatment_response is None
    assert conv.diagnosis_evaluation is None


def test_completed_conversation_rejects_turns_and_examinations(sqlite_client: TestClient):
    conversation_id = _conversation(sqlite_client)
    assert sqlite_client.post(
        f"/api/conversations/{conversation_id}/finish", headers=AUTH,
        json={"treatment_plan": "Plan"},
    ).status_code == 200
    assert sqlite_client.post(
        f"/api/conversations/{conversation_id}/turns", headers=AUTH,
        json={"content": "Pytanie"},
    ).status_code == 409
    assert sqlite_client.post(
        f"/api/conversations/{conversation_id}/examinations", headers=AUTH,
        json={"examination": "EKG"},
    ).status_code == 409


def test_finish_enforces_ownership_and_spend_cap(sqlite_client: TestClient, db_session: Session):
    conversation_id = _conversation(sqlite_client)
    other = User(id=uuid.uuid4(), email="other@example.com", is_approved=True, spend_cap_usd=10)
    sqlite_client.app.dependency_overrides[get_current_user] = lambda: other
    try:
        assert sqlite_client.post(
            f"/api/conversations/{conversation_id}/finish", json={"treatment_plan": "Plan"}
        ).status_code == 404
    finally:
        del sqlite_client.app.dependency_overrides[get_current_user]
    conv = db_session.get(Conversation, uuid.UUID(conversation_id))
    db_session.add(UsageLedger(user_id=conv.user_id, action_type="test", cost_usd=10))
    db_session.commit()
    capped = sqlite_client.post(
        f"/api/conversations/{conversation_id}/finish", headers=AUTH,
        json={"treatment_plan": "Plan"},
    )
    assert capped.status_code == 403
    assert capped.json()["detail"]["code"] == "spend_cap_exceeded"
