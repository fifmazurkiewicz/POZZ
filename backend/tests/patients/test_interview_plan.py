import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.models import Conversation, UsageLedger, User

AUTH = {"Authorization": "Bearer dev-token"}


def _manual_conversation(client: TestClient, total_msgs: int = 0) -> str:
    created = client.post(
        "/api/patients/manual",
        headers=AUTH,
        json={
            "title": "Ból brzucha",
            "scenario": "Pacjent zgłasza ból brzucha od tygodnia, nasilający się po jedzeniu.",
        },
    )
    assert created.status_code == 200, created.text
    conversation_id = created.json()["conversation_id"]
    for i in range(total_msgs):
        turn = client.post(
            f"/api/conversations/{conversation_id}/turns",
            headers=AUTH,
            json={"content": f"Pytanie {i + 1}?"},
        )
        assert turn.status_code == 200, turn.text
    return conversation_id


def test_plan_generates_and_persists_interview_summary(
    sqlite_client: TestClient, db_session: Session, monkeypatch
):
    conversation_id = _manual_conversation(sqlite_client, total_msgs=2)
    captured = {}

    class Provider:
        def complete(self, messages):
            captured["messages"] = messages
            return "**WYWIAD**\n- Skargi: ból brzucha.\n\n**ZALECANE BADANIA**\n- USG jamy brzusznej."

    monkeypatch.setattr("app.api.routes.conversations.get_text_provider", lambda: Provider())

    response = sqlite_client.post(f"/api/conversations/{conversation_id}/plan", headers=AUTH, json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interview_summary"].startswith("**WYWIAD**")
    assert "scenario" not in body
    assert "treatment_plan" not in body

    prompt = captured["messages"][1]["content"]
    assert "ból brzucha" in prompt  # scenario content present
    assert "Pytanie 1" in prompt  # transcript present

    db_session.expire_all()
    conv = db_session.get(Conversation, uuid.UUID(conversation_id))
    assert conv is not None
    assert conv.interview_summary == body["interview_summary"]


def test_plan_rejects_completed_conversation(sqlite_client: TestClient):
    conversation_id = _manual_conversation(sqlite_client)
    assert (
        sqlite_client.post(
            f"/api/conversations/{conversation_id}/finish",
            headers=AUTH,
            json={"treatment_plan": "Plan"},
        ).status_code
        == 200
    )
    response = sqlite_client.post(f"/api/conversations/{conversation_id}/plan", headers=AUTH, json={})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "conversation_completed"


def test_plan_enforces_ownership_and_spend_cap(
    sqlite_client: TestClient, db_session: Session
):
    conversation_id = _manual_conversation(sqlite_client)
    other = User(id=uuid.uuid4(), email="other@example.com", is_approved=True, spend_cap_usd=10)
    sqlite_client.app.dependency_overrides[get_current_user] = lambda: other
    try:
        assert sqlite_client.post(
            f"/api/conversations/{conversation_id}/plan", headers=AUTH, json={}
        ).status_code == 404
    finally:
        del sqlite_client.app.dependency_overrides[get_current_user]

    conv = db_session.get(Conversation, uuid.UUID(conversation_id))
    db_session.add(UsageLedger(user_id=conv.user_id, action_type="test", cost_usd=10))
    db_session.commit()
    capped = sqlite_client.post(f"/api/conversations/{conversation_id}/plan", headers=AUTH, json={})
    assert capped.status_code == 403
    assert capped.json()["detail"]["code"] == "spend_cap_exceeded"