import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Conversation, InterviewTranscript, Message

AUTH = {"Authorization": "Bearer dev-token"}


def test_recording_persists_full_transcript_without_audio(
    sqlite_client: TestClient, db_session: Session, monkeypatch
):
    async def fake_transcribe(audio, settings, *, client):
        assert audio.filename == "wizyta.webm"
        return "Dzień dobry. Od kiedy boli brzuch? Od wczoraj wieczorem."

    class Provider:
        def complete(self, messages):
            assert "Porządkujesz polską transkrypcję" in messages[0]["content"]
            return (
                '{"turns":['
                '{"speaker":"doctor","text":"Dzień dobry. Od kiedy boli brzuch?"},'
                '{"speaker":"patient","text":"Od wczoraj wieczorem."}'
                "]}"
            )

    monkeypatch.setattr("app.api.routes.interviews.transcribe_upload", fake_transcribe)
    monkeypatch.setattr(
        "app.api.routes.interviews.get_text_provider", lambda: Provider()
    )

    response = sqlite_client.post(
        "/api/interviews/recordings",
        headers=AUTH,
        data={"title": "Ból brzucha"},
        files={"audio": ("wizyta.webm", b"audio", "audio/webm")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "recorded_interview"
    assert body["title"] == "Ból brzucha"
    assert body["recorded_transcript"]["raw_text"].startswith("Dzień dobry")
    assert body["recorded_transcript"]["turns"][1]["speaker"] == "patient"
    conversation = db_session.get(Conversation, uuid.UUID(body["conversation_id"]))
    assert conversation is not None
    assert conversation.audio_ref is None
    transcript = db_session.query(InterviewTranscript).one()
    assert transcript.transcript_json == body["recorded_transcript"]


def test_plan_uses_recorded_transcript_and_returns_it(
    sqlite_client: TestClient, db_session: Session, monkeypatch
):
    raw_text = "Lekarz pyta o gorączkę. Pacjent zaprzecza."

    async def fake_transcribe(audio, settings, *, client):
        return raw_text

    prompts = []

    class Provider:
        def complete(self, messages):
            prompts.append(messages)
            if "Porządkujesz polską transkrypcję" in messages[0]["content"]:
                return '{"turns":[{"speaker":"unknown","text":"Cała rozmowa"}]}'
            return "## WYWIAD\n- Bez gorączki\n\n## PROPONOWANE LEKI\n- Brak podstaw"

    provider = Provider()
    monkeypatch.setattr("app.api.routes.interviews.transcribe_upload", fake_transcribe)
    monkeypatch.setattr("app.api.routes.interviews.get_text_provider", lambda: provider)
    monkeypatch.setattr(
        "app.api.routes.conversations.get_text_provider", lambda: provider
    )
    created = sqlite_client.post(
        "/api/interviews/recordings",
        headers=AUTH,
        files={"audio": ("wizyta.wav", b"audio", "audio/wav")},
    ).json()
    # Any later annotations/results must supplement, not replace, the source
    # transcript when the description is generated.
    db_session.add(
        Message(
            conversation_id=uuid.UUID(created["conversation_id"]),
            role="assistant",
            content="Dodatkowa notatka kliniczna.",
        )
    )
    db_session.commit()

    response = sqlite_client.post(
        f"/api/conversations/{created['conversation_id']}/plan",
        headers=AUTH,
        json={},
    )

    assert response.status_code == 200, response.text
    assert response.json()["recorded_transcript"]["raw_text"] == raw_text
    assert response.json()["interview_summary"].startswith("## WYWIAD")
    assert raw_text in prompts[-1][1]["content"]
