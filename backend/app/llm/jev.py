from __future__ import annotations

import time
from typing import Any, Literal

import httpx

from app.settings import get_settings


class VoiceTurnDecisionError(Exception):
    """JEV did not return a reliable utterance-boundary decision."""


def training_signals(evaluation: str) -> dict[str, Any] | None:
    key = get_settings().openrouter_api_key
    if not key:
        return None
    payload = {"model": "typesafe/jev-1.13", "state": {"evaluation": evaluation}, "questions": {"training_feedback_completeness": {"type": "score", "instructions": "How complete is this non-clinical training feedback?", "criteria": ["Missing essential feedback", "Partly complete feedback", "Clear, complete training feedback"]}, "draft_has_clinician_review_boundary": {"type": "noul", "instructions": "Does this text clearly state that clinician review is required?"}}}
    for attempt in range(3):
        try:
            response = httpx.post("https://openrouter.ai/api/alpha/decisions", headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=5)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1)); continue
                return None
            response.raise_for_status()
            return response.json().get("answers")
        except (httpx.HTTPError, ValueError, TypeError):
            return None
    return None


def voice_turn_decision(transcript: str) -> Literal["complete", "continue"]:
    """Return only a strict, non-clinical turn-boundary decision."""
    key = get_settings().openrouter_api_key
    if not key:
        raise VoiceTurnDecisionError()
    payload = {
        "model": "typesafe/jev-1.13",
        "state": {"transcript": transcript},
        "questions": {
            "voice_turn_complete": {
                "type": "noul",
                "instructions": "Is this Polish speaker utterance complete? Answer yes only if the speaker has clearly finished; answer no if it is incomplete.",
            }
        },
    }
    try:
        response = httpx.post("https://openrouter.ai/api/alpha/decisions", headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=5)
        response.raise_for_status()
        value = response.json().get("answers", {}).get("voice_turn_complete", {}).get("value")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        raise VoiceTurnDecisionError() from exc
    if value is True:
        return "complete"
    if value is False:
        return "continue"
    raise VoiceTurnDecisionError()
