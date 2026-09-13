from __future__ import annotations

import time
from typing import Protocol

import httpx

from app.settings import get_settings

MOCK_SCENARIO = """**Dane demograficzne:** Anna Nowak, 42 lata, księgowa.

**Powód wizyty:** kaszel suchy od tygodnia, gorączki nie ma.

**Historia obecnej choroby (HPI):** kaszel nasila się w nocy; duszności nie zgłasza spontanicznie.

**Przeszłość medyczna (PMH):** nadciśnienie tętnicze, ramipryl 5 mg; appendektomia 2010; alergia na penicylinę.

**Wywiad rodzinny i społeczny:** ojciec zawał w wieku 60 lat.

**Ukryte informacje:** pali około 10 papierosów dziennie — powie tylko, gdy lekarz zapyta wprost.

**Karta pacjenta:**
- Imię i nazwisko: Anna Nowak
- Wiek: 42 lata
- Historia w punkcie: Tak
- Choroby przewlekłe: nadciśnienie tętnicze
- Operacje: appendektomia 2010
- Alergie: penicylina
- Wywiad rodzinny: ojciec zawał w wieku 60 lat
"""

MOCK_FIRST_TIME_SCENARIO = """**Dane demograficzne:** Piotr Wiśniewski, 29 lat, kurier.

**Powód wizyty:** ból głowy od trzech dni.

**Historia obecnej choroby (HPI):** ból tępy, nasila się wieczorem.

**Przeszłość medyczna (PMH):**
- Choroby przewlekłe: nieznane – do zebrania podczas wywiadu
- Operacje: nieznane – do zebrania podczas wywiadu
- Alergie: nieznane – do zebrania podczas wywiadu
- Leki: nieznane – do zebrania podczas wywiadu

**Ukryte informacje:** pracuje na nocne zmiany i pije dużo energii — nie powie sam.

**Karta pacjenta:**
- Imię i nazwisko: Piotr Wiśniewski
- Wiek: 29 lat
- Historia w punkcie: Nie
"""

MOCK_TREATMENT_PLAN = (
    "**DIAGNOZA:** kaszel poinfekcyjny / podejrzenie palenia.\n"
    "**LEKI:** objawowo; rozważyć odstawienie ACE-I jeśli kaszel suchy utrzymuje się.\n"
    "**ZALECENIA:** rzucenie palenia, nawodnienie.\n"
    "**BADANIA:** RTG klatki jeśli objawy się nasilą."
)

MOCK_FIRST_TIME_PLAN = (
    "**DIAGNOZA:** ból głowy napięciowy vs. niewyspanie.\n"
    "**LEKI:** paracetamol doraźnie.\n"
    "**ZALECENIA:** higiena snu, ograniczenie energetyków.\n"
    "**BADANIA:** brak w pierwszej kolejności."
)


class TextCompletionProvider(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenRouterProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://pozz.fmazurkiewicz.dev",
                        "X-Title": "POZZ",
                    },
                    json={"model": self.model, "messages": messages},
                    timeout=60.0,
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))
                continue

            if response.status_code != 429 and response.status_code < 500:
                break
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))

        if response is None:
            raise httpx.RequestError("OpenRouter request did not return a response")
        if response.is_error:
            detail = response.text[:500]
            raise httpx.HTTPStatusError(
                f"OpenRouter {response.status_code} for model {self.model!r}: {detail}",
                request=response.request,
                response=response,
            )
        return _openrouter_completion_text(response)


def _openrouter_completion_text(response: httpx.Response) -> str:
    """Normalize OpenRouter content and classify malformed 200s as provider failures."""
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise httpx.RemoteProtocolError(
            "OpenRouter returned a malformed completion response",
            request=response.request,
        ) from exc

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {None, "text"}
        ).strip()
    else:
        text = ""
    if not text:
        raise httpx.RemoteProtocolError(
            "OpenRouter returned an empty completion",
            request=response.request,
        )
    return text


class GoogleGeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model.removeprefix("google/")

    def complete(self, messages: list[dict[str, str]]) -> str:
        system_parts = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]
        payload: dict[str, object] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(
            part.get("text", "")
            for part in data["candidates"][0]["content"]["parts"]
            if isinstance(part, dict)
        )


class FallbackTextProvider:
    def __init__(
        self, primary: TextCompletionProvider, fallback: TextCompletionProvider
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.primary.complete(messages)
        except httpx.HTTPError:
            return self.fallback.complete(messages)


class MockTextProvider:
    def complete(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "generatorem scenariuszy" in system:
            if "PIERWSZY RAZ" in system or "pierwszorazowy" in system:
                return MOCK_FIRST_TIME_SCENARIO
            return MOCK_SCENARIO
        if "aktorem odgrywającym pacjenta" in system:
            return "Kaszel mam od tygodnia, doktorze, głównie w nocy. Gorączki raczej nie czuję."
        if "lekarza rodzinnego" in system:
            return "Od kiedy ten kaszel się zaczął i czy coś go nasila?"
        if "klinicznym mentorem" in system:
            return "Dopytaj o palenie i leki na nadciśnienie — kaszel suchy bywa związany z ACE-I."
        return f"Rozumiem. {user[:80]}".strip()


def get_text_provider() -> TextCompletionProvider:
    settings = get_settings()
    if settings.openrouter_api_key:
        primary = OpenRouterProvider(settings.openrouter_api_key, settings.text_model)
        if settings.google_api_key:
            return FallbackTextProvider(
                primary,
                GoogleGeminiProvider(settings.google_api_key, settings.text_model),
            )
        return primary
    if settings.google_api_key:
        return GoogleGeminiProvider(settings.google_api_key, settings.text_model)
    return MockTextProvider()
