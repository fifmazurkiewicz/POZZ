from __future__ import annotations

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
        if response.is_error:
            detail = response.text[:500]
            raise httpx.HTTPStatusError(
                f"OpenRouter {response.status_code} for model {self.model!r}: {detail}",
                request=response.request,
                response=response,
            )
        return response.json()["choices"][0]["message"]["content"]


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
        return OpenRouterProvider(settings.openrouter_api_key, settings.text_model)
    return MockTextProvider()
