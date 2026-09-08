from __future__ import annotations

import re
from typing import Any


def parse_patient_card_from_scenario(scenario: str) -> dict[str, Any] | None:
    """Parse the Karta pacjenta block. Same fields as the Streamlit prototype."""
    header_match = re.search(r"karta\s+pacjenta\s*:?(?:\s*\(wymagana\))?\s*", scenario, re.IGNORECASE)
    if header_match:
        start_idx = header_match.end()
    else:
        low = scenario.lower()
        idx = low.find("karta pacjenta")
        if idx == -1:
            return None
        line_end = scenario.find("\n", idx)
        start_idx = (line_end + 1) if line_end != -1 else idx

    after = scenario[start_idx : start_idx + 2000]
    for marker in ["\n🩺", "\nSymulator", "\nSłowa kluczowe", "\nSlowa kluczowe"]:
        pos = after.find(marker)
        if pos != -1:
            after = after[:pos]
            break

    def normalize_line(line: str) -> str:
        line = re.sub(r"^[\s>*\-•\u2022]*", "", line)
        line = line.strip()
        return line.replace("**", "").replace("*", "")

    field_map = {
        "imię i nazwisko": "name",
        "imie i nazwisko": "name",
        "wiek": "age",
        "historia w punkcie": "has_history_here",
        "choroby przewlekłe": "chronic_diseases",
        "choroby przewlekle": "chronic_diseases",
        "operacje": "operations",
        "alergie": "allergies",
        "wywiad rodzinny": "family_history",
    }

    result: dict[str, Any] = {
        "name": "",
        "age": "",
        "has_history_here": None,
        "chronic_diseases": "",
        "operations": "",
        "allergies": "",
        "family_history": "",
    }

    for raw_line in after.splitlines():
        line = normalize_line(raw_line)
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip().lower()
        value = value.strip()
        key = field_map.get(label)
        if not key:
            continue
        if key == "has_history_here":
            low = value.lower()
            if "tak" in low:
                result["has_history_here"] = True
            elif "nie" in low:
                result["has_history_here"] = False
        else:
            result[key] = value

    if (result["name"] or result["age"]) and result["has_history_here"] is not None:
        return result
    return None


def public_card(card: dict[str, Any] | None) -> dict[str, Any]:
    """Fields the doctor may see. First-time patients: name, age, historia = Nie."""
    empty = {
        "name": "",
        "age": "",
        "has_history_here": None,
        "chronic_diseases": "",
        "operations": "",
        "allergies": "",
        "family_history": "",
    }
    if not card:
        return empty
    if card.get("has_history_here") is False:
        return {
            **empty,
            "name": card.get("name") or "",
            "age": card.get("age") or "",
            "has_history_here": False,
        }
    return {**empty, **card}
