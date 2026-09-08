from app.patients.card import parse_patient_card_from_scenario


RETURNING_SCENARIO = """
## Powód wizyty
Kaszel od tygodnia.

**Karta pacjenta:**
- Imię i nazwisko: Anna Nowak
- Wiek: 42 lata
- Historia w punkcie: Tak
- Choroby przewlekłe: nadciśnienie tętnicze
- Operacje: appendektomia 2010
- Alergie: penicylina
- Wywiad rodzinny: ojciec zawał w wieku 60 lat
"""

FIRST_TIME_SCENARIO = """
**Karta pacjenta (wymagana)**
* Imię i nazwisko: Piotr Wiśniewski
* Wiek: 29 lat
* Historia w punkcie: Nie
"""


def test_parse_returning_patient_card():
    card = parse_patient_card_from_scenario(RETURNING_SCENARIO)
    assert card is not None
    assert card["name"] == "Anna Nowak"
    assert card["age"] == "42 lata"
    assert card["has_history_here"] is True
    assert card["chronic_diseases"] == "nadciśnienie tętnicze"
    assert card["allergies"] == "penicylina"


def test_parse_first_time_card_name_and_age_only():
    card = parse_patient_card_from_scenario(FIRST_TIME_SCENARIO)
    assert card is not None
    assert card["name"] == "Piotr Wiśniewski"
    assert card["age"] == "29 lat"
    assert card["has_history_here"] is False


def test_parse_returns_none_without_card_section():
    assert parse_patient_card_from_scenario("Brak karty w tekście") is None
