from typing import List, Dict, Optional


def generate_patient_scenario_prompt(
    keywords: Optional[str] = None,
    first_time_missing_basics: bool = False,
) -> List[Dict[str, str]]:
    """Build a system message that forces Polish-language output for the patient scenario.

    If first_time_missing_basics is True, the prompt will instruct the model to
    generate a first-time patient at the point of care with missing baseline data
    (chronic diseases, operations, allergies, family history), explicitly marked
    as "nieznane/nieudokumentowane – do zebrania podczas wywiadu".
    """

    base_prompt_content = (
        "Jesteś zaawansowanym generatorem scenariuszy medycznych dla szkoleń w POZ. "
        "Twoim zadaniem jest stworzenie kompletnego, realistycznego i wewnętrznie spójnego profilu pacjenta. "
        "Profil musi zawierać subtelne pułapki diagnostyczne. Działasz jak kreator postaci do RPG dla lekarza.\n\n"
        "Wygeneruj odpowiedź WYŁĄCZNIE po polsku. Użyj formatowania Markdown.\n\n"
        "**WAŻNE - RÓŻNORODNOŚĆ DANYCH:**\n"
        "ZAWSZE używaj RÓŻNYCH, losowych polskich imion i nazwisk dla każdego pacjenta. "
        "Używaj szerokiej gamy popularnych i mniej popularnych polskich imion (np. Anna, Piotr, Maria, Tomasz, Katarzyna, "
        "Aleksander, Joanna, Michał, Agata, Łukasz, Ewa, Marcin, Magdalena, Paweł, Natalia, Krzysztof, itd.). "
        "Używaj różnych polskich nazwisk (np. Kowalski, Nowak, Wiśniewski, Wójcik, Kowalczyk, Mazur, Krawczyk, Kaczmarek, itd.). "
        "Wiek pacjenta MUSI być różnorodny - losuj z różnych grup wiekowych (18-95 lat), unikaj powtarzania tych samych wartości. "
        "NIGDY nie używaj tych samych kombinacji imię-nazwisko-wiek w kolejnych scenariuszach.\n\n"
        "Profil musi zawierać następujące sekcje:\n"
        "- **Dane demograficzne:** Wiek, płeć, zawód, sytuacja życiowa.\n"
        "- **Powód wizyty:** Jeden główny objaw zgłaszany przez pacjenta.\n"
        "- **Historia obecnej choroby (HPI):** Początek, charakter, czynniki nasilające/łagodzące.\n"
    )
    if not first_time_missing_basics:
        base_prompt_content += (
            "- **Przeszłość medyczna (PMH):** Choroby przewlekłe, operacje, alergie, leki (nazwy i dawki).\n"
            "- **Wywiad rodzinny i społeczny:** Choroby w rodzinie, papierosy, alkohol, styl życia.\n"
        )
    else:
        base_prompt_content += (
            "- **Przeszłość medyczna (PMH):** \n"
            "  - Choroby przewlekłe: nieznane – do zebrania podczas wywiadu\n"
            "  - Operacje: nieznane – do zebrania podczas wywiadu\n"
            "  - Alergie: nieznane – do zebrania podczas wywiadu\n"
            "  - Leki: nieznane – do zebrania podczas wywiadu\n"
            "- **Wywiad rodzinny i społeczny:** w znacznej części nieudokumentowany – do zebrania podczas wywiadu.\n"
        )
    base_prompt_content += (
        "- **Ukryte informacje:** Kluczowe fakty ujawniane tylko przy celnych pytaniach\n"
        "  (np. stres, problemy w domu, lęk zdrowotny, niestosowanie zaleceń, wstydliwy objaw). To sekcja kluczowa.\n\n"
        "- **Karta pacjenta (wymagana):**\n"
        "  ZAWSZE dołącz sekcję 'Karta pacjenta' na końcu scenariusza, w formie listy pozycji.\n"
        "  - Imię i nazwisko: <wartość> - UŻYJ RÓŻNEGO, losowego polskiego imienia i nazwiska (nie powtarzaj poprzednich)\n"
        "  - Wiek: <wartość> - UŻYJ RÓŻNEGO wieku (losuj z zakresu 18-95 lat, unikaj powtórzeń)\n"
        "  - Historia w punkcie: <Tak/Nie>\n"
        "  - Choroby przewlekłe: <wartość>\n"
        "  - Operacje: <wartość>\n"
        "  - Alergie: <wartość>\n"
        "  - Wywiad rodzinny: <wartość>\n\n"
        "  Jeśli pacjent jest PIERWSZY RAZ w punkcie (brak historii), ustaw: 'Historia w punkcie: Nie' i w tej sekcji podaj TYLKO\n"
        "  Imię i nazwisko oraz Wiek (pozostałe pozycje w tej sekcji pomiń — nie wpisuj 'brak danych').\n"
        "  Jeśli pacjent ma historię (drugi lub więcej raz), uzupełnij WSZYSTKIE powyższe pozycje konkretnymi danymi spójnymi z opisem."
    )

    if first_time_missing_basics:
        base_prompt_content += (
            "\n\n**STATUS PACJENTA:** Pacjent jest pierwszorazowy w tym punkcie (brak historii w dokumentacji). "
            "W sekcjach wymagających danych bazowych zaznacz braki wprost jako nieznane/nieudokumentowane. "
            "Pacjent nie podaje tych informacji spontanicznie – ujawnia je tylko po celnych pytaniach lekarza."
        )

    if keywords and keywords.strip():
        keyword_instruction = (
            f"\n\n**ZADANIE SPECJALNE:** Wygeneruj scenariusz ściśle powiązany z następującymi słowami kluczowymi: \"{keywords}\". "
            "Słowa te mają być centralnym elementem przypadku. Interpretuj kreatywnie, zachowując realizm kliniczny. "
            "Zapewnij spójność opowieści i logiczne powiązanie koncepcji."
        )
        final_prompt_content = base_prompt_content + keyword_instruction
    else:
        random_instruction = (
            "\n\n**ZADANIE:** Wygeneruj typowy przypadek z codziennej praktyki POZ (np. infekcja, problem w chorobie przewlekłej "
            "lub nowy niepokojący objaw)."
        )
        final_prompt_content = base_prompt_content + random_instruction

    return [{"role": "system", "content": final_prompt_content}]


def create_simulation_prompt(
    role_to_play: str,
    patient_scenario: str,
    chat_history: List[Dict[str, str]],
    question: str,
) -> List[Dict[str, str]]:
    """Build a prompt for the live interview simulation; force Polish responses."""

    if role_to_play == "patient":
        system_prompt = (
            "Jesteś aktorem odgrywającym pacjenta. NIE jesteś asystentem AI. Odpowiadasz krótko, naturalnie i zgodnie z rolą.\n"
            "Wszystkie odpowiedzi udzielaj WYŁĄCZNIE po polsku.\n"
            "Oto Twój tajny scenariusz postaci (lekarz go nie zna). Używaj go do kształtowania odpowiedzi:\n"
            "---\n"
            f"{patient_scenario}\n"
            "---\n"
            "Poniżej znajduje się dotychczasowa rozmowa z lekarzem.\n\n"
            "Zadanie: odpowiedz na ostatnie pytanie lekarza ściśle z perspektywy pacjenta. Nie ujawniaj informacji, "
            "o które nie poproszono wprost. Bądź realistyczny — możesz być zdenerwowany, zdezorientowany lub małomówny."
        )
        messages = [{"role": "system", "content": system_prompt}] + chat_history + [
            {"role": "user", "content": question}
        ]
        return messages

    if role_to_play == "doctor":
        system_prompt = (
            "Grasz rolę lekarza rodzinnego prowadzącego celowany, sprawny i empatyczny wywiad. Odpowiadaj po polsku."
        )
        return [{"role": "system", "content": system_prompt}] + chat_history + [
            {"role": "user", "content": question}
        ]

    if role_to_play == "meta":
        system_prompt = (
            "Jesteś klinicznym mentorem. Udzielasz krótkich, opartych na dowodach wskazówek na podstawie rozmowy. Odpowiadaj po polsku."
        )
        return [{"role": "system", "content": system_prompt}] + chat_history + [
            {"role": "user", "content": question}
        ]

    # Default pass-through
    return chat_history + [{"role": "user", "content": question}]


def generate_patient_summary_prompt(scenario: str) -> List[Dict[str, str]]:
    """Generate a prompt asking LLM to extract summary from patient scenario.
    
    Returns format: "Imię Nazwisko, wiek lat — główny objaw"
    """
    system_prompt = (
        "Jesteś asystentem wyciągającym kluczowe informacje ze scenariusza pacjenta. "
        "Wygeneruj krótkie podsumowanie w formacie:\n"
        '"Imię Nazwisko, wiek lat — główny objaw"\n\n'
        "Wyciągnij z podanego scenariusza:\n"
        "- Imię i nazwisko (jeśli nie ma, użyj 'NN')\n"
        "- Wiek (jeśli nie ma, użyj '?')\n"
        "- Główny objaw/powód wizyty (jedno zdanie, maksymalnie 50 znaków)\n\n"
        "Odpowiedz TYLKO w formacie: \"Imię Nazwisko, wiek lat — główny objaw\"\n"
        "Bez dodatkowych wyjaśnień, tylko ten format."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Scenariusz pacjenta:\n\n{scenario}"},
    ]


def generate_patient_card_extract_prompt(scenario: str, must_fill: bool = False) -> List[Dict[str, str]]:
    """Ask LLM to extract basic patient card info in strict JSON.

    Expected keys: name, age, chronic_diseases, operations, allergies, family_history.
    If must_fill is True (pacjent ma historię), wypełnij WSZYSTKIE pola realistycznymi
    danymi spójnymi ze scenariuszem (nie zostawiaj pustych). Gdy brak w tekście,
    uzupełnij najbardziej prawdopodobnymi informacjami, zachowując realizm kliniczny.
    """
    base = (
        "Jesteś ekstraktorem danych medycznych. Z poniższego scenariusza pacjenta wyodrębnij "
        "pola do karty pacjenta i ZWRÓĆ WYŁĄCZNIE poprawny JSON (bez komentarzy, bez markdown).\n\n"
        "Wynikowy JSON ma mieć dokładnie klucze:\n"
        "name (string), age (string), chronic_diseases (string), operations (string), allergies (string), family_history (string).\n\n"
    )
    if must_fill:
        base += (
            "Pacjent ma historię w punkcie — wszystkie pola MUSZĄ być uzupełnione KONKRETNYMI danymi z tekstu scenariusza. "
            "Wyciągnij WSZYSTKIE dane z tekstu:\n"
            "- name: pełne imię i nazwisko (jeśli nie ma, wywnioskuj z kontekstu np. 'Jan Kowalski')\n"
            "- age: konkretny wiek (np. '45 lat', nie '?')\n"
            "- chronic_diseases: konkretne choroby (np. 'nadciśnienie tętnicze, cukrzyca typu 2' lub 'brak')\n"
            "- operations: konkretne operacje (np. 'appendektomia 2010' lub 'brak')\n"
            "- allergies: konkretne alergie (np. 'penicylina' lub 'brak')\n"
            "- family_history: konkretny wywiad (np. 'ojciec zawał w wieku 60 lat' lub 'brak istotnych obciążeń')\n\n"
            "ZABRONIONE: placeholdery typu 'dane dostępne w dokumentacji', '?', 'Pacjent znany'. "
            "Zawsze konkretne wartości lub 'brak'/'nie zgłasza' jeśli rzeczywiście nie ma danych.\n"
        )
    else:
        base += "Jeśli czegoś nie ma w tekście, wpisz pusty string.\n"
    system_prompt = base
    user_prompt = f"Scenariusz pacjenta:\n\n{scenario}\n\nZwróć WYŁĄCZNIE JSON."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_minimal_identity_prompt(scenario: str) -> List[Dict[str, str]]:
    """Ask LLM to produce minimal identity: name and age.

    Always return JSON with keys: name, age. If scenario lacks them,
    wymyśl realistyczne polskie imię i wiek spójny z kontekstem.
    """
    system_prompt = (
        "Jesteś ekstraktorem danych. Zwróć TYLKO poprawny JSON (bez komentarzy, bez markdown).\n\n"
        "Wygeneruj klucze: name (string), age (string).\n"
        "Jeśli scenariusz nie zawiera tych danych, wymyśl realistyczne polskie imię i wiek spójny z kontekstem."
    )
    user_prompt = f"Scenariusz pacjenta:\n\n{scenario}\n\nZwróć WYŁĄCZNIE JSON."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def generate_interview_summary_prompt(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Generate a prompt for summarizing the interview conversation."""
    system_prompt = (
        "Jesteś lekarzem tworzącym podsumowanie wywiadu medycznego. "
        "Przeanalizuj poniższą rozmowę lekarza z pacjentem i stwórz zwięzłe podsumowanie.\n\n"
        "Podsumowanie powinno zawierać:\n"
        "- Główny powód wizyty/podejrzenie diagnostyczne\n"
        "- Kluczowe informacje z wywiadu\n"
        "- Ważne objawy, czynniki ryzyka, historia medyczna\n\n"
        "Bądź zwięzły, profesjonalny i skup się na faktach medycznych."
    )
    # Format chat history for summary
    conversation_text = ""
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        speaker = "Lekarz" if role == "user" else "Pacjent"
        conversation_text += f"{speaker}: {content}\n\n"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Rozmowa:\n\n{conversation_text}"},
    ]


def generate_recommendations_prompt(chat_history: List[Dict[str, str]], summary: str) -> List[Dict[str, str]]:
    """Generate a prompt for creating medical recommendations based on interview."""
    system_prompt = (
        "Jesteś doświadczonym lekarzem rodzinnym. "
        "Na podstawie podsumowania wywiadu medycznego, stwórz listę zaleceń.\n\n"
        "Zalecenia powinny zawierać:\n"
        "- Proponowane badania diagnostyczne (z uzasadnieniem)\n"
        "- Zalecenia terapeutyczne\n"
        "- Dalsze kroki w postępowaniu\n"
        "- Monitorowanie/obserwacja\n\n"
        "Formatuj jako listę punktowaną. Bądź konkretny i praktyczny."
    )
    conversation_text = ""
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        speaker = "Lekarz" if role == "user" else "Pacjent"
        conversation_text += f"{speaker}: {content}\n\n"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Podsumowanie wywiadu:\n{summary}\n\nRozmowa:\n\n{conversation_text}"},
    ]


def generate_treatment_plan_prompt(
    patient_scenario: str,
    chat_history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Generate a prompt for creating the correct treatment plan (medications, recommendations, tests) based on patient scenario and interview."""
    system_prompt = (
        "Jesteś doświadczonym lekarzem rodzinnym tworzącym kompleksowy plan postępowania medycznego.\n\n"
        "Na podstawie scenariusza pacjenta i całej historii wywiadu, wygeneruj POPRAWNY plan postępowania.\n\n"
        "Plan powinien zawierać następujące sekcje:\n\n"
        "**DIAGNOZA:**\n"
        "- Główna diagnoza (podejrzenie) na podstawie wywiadu\n"
        "- Diagnozy różnicowe (jeśli istotne)\n"
        "- Uzasadnienie diagnostyczne\n\n"
        "**LEKI (jeśli wskazane):**\n"
        "- Konkretne nazwy leków z dawkami\n"
        "- Uzasadnienie wyboru\n\n"
        "**ZALECENIA:**\n"
        "- Zalecenia dotyczące stylu życia\n"
        "- Zalecenia dotyczące diety\n"
        "- Inne zalecenia terapeutyczne\n\n"
        "**BADANIA:**\n"
        "- Badania diagnostyczne do wykonania\n"
        "- Uzasadnienie każdego badania\n\n"
        "Formatuj odpowiedź jako strukturę z sekcjami. Bądź konkretny, profesjonalny i oparty na najlepszych praktykach medycznych.\n"
        "Odpowiedz WYŁĄCZNIE po polsku."
    )
    
    conversation_text = ""
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        speaker = "Lekarz" if role == "user" else "Pacjent"
        conversation_text += f"{speaker}: {content}\n\n"
    
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Scenariusz pacjenta:\n\n{patient_scenario}\n\n---\n\nHistoria wywiadu:\n\n{conversation_text}",
        },
    ]


def generate_diagnosis_evaluation_prompt(
    correct_plan: str,
    user_response: str,
    patient_scenario: str,
    chat_history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Generate a prompt for evaluating user's diagnosis and treatment plan against the correct one."""
    system_prompt = (
        "Jesteś doświadczonym lekarzem-mentorem oceniającym pracę lekarza.\n\n"
        "Porównaj odpowiedź z poprawnym planem postępowania i oceń:\n\n"
        "1. **Czy diagnoza jest prawidłowa?** (Tak/Nie/U częściowo)\n"
        "2. **Czy leki są odpowiednie?** (Zaznacz co jest dobre, co brakuje, co jest nieprawidłowe)\n"
        "3. **Czy badania są odpowiednie?** (Co jest dobre, co brakuje, co jest niepotrzebne)\n"
        "4. **Czy zalecenia są odpowiednie?** (Co jest dobre, co brakuje)\n\n"
        "Następnie stwórz konstruktywną ocenę:\n"
        "- Co zostało zrobione dobrze?\n"
        "- Co można poprawić?\n"
        "- Jakie kluczowe elementy zostały pominięte?\n"
        "- Jakie dodatkowe informacje mogą być przydatne?\n\n"
        "Bądź konstruktywny, profesjonalny i edukacyjny. Odpowiedz WYŁĄCZNIE po polsku."
    )
    
    conversation_text = ""
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        speaker = "Lekarz" if role == "user" else "Pacjent"
        conversation_text += f"{speaker}: {content}\n\n"
    
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Scenariusz pacjenta:\n\n{patient_scenario}\n\n"
                f"---\n\nHistoria wywiadu:\n\n{conversation_text}\n\n"
                f"---\n\nPOPRAWNY plan postępowania:\n\n{correct_plan}\n\n"
                f"---\n\nOdpowiedź lekarza:\n\n{user_response}\n\n"
                f"---\n\nOceń odpowiedź lekarza w porównaniu z poprawnym planem."
            ),
        },
    ]


