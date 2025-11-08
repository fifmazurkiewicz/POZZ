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


def assign_roles_and_suggest(
    transcript: str,
    context: Optional[str] = None,
    previous_chunks: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Generuje prompt do przypisania ról (Lekarz/Pacjent) i sugestii pytań.
    
    Args:
        transcript: Surowa transkrypcja z Whisper
        context: Kontekst wywiadu (scenariusz pacjenta)
        previous_chunks: Poprzednie chunki z przypisanymi rolami
        
    Returns:
        Lista wiadomości dla LLM
    """
    previous_context = ""
    if previous_chunks:
        previous_context = "\n\nPoprzednie fragmenty wywiadu:\n"
        for chunk in previous_chunks[-2:]:  # Ostatnie 2 chunki dla kontekstu
            for segment in chunk.get("transcript", []):
                role = segment.get("role", "unknown")
                text = segment.get("text", "")
                previous_context += f"{role.upper()}: {text}\n"
    
    context_text = f"\n\nKontekst pacjenta:\n{context}" if context else ""
    
    return [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem medycznym analizującym transkrypcję wywiadu lekarskiego. "
                "Twoim zadaniem jest:\n"
                "1. Przypisać role (Lekarz/Pacjent) do każdej wypowiedzi na podstawie treści\n"
                "2. Wygenerować sugestie pytań lub informacji do zebrania\n\n"
                "Zasady przypisywania ról:\n"
                "- Lekarz zwykle zadaje pytania, używa terminologii medycznej, proponuje badania/leki\n"
                "- Pacjent opisuje objawy, odpowiada na pytania, używa języka potocznego\n"
                "- Pierwsza osoba mówiąca w wywiadzie to zwykle lekarz\n"
                "- Analizuj kontekst i wzorce językowe\n\n"
                "Odpowiedz TYLKO w formacie JSON:\n"
                "{\n"
                '  "transcript_with_roles": [\n'
                '    {"role": "doctor"|"patient", "text": "...", "timestamp": 0.0},\n'
                '    ...\n'
                '  ],\n'
                '  "suggestions": "Sugestie pytań lub informacji do zebrania..."\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Przeanalizuj poniższą transkrypcję wywiadu i przypisz role.{previous_context}{context_text}\n\n"
                f"Transkrypcja do analizy:\n{transcript}\n\n"
                "Przypisz role i wygeneruj sugestie."
            ),
        },
    ]


def generate_interview_summary(
    transcripts_with_roles: List[Dict],
    patient_scenario: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generuje prompt do podsumowania całego wywiadu.
    
    Args:
        transcripts_with_roles: Lista wszystkich transkrypcji z przypisanymi rolami
        patient_scenario: Scenariusz pacjenta (opcjonalnie)
        
    Returns:
        Lista wiadomości dla LLM
    """
    # Połącz wszystkie transkrypcje w jeden tekst
    full_transcript = ""
    for chunk in transcripts_with_roles:
        for segment in chunk.get("transcript", []):
            role = segment.get("role", "unknown")
            text = segment.get("text", "")
            timestamp = segment.get("timestamp", 0.0)
            role_label = "Lekarz" if role == "doctor" else "Pacjent"
            full_transcript += f"[{timestamp:.1f}s] {role_label}: {text}\n"
    
    context_text = f"\n\nScenariusz pacjenta:\n{patient_scenario}" if patient_scenario else ""
    
    return [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem medycznym tworzącym podsumowanie wywiadu lekarskiego. "
                "Stwórz zwięzłe, profesjonalne podsumowanie zawierające:\n"
                "- Główny powód wizyty\n"
                "- Kluczowe objawy i dolegliwości\n"
                "- Ważne informacje z wywiadu (choroby przewlekłe, leki, alergie)\n"
                "- Proponowane działania (badania, leki, zalecenia)\n\n"
                "Użyj języka medycznego, ale zrozumiałego. Odpowiedz po polsku."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Stwórz podsumowanie poniższego wywiadu.{context_text}\n\n"
                f"Pełna transkrypcja wywiadu:\n{full_transcript}\n\n"
                "Wygeneruj podsumowanie wywiadu."
            ),
        },
    ]


def extract_medical_info(
    transcripts_with_roles: List[Dict],
    doctor_proposals: Optional[str] = None,
    patient_scenario: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generuje prompt do ekstrakcji informacji medycznych i oceny propozycji lekarza.
    
    Args:
        transcripts_with_roles: Lista wszystkich transkrypcji z przypisanymi rolami
        doctor_proposals: Propozycje lekarza (leki, zalecenia, badania) - opcjonalnie
        patient_scenario: Scenariusz pacjenta (opcjonalnie)
        
    Returns:
        Lista wiadomości dla LLM
    """
    # Połącz wszystkie transkrypcje
    full_transcript = ""
    for chunk in transcripts_with_roles:
        for segment in chunk.get("transcript", []):
            role = segment.get("role", "unknown")
            text = segment.get("text", "")
            role_label = "Lekarz" if role == "doctor" else "Pacjent"
            full_transcript += f"{role_label}: {text}\n"
    
    context_text = f"\n\nScenariusz pacjenta:\n{patient_scenario}" if patient_scenario else ""
    proposals_text = (
        f"\n\nPropozycje lekarza:\n{doctor_proposals}" if doctor_proposals else ""
    )
    
    return [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem medycznym ekstrahującym informacje z wywiadu lekarskiego. "
                "Twoim zadaniem jest:\n"
                "1. Wyekstrahować leki, zalecenia i badania z wywiadu\n"
                "2. Jeśli podano propozycje lekarza - ocenić je w kontekście wywiadu\n"
                "3. Zaproponować poprawki jeśli potrzebne\n\n"
                "Odpowiedz TYLKO w formacie JSON:\n"
                "{\n"
                '  "leki": ["lek1", "lek2", ...],\n'
                '  "zalecenia": ["zalecenie1", "zalecenie2", ...],\n'
                '  "badania": ["badanie1", "badanie2", ...],\n'
                '  "ocena_propozycji_lekarza": "Ocena propozycji lekarza...",\n'
                '  "sugestie_poprawek": "Sugestie poprawek jeśli potrzebne..."\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Wyekstrahuj informacje medyczne z poniższego wywiadu.{context_text}{proposals_text}\n\n"
                f"Transkrypcja wywiadu:\n{full_transcript}\n\n"
                "Wyekstrahuj leki, zalecenia, badania i oceń propozycje lekarza."
            ),
        },
    ]


def process_interview_transcript(
    transcript: str,
    patient_scenario: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generuje prompt do przetworzenia całej transkrypcji wywiadu w jednym wywołaniu LLM.
    
    LLM wykonuje:
    1. Przypisanie ról (Lekarz/Pacjent) do wypowiedzi
    2. Stworzenie podsumowania wywiadu
    3. Ekstrakcję informacji medycznych (leki, zalecenia, badania)
    
    Args:
        transcript: Surowa transkrypcja z Whisper
        patient_scenario: Scenariusz pacjenta (opcjonalnie)
        
    Returns:
        Lista wiadomości dla LLM
    """
    context_text = f"\n\nScenariusz pacjenta (jeśli dostępny):\n{patient_scenario}" if patient_scenario else ""
    
    return [
        {
            "role": "system",
            "content": (
                "Jesteś asystentem medycznym analizującym transkrypcję wywiadu lekarskiego. "
                "Twoim zadaniem jest:\n\n"
                "1. PRZYPISANIE RÓL:\n"
                "   - Przypisz role (Lekarz/Pacjent) do każdej wypowiedzi na podstawie treści\n"
                "   - Lekarz zwykle zadaje pytania, używa terminologii medycznej, proponuje badania/leki\n"
                "   - Pacjent opisuje objawy, odpowiada na pytania, używa języka potocznego\n"
                "   - Pierwsza osoba mówiąca w wywiadzie to zwykle lekarz\n"
                "   - Dla każdej wypowiedzi przypisz timestamp (w sekundach od początku)\n\n"
                "2. PODSUMOWANIE:\n"
                "   - Stwórz zwięzłe, profesjonalne podsumowanie zawierające:\n"
                "     * Główny powód wizyty\n"
                "     * Kluczowe objawy i dolegliwości\n"
                "     * Ważne informacje z wywiadu (choroby przewlekłe, leki, alergie)\n"
                "     * Proponowane działania (badania, leki, zalecenia)\n\n"
                "3. EKSTRAKCJA INFORMACJI:\n"
                "   - Wyekstrahuj leki, zalecenia i badania z wywiadu\n"
                "   - Oceń propozycje lekarza w kontekście wywiadu\n"
                "   - Zaproponuj poprawki jeśli potrzebne\n\n"
                "Odpowiedz TYLKO w formacie JSON:\n"
                "{\n"
                '  "transcript_with_roles": [\n'
                '    {"role": "doctor"|"patient", "text": "...", "timestamp": 0.0},\n'
                '    ...\n'
                '  ],\n'
                '  "summary": "Podsumowanie wywiadu...",\n'
                '  "extracted_info": {\n'
                '    "leki": ["lek1", "lek2", ...],\n'
                '    "zalecenia": ["zalecenie1", "zalecenie2", ...],\n'
                '    "badania": ["badanie1", "badanie2", ...],\n'
                '    "ocena_propozycji_lekarza": "Ocena propozycji lekarza...",\n'
                '    "sugestie_poprawek": "Sugestie poprawek jeśli potrzebne..."\n'
                '  }\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Przeanalizuj poniższą transkrypcję wywiadu lekarskiego.{context_text}\n\n"
                f"Transkrypcja do analizy:\n{transcript}\n\n"
                "Przypisz role, stwórz podsumowanie i wyekstrahuj informacje medyczne. "
                "Odpowiedz w formacie JSON zgodnie z instrukcjami."
            ),
        },
    ]


