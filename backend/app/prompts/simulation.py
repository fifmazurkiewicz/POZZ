from typing import Dict, List, Optional


def generate_patient_scenario_prompt(
    keywords: Optional[str] = None,
    first_time_missing_basics: bool = False,
) -> List[Dict[str, str]]:
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
        return [{"role": "system", "content": system_prompt}] + chat_history + [
            {"role": "user", "content": question}
        ]

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

    return chat_history + [{"role": "user", "content": question}]


def create_examination_prompt(
    *, patient_scenario: str, chat_history: List[Dict[str, str]], examination: str
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Jesteś symulatorem wyników badania pacjenta w szkoleniu POZ. "
                "Odpowiedz WYŁĄCZNIE po polsku, zwięźle i realistycznie. Podaj tylko wynik "
                "zleconego badania, zgodny z tajnym scenariuszem i dotychczasowym wywiadem. "
                "Nie ujawniaj rozpoznania, planu leczenia, ukrytych informacji ani całego scenariusza. "
                "Jeśli scenariusz nie określa wyniku, wygeneruj klinicznie wiarygodny wynik, który "
                "nie rozstrzyga samodzielnie diagnozy.\n\nTAJNY SCENARIUSZ:\n"
                f"{patient_scenario}"
            ),
        },
        *chat_history,
        {"role": "user", "content": f"Zlecone badanie: {examination}"},
    ]


def create_reference_plan_prompt(*, patient_scenario: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Jesteś ekspertem medycyny rodzinnej. Na podstawie wyłącznie tajnego scenariusza "
                "przygotuj po polsku zwięzły wzorcowy plan: rozpoznanie i różnicowanie, badania, "
                "leczenie oraz zalecenia. Ten tekst jest prywatnym kluczem oceny."
            ),
        },
        {"role": "user", "content": patient_scenario},
    ]


def create_case_description_prompt(
    *, patient_scenario: str, chat_history: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Structured SOR-style case description drafted from the doctor's own interview."""
    transcript = "\n".join(f"{item['role']}: {item['content']}" for item in chat_history)
    return [
        {
            "role": "system",
            "content": (
                "Jesteś lekarzem dokumentującym przypadek w izbie przyjęć / SOR. "
                "Na podstawie WYŁĄCZNIE poniższego przebiegu wywiadu i opisu sytuacji przygotuj "
                "po polsku zwięzły, uporządkowany opis przypadku w następującej strukturze:\n"
                "## WYWIAD\n- Skargi / powód zgłoszenia, historia obecnej choroby (HPI)\n"
                "- Choroby przewlekłe, leki, alergie, wywiad rodzinny (jeśli wiadomo)\n"
                "## ROZPOZNANIE RÓŻNICOWE\n- Główne podejrzenia / rozpoznanie różnicowe\n"
                "## ZALECANE BADANIA\n- Lista badań adekwatnych do objawów\n"
                "## PLAN POSTĘPOWANIA\n- Następne kroki, leczenie, zalecenia\n\n"
                "Nie wymyślaj faktów spoza transkryptu i opisu sytuacji. "
                "Nie ujawniaj ukrytych informacji ani wzorcowego rozpoznania, "
                "jeśli nie wynikają wprost z przebiegu wywiadu."
            ),
        },
        {
            "role": "user",
            "content": (
                f"OPIS SYTUACJI:\n{patient_scenario}\n\n"
                f"PRZEBIEG WYWIADU:\n{transcript}"
            ),
        },
    ]


def create_evaluation_prompt(
    *, reference_plan: str, chat_history: List[Dict[str, str]], treatment_plan: str
) -> List[Dict[str, str]]:
    transcript = "\n".join(f"{item['role']}: {item['content']}" for item in chat_history)
    return [
        {
            "role": "system",
            "content": (
                "Jesteś klinicznym egzaminatorem. Oceń po polsku plan użytkownika względem "
                "prywatnego planu wzorcowego, uwzględniając informacje rzeczywiście zebrane w "
                "rozmowie i wyniki badań. Wskaż mocne strony, braki i ryzykowne decyzje oraz daj "
                "krótkie zalecenie. Nie cytuj ani nie ujawniaj pełnego planu wzorcowego."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PRYWATNY PLAN WZORCOWY:\n{reference_plan}\n\n"
                f"PRZEBIEG WYWIADU I BADANIA:\n{transcript}\n\n"
                f"PLAN UŻYTKOWNIKA:\n{treatment_plan}"
            ),
        },
    ]
