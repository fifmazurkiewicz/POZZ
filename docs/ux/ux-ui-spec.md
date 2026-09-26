# UX / UI Specification — MVP

Skonsolidowana wersja ustaleń UX/UI. Opis biznesowo-techniczny: [`../architecture-for-cursor.md`](../architecture-for-cursor.md). Kierunek wizualny SoT: [`ux-ui-decisions.md`](./ux-ui-decisions.md) (Classical, clinical).

Copy produktu jest **po polsku** (preserve). To nie jest aplikacja medyczna — w Menu i na ekranie logowania krótki disclaimer: trening, nie urządzenie medyczne.

## 1. Zakres MVP (Etap 1)

- **Symulacja** — karta pacjenta, czat (tekst zawsze; mikrofon opcjonalnie), tryby Lekarz / Pacjent / Dopytaj AI, Zakończ wywiad → plan leczenia → ocena vs złoty standard.
- **Wywiad** — nagranie jednego ujęcia (Start → Stop → przetwarzanie) **oraz** ręczny konstruktor linii Lekarz/Pacjent → podsumowanie i zalecenia. Ręczny konstruktor ma **przycisk mikrofonu** (STT) obok pola tekstowego — dyktowanie zamiast pisania. Bez TTS pacjenta, bez lampy Live. ADR: [`../technical/decisions/2026-09-13-interview-voice-input.md`](../technical/decisions/2026-09-13-interview-voice-input.md). **Formularz tworzenia przypadku** (tytuł + opis pacjenta) ma też mikrofony przy obu polach — dictation przed "Rozpocznij wywiad".
- **Menu → Sesje** — lista własnych wywiadów (symulacja / nagrany / ręczny) + szczegół.
- Google OAuth, bramka akceptacji, Admin (cap, kolejka Accept, masowe generowanie pacjentów).
- **Głos w Symulacji:** chained STT + TTS. Composer ma przełącznik **Wiadomości / Rozmowa**. W Rozmowie lokalny VAD wykrywa 1,5 s ciszy, a następnie JEV sprawdza niekliniczną granicę wypowiedzi; błąd prosi o powtórzenie i nic nie zapisuje. Transkrypt + composer są zawsze widoczne i historia automatycznie podąża za nową turą, dopóki użytkownik nie przewinie jej ręcznie. Spec: [`../superpowers/specs/2026-09-26-jev-voice-turns-design.md`](../superpowers/specs/2026-09-26-jev-voice-turns-design.md).

**Poza MVP (Etap 2):** podpowiedzi co 10 s podczas nagrania, diarization GPU, wipe bazy w UI, resume tej samej `conversation_id`.

## 2. Platforma

- Jedna responsywna PWA (Next.js) — desktop i telefon.
- Na telefonie: manifest + service worker, `standalone`, touch targets ≥ 44 px, safe-area.
- Bez sklepów / natywnych app.
- Mikrofon wymaga HTTPS (prod) / localhost (dev).

## 3. Nawigacja — dolny pasek, 3 zakładki

| Zakładka | Zawartość |
|---|---|
| **Symulacja** | Domyślna po zalogowaniu (gdy approved) — karta + czat |
| **Wywiad** | Nagranie albo konstruktor ręczny |
| **Menu** | Sesje, profil, wygląd, Admin, Wyloguj |

Zimny API: globalny banner **Waking up…** (ApiPulse), zanim czat/mikrofon są używalne.

## 4. Onboarding / dostęp

1. Google OAuth (lub Login dev lokalnie).
2. Jeśli `is_approved = false` — pełnoekranowy wait (poll 15 s). Bez czatu.
3. Brak wielojęzycznego onboardingu Langy — POZZ jest zawsze PL. Opcjonalnie krótki disclaimer + „Zaczynam”.

## 5. Symulacja

- Pole **Słowa kluczowe** (opcjonalne) + **Następny pacjent**.
- **Karta pacjenta** zwijana. First-time: imię, wiek, historia w punkcie = Nie. Returning: pełna karta. Scenariusz ukryty (nie pokazujemy gold planu ani pełnego HPI lekarzowi).
- Powłoka ma stałą wysokość viewportu. Tylko transkrypt się przewija; composer, akcje rozmowy i globalny dolny pasek pozostają widoczne, również nad mobile safe area.
- Transkrypt **zawsze widoczny** w aktywnej sesji (linie Lekarz / Pacjent / AI).
- Radio trybu pod czatem: Lekarz | Pacjent | Dopytaj AI.
- Composer: text input zawsze; Listening opcjonalne. Obok pola są opisane ikony mikrofonu i głośnika ze stanem dostępnym także bez rozpoznawania koloru. Nie stosujemy nieopisanych kropek statusu głosu.
- **Zatrzymaj** anuluje bieżące żądanie przeglądarki, wycisza TTS / mowę przeglądarki, zatrzymuje mikrofon i odrzuca niedokończony input. Praca synchroniczna uruchomiona już na serwerze może dobiec końca, ale jej spóźniona odpowiedź nie zmienia zatrzymanego widoku.
- **Zrób badanie** otwiera dialog z widoczną etykietą. Wynik jest tekstem zapisanym w historii; nie jest odczytywany głosem i nie ujawnia ukrytej diagnozy ani wzorcowego planu.
- **Resetuj wywiad** — nowa konwersacja, ten sam pacjent.
- **Zakończ wywiad** (gdy jest ≥ 1 tura) zatrzymuje audio i otwiera dialog planu (leki / zalecenia / badania) → **Zakończ i oceń**. Sukces zapisuje ocenę i czas zakończenia; rozmowa staje się tylko do odczytu. Błąd pozostawia ją otwartą do ponowienia.

## 6. Wywiad (nagrany + ręczny)

- Sub-tryby: **Nagraj** | **Ręcznie** (kompaktowy toggle, `text-sm`).
- Nagraj: opcjonalny tytuł + przycisk **Rozpocznij nagrywanie** → live capture (`useVoiceController`) z pulsującą kropką i timerem → **Zatrzymaj i transkrybuj** (albo **Anuluj**) → POST `/api/interviews/recordings` z blobem w pamięci → transkrypt z rolami + podsumowanie + ekstrakcja (leki / zalecenia / badania). Nic nie jest zapisywane po stronie klienta. Implementacja: [`frontend/src/components/interview/RecordedRecorder.tsx`](../../frontend/src/components/interview/RecordedRecorder.tsx).
- Ręcznie: textarea scenariusza opcjonalna; rozmowa tekstowa; te same akcje **Zatrzymaj**, **Zrób badanie** i **Zakończ wywiad** co w Symulacji. Wyniki badań trafiają do historii, a ukończona rozmowa jest tylko do odczytu.

## 7. Sesje

Lista własnych pozycji (data, etykieta). Tap → szczegół (transkrypt / czat / ocena / ekstrakcja). Back do listy. Bez usuwania w MVP (można dodać później).

## 8. Menu

Sesje · Wygląd (System / Jasny / Ciemny) · Głos · (Admin) · Wyloguj.

**Głos:** wybór Live / TTS znajduje się tutaj, nie w wierszu rozmowy. Obecna wersja pokazuje Live jako niedostępny i wybiera TTS. Opcjonalne pole ElevenLabs Voice ID jest zapisane lokalnie w tej przeglądarce; puste pole oznacza domyślny głos serwera, a format jest walidowany.

**Admin:** lista użytkowników (`is_approved` toggle), spend cap, masowe generowanie N pacjentów, status puli. Bez wipe i bez osobnego przycisku Menu w prawym górnym rogu; powrót zapewnia stały dolny pasek.

## 9. Stany puste / błędy / cap

- Brak pacjenta: CTA Następny pacjent.
- Brak sesji: „Nie masz jeszcze wywiadów.”
- Cap: toast / banner — kosztowe akcje wstrzymane do następnego miesiąca kalendarzowego; Sesje pozostają.
- Błędy: co się stało + co zrobić. Bez zawstydzania. Mic denied: instrukcja uprawnień przeglądarki.

## 10. A11y

`prefers-reduced-motion`. Hit targets ≥ 44 px. Kontrast kliniczny (Taste VARIANCE 3). Nie używać koloru jako jedynego sygnału oceny.
