# UX / UI Specification — MVP

Skonsolidowana wersja ustaleń UX/UI. Opis biznesowo-techniczny: [`../architecture-for-cursor.md`](../architecture-for-cursor.md). Kierunek wizualny SoT: [`ux-ui-decisions.md`](./ux-ui-decisions.md) (Classical, clinical).

Copy produktu jest **po polsku** (preserve). To nie jest aplikacja medyczna — w Menu i na ekranie logowania krótki disclaimer: trening, nie urządzenie medyczne.

## 1. Zakres MVP (Etap 1)

- **Symulacja** — karta pacjenta, czat (tekst zawsze; mikrofon opcjonalnie), tryby Lekarz / Pacjent / Dopytaj AI, Zakończ wywiad → plan leczenia → ocena vs złoty standard.
- **Wywiad** — nagranie jednego ujęcia (Start → Stop → przetwarzanie) **oraz** ręczny konstruktor linii Lekarz/Pacjent → podsumowanie i zalecenia.
- **Menu → Sesje** — lista własnych wywiadów (symulacja / nagrany / ręczny) + szczegół.
- Google OAuth, bramka akceptacji, Admin (cap, kolejka Accept, masowe generowanie pacjentów).
- **Głos w Symulacji:** lampa Live Gemini (ON) vs TTS (OFF); transkrypt + composer zawsze. Spec: [`../superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`](../superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md).

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
- Chrome Symulacji (zablokowane, jak Langy Chat): nagłówek (skrót pacjenta + Następny) · **wiersz statusu** (lampa Live/TTS lewo · Ready/Słuchanie środek · kropki Głos pacjenta + Słuchanie prawo) · tylko transkrypt się scrolluje · composer + dolny pasek.
- Transkrypt **zawsze widoczny** w aktywnej sesji (linie Lekarz / Pacjent / AI).
- Radio trybu pod czatem: Lekarz | Pacjent | Dopytaj AI.
- Composer: text input zawsze; Listening opcjonalne. Send → Stop gdy pacjent mówi lub model pisze.
- Lampa: ON = Gemini Live; OFF = STT + TTS pacjenta. `VOICE_MODE=chained` wyłącza lampę. Preference w `localStorage`.
- **Resetuj wywiad** — nowa konwersacja, ten sam pacjent.
- **Zakończ wywiad** (gdy jest ≥ 1 tura) → textarea planu (leki / zalecenia / badania) + opcjonalny mic → **Wyślij** → ocena + expander złotego planu. **Spróbuj ponownie** wraca do textarea.

## 6. Wywiad (nagrany + ręczny)

- Sub-tryby: **Nagraj** | **Ręcznie**.
- Nagraj: Start (tworzy sesję) → nagrywanie → Stop (upload + spinner transkrypcja/analiza) → transkrypt z rolami + podsumowanie + ekstrakcja (leki / zalecenia / badania).
- Ręcznie: textarea scenariusza opcjonalna; radio Lekarz/Pacjent; Dodaj linię; Koniec wywiadu → podsumowanie + zalecenia.

## 7. Sesje

Lista własnych pozycji (data, etykieta). Tap → szczegół (transkrypt / czat / ocena / ekstrakcja). Back do listy. Bez usuwania w MVP (można dodać później).

## 8. Menu

Sesje · Wygląd (System / Jasny / Ciemny) · (Admin) · Wyloguj.

**Admin:** lista użytkowników (`is_approved` toggle), spend cap, masowe generowanie N pacjentów, status puli. Bez wipe.

## 9. Stany puste / błędy / cap

- Brak pacjenta: CTA Następny pacjent.
- Brak sesji: „Nie masz jeszcze wywiadów.”
- Cap: toast / banner — kosztowe akcje wstrzymane do następnego miesiąca kalendarzowego; Sesje pozostają.
- Błędy: co się stało + co zrobić. Bez zawstydzania. Mic denied: instrukcja uprawnień przeglądarki.

## 10. A11y

`prefers-reduced-motion`. Hit targets ≥ 44 px. Kontrast kliniczny (Taste VARIANCE 3). Nie używać koloru jako jedynego sygnału oceny.
