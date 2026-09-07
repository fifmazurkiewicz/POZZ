# Domain user journey — design (2026-09-07)

Approved as the POZZ domain contract. Durable copy also synced into `docs/architecture-for-cursor.md` and `docs/ux/ux-ui-spec.md`.

## Happy path

1. Google OAuth → if not `is_approved`, waiting screen → after Accept, disclaimer (simulator, not a medical device) → Simulation.
2. **Następny pacjent** loads a shared catalog row this user has not completed/skipped, or generates one. Optional keywords force a new generate.
3. Doctor sees the **card** (not the hidden scenario / gold plan). First-time patients: name + age only.
4. Simulation: text always; optional mic → STT. Modes: Lekarz / Pacjent / Dopytaj AI. Transcript visible.
5. **Zakończ wywiad** → doctor writes medications / advice / tests → LLM evaluation vs gold → gold plan revealed. `patient_user_state = completed`.
6. Parallel path: Interview → record one WAV → STT + roles + summary + extract; or Manual lines → summary + recommendations.
7. Menu → Sesje lists **this user’s** conversations only.
8. New user `spend_cap_usd = 10` (monthly, Europe/Warsaw). At cap: costly actions blocked; Sesje still readable.

## Given / When / Then (core)

| Given | When | Then |
|---|---|---|
| Approved user on Simulation | Taps Następny pacjent with empty keywords | Loads unused catalog patient or generates one; card visible; gold plan hidden |
| Keywords filled | Taps Następny pacjent | New `patients` row from keywords; conversation opened |
| First-time patient | Card renders | Name, age, historia = Nie; no chronic/ops/allergies/family |
| Active simulation | User sends text | User + assistant messages persist; transcript shows Lekarz vs Pacjent vs AI per mode |
| ≥ 1 turn | Taps Zakończ wywiad | Waiting-for-plan UI; gold still hidden |
| Plan submitted | Evaluation job/LLM returns | Evaluation + gold expander; state completed |
| User at monthly cap | Tries generate / turn / STT / evaluate | Blocked; Sesje remain |
| Recorded interview | Stop after WAV | Transcript roles + summary + extract; no audio row |
| User A has conversations | User B opens Sesje | User B never sees User A rows |
| Unapproved signup | Hits any feature API | 403; waiting UI polls `/api/auth/me` |
| Admin | PATCH `is_approved` true | Waiting UI proceeds on next poll |

## Out of scope here

Patient TTS, Gemini Live, chunked live suggestions, diarization GPU, resume conversation, product-UI database wipe, inbound audio storage.
