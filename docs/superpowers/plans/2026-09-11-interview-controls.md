# Interview controls implementation plan

> Execute with superpowers:subagent-driven-development; independent ownership of backend, voice lifecycle, and UI integration.

**Goal:** Interrupt speech/listening, perform examinations, complete and evaluate interviews, and choose a voice in Menu.

**Architecture:** Existing FastAPI conversation routes persist examinations and evaluation in existing columns. A cancellable browser voice controller owns audio resources. Shared React action dialogs serve both interview surfaces.

**Tech Stack:** FastAPI, SQLAlchemy, Next.js, React, native dialog, ElevenLabs, Vitest, pytest.

## Global constraints

- Polish product UI; English identifiers/docs. Preserve existing clinical design.
- Vercel frontend, Render backend, Supabase schema; no Streamlit changes.
- Auth ownership and spending cap enforced for every new costly action.
- Completed sessions read-only; stopped operations cannot restart audio.
- No secrets in frontend; voice preference is a validated identifier only.

## Task 1: Backend lifecycle and voice override

- [x] Add regression tests for examination persistence, hidden data isolation, end/retry/idempotence, ownership and cap checks, ended-turn rejection, voice override validation.
- [x] Implement POST /api/conversations/{id}/examinations with {examination}, and POST /api/conversations/{id}/finish with {treatment_plan}; return the extended conversation payload.
- [x] Persist reference plan privately, evaluation and ended_at; reuse saved completion on retries. Return ended_at, user_treatment_response, diagnosis_evaluation in payload (evaluation only after completion).
- [x] Validate optional voice_id on /api/voice/speech; empty uses configured default.
- [x] Run backend pytest suite.

## Task 2: Audio lifecycle

- [x] Write Vitest tests proving abort stops audio and discards stale async work.
- [x] Implement a reusable voice controller with play(text, token), startRecording(onBlob), finishRecording(), stop(), dispose(). Abort TTS requests; cancel browser speech; pause audio and revoke URLs; stop tracks; discard canceled recordings.
- [x] Add browser voice ID read/save helpers and validation.
- [x] Run focused Vitest suite.

## Task 3: Shared UI and integration

- [x] Extend ApiOptions with signal and simulation API with examination/finish methods and completion fields.
- [x] Add shared native-dialog action component for examination/finish, cancellation and result rendering. Gate inputs after completion.
- [x] Wire Stop to pending text requests and voice lifecycle in Simulation and Interview; prevent stale state updates. Keep recording finish/send distinct from Stop/discard.
- [x] Add voice ID field and Save/reset behavior to Menu with local persistence and clear provider copy.
- [x] Run frontend lint, tests and production build; review backend/frontend contract and cancellation races.

## Verification

Run backend tests with an isolated SQLite test engine; frontend lint and Vitest then Next production build. Review UI layout and native dialog semantics. Document any environment limitations and do not claim production deployment.
