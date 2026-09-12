---
name: 2026-09-11 interview error handling
overview: Polish JSON contract for unhandled provider failures on /examinations and /finish, then surface real error/cancel states in the interview modals and let doctors dictate the diagnosis.
todos:
  - id: diagnose
    content: Reproduce 500 vs Polish fallback confusion; identify provider 4xx/5xx, network, abort and cold start as distinct failure modes
    status: completed
  - id: handler
    content: backend/app/main.py exception_handler(Exception) returns 502 on httpx errors and 503 otherwise, with {code: provider_error, message: Polish}
    status: completed
  - id: fixture
    content: conftest.py sqlite_client_no_raise for tests that assert on handler output
    status: completed
  - id: backend-tests
    content: TDD: test_failed_evaluation_leaves_conversation_open (now 503) + test_examination_returns_502_when_provider_errors + test_existing_polish_codes_still_flow_through_handler
    status: completed
  - id: hook-rewrite
    content: useAbortableAction distinguishes cancel (silent) from network/transport abort (fallback), exposes cancelled + latestError + reset
    status: completed
  - id: modal-mic
    content: InterviewActions modal: inline status text, separate 'Anuluj' label, microphone button for STT dictation (examination + finish)
    status: completed
  - id: modal-error-gate
    content: InterviewActions suppresses modal error display when cancelled=true so Cancel never shows 'Spróbuj ponownie'
    status: completed
  - id: tooling
    content: vitest.config.ts (jsdom env + @ alias); dev deps @testing-library/react, jsdom
    status: completed
  - id: verify
    content: pytest 43/43, vitest 31/31, eslint 0, tsc 0, next build OK
    status: completed
---

## Decisions

- **2026-09-11 — Polish JSON contract for unhandled provider failures.** `app.exception_handler(Exception)` in `backend/app/main.py` returns 502 on `httpx.HTTPError` and 503 otherwise, with `{"detail":{"code":"provider_error","message":"Nie udało się wykonać operacji. Spróbuj ponownie."}}`. Existing structured codes (`spend_cap_exceeded`, `conversation_completed`) still flow through unchanged. Tracebacks remain in the logs via `logging.exception`.
- **2026-09-11 — test fixture `sqlite_client_no_raise`.** TestClient constructor flag only, so a parallel fixture was added in `tests/conftest.py` to assert on handler output without re-raising.
- **2026-09-11 — `useAbortableAction` distinguishes user cancel from transport abort.** `cancel()` sets an internal `cancelled` flag and the catch handler short-circuits on `signal.aborted` (no error surface). Network `TypeError` still uses the fallback. A new `getError()` exposes `{message, code?}` for callers that want to branch on `provider_error` vs other codes.
- **2026-09-11 — `InterviewActions` modal gains inline status + microphone.** Each modal (examination + finish) owns its own `useVoiceController` so dictation is independent of the chat composer. `POST /api/voice/transcribe` is reused; the transcript is appended to the active textarea with the existing max-length cap. Cancel button no longer doubles as Stop and never displays the fallback after a user cancel.
- **Out of scope (deferred):** local SQLite bootstrap so `uvicorn` against a fresh DB does not 500 on first call. Deployment target is Render + Postgres where migrations run elsewhere; only blocks local repro.