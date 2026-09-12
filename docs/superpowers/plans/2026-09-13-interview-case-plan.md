---
name: 2026-09-13 interview case plan
overview: Interview tab sub-feature — generate a structured SOR-style "opis i plan" document from a doctor-authored manual interview and persist it on the conversation.
todos:
  - id: tdd-test
    content: TDD test_interview_plan.py — POST /conversations/{id}/plan generates, persists interview_summary, 409/404/403 guards
    status: completed
  - id: prompt
    content: create_case_description_prompt in prompts/simulation.py (WYWIAD / ROZPOZNANIE RÓŻNICOWE / ZALECANE BADANIA / PLAN POSTĘPOWANIA, Polish, no invented facts)
    status: completed
  - id: backend
    content: POST /plan endpoint in conversations.py (open-only, ownership, spend cap) + interview_summary in conversation_payload
    status: completed
  - id: frontend-api
    content: generateCasePlan client + SimulationSession.interview_summary + api.test.ts contract test
    status: completed
  - id: frontend-ui
    content: interview/page.tsx "Opis i plan" card (Generuj → Odśwież/Kopiuj) wired to generateCasePlan
    status: completed
  - id: verify
    content: pytest 47/47, vitest 32/32, eslint 0, next build OK, browser smoke on /interview
    status: completed
---

## Decisions

- **2026-09-13 — feature lives in the Interview tab only.** The "Opis i plan" generator is a sub-feature of the doctor-authored manual interview workspace (`kind="manual_interview"`). The Simulation tab and its InterviewActions are untouched.
- **2026-09-13 — single POST regenerates.** `POST /api/conversations/{id}/plan` always regenerates from the latest transcript and overwrites `conversation.interview_summary`. The UI's "Generuj opis i plan" button becomes "Odśwież opis" once a document exists — same endpoint, same click.
- **2026-09-13 — persist to existing unused column.** `Conversation.interview_summary` (added earlier, never written) is reused; no migration, no new columns. `conversation_payload` now always includes `interview_summary`.
- **2026-09-13 — no speaking.** The speaking/reading-aloud aspect was discussed and explicitly deferred by the user for this iteration.