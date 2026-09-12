# Interview case plan generator — POZZ

**Date:** 2026-09-13
**Status:** approved

A sub-feature of the **Interview (Wywiad) tab** for **doctor-authored manual cases** (`POST /api/patients/manual` → `kind="manual_interview"`). After a doctor writes their own interview (form + text chat), they can click **"Generuj opis i plan"** to produce a structured SOR-style document (WYWIAD / ROZPOZNANIE RÓŻNICOWE / ZALECANE BADANIA / PLAN POSTĘPOWANIA) generated from the transcript. The document persists on the conversation and can be regenerated ("Odśwież opis") and copied.

## Out of scope

The Simulation tab and its `InterviewActions` bar; speaking the document aloud; evaluation/finish for manual cases; hidden-gold interplay; editing the generated document in place.

## Requirements (Given / When / Then)

1. **Given** an approved, open manual-interview conversation owned by the caller, **When** they `POST /api/conversations/{id}/plan`, **Then** the server generates a Polish structured case description from `patient.scenario` + the full transcript, persists it to `conversation.interview_summary`, and returns the updated conversation payload containing `interview_summary`. The LLM call must not reveal hidden diagnosis/treatment gold beyond what the transcript states.
2. **Given** a completed conversation (`ended_at` set), **When** they call `/plan`, **Then** the API returns 409 `conversation_completed` (matching `/turns` semantics).
3. **Given** a caller who does not own the conversation, **When** they call `/plan`, **Then** the API returns 404 (no ownership leak).
4. **Given** a user at/over their monthly spend cap, **When** they call `/plan`, **Then** the API returns 403 `spend_cap_exceeded`.
5. **Given** a doctor with an open manual interview, **When** they open the Interview tab, **Then** the "Opis i plan" card is present; before first generation it shows **"Generuj opis i plan"**; after generation it shows the document plus **"Odśwież opis"** and **"Kopiuj"** buttons.

## API

New endpoint in `backend/app/api/routes/conversations.py`:

- `POST /api/conversations/{conversation_id}/plan`
  - Auth: approved user + ownership (`get_owned_conversation_for_update`).
  - Guards: open-only (`_assert_open`), spend cap (`assert_under_cap`).
  - Calls `get_text_provider().complete(create_case_description_prompt(patient_scenario, chat_history))`.
  - Persists the stripped result to `conv.interview_summary`, commits, returns `conversation_payload`.

`conversation_payload` (in `patients/service.py`) now always includes `interview_summary` (was previously absent/unused).

## Prompt

New `create_case_description_prompt(*, patient_scenario, chat_history)` in `backend/app/prompts/simulation.py`:

- System: announces a doctor documenting an emergency-department (SOR / izba przyjęć) case; instructs Polish output with exactly the four sections (`## WYWIAD`, `## ROZPOZNANIE RÓŻNICOWE`, `## ZALECANE BADANIA`, `## PLAN POSTĘPOWANIA`); forbids inventing facts outside the transcript/scenario and forbids leaking hidden diagnosis/gold.
- User: `OPIS SYTUACJI:\n{scenario}\n\nPRZEBIEG WYWIADU:\n{transcript}` where transcript is `role: content` lines of all messages.

## UI

In `frontend/src/app/interview/page.tsx`, beneath the transcript `<ol>` and above the composer:

- Empty state card (aria-label `Opis i plan`): heading "Opis i plan", helper text, primary button **"Generuj opis i plan"** → `generateCasePlan(access, conversation_id)`; disabled + "Generowanie…" while in flight.
- Generated state card: heading "Opis i plan", buttons **"Odśwież opis"** (same action) and **"Kopiuj"** (`navigator.clipboard.writeText(interview_summary)`), with the document rendered in a bordered `<pre className="whitespace-pre-wrap">`.
- Errors use the page's existing `error` line (Polish, `role="alert"`).

New client in `frontend/src/lib/simulation/api.ts`:

```ts
export function generateCasePlan(token: string, id: string, signal?: AbortSignal) {
  return apiFetch<SimulationSession>(`/api/conversations/${id}/plan`, {
    method: "POST", token, signal, body: {},
  });
}
```

`SimulationSession` gains `interview_summary?: string | null`.

## Tests

- `backend/tests/patients/test_interview_plan.py`
  - generates + persists `interview_summary`, prompt contains scenario + transcript, no `scenario`/`treatment_plan` leakage;
  - 409 on completed; 404 on cross-user ownership; 403 `spend_cap_exceeded`.
- `frontend/src/lib/simulation/api.test.ts` — `generateCasePlan` POSTs to `/plan` with auth headers and returns `interview_summary`.

## Decisions

- **2026-09-13 — Interview tab only.** The generator is a sub-feature of the doctor-authored manual interview workspace; the Simulation tab and its InterviewActions are untouched.
- **2026-09-13 — single POST regenerates.** `POST /plan` always regenerates from the latest transcript and overwrites `interview_summary`. "Odśwież opis" is the same endpoint after first generation.
- **2026-09-13 — persist to existing unused column.** Reuses `Conversation.interview_summary`; no migration, no new columns.
- **2026-09-13 — no speaking.** Reading the document aloud was discussed and explicitly deferred by the user for this iteration.