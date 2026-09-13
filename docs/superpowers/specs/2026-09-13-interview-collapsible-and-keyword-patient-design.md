# Collapsible "Opis i plan" + keyword-driven patient generation

**Date:** 2026-09-13
**Status:** approved

Two small UI deltas on existing surfaces. No new backend endpoints, no schema changes.

## Out of scope

- Editing the generated "Opis i plan" document in place (existing decision: read-only, "Odśwież opis" regenerates).
- Persisting the Wywiad "Opis i plan" collapse state across page reloads.
- Filtering the existing shared catalog by keyword — keywords always generate a **private** patient.
- Backend changes — `POST /api/patients/next` already accepts an optional `keywords` string and routes to `pick_or_generate_patient`, which generates a private patient when keywords are non-empty (`backend/app/api/routes/patients.py:28-42`, `backend/app/patients/service.py:80-85`).

## Requirements (Given / When / Then)

### A. Wywiad — collapsible "Opis i plan"

1. **Given** a doctor with a generated `interview_summary` on a manual-interview conversation, **When** they open the Wywiad tab, **Then** the "Opis i plan" card shows the document body by default.
2. **Given** the card is open, **When** the doctor clicks **"Ukryj opis"**, **Then** the `<pre>` body is removed from view and the button label switches to **"Pokaż opis"**.
3. **Given** the card is hidden, **When** the doctor clicks **"Pokaż opis"**, **Then** the `<pre>` body reappears and the button label switches to **"Ukryj opis"**.
4. **Given** the card is hidden or shown, **When** the doctor clicks **"Odśwież opis"** or **"Kopiuj"**, **Then** those actions still work (the toggle button is independent).
5. **Given** the conversation has no `interview_summary` yet, **When** the doctor opens the Wywiad tab, **Then** the empty-state card with **"Generuj opis i plan"** is unchanged (no toggle before first generation).
6. **Given** the doctor navigates away and back, **When** the Wywiad tab is reopened, **Then** the card re-opens (collapse state is in-memory only).

### B. Symulacja — "Wygeneruj pacjenta" with keyword input

1. **Given** the Symulacja header has no active conversation, **When** the doctor types keywords (e.g. `zaburzenia neurologiczne`) into the new input and clicks **"Wygeneruj pacjenta"**, **Then** `fetchNextPatient(access, keywords)` is called with the trimmed keywords and a new simulation is opened on a freshly generated private patient.
2. **Given** the keyword input is empty (whitespace only), **When** the doctor clicks **"Wygeneruj pacjenta"**, **Then** it behaves exactly like the existing **"Następny pacjent"** button — picks an unused shared-catalog row or generates on demand.
3. **Given** an existing **"Następny pacjent"** button, **When** the doctor clicks it, **Then** it still calls `fetchNextPatient(access, undefined)` and is unchanged in behavior.
4. **Given** a busy state (existing `busy` flag), **When** the doctor types or clicks either button, **Then** the input and both buttons are disabled, and "Wygeneruj pacjenta" shows the busy indicator already in use on **"Następny pacjent"** (`useAbortableAction`).
5. **Given** the doctor presses **Escape** while focused on the keyword input, **When** the input is non-empty, **Then** the input clears (does not submit).

## Architecture

Two independent UI-only changes. Both reuse existing infrastructure:

| Surface | Component | New local state | Existing infrastructure |
|---|---|---|---|
| Wywiad "Opis i plan" | `frontend/src/app/interview/page.tsx` | `planOpen: boolean` (default `true`) | `generatePlan`, `copyPlan` already wired |
| Symulacja header | `frontend/src/components/simulation/SimulationClient.tsx` | `keywords: string` | `fetchNextPatient(access, keywords?, signal)` already accepts keywords |

`frontend/src/lib/simulation/api.ts` already exports:

```ts
export function fetchNextPatient(token: string, keywords?: string | null, signal?: AbortSignal)
```

No changes to `api.ts`. No new client function.

## UI

### A. Wywiad "Opis i plan" — generated state card

In the generated branch (around `frontend/src/app/interview/page.tsx:299-313`), keep the heading + body + the existing **"Odśwież opis"** / **"Kopiuj"** buttons, and add a third button **"Ukryj opis"** / **"Pokaż opis"** next to them.

- Buttons row order (right-aligned, 8 px gap): `[Ukryj/Pokaż] [Odśwież] [Kopiuj]`.
- When hidden, render only the heading + button row inside the `<section>`; omit the `<pre>`.
- A11y:
  - Toggle button gets `aria-expanded={planOpen}`, `aria-controls="interview-plan-body"`.
  - `id="interview-plan-body"` on the `<pre>` (only rendered when open).
  - Label text flips between "Ukryj opis" and "Pokaż opis".

### B. Symulacja header — keyword input + "Wygeneruj pacjenta"

Replace the single `<button>Następny pacjent</button>` in the header (around `frontend/src/components/simulation/SimulationClient.tsx:83`) with:

```tsx
<header>
  <div>
    <h1>Symulacja</h1>
    <p>Pacjent i ocena są generowane przez AI</p>
  </div>
  <div className="flex shrink-0 items-center gap-2">
    <input
      type="text"
      value={keywords}
      maxLength={500}
      disabled={busy}
      aria-label="Słowa kluczowe pacjenta"
      placeholder="np. zaburzenia neurologiczne, ból w klatce"
      onChange={(e) => setKeywords(e.target.value)}
      onKeyDown={(e) => { if (e.key === "Escape") setKeywords(""); }}
      className="min-h-11 w-44 rounded border border-[var(--color-divider)] bg-[var(--color-bg)] px-2 text-sm"
    />
    <button type="button" className="classical-btn text-sm" disabled={busy} onClick={nextPatient}>
      Następny pacjent
    </button>
    <button type="button" className="classical-btn classical-btn-primary text-sm" disabled={busy} onClick={generateFromKeywords}>
      {busy ? "Generowanie…" : "Wygeneruj pacjenta"}
    </button>
  </div>
</header>
```

- `keywords` state lives in the component (`useState<string>("")`).
- `generateFromKeywords` calls `fetchNextPatient(access, keywords.trim() || undefined, signal)` through the existing `run` action so cancel/busy/error reuse works.
- Both buttons funnel through the existing `run` callback — same abort/cancel/error behavior as the existing `nextPatient`.
- The **"Wygeneruj pacjenta"** button label flips to **"Generowanie…"** while `busy` (primary action affordance). **"Następny pacjent"** stays unchanged (existing behavior — `disabled={busy}` only).
- Keyboard: pressing **Escape** while focused on the input clears it (no-op when already empty).

## Tests (TDD — red first)

### A. Wywiad "Opis i plan"

Extend or create `frontend/src/app/interview/page.test.tsx` (if absent, create one covering the page smoke):

- `test_plan_card_hides_body_when_ukryj_clicked`: render page with a `session` that has `interview_summary`. Find the **"Ukryj opis"** button. Click it. Assert `<pre>` with the summary is no longer in the document. Assert button label is now **"Pokaż opis"**.
- `test_plan_card_shows_body_when_pokaz_clicked`: continue from above. Click **"Pokaż opis"**. Assert `<pre>` is back, button label is **"Ukryj opis"**.
- `test_plan_toggle_does_not_affect_refresh_and_copy`: with body hidden, click **"Kopiuj"** (mock `navigator.clipboard.writeText`); assert the summary text is written.
- `test_empty_state_card_has_no_toggle`: render page with `session.interview_summary = null`. Assert **"Ukryj opis"** / **"Pokaż opis"** buttons are absent; **"Generuj opis i plan"** is present.

### B. Symulacja keyword input

Extend `frontend/src/components/simulation/SimulationClient.test.tsx` (or create) with a mock of `fetchNextPatient`:

- `test_wygeneruj_with_keywords_passes_them_to_fetchNextPatient`: type `"zaburzenia neurologiczne"` into the input, click **"Wygeneruj pacjenta"**. Assert `fetchNextPatient` was called with those exact keywords (not `undefined`, not empty string).
- `test_wygeneruj_with_empty_input_behaves_like_nastepny`: input is empty. Click **"Wygeneruj pacjenta"**. Assert `fetchNextPatient` was called with `undefined` (no keywords path → shared catalog).
- `test_nastepny_pacjent_still_works`: click **"Następny pacjent"**. Assert `fetchNextPatient` was called with `undefined`.
- `test_escape_clears_keywords`: focus input, type `"x"`, press `Escape`. Assert input value is `""`; assert no fetch happened.
- `test_buttons_disabled_while_busy`: while `run` action is in flight, assert both buttons and the input are `disabled`.

## Decisions

- **2026-09-13 — collapse is local UI state**, no localStorage / URL persistence. Matches existing `cardOpen` precedent on Simulation.
- **2026-09-13 — two buttons in Symulacja header**: **"Następny pacjent"** (random catalog) and **"Wygeneruj pacjenta"** (keyword-driven). Both call `POST /api/patients/next`; only the second passes a non-empty `keywords` string. Keeping both preserves the existing affordance.
- **2026-09-13 — keywords always generate a private patient.** No keyword filtering on the shared catalog — out of scope, can be added later.
- **2026-09-13 — empty-state Wywiad card has no toggle** — there is no body to hide before first generation.
