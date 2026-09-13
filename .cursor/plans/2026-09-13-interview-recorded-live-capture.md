---
name: 2026-09-13 interview recorded live capture
overview: Wywiad (Interview) tab — "Nagranie" mode becomes a live recorder (mic → STT + diarization) instead of a file-upload form; smaller mode-toggle buttons.
todos: []
---

## Goal

Change the "Nagranie" surface on the Wywiad (Interview) tab from "upload a
pre-recorded audio file" to **live recording in the browser** (tap to
record, tap to stop & transcribe). Keep the rest of the tab intact.
Shrink the mode-toggle buttons to match the rest of the page.

## Why now

- Doctor asked (Session 2026-09-13, ongoing): "apka nagrywa wywiad który na
  żywo i transkrybuje a następnie robimy analizę itp. Czyli nie chce
  wrzucać gotowego nagrania ale ma to być nagrywane." The current
  upload form is the wrong UX for that.
- All building blocks exist: `useVoiceController` (unlimited-length
  recording), `POST /api/interviews/recordings` (STT + diarization +
  creates the conversation), `uploadRecordedInterview` FE wrapper.
  Wiring is the missing piece.

## Architecture

- **No backend change.** `POST /api/interviews/recordings` already
  accepts a `multipart/form-data` upload of an audio `Blob` plus an
  optional `title`. Live recording → in-memory `Blob` → same endpoint.
- **Reuse `useVoiceController`.** It's already designed for
  arbitrary-length capture (no per-chunk truncation, no timeouts).
- **Inline a small recorder widget** on the Wywiad page in place of the
  `<input type="file">` block. The existing `useInterviewVoiceInput`
  hook is for short dictation + `/api/voice/transcribe`; we want
  arbitrary-length + `/api/interviews/recordings`, so this widget uses
  `useVoiceController` directly with a custom submit callback
  (`uploadRecordedInterview`).
- **Smaller mode toggle.** Drop the `min-h-11` (44 px) from the
  `Nagranie` / `Ręcznie` buttons; they become `text-sm` matching the
  "Nowy przypadek" button elsewhere on the page. Tighter pill group
  (`p-0.5` instead of `p-1`).
- **Title field stays.** One optional "Tytuł (opcjonalny)" input above
  the recorder, same shape as before.
- **No "Opis sytuacji"** on the Nagranie tab — that's only on Ręcznie.

## Tech Stack

- Frontend only: React + existing Tailwind tokens +
  `useVoiceController` + `uploadRecordedInterview`.

## Global Constraints

- Voice/STT wiring uses `useVoiceController` (single source of truth for
  mic lifecycle, abort, cleanup). No ad-hoc `MediaRecorder` in the page.
- Polish-only UI strings. Polish error messages from backend are
  surfaced as-is via the existing inline banner.
- Tailwind tokens from `frontend/src/app/globals.css` only — no
  hard-coded colors.
- Mic-disabled states must match the existing pattern
  (`busy || voice.listening || voice.busy`).
- `prefers-reduced-motion` already enforced by the design dials (this
  feature is essentially static — only an animated recording dot, gated
  on reduced motion).

## File map

| File | Change |
|---|---|
| `frontend/src/app/interview/page.tsx` | Replace upload form with live-recorder UI; shrink mode toggle; remove unused `audioFile` state |
| `frontend/src/lib/voice/useVoiceController.ts` | Unchanged (already supports arbitrary-length) |
| `frontend/src/lib/interview/api.ts` | Unchanged (already accepts a `Blob`-shaped `File`) |
| `frontend/src/lib/interview/api.test.ts` | Extend contract test to cover `Blob` input (not just `File`) |
| (new) `frontend/src/components/interview/RecordedRecorder.tsx` | Standalone live-recorder widget — Mic permission, idle / recording / transcribing states |
| (new) `frontend/src/components/interview/RecordedRecorder.test.tsx` | Component-level test (state transitions, abort on cancel, error surface) |

---

## Tasks

### Task 1 — `useInterviewVoiceInput` Blob contract + RecordedRecorder component

**Files:**
- Create: `frontend/src/components/interview/RecordedRecorder.tsx`
- Create: `frontend/src/components/interview/RecordedRecorder.test.tsx`
- Modify: `frontend/src/lib/interview/api.ts` (allow `Blob` instead of `File` if needed)
- Modify: `frontend/src/lib/interview/api.test.ts`

**Interfaces:**
- `<RecordedRecorder token={string} title?: string onCreated: (session: SimulationSession) => void onError: (msg: string) => void />`
- Internal: `useVoiceController`; on `finishRecording` → `uploadRecordedInterview(token, blob, title)`.

- [ ] **Step 1.1 — Write the failing component test**

  Render `RecordedRecorder`, stub `MediaRecorder` + `getUserMedia` (same
  pattern as `useVoiceController.test.ts`). Simulate the idle → recording
  → transcribing → created flow. Assert:
  - Idle: button shows "Rozpocznij nagrywanie".
  - Recording: button shows "Zatrzymaj i transkrybuj"; timer ticks.
  - After `stop`: button shows "Transkrypcja…"; spinner visible.
  - On success: `onCreated` fires with the session.

- [ ] **Step 1.2 — Run test to verify it fails**

  Run: `cd frontend && npm test -- --run RecordedRecorder.test`
  Expected: FAIL — module not found.

- [ ] **Step 1.3 — Implement `RecordedRecorder`**

  Self-contained client component. Reuses `useVoiceController`. On
  `finishRecording` callback: calls
  `uploadRecordedInterview(token, blob, title)`; on success →
  `onCreated(session)`; on error → `onError(message)` (Polish from
  `ApiError.message`).

  States:
  - `idle`: full-width primary button "Rozpocznij nagrywanie" + status
    helper text.
  - `listening`: pulsing red dot, `mm:ss` timer, button "Zatrzymaj i
    transkrybuj".
  - `busy`: muted spinner + "Transkrypcja i rozpoznawanie rozmówców…".

  Reduced-motion: replace pulsing dot with a static one.

- [ ] **Step 1.4 — Run test to verify it passes**

  Run: `cd frontend && npm test -- --run RecordedRecorder.test`
  Expected: PASS.

- [ ] **Step 1.5 — Extend `api.test.ts` to cover `Blob`**

  Add a test that calls `uploadRecordedInterview(token, new Blob([...]),
  "Tytuł")` and asserts the FormData `audio` field is a `Blob` (not
  required to be a `File`).

- [ ] **Step 1.6 — Run `api.test.ts`**

  Run: `cd frontend && npm test -- --run interview/api.test`
  Expected: PASS (no change needed if `api.ts` already typed `File` — a
  `Blob` is assignable to it; otherwise loosen the parameter type to
  `Blob`).

- [ ] **Step 1.7 — Commit**

  ```bash
  git add frontend/src/components/interview/ frontend/src/lib/interview/api.ts frontend/src/lib/interview/api.test.ts
  git commit -m "feat(interview): RecordedRecorder — live mic capture for Nagranie tab"
  ```

### Task 2 — Wywiad page wiring + smaller mode toggle

**Files:**
- Modify: `frontend/src/app/interview/page.tsx`

- [ ] **Step 2.1 — Shrink the mode toggle**

  Replace the `min-h-11` buttons with `text-sm`, tighter padding
  (`px-3`), tighter group padding (`p-0.5`). Visually identical
  treatment to the secondary "Nowy przypadek" button on the same page.

- [ ] **Step 2.2 — Replace upload form with `RecordedRecorder`**

  - Drop `audioFile` state and the `uploadRecording` handler.
  - In the `mode === "recorded"` branch, render a card containing:
    - `<h2>Transkrypcja nagrania</h2>`
    - `<p>Dodaj pełne nagranie rozmowy lekarza z pacjentem. Plik jest przetwarzany w pamięci i nie jest zapisywany.</p>`
    - "Tytuł (opcjonalny)" input (`maxLength={120}`, same `np. Wizyta kontrolna` placeholder).
    - `<RecordedRecorder token={access} title={title} onCreated={setSession} onError={setError} />`
  - When `session` is created: also `setMessages([])` (matches the
    existing `uploadRecording` behavior).

- [ ] **Step 2.3 — Run lint + test + build**

  Run: `cd frontend && npm run lint && npm test -- --run && npm run build`
  Expected: PASS.

- [ ] **Step 2.4 — Browser smoke**

  Spin up backend + frontend; navigate to `/interview`; toggle to
  Nagranie; tap "Rozpocznij nagrywanie"; speak for ~5 s; tap "Zatrzymaj
  i transkrybuj"; confirm the conversation appears with a transcript.

- [ ] **Step 2.5 — Commit**

  ```bash
  git add frontend/src/app/interview/page.tsx
  git commit -m "feat(interview): Wywiad Nagranie — live recorder; shrink mode toggle"
  ```

### Task 3 — Docs sync

**Files:**
- Modify: `docs/technical/decisions/2026-09-13-interview-voice-input.md` (add Phase 3 follow-up section)
- Modify: `docs/ux/ux-ui-spec.md` (Wywiad section — Nagranie is now live capture)
- Modify: `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md` Task 6 line (note: recorded one-shot capture now lives in the Wywiad tab as live mic capture)

- [ ] **Step 3.1 — ADR Phase 3**

  Append Phase 3 to the existing ADR:
  - Phase 3 — live recording on Nagranie tab. Decision: replace file
    upload with live recorder (mic → Blob → same endpoint). Why:
    explicit user request. Consequences: existing upload form is gone;
    `useInterviewVoiceInput` stays unchanged (it's for short
    dictation); a new `RecordedRecorder` component owns the
    recorder-state machine.

- [ ] **Step 3.2 — UX spec update**

  In the Wywiad section, replace "Nagranie — upload WAV/MP3/..." with
  "Nagranie — nagrywanie na żywo (przycisk mikrofonu → stop →
  transkrypcja)."

- [ ] **Step 3.3 — Commit**

  ```bash
  git add docs/
  git commit -m "docs(interview): Nagranie tab is now live recorder"
  ```

---

## Decisions

- **2026-09-13 — Live recorder on Nagranie tab.** Replaces the file-upload
  form. Same backend endpoint, in-memory `Blob`. No persistent audio
  storage (matches MVP constraint).
- **2026-09-13 — Title input stays on Nagranie.** Same shape as before
  (`maxLength=120`, optional). Justified: doctor often knows the
  encounter name in advance.
- **2026-09-13 — No "Opis sytuacji" on Nagranie.** The scenario lives in
  the recorded audio; adding a separate description field would be
  noise on this surface.
- **2026-09-13 — New `RecordedRecorder` component instead of extending
  `useInterviewVoiceInput`.** The hook is for short dictation +
  `/api/voice/transcribe`; this is arbitrary-length +
  `/api/interviews/recordings`. Different submit endpoint and longer
  UX (timer, animated state) — worth a dedicated component.
- **2026-09-13 — Mode toggle shrinks.** `text-sm` + `px-3` instead of
  `min-h-11`. Matches the rest of the page; no 44 px touch-target
  requirement on this surface (it's a desktop clinical tool, not a
  tablet).
- **2026-09-13 — Animated dot respects `prefers-reduced-motion`.**
  Static dot under reduce.

## Out of scope (deferred)

- Pause / resume during a recording.
- Live waveform visualisation.
- Streaming / chunked upload to backend.
- Storing audio blobs.
- Patient TTS on the recorded conversation (the doctor already knows
  the patient's voice from the live encounter).