# Wywiad — voice input (mic → STT) (2026-09-13)

**Status:** draft (supercedes the "Wywiad is text-only" decision in
`docs/superpowers/specs/2026-09-08-refactor-build-order-design.md` Task 6).

## Goal

Add a **mic-only** speech-to-text input on the Wywiad (Interview) tab so the
doctor can dictate questions instead of typing. No patient TTS, no Live lamp —
this stays a private manual case composer + recorded one-shot surface.

## Why now

- Doctor asked for the ability to speak their manual case (Session 2026-09-13).
- All building blocks exist: `useVoiceController`, `POST /api/voice/transcribe`,
  `ConversationComposer` pattern. Wiring is the missing piece.
- The Wywiad tab already has a text input + "Wyślij"; adding a mic is additive.

## Requirements (Given / When / Then)

1. **Given** a doctor on `/interview` with an active session, **when** they
   tap the new "Mikrofon" button, **then** the browser requests mic permission
   and the button label switches to "Wyślij nagranie" (matches Symulacja).
2. **Given** recording is in progress, **when** the doctor taps the button
   again, **then** the recording is finalized and sent to
   `POST /api/voice/transcribe` (multipart `audio` field).
3. **Given** STT returns `{ "text": "…" }`, **when** the response resolves,
   **then** the recognised text is **appended** to the existing `draft`
   input (whitespace separator, trim) — the doctor reviews/edits before
   tapping "Wyślij". No auto-submit.
4. **Given** STT returns an error (4xx/5xx, network, abort), **when** the
   promise rejects, **then** the existing `setError` surface shows the
   Polish message and `draft` is left untouched. The button returns to
   "Mikrofon".
5. **Given** the doctor has not yet started a session (no `session`), **when**
   the form is on the create-case step, **then** the mic button is **hidden**
   (no point recording before the case exists).
6. **Given** the session is `ended_at` set or `busy` is true, **when** the
   doctor looks at the composer, **then** the mic button is disabled (same
   pattern as `Wyślij`).
7. **Given** voice controller reports `listening === true`, **when** the
   component renders, **then** the status line shows "Nagrywanie…" (Polish,
   reuse `text-xs text-[var(--color-soft)] role="status"`).
8. **Given** voice controller reports `speaking` (only possible if a future
   caller invokes `voice.play`), **when** rendering, **then** the mic button
   is disabled — we never have a patient voice on this surface, but defense
   in depth.
9. **Given** the test `InterviewVoice.test.tsx` mounts the page with a
   session, **when** it stubs `MediaRecorder` + `fetch /api/voice/transcribe`
   and simulates a recording cycle, **then** the input value contains the
   recognised text and `Wyślij` is enabled.

## Contract (already on backend)

| Direction | Path | Method | Body | Response |
|---|---|---|---|---|
| FE → BE | `/api/voice/transcribe` | POST | `multipart/form-data; audio=<webm blob>` | `{ "text": "…" }` |

Auth: `Authorization: Bearer <access>`. Errors follow the same `{ code, message }`
shape the rest of the app uses. Reuse `useVoiceController.startRecording` +
manual `fetch(apiUrl("/api/voice/transcribe"), …)` mirroring
`SimulationClient.submitTurn(audio)` (`frontend/src/components/simulation/SimulationClient.tsx:60-72`).

## Plan

1. **Red test #1 — pure helper.** New file
   `frontend/src/lib/voice/transcribeAudio.ts` exporting
   `transcribeAudio(token, blob, signal?) => Promise<{ text: string }>`
   using the same `fetch` + `apiUrl` + `{ code, message }` error shape as
   `SimulationClient.submitTurn(audio)` (`frontend/src/components/simulation/SimulationClient.tsx:60-72`).
   Test `transcribeAudio.test.ts`: multipart body, auth header, abort
   before fetch, 4xx `ApiError` with Polish message, 200 happy path.
2. **Red test #2 — hook.** New file
   `frontend/src/lib/voice/useInterviewVoiceInput.ts` exporting
   `useInterviewVoiceInput({ token, appendTranscript })` returning
   `{ listening, busy, error, toggle }`. Internals: `useVoiceController`
   + `useAbortableAction` + `transcribeAudio`. When toggled on → call
   `voice.startRecording((blob) => run(transcribe → append))`; toggled off
   → `voice.finishRecording()`. Second toggle-on cancels prior in-flight
   transcription (`useAbortableAction` already does this).
   Test `useInterviewVoiceInput.test.tsx` (the **first** component/hook
   test in the repo): stub `MediaRecorder` + `getUserMedia` + `fetch`,
   render with `renderHook`, assert (a) `listening` toggles, (b) on
   finishRecording, `appendTranscript` is called with the recognised text,
   (c) second toggle-on cancels prior fetch, (d) `error` surfaces on 4xx.
3. **Wire component** — in `frontend/src/app/interview/page.tsx`:
   - import the hook + `useVoiceController` (just for the listening
     visual state — already returned by the hook).
   - add `const voice = useInterviewVoiceInput({ token, appendTranscript: (t) => setDraft((d) => (d ? `${d} ${t}` : t)) });`
   - replace the bare `<input> + <button>Wyślij</button>` row with the same
     `flex gap-2` row, plus a secondary "Mikrofon" button beside "Wyślij"
     with `aria-pressed={voice.listening}`, label flips
     `voice.listening ? "Wyślij nagranie" : "Mikrofon"`. Same SVG icon as
     `ConversationComposer`. A small status line under the row shows
     "Nagrywanie…" / "Transkrypcja…" (Polish).
   - hide the mic button entirely when `!session` (req #5).
   - disable mic when `busy || session.ended_at || voice.speaking || voice.busy`.
4. **Cleanup** — `useVoiceController`'s `useEffect` cleanup already
   disposes on unmount. `useAbortableAction`'s unmount effect aborts.
5. **Verify** — `npm run lint && npm test && npm run build` from
   `frontend/`. Manually smoke in the browser: create a case, hit mic,
   dictate, confirm text appears in input, hit Wyślij.
6. **Docs** — update `docs/ux/ux-ui-spec.md` Wywiad section (note: now
   also offers mic input). Update `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md`
   Task 6 line ("manual constructor") — append "(+ dictation via mic)".

## Decisions

- **2026-09-13** — Wywiad gets a mic button. Scope: STT-only, no patient TTS,
  no lamp. Spec/ADR: `docs/technical/decisions/2026-09-13-interview-voice-input.md`.
  Why: explicit user request; reuse existing voice controller; no extra
  backend work.
- **2026-09-13** — Do **not** split Wywiad into "Manual" + "Recorded" tabs
  now. Why: user opted to defer the split; the spec already lists Recorded
  Interview as a separate surface but not as a separate tab. Keep one tab,
  add mic. Recorded one-shot capture (real audio blob stored alongside the
  transcript) remains a future task — see
  `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md` Task 6.

## Out of scope (deferred)

- Patient TTS on Wywiad (text-only).
- Gemini Live on Wywiad.
- Splitting into two tabs.
- Storing audio blobs (`audio_ref` column on conversations).

## Risks

- Mic permission denied → `voice.error("Brak dostępu do mikrofonu.")`
  already handled by `VoiceController`. Test path covers.
- STT endpoint fails (e.g. `spend_cap_exceeded`, `provider_error`) →
  existing `fetch` → `ApiError` path catches; show Polish message via
  `setError`.
- Two recordings fired quickly → `startRecording` calls `stop()` first,
  which aborts any pending request. Safe.
- Test environment lacks `MediaRecorder` → vitest stubs needed (same as
  `voiceController.test.ts`).

---

## Phase 2 — dictation on the create-case form (2026-09-13 follow-up)

**Scope delta.** Doctor asked to also dictate the **case title** and
**patient scenario description** on the form (before "Rozpocznij wywiad").
This is additive — the mic on the running-interview composer (Phase 1)
stays exactly as is.

### Decisions

- **2026-09-13 (follow-up)** — Two independent mic buttons on the
  create-case form: one next to `Tytuł przypadku`, one next to
  `Opis pacjenta i sytuacji`. Each appends the recognised text to its
  own field (the doctor reviews/edits, like Phase 1).
- **2026-09-13 (follow-up)** — Reuse the existing `useInterviewVoiceInput`
  hook with a per-field `appendTranscript` callback
  (`setTitle` / `setScenario`). No new hook needed — the hook is already
  field-agnostic.
- **2026-09-13 (follow-up)** — `Rozpocznij wywiad` activation condition
  stays `scenario.trim().length >= 20`. Dictated text counts toward that
  limit immediately after STT resolves (no extra guard).
- **2026-09-13 (follow-up)** — Mic disabled when `busy` (form submit in
  flight). Polish error from STT (e.g. `spend_cap_exceeded`) routes to the
  existing inline banner (same path as Phase 1).
- **2026-09-13 (follow-up)** — No recording two fields at once:
  starting the title mic while scenario mic is `listening` is allowed
  (each is its own hook); but if both fire close together, the page
  shows both listening states (each hook has its own `MediaRecorder`).

### Requirements (Given / When / Then)

1. **Given** the create-case form is shown, **when** the doctor taps the
   mic next to `Tytuł przypadku`, **then** recording starts and the button
   label flips to "Wyślij nagranie".
2. **Given** recording on the title field, **when** the doctor taps the
   button again, **then** the recognised text is **appended** to the
   title input (whitespace separator, trim).
3. **Given** the create-case form, **when** the doctor taps the mic next
   to `Opis pacjenta i sytuacji`, **then** recording starts independently
   of the title mic (two `useInterviewVoiceInput` instances coexist).
4. **Given** STT fails on the scenario mic, **when** the error resolves,
   **then** the existing inline error banner shows the Polish message
   and the textarea stays untouched.
5. **Given** `busy` is true (form submit), **when** the doctor looks at
   the form, **then** both mic buttons are disabled.
6. **Given** dictated text brings `scenario.length` to ≥ 20, **when** the
   doctor reviews the form, **then** `Rozpocznij wywiad` is enabled.

### Plan (Phase 2)

1. **Doc sync** — update ADR `2026-09-13-interview-voice-input.md` (add
   Phase 2 section) + UX spec ("Wywiad" bullet mentions both mics).
2. **Test delta** — extend `useInterviewVoiceInput.test.ts` with one
   test that mounts two hooks in the same component and proves they
   don't cross-contaminate (append goes to the right callback). Reuses
   the existing fake MediaRecorder (each `useVoiceController` instance
   gets its own).
3. **Wire page** — in `frontend/src/app/interview/page.tsx`:
   - instantiate `titleVoice = useInterviewVoiceInput({ token, appendTranscript: setTitle })`
   - instantiate `scenarioVoice = useInterviewVoiceInput({ token, appendTranscript: (t) => setScenario((s) => (s ? `${s} ${t}` : t)) })`
   - render two mic buttons (title + scenario), same SVG / classes as
     Phase 1, with their own status lines.
   - mirror errors via the page-level `banner = voice.error ?? error`
     union (combine `titleVoice.error || scenarioVoice.error`).
4. **Verify** — `npm run lint && npm test && npm run build`. Smoke in
   the browser: fill title + scenario by voice, hit Rozpocznij wywiad.
5. **Commit + push**.

### Risks (Phase 2)

- Two `useVoiceController` instances → two `MediaRecorder`s; the OS may
  only allow one mic capture at a time. UX implication: the second tap
  may silently fail. We accept this for MVP; a future follow-up could
  share a single recorder and route to whichever field is active.
- `setTitle` as append: dictating twice appends. Doctor can clear the
  field between dictations if they want a clean replacement.
- Larger `scenario` textarea grows past 12000 char `maxLength` → STT
  appends after the cap will be silently dropped by the textarea
  (browser native behavior). Acceptable.