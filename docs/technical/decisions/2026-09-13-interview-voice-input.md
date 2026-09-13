# ADR: Voice input (STT only) on Wywiad (Interview) tab

**Status:** accepted (2026-09-13)
**Supersedes:** n/a (additive — narrows the "text-only Wywiad" implication of
`docs/superpowers/specs/2026-09-08-refactor-build-order-design.md` Task 6
but does not replace it).

## Context

The Wywiad (Interview) tab is the doctor-private manual case composer +
recorded one-shot surface (`docs/superpowers/specs/2026-09-08-refactor-build-order-design.md`
Task 6). It currently exposes only a text input and a "Wyślij" button.

Doctors practising POZ interviews benefit from being able to **dictate**
their questions rather than type. The Symulacja surface already wires
`useVoiceController` + `POST /api/voice/transcribe` for the same UX. The
back-end contract, error shape, and authentication are already in place.

## Decision

Add a **mic-only** speech-to-text input to the Wywiad composer:

- Tap "Mikrofon" → record via `MediaRecorder` (same `VoiceController` used
  by Symulacja).
- Tap again → finalize recording → `POST /api/voice/transcribe` with the
  recorded webm blob → append recognised text to the existing text input.
- Doctor reviews / edits, then taps "Wyślij" — **no auto-submit**.
- Button disabled when the session is not active, `busy` is true, or
  `voice.speaking` (defense in depth; we never call `voice.play` here).

Out of scope: patient TTS, Gemini Live, splitting the tab, storing audio
blobs. Those are explicitly deferred.

## Why

- Explicit user request (Session 2026-09-13) to dictate the manual case.
- Reuses existing building blocks (`useVoiceController`,
  `/api/voice/transcribe`, `ConversationComposer` style) — no new backend
  code, no new env vars, no new dependencies.
- Stays consistent with `docs/superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`
  for auth, error shape, and spend-cap behavior (STT counts toward the
  monthly cap — already documented in the spec).

## Consequences

- Wywiad is **no longer text-only** — needs a doc note in
  `docs/ux/ux-ui-spec.md` and a clarification of Task 6 in
  `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md`.
- Same `spend_cap_exceeded` / `provider_error` Polish error contract as
  Symulacja; toasts and inline errors continue to use the backend's
  `{ code, message }`.
- Recorded one-shot capture (real audio persisted alongside the
  transcript) remains a future task and is still listed in Task 6.

## Alternatives considered

- **Full voice surface (mic + patient TTS + lamp).** Rejected: makes
  Wywiad a second Symulacja, violating the "Recorded interview is a
  separate surface from the simulated patient chat" rule.
- **TTS read-aloud of patient replies only.** Rejected: Wywiad's "patient"
  is the LLM via `/api/conversations/{id}/turns`; reading its answers
  aloud is the same as Symulacja's TTS and confuses the two surfaces.
- **Split into "Manual" + "Recorded" tabs.** Deferred by user choice
  (see plan Decisions §2). Recorded one-shot capture stays a future
  Task 6 sub-task.

## Follow-up: Phase 2 — dictation on the create-case form (2026-09-13)

Doctor asked to also dictate the **case title** and **patient scenario
description** before tapping "Rozpocznij wywiad". Decision:

- Add a second mic button on the create-case form, one next to the title
  and one next to the scenario textarea. Each appends the recognised
  text to its own field (same UX as the running-interview mic).
- Reuse `useInterviewVoiceInput` with a per-field callback (the hook is
  field-agnostic). No new hook.
- Polish error contract and `spend_cap` behavior inherited from Phase 1.
- Out of scope: shared `MediaRecorder` (two instances may coexist; the
  browser decides if both can capture at once — accepted for MVP).

See `.cursor/plans/2026-09-13-interview-voice-input.md` Phase 2 section
for the full Given/When/Then and risks.

## Follow-up: Phase 3 — live recording on Nagranie tab (2026-09-13)

Doctor asked for the Wywiad "Nagranie" surface to record the encounter
in the browser rather than upload a pre-recorded file:

> "Po 1 chciałbym mieć mniejsze te przyciski. Tryb recorded ma
> wyglądać że apka nagrywa wywiad który na żywo i transkrybuje a
> następnie robimy analizę itp. Czyli nie chce wrzucać gotowego
> nagrania ale ma to być nagrywane."

Decision:

- Replace the file-upload form on the "Nagranie" tab with a live
  recorder (`RecordedRecorder` component). Mic lifecycle is owned by
  the same `useVoiceController` used elsewhere; on stop, the resulting
  `Blob` is POSTed to the existing `POST /api/interviews/recordings`
  endpoint (no backend change — the endpoint already accepts a
  multipart audio blob and runs STT + speaker diarization).
- Title field stays (optional, `maxLength=120`).
- No "Opis sytuacji" on the Nagranie tab — that field lives on the
  Ręcznie tab; the recording itself is the description on this
  surface.
- Mode toggle shrinks (`text-sm` + `px-3`, `p-0.5` group) to match
  the rest of the page; touch-target size 44 px is not required on
  this desktop clinical surface.
- New component `frontend/src/components/interview/RecordedRecorder.tsx`
  encapsulates the recorder state machine (idle → recording →
  submitting). `useInterviewVoiceInput` is unchanged — it's still
  the right hook for short dictation + `/api/voice/transcribe`.
- Animated recording dot uses `motion-safe:animate-pulse` and
  degrades to a static dot under `prefers-reduced-motion`.

Why:

- Explicit user request.
- Reuses every existing building block (`useVoiceController`,
  `/api/interviews/recordings`, `uploadRecordedInterview`). No new
  backend code, no new env vars, no new dependencies.

Consequences:

- The legacy upload form is gone. Files already uploaded via the
  previous flow remain visible in sessions (`kind="recorded_interview"`
  rows continue to load and render their existing transcript).
- The doctor still sees a "Tytuł (opcjonalny)" field — backend uses
  it if present, falls back to a timestamped filename otherwise
  (auto-generated client-side as `nagranie-YYYYMMDD-HHMMSS.webm`).
- No audio is persisted client-side (matches MVP "no durable audio
  blobs" rule).

Out of scope (still deferred):

- Pause / resume during a recording.
- Live waveform visualisation.
- Streaming / chunked upload.
- Storing audio blobs.

See `.cursor/plans/2026-09-13-interview-recorded-live-capture.md`
for the implementation plan and test contract.