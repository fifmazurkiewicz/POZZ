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