# ADR — Simulation Live vs TTS lamp

**Date:** 2026-09-08  
**Status:** superseded on 2026-09-11 by `2026-09-11-interview-controls-and-voice-settings.md`

## Context

The Streamlit prototype never spoke the patient. The 2026-09-07 contract deferred TTS and Gemini Live. The product now requires both, plus a single switcher, while rewriting the UI off Streamlit.

## Decision

- Simulation status-row lamp: ON = Gemini Live, OFF = chained STT + patient TTS.
- `VOICE_MODE` still caps Live (`speech_to_speech` vs `chained`).
- Preference in `localStorage` (`pozz-sim-live-gemini`), default ON.
- Recorded Interview tab stays capture + STT (no talking patient).
- Build order: scaffold → auth → text simulation → chained voice → Live → evaluation → Interview → admin.

## Consequences

Package 0 ships lamp chrome without connecting Live or playing TTS. Packages 3–4 add the audio backends. Spend cap counts TTS + ASR + GenAI.

The lamp UI was later removed in favor of voice settings in Menu. The checked-out product has chained TTS but no functional Gemini Live browser client/socket.
