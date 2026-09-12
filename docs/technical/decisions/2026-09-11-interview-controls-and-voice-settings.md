# ADR — Interview controls and voice settings

**Date:** 2026-09-11  
**Status:** accepted; supersedes the September 8 status-row lamp UI

## Context

Simulation and manual Interview need the same safe controls for interrupting work, ordering scenario-constrained examinations and completing an interview. The checked-out frontend implements chained speech, while the earlier lamp design described a Gemini Live client that does not yet exist.

## Decision

- Both conversation surfaces expose **Zatrzymaj**, **Zrób badanie** and **Zakończ wywiad**.
- Stop aborts the browser request, stops audio/browser speech and microphone capture, and discards unfinished input. It cannot guarantee cancellation of synchronous provider work already running on the server.
- `POST /api/conversations/{id}/examinations` accepts `{ "examination": string }`, generates a scenario-constrained text result and persists both request and result in message history. It does not speak the result or reveal hidden scenario gold.
- `POST /api/conversations/{id}/finish` accepts `{ "treatment_plan": string }`, persists the plan, evaluation and `ended_at`, and makes the conversation read-only. Failed evaluation leaves it open. Missing reference gold is generated independently from the scenario.
- The fixed viewport shell scrolls only the transcript. Composer, conversation actions and bottom navigation remain visible above the mobile safe area.
- Labeled microphone and speaker icons replace opaque voice-status dots near the composer.
- Live/TTS selection moves to Menu. Until a Gemini Live client/socket exists, Live is visibly unavailable and TTS is selected.
- Menu stores an optional validated ElevenLabs voice ID in local browser storage. Empty means the server default.
- Admin relies on the global bottom navigation and has no separate top-right Menu button.

## Consequences

Completed conversations remain readable but reject turns and examinations. Aborting a browser request protects the interface from late results even when server-side synchronous work continues. The application does not claim functional Live voice based only on server configuration.
