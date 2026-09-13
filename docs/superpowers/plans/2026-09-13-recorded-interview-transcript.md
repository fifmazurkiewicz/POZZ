---
name: 2026-09-13 recorded interview transcript
overview: Upload a real interview recording, review the full transcript with inferred speakers, then generate a clinician-reviewed structured draft.
todos:
  - id: transcription-service
    content: Extract reusable in-memory Groq transcription service with Polish structured errors and 25 MB limit
    status: in_progress
  - id: recorded-api
    content: Add owned recorded-interview upload endpoint and persist transcript JSON without audio
    status: pending
  - id: plan-input
    content: Generate the clinical draft from a recorded transcript and expand medication/safety sections
    status: pending
  - id: frontend
    content: Add Nagranie/Ręcznie modes, file upload, inferred turns, raw transcript and clinical draft disclosure
    status: pending
  - id: tests
    content: Add backend and frontend contracts, then run full verification
    status: pending
---

## Decisions

- Reuse the existing `interview_transcripts.transcript_json` and `conversations.interview_summary` columns; no migration is needed.
- Preserve manual Interview as a separate sub-mode and leave Simulation untouched.
- Infer speaker turns with the text model but always preserve and display the raw STT text because the configured Whisper endpoint does not provide acoustic diarization.
- Store no audio in MVP (`audio_ref` remains null).
