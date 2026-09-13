# Recorded interview transcription and clinical draft

## Goal

The **Wywiad** tab supports a real doctor-patient audio file as a separate workflow from simulated patient chat. An approved doctor uploads one recording, reviews the complete Polish transcript, and then explicitly asks POZZ to draft clinical documentation and recommendations.

## User flow

1. Choose **Nagranie** or **Ręcznie** in the Interview tab.
2. In **Nagranie**, select one audio file up to 25 MB and optionally enter a title.
3. POZZ transcribes the whole file in memory. The audio file is not stored.
4. POZZ infers Doctor/Patient turns from the transcript text. Because this is not acoustic diarization, the UI labels speaker attribution as AI-inferred and keeps the untouched full transcript visible.
5. The doctor reviews the transcript and clicks **Generuj opis i plan**.
6. POZZ drafts Polish sections for interview summary, differential diagnosis, investigations, medications, non-drug recommendations, safety-netting and next steps. The UI identifies the output as a draft requiring clinician verification.

## API contract

- `POST /api/interviews/recordings` accepts multipart `audio` and optional `title`.
- It enforces approval, ownership scope, spend cap, non-empty input and the 25 MB limit.
- It creates a private `recorded_interview` conversation, persists one `interview_transcripts` row containing `raw_text` and inferred `turns`, and never persists audio bytes or `audio_ref`.
- Conversation detail and plan responses include `recorded_transcript` for recorded interviews.
- `POST /api/conversations/{id}/plan` uses the recorded transcript when the conversation has no manual chat messages.

## Privacy and safety

- Audio is sent to the configured STT provider and held only in memory.
- Transcript and generated draft are per-user data protected by existing ownership checks and RLS.
- Medication content is a proposal, not an automatically accepted prescription.
- Raw transcript remains visible so incorrect speaker attribution or wording can be caught before clinical use.

## Out of scope

- Durable audio storage.
- True acoustic speaker diarization or speaker enrollment.
- Live chunked suggestions during recording.
- Automatic prescribing or submission to an EHR.
