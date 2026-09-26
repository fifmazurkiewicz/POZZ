# Superseded: JEV voice-turn detection

Superseded on 2026-09-26. Simulation no longer determines whether a spoken
utterance is complete and has no automatic send path.

The single Simulation composer accepts typed text or microphone input. Stopping
the microphone transcribes its in-memory recording into the editable draft.
The doctor can edit the draft and must explicitly select **Wyślij** to create a
conversation turn. No JEV request or server endpoint is involved in deciding a
voice-turn boundary.

The existing `/api/voice/transcribe` path continues to process audio in memory
only. On a transcription failure, the draft is preserved and no turn is sent.
