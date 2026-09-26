# JEV voice-turn detection — design

## Goal

Add an optional **Rozmowa** mode to Simulation. It turns a spoken doctor
utterance into one ordinary simulation turn without a separate Send action.
After an initial 2.5 seconds of captured speech, JEV determines whether the
transcribed utterance is complete. A `continue` result captures another second
and repeats the check. The maximum capture duration is 15 seconds.

The feature is a conversation-boundary detector only. It makes no clinical,
diagnostic, triage, treatment, or training-evaluation decision.

## User flow

1. The Simulation composer offers a keyboard-accessible `Wiadomości | Rozmowa`
   toggle. `Wiadomości` remains the default and keeps the current controls.
2. In `Rozmowa`, the doctor starts capture. The UI announces that it is
   listening and exposes a visible stop/cancel control.
3. After 2.5 seconds, the client submits the current in-memory audio to the
   voice-turn checkpoint endpoint.
4. The endpoint transcribes the submitted bytes and asks JEV whether the text
   is a completed Polish utterance. `complete` returns the normalized text;
   `continue` returns no user-visible turn.
5. On `continue`, the client captures one more second and checks the newly
   accumulated audio again. It repeats until `complete` or 15 seconds.
6. On `complete`, the client posts that text to the existing conversation-turn
   endpoint. The existing patient response and TTS playback then run unchanged.
   Listening never overlaps patient playback.
7. Any STT/JEV failure, ambiguous answer, cap failure, cancellation, or the
   15-second limit discards the local audio and partial text. The UI announces
   a Polish error asking the doctor to repeat the utterance; it never posts a
   partial conversation message.

## API and privacy

`POST /api/voice/turn-check` is authenticated and approved-user only. It takes
one multipart `audio` field and returns either:

```json
{ "decision": "complete", "text": "..." }
```

or:

```json
{ "decision": "continue" }
```

The server processes uploaded audio and transcripts in memory only. It passes
only the current transcript to OpenRouter Decisions; it stores neither audio,
partial text, nor an audit copy of the transcript. The existing monthly spend
cap is checked for every checkpoint. A malformed, unavailable, or uncertain
JEV response is a structured Polish error, not a fallback completion.

## Implementation boundaries

- Add a narrowly scoped JEV `voice_turn_complete` decision helper alongside
  existing JEV integration, with a fixed model and an English binary question.
- Reuse the existing Groq transcription path rather than introducing a browser
  speech-recognition dependency or durable audio storage.
- Extend the browser voice controller to retain its current `MediaRecorder`
  stream in memory, emit checkpoints at 2.5 seconds then 1-second increments,
  and stop/release tracks on every terminal path.
- Keep existing manual mic recording, text composer, TTS, and conversation API
  behavior intact.
- Make the message transcript a deliberate scrolling region: automatically
  follow new turns only while the reader is at the bottom; expose a labelled
  jump-to-latest control when they have scrolled away.

## Failure and accessibility requirements

- Every error reaches the existing Polish JSON error boundary and is announced
  through `role=alert`; UI copy never exposes raw provider errors.
- Switching mode, ending an interview, selecting a new patient, or pressing
  Stop cancels timers/requests, discards audio, and releases the microphone.
- The mode control uses native buttons with `aria-pressed`. Listening and
  checkpointing status use a polite live region. Controls remain usable by
  keyboard and on narrow mobile screens.

## Verification

- Backend tests mock STT and Decisions for complete, continue, uncertain,
  provider failure, approval/auth, and no durable partial transcript.
- Frontend/controller tests cover the 2.5-second first check, 1-second retries,
  15-second limit, stop/mode change cleanup, and no message post on error.
- Component tests cover mode state, accessible status/error, automatic
  scroll-follow, and the jump-to-latest control.
- Run backend pytest plus frontend lint, tests, and build before push.

## Non-goals

- Gemini Live, browser-native speech recognition, server-side audio storage,
  clinical decisions, and a change to the text-message workflow.
