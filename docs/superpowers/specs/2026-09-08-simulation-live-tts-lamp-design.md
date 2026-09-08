# Simulation Live vs TTS lamp — design (2026-09-08)

**Status:** approved (product request: rewrite Streamlit + add TTS and a Gemini Live switcher)

Replaces the 2026-09-07 “no TTS / no Live in MVP” decision. Synced into `docs/architecture-for-cursor.md` and `docs/ux/ux-ui-spec.md`.

## Problem

The Streamlit prototype is text-out only (doctor mic → STT → LLM text). Doctors training POZ interviews need to **hear** the patient. They also need a single control to pick **Gemini Live** (speech-to-speech) vs a fully independent **chained TTS** path — same metaphor as Langy’s Chat lamp.

## Goals

- Status-row lamp (left): **ON (lit) = Gemini Live**, **OFF = chained TTS** (STT → LLM → patient TTS).
- TTS path never opens Live (`live-token` / connect / PCM).
- Persist preference in `localStorage` key `pozz-sim-live-gemini`.
- Switch mid-session: OFF disconnects Live immediately; ON reconnects when Patient voice is on.
- Text composer **always** present.
- Mute / pause Listening while the patient speaks (TTS or Live) so the doctor mic does not capture patient audio.

## Non-goals (this spec)

- Profile/server sync of the lamp preference.
- Chunked live suggestions during recorded Interview.
- Diarization GPU (`diarization_test/`).
- Patient TTS on the **recorded Interview** tab (that tab is capture + STT, not a talking patient).

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-08 | Lamp on Simulation status row, left | Always visible; matches Langy Chat chrome without copying Chat IA |
| 2026-09-08 | Lit = Live, unlit = TTS | Same metaphor as Langy |
| 2026-09-08 | `localStorage` `pozz-sim-live-gemini` | Persist without backend |
| 2026-09-08 | Default ON | Matches `VOICE_MODE=speech_to_speech` |
| 2026-09-08 | Env `VOICE_MODE` still caps Live | `chained` forces lamp off / disabled |
| 2026-09-08 | Polish TTS | Patient replies in Polish; `TTS_PROVIDER=elevenlabs` (product) or `browser` |
| 2026-09-08 | `GET /api/voice/config` | Frontend learns `voice_mode`, `live_available`, `tts_provider` |
| 2026-09-08 | Spend cap counts TTS + ASR + GenAI | Same as Langy ledger |

## Architecture

```
Doctor text or mic
        │
        ├─ lamp ON  and live_available → Gemini Live (browser ↔ Live; Render = auth, agenda, tools, persist)
        └─ lamp OFF → POST /api/stt (if audio) → POST /api/simulation/turns → TTS (ElevenLabs or browser)
```

- `readLiveGeminiPreference` / `writeLiveGeminiPreference`.
- `LiveGeminiLamp`: hit target ≥44px, `aria-pressed`, name “Live Gemini”.
- Simulation chrome: locked header (patient summary + Next) · locked status row (lamp left · Ready/Listening center · Patient-voice + Listening dots right) · scroll-only transcript · locked composer + bottom nav.
- `connectLive` no-ops when lamp OFF or `live_available` is false.
- `withMicSuspended` around patient TTS / Live playback.

## Given / When / Then

| Given | When | Then |
|---|---|---|
| Lamp ON, `live_available`, Patient voice on | Session active | Simulation may connect Gemini Live |
| Lamp OFF | Session active | No Live token; turns use text/STT + patient TTS |
| Live connected | User turns lamp OFF | Live disconnects immediately; further turns stay on TTS path |
| Preference in localStorage | User revisits Simulation | Lamp matches saved state |
| `VOICE_MODE=chained` | UI loads | Lamp disabled/off; TTS path only |
| Listening on + patient TTS | Patient audio plays | Recognition suspended until utterance ends |
| Listening on + Live PCM | Patient audio plays | Recognition suspended until turnComplete + idle |
| Active Simulation | User scrolls | Only transcript moves; header, status row, composer, nav stay fixed |

## Copy (Polish UI)

- Lamp title: „Live Gemini” / „Tylko TTS”
- Dots: **Głos pacjenta** · **Słuchanie**
- Banner: „Uruchamianie API…”
