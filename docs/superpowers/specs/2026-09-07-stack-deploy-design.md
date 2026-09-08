# Stack, deploy & voice — design (2026-09-07)

Approved hosting + provider contract. Synced into `docs/architecture-for-cursor.md` and ADRs.

## Hosting

| Piece | Choice |
|---|---|
| Frontend | Vercel → `pozz.fmazurkiewicz.dev` |
| Backend | Render Free (Docker) → `api-pozz.fmazurkiewicz.dev` |
| DB / Auth | Supabase |
| Cold start | Accepted; UI shows **Waking up…** until API is ready |
| Redis | **Not in MVP** — async jobs via **Postgres** |
| Langfuse | **Cloud**; **runtime SoT for prompts**; repo holds promptfoo fixtures only |
| promptfoo | Pre-deploy / CI against repo fixtures (after backend exists) |

## Voice (MVP) — **MODIFIED 2026-09-08**

- Default `VOICE_MODE=speech_to_speech`. `chained` still supported (forces lamp off).
- **Lamp ON:** Gemini Live (browser ↔ Live; short-lived token from API).
- **Lamp OFF:** STT (Groq) → LLM → patient TTS (ElevenLabs or browser). Never opens Live.
- Recorded Interview tab: capture → STT only (no talking patient).
- Full **transcript** in Postgres; no audio files in MVP.
- Text composer always available.
- Spec: `docs/superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`.

## Chat / simulation UX (listening)

- Status row: Live/TTS lamp left · Ready/Listening center · Głos pacjenta + Słuchanie dots right.
- Optional Listening (on/off). Not push-to-talk as the only path.
- iOS: turning Listening **on** (or Start recording) is the user gesture that unlocks the mic.
- Mute Listening while the patient speaks (TTS or Live).
- Cold API: **Uruchamianie API…** before recorder or send is usable.

## Admin

- Sole admin via `ALLOWED_ADMIN_EMAILS=fifmazurkiewicz@gmail.com` (`is_admin` on login).
- Approval queue, spend cap, bulk patient generate.

## Explicit non-goals for production

- AWS Secrets Manager as the secret store
- `cloudflared` quick tunnel as the public URL
- Streamlit `app.py` as the Render/Vercel entrypoint
