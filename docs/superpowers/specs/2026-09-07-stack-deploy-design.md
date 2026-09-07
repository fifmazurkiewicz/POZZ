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

## Voice (MVP)

- `VOICE_MODE=chained` only.
- Doctor (or recorded interview) mic → short WAV → Groq Whisper (`STT_PROVIDER=groq`) → text.
- LLM replies are **text**. No patient TTS. No Gemini Live.
- Full **transcript** stored in Postgres; no audio files in MVP.
- Text composer always available.

## Chat / simulation UX (listening)

- Optional recorder control (start/stop). Not a required per-turn mic.
- Phase 2 may add VAD hands-free / Live; do not block MVP.
- iOS: the Start recording tap is the user gesture that unlocks the mic.
- Cold API: **Waking up…** before recorder or send is usable.

## Admin

- Sole admin via `ALLOWED_ADMIN_EMAILS=fifmazurkiewicz@gmail.com` (`is_admin` on login).
- Approval queue, spend cap, bulk patient generate.

## Explicit non-goals for production

- AWS Secrets Manager as the secret store
- `cloudflared` quick tunnel as the public URL
- Streamlit `app.py` as the Render/Vercel entrypoint
