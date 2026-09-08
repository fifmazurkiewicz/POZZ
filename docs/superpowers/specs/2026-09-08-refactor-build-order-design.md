# Streamlit rewrite + voice — what next (2026-09-08)

**Status:** approved order. First vertical slice after this plan: Package 0 (this PR’s code).

The Streamlit app is **not** ported file-by-file. Production is greenfield Next.js + FastAPI. Prompts and flows come from `app.py` / `modules/`.

## Why this order

Voice (TTS + Live) sits **on** Simulation turns. Turns need auth, a patient, and a conversation. Those need a running API and a PWA shell. So: shell → schema/auth → text simulation → chained STT/TTS → Live lamp wiring → evaluation → recorded interview → admin.

Do **not** start Gemini Live before a text turn works. Do **not** rewrite Streamlit in place.

## Packages

| # | Package | Done when |
|---|---|---|
| **0** | Scaffold | `GET /api/health` + `GET /api/voice/config`; Next.js tabs; ApiPulse; Live/TTS lamp (preference only); CI pytest + frontend lint/test/build |
| **1** | Schema + auth | `001_initial.sql`; `dev-token` / `/api/auth/me`; AuthGate waiting screen |
| **2** | Simulation text | Next patient, card, modes, text turns, transcript — **no** Streamlit |
| **3** | Chained voice | Mic → STT; patient TTS (`TTS_PROVIDER`); mute Listening during TTS |
| **4** | Gemini Live | Ephemeral token; lamp ON connects; lamp OFF never opens Live |
| **5** | End interview | Gold plan hidden until evaluation; retry |
| **6** | Interview tab | Recorded one-shot + manual constructor |
| **7** | Sessions + Admin | Per-user history; approval; spend cap; bulk generate |
| **8** | promptfoo | Critical prompts in CI |

Package 0 is this change. Packages 1–8 are later PRs (one package per PR unless a package is tiny).

## Explicitly later (not “next”)

Chunked live suggestions, diarization GPU, audio blob storage, resume `conversation_id`, product-UI database wipe.
