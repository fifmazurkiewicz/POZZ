---
name: 2026-09-08 package 0 scaffold
overview: FastAPI health + Next.js PWA shell with Simulation Live/TTS lamp chrome. Streamlit stays reference-only.
todos:
  - id: backend
    content: FastAPI /api/health + /api/voice/config + Dockerfile + pytest CI
    status: completed
  - id: frontend
    content: Next.js tabs, ApiPulse, LiveGeminiLamp preference
    status: completed
  - id: verify
    content: pytest, frontend lint/test/build
    status: completed
---

# Plan — Package 0 scaffold (working copy)

Canonical: [`docs/superpowers/plans/2026-09-08-package-0-scaffold.md`](../../docs/superpowers/plans/2026-09-08-package-0-scaffold.md).
Voice spec: [`docs/superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`](../../docs/superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md).

## Decisions

- **2026-09-11 — production runbook.** Operator SoT is `docs/technical/production-deploy.md`. Stack unchanged. Render must set `CORS_ORIGINS=https://pozz.fmazurkiewicz.dev`.
