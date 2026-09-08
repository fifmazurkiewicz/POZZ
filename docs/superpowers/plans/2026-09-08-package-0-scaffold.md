# Package 0 — Greenfield scaffold + Live/TTS chrome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit-only run path with a FastAPI + Next.js PWA skeleton that already shows the Simulation Live/TTS lamp chrome.

**Architecture:** `backend/` Dockerized FastAPI: liveness, readiness (no DB URL → 503), voice config. `frontend/` Next.js App Router: three-tab nav, ApiPulse, Simulation status row with lamp + preference in localStorage. Streamlit stays at repo root as reference. No Live connect and no TTS playback in this package.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, pydantic-settings, pytest, Next.js 16, React 19, TypeScript, Tailwind 4, Vitest.

## Global Constraints

- Stack is Vercel + Render + Supabase only (`deployment-standard.mdc`).
- No Redis. No durable disk. No Streamlit as the production entrypoint.
- Product UI copy is Polish; identifiers and agent-written code are English.
- `GET /api/health` returns `{"status":"ok","service":"pozz"}`.
- `GET /api/voice/config` returns `voice_mode`, `live_available` (`voice_mode == "speech_to_speech"`), `tts_provider`.
- Live/TTS lamp: ON = Live, OFF = TTS; key `pozz-sim-live-gemini`; default ON.
- `DEV_AUTH_ENABLED` never set on Render. Python 3.12, Node 22.
- Taste dials VARIANCE 3 / MOTION 2 / DENSITY 6.
- Do not read or commit `.env` files other than `*.env.example`.

---

### Task 0: FastAPI health + voice config + Dockerfile

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pytest.ini`
- Create: `backend/app/__init__.py`
- Create: `backend/app/settings.py`
- Create: `backend/app/main.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/router.py`
- Create: `backend/app/api/routes/__init__.py`
- Create: `backend/app/api/routes/health.py`
- Create: `backend/app/api/routes/voice.py`
- Create: `backend/tests/test_health.py`
- Create: `backend/tests/test_voice_config.py`
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`
- Modify: `backend/AGENTS.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/dependabot.yml`

**Interfaces:**
- Consumes: `DATABASE_URL`, `VOICE_MODE`, `TTS_PROVIDER`, `CORS_ORIGINS`
- Produces: `GET /api/health` → `{status, service: "pozz"}`; `GET /api/health/ready` → 200 or 503; `GET /api/voice/config` → `{voice_mode, live_available, tts_provider}`

- [x] **Step 1–6:** Implemented in this PR (pytest + Docker + CI backend job).

---

### Task 1: Next.js PWA shell + ApiPulse + Live/TTS lamp chrome

**Files:**
- Create: `frontend/package.json` and App Router app
- Create: pulse provider/banner, BottomNav, Simulation status row + `LiveGeminiLamp`
- Create: `frontend/src/lib/voice/liveGeminiPreference.ts` + test
- Modify: `frontend/AGENTS.md`, CI frontend job, Dependabot npm `/frontend`

**Interfaces:**
- Consumes: `NEXT_PUBLIC_API_URL`; `GET /api/health`; `GET /api/voice/config`
- Produces: `useApiPulse()`; Polish nav Symulacja / Wywiad / Menu; lamp preference helpers

- [x] **Step 1–4:** Implemented in this PR (lint, vitest, build).

---

Later packages (do not implement in this PR): schema/auth, text simulation, chained STT+TTS, Live token, evaluation, Interview tab, Sessions/Admin, promptfoo — see `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md`.
