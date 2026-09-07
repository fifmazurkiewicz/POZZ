# Greenfield scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit-only run path with a FastAPI + Next.js PWA skeleton on the Vercel/Render/Supabase contract, without deleting the Streamlit prototype.

**Architecture:** `backend/` is a Dockerized FastAPI app with liveness/readiness, CORS, and JWT-ready settings. `frontend/` is a Next.js App Router PWA shell with AuthGate placeholder, ApiPulse, and three-tab nav. Schema lives in `supabase/migrations/001_initial.sql`. Streamlit stays at repo root as reference.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, SQLAlchemy, PyJWT, Next.js (App Router), TypeScript, Supabase JS (optional until keys exist), GitHub Actions.

## Global Constraints

- Stack is Vercel + Render + Supabase only (`deployment-standard.mdc`).
- No Redis. No durable disk. No Streamlit as the production entrypoint.
- Product UI copy is Polish; identifiers and agent-written code are English.
- `GET /api/health` returns `{"status":"ok","service":"pozz"}`.
- `DEV_AUTH_ENABLED` never set on Render. Python 3.12, Node 22.
- Taste dials VARIANCE 3 / MOTION 2 / DENSITY 6. Do not add a parallel color rule file.
- Do not read or commit `.env` files other than `*.env.example`.

---

### Task 0: FastAPI health + Dockerfile

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/settings.py`
- Create: `backend/tests/test_health.py`
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`
- Modify: `backend/AGENTS.md` (replace placeholder with the same commands once files exist)
- Modify: `.github/workflows/ci.yml` (add backend job)
- Modify: `.github/dependabot.yml` (pip `backend/`)

**Interfaces:**
- Consumes: env names from `docs/technical/configuration.md`
- Produces: `GET /api/health` → `{status: "ok", service: "pozz"}`; `GET /api/health/ready` → 200 `{"status":"ok"}` when `DATABASE_URL` connects, else 503

- [ ] **Step 1: Write the failing health test**

```python
# backend/tests/test_health.py
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "pozz"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_health.py -v`

Expected: FAIL (module `app.main` not found) until Step 3.

- [ ] **Step 3: Write minimal FastAPI app**

`backend/requirements.txt`:

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sqlalchemy>=2.0.36
psycopg[binary]>=3.2.1
pyjwt[crypto]>=2.9.0
httpx>=0.27.0
pydantic-settings>=2.6.0
pytest>=8.3.0
```

`backend/app/settings.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = ""
    database_url: str = ""
    dev_auth_enabled: bool = False
    allowed_admin_emails: str = ""
    cors_origins: str = "http://localhost:3000"


settings = Settings()
```

`backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.settings import settings

app = FastAPI(title="POZZ")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pozz"}


@app.get("/api/health/ready")
def ready():
    if not settings.database_url:
        return JSONResponse({"status": "degraded", "reason": "no_database_url"}, status_code=503)
    return {"status": "ok"}
```

- [ ] **Step 4: Dockerfile (Render)**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`: `__pycache__`, `.venv`, `tests`, `.env`

Health check path for Render: `/api/health`. Root Directory: `backend`. Runtime: Docker.

- [ ] **Step 5: Run tests and wire CI**

Run: `cd backend && pip install -r requirements.txt && python -m pytest -q`

Expected: PASS.

Add to `.github/workflows/ci.yml` a `backend` job: Python 3.12, `pip install -r backend/requirements.txt`, `python -m pytest -q` with `working-directory: backend`.

Dependabot: add `package-ecosystem: pip` directory `/backend`.

- [ ] **Step 6: Commit**

```bash
git add backend .github/workflows/ci.yml .github/dependabot.yml
git commit -m "feat: add FastAPI health skeleton for Render"
```

---

### Task 1: Supabase initial schema

**Files:**
- Create: `supabase/migrations/001_initial.sql`
- Create: `backend/scripts/create_tables.py` (optional local apply via SQLAlchemy text of the same DDL, or document SQL editor only — prefer one file: the migration is SoT)

**Interfaces:**
- Consumes: data model in `docs/architecture-for-cursor.md` §6
- Produces: tables `users`, `usage_ledger`, `patients`, `conversations`, `messages`, `interview_transcripts`, `interview_suggestions`, `jobs`, `patient_user_state` + RLS policies

- [ ] **Step 1: Write `001_initial.sql`**

Include `create extension if not exists pgcrypto;` and `users.is_approved boolean not null default false`. Enable RLS on user-owned tables with `auth.uid() = user_id`. `patients` select for `authenticated`; insert restricted to service role (backend). `jobs` no client access.

- [ ] **Step 2: Local apply**

Run against local Postgres or Supabase SQL editor. Do not commit connection strings.

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/001_initial.sql
git commit -m "feat: add initial Supabase schema for POZZ"
```

---

### Task 2: Next.js PWA shell + ApiPulse

**Files:**
- Create: `frontend/package.json` and App Router app (`src/app/layout.tsx`, `src/app/page.tsx`, `src/app/simulation/page.tsx`, `src/app/interview/page.tsx`, `src/app/menu/page.tsx`)
- Create: `frontend/src/lib/api/pulse.ts`, `ApiPulseProvider`, `ApiPulseBanner`
- Create: `frontend/src/components/BottomNav.tsx`
- Modify: `frontend/AGENTS.md`
- Modify: CI frontend job + Dependabot npm `/frontend`

**Interfaces:**
- Consumes: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`); `GET /api/health`
- Produces: `useApiPulse()` → `{ isHealthy, isWaking, status, checkNow }`; Polish labels Symulacja / Wywiad / Menu; banner „Uruchamianie API…”

- [ ] **Step 1: Scaffold Next.js**

`npx create-next-app@latest frontend --typescript --app --src-dir --no-tailwind` is allowed if followed by Taste/Classical tokens. Prefer matching Langy’s pulse contract, not Langy’s Chat screens.

Pulse intervals: 5 s unhealthy, 30 s healthy, 8 s abort.

- [ ] **Step 2: Bottom nav + placeholder routes**

Default `/` redirects to `/simulation`. Bottom nav locked. Empty states in Polish.

- [ ] **Step 3: Lint, test, build**

Run: `cd frontend && npm run lint && npm test && npm run build`

Expected: PASS. Add CI job Node 22.

- [ ] **Step 4: Commit**

```bash
git add frontend .github
git commit -m "feat: add Next.js PWA shell with API pulse"
```

---

### Task 3: Dev auth + AuthGate waiting screen

**Files:**
- Create: `backend/app/auth.py` (`get_current_user`, `require_approved`)
- Create: `backend/app/routers/auth.py` (`GET /api/auth/me`)
- Create: `frontend/src/components/AuthGate.tsx`
- Test: `backend/tests/test_auth_me.py`

**Interfaces:**
- Consumes: `DEV_AUTH_ENABLED`, empty `SUPABASE_URL` → accept `Authorization: Bearer dev-token` as user `00000000-0000-0000-0000-000000000001`, email `dev@localhost`, `is_approved=true`
- Produces: `/api/auth/me` → `{ id, email, is_admin, is_approved, spend_cap_usd }`; production with `SUPABASE_URL` set ignores `DEV_AUTH_ENABLED`

- [ ] **Step 1: Failing test** — without header, `/api/auth/me` is 401; with `dev-token` and flag, 200 approved.

- [ ] **Step 2: Implement** JWT JWKS path as a stub that 501s until Supabase keys exist; document in `local-setup.md`.

- [ ] **Step 3: AuthGate** — Polish waiting copy when `is_approved === false`; poll 15s.

- [ ] **Step 4: Commit**

```bash
git add backend frontend docs/technical/local-setup.md
git commit -m "feat: add dev-token auth and approval waiting gate"
```

---

### Task 4: Simulation next-patient + text turn (first vertical slice)

**Files:**
- Create: `backend/app/routers/patients.py`, `backend/app/routers/simulation.py`
- Create: prompt wrappers that port `generate_patient_scenario_prompt` / `create_simulation_prompt` from `modules/prompt_manager.py` (reimplement in `backend/app/prompts/`; do not import Streamlit)
- Create: frontend Simulation page wired to REST
- Test: generate + turn with mocked LLM

**Interfaces:**
- `POST /api/patients/next` `{ keywords?: string }` → `{ patient_id, card, conversation_id }` (no `scenario`, no `treatment_plan` in the client payload)
- `POST /api/conversations/{id}/turns` `{ text, mode }` → `{ assistant_text }`
- Hidden gold plan stays server-side

After this task a doctor can run a text-only interview locally. Mic, evaluation, recorded interview, admin generate follow as later plans (do not expand this file).

---

## Self-review

1. Spec coverage: constitution docs are a separate already-landed change. This plan covers architecture §9 steps 1–6 (scaffold, schema, shell, gate, providers start, simulation text). STT, evaluation, recorded interview, admin bulk, promptfoo are **next plans**, not placeholders inside Task 4.
2. No TBD in task steps 0–3. Task 4 names exact routes.
3. Health JSON `service: pozz` is consistent across architecture, AGENTS.md, and Task 0.
