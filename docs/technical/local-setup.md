# Local setup

## Prerequisites

- Python 3.12+
- Node.js 22+ (after greenfield frontend exists; Supabase JS requires `>=22`)
- Postgres (local) **or** Supabase Cloud dev project
- `uv` optional for the Streamlit prototype

## 1. Environment

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

Fill OpenRouter (+ Groq for STT). Variable **names**: [configuration.md](./configuration.md). Never copy values from a real `.env` into chat or git.

For UI-only dev without Supabase (after scaffold), leave `NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` empty **and** set `DEV_AUTH_ENABLED=true` with an empty `SUPABASE_URL` in `.env`. The API rejects `dev-token` without that opt-in.

## 2. Legacy Streamlit prototype (today)

Works **now**, without `backend/` / `frontend/` code:

```bash
uv sync
# DATABASE_URL → local Postgres; OPENROUTER_API_KEY set
uv run streamlit run app.py --server.port 8501
```

Open http://localhost:8501 — tabs Symulacja / Wywiad / Przeglądanie / Admin. Mic needs a secure context (localhost is OK).

This path is **not** the production deploy. Do not point Vercel or Render at `app.py`.

## 3. Database (greenfield)

After Task 0, apply `supabase/migrations/001_initial.sql` in Supabase SQL editor, **or** locally:

```bash
cd backend
python -m pip install -r requirements.txt
# set DATABASE_URL in .env (Postgres)
python -m scripts.create_tables
```

## 4. Backend (greenfield)

```bash
cd backend
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/api/health
```

Expected: `{"status":"ok","service":"pozz"}`.

## 5. Frontend (greenfield)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — Login (dev) → waiting screen until admin Accept (allowlist / `dev-token` auto-approved on first insert) → Simulation / Interview / Menu.

## 6. Smoke test (greenfield, once scaffold exists)

1. Health returns `{"status":"ok","service":"pozz"}`
2. Sign in → approved
3. Simulation → Next patient → one text turn → End interview → write a plan → see evaluation
4. Sessions lists that conversation

## 7. Supabase OAuth redirect URLs

Add to Supabase → Authentication → URL Configuration → **Redirect URLs**:

- `https://pozz.fmazurkiewicz.dev/auth/callback`
- `http://localhost:3000/auth/callback` (local)

## 8. Production alignment

- Frontend: Vercel (`pozz.fmazurkiewicz.dev`)
- Backend: Render Docker (`api-pozz.fmazurkiewicz.dev`)
- `DATABASE_URL` via Supavisor pooler on Render
- `SPEND_CAP_TZ=Europe/Warsaw`
- Render Root Directory `backend`, Runtime **Docker**
