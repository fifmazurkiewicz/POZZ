# Local setup

## Prerequisites

- Python 3.12+
- Node.js 22+
- Postgres (local) **or** Supabase Cloud (needed from Package 1)
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

## 3. Greenfield API + PWA (Package 0)

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/api/health
# {"status":"ok","service":"pozz"}
```

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — redirects to Symulacja. Bottom nav: Symulacja / Wywiad / Menu. Lamp on the status row (Live vs TTS preference). Banner „Uruchamianie API…” until health is 200.

Auth, next-patient, and spoken audio are later packages.

## 4. Smoke test (Package 0)

1. `GET /api/health` returns `{"status":"ok","service":"pozz"}`
2. `GET /api/voice/config` includes `live_available: true` when `VOICE_MODE=speech_to_speech`
3. Frontend `/simulation` shows the lamp; toggle persists in `localStorage` (`pozz-sim-live-gemini`)

## 5. Supabase OAuth redirect URLs

Add to Supabase → Authentication → URL Configuration → **Redirect URLs**:

- `https://pozz.fmazurkiewicz.dev/auth/callback`
- `http://localhost:3000/auth/callback` (local)

## 6. Production alignment

Step-by-step cutover: [production-deploy.md](./production-deploy.md).

- Frontend: Vercel (`pozz.fmazurkiewicz.dev`)
- Backend: Render Docker (`api-pozz.fmazurkiewicz.dev`)
- `DATABASE_URL` via Supavisor pooler on Render
- `SPEND_CAP_TZ=Europe/Warsaw`
- Render Root Directory `backend`, Runtime **Docker**
- Render `CORS_ORIGINS` includes `https://pozz.fmazurkiewicz.dev`
