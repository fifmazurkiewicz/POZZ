# POZZ backend

FastAPI on Render Docker. Postgres via `DATABASE_URL`. Do not store durable state on disk.

`app/` is the production package root. Keep FastAPI routes and provider
clients as outer adapters; put orchestration outside routes and keep clinical
product rules independent of FastAPI, database, and provider SDKs. The legacy
Streamlit prototype remains outside this boundary.

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
python -m pytest
```

Health: `GET /api/health` (liveness), `GET /api/health/ready` (DB URL present). Voice: `GET /api/voice/config`. Env names: `docs/technical/configuration.md`.
