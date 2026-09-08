# POZZ backend

FastAPI on Render Docker. Postgres via `DATABASE_URL`. Do not store durable state on disk.

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
python -m pytest
```

Health: `GET /api/health` (liveness), `GET /api/health/ready` (DB URL present). Voice: `GET /api/voice/config`. Env names: `docs/technical/configuration.md`.
