# ADR — Greenfield build; Streamlit reference only

**Date:** 2026-09-07  
**Status:** accepted

## Context

POZZ today is a Streamlit monolith (`app.py` + `modules/`) with Postgres, OpenRouter, Groq STT, and AWS/cloudflared hosting scripts. Streamlit cannot be the Vercel PWA. Lifting Streamlit onto Render would freeze the UX in a non-PWA server UI and skip Auth/RLS/ApiPulse.

Langy is the **documentation and hosting template**, not a code fork. Copying Langy’s Chat/Memo/FSRS would be the wrong product.

## Decision

- **No Streamlit production deploy** (not Vercel, not Render as the product UI, not Cloudflare quick tunnel as the canonical URL).
- **Streamlit = reference only** (prompts in `modules/prompt_manager.py`, schema in `modules/db.py`, tab flows in `app.py`).
- **Greenfield scaffold:** `backend/` (FastAPI) + `frontend/` (Next.js PWA) built from `docs/architecture-for-cursor.md` and UX specs.
- Keep/adapt/drop table (`.cursor/plans/2026-09-07-streamlit-keep-adapt-drop.md`) is a **reference map**, not an import checklist.
- Do not add new product features to `app.py` once Task 0 starts.

## Consequences

- Task 0 in implementation plans = greenfield MVP skeleton (health, auth, Simulation shell), not wrapping Streamlit.
- Legacy `start_up.sh` / `pozz-monitor.service` stay in git as historical ops; they are not the deploy playbook.
- `diarization_test/` stays experimental and out of MVP.
