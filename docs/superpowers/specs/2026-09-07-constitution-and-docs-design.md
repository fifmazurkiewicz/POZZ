# Constitution, docs, and repo hygiene — design (2026-09-07)

**Status:** approved (this change implements it)

## Problem

POZZ could not be deployed on Vercel / Render / Supabase as-is: no Cursor constitution, no `docs/` contract, Streamlit+AWS as the only runbook, no secret scanning, no env-name catalog.

## Decision

Copy Langy’s **process and hosting constitution**, rewrite the **product** contract for the POZ simulator.

| Artifact | Role |
|---|---|
| `.cursor/rules/*` | Superpowers, Graft, Taste, language (EN agent / PL UI), deployment standard, secrets |
| `AGENTS.md` | Commands + stack + learned facts |
| `docs/architecture-for-cursor.md` | Business + technical SoT |
| `docs/ux/*` | Behavior + Classical clinical DS |
| `docs/technical/*` | local-setup, configuration names, ADRs |
| `docs/superpowers/specs/*` | Journey, stack, approval gate |
| `.env.example` + `frontend/.env.example` | Placeholders only |
| CI Gitleaks + Dependabot | Hygiene before code exists |
| `backend/AGENTS.md` / `frontend/AGENTS.md` | Placeholders until Task 0 |

## Non-goals of this change

- Implementing FastAPI / Next.js (that is the next plan)
- Moving or deleting Streamlit (`app.py` stays as reference)
- Creating HTML UX mock screens
- Provisioning Vercel/Render/Supabase projects (dashboards stay human)

## Success

An agent cloning only `fifmazurkiewicz/POZZ` can follow `AGENTS.md` + `deployment-standard.mdc` and start Task 0 of the greenfield plan without guessing the stack.
