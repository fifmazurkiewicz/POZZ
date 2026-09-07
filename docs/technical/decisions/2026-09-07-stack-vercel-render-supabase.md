# ADR — Stack: Vercel, Render, Supabase

**Date:** 2026-09-07  
**Status:** accepted

## Context

The prototype runs Streamlit on a long-lived host, reads secrets from AWS Secrets Manager, and exposes HTTPS via `cloudflared` quick tunnels. That path does not match the house deployment standard and cannot be the Vercel frontend.

## Decision

| Layer | Platform |
|---|---|
| Frontend | Vercel (Next.js PWA) → `pozz.fmazurkiewicz.dev` |
| Backend | Render Docker Free (FastAPI) → `api-pozz.fmazurkiewicz.dev` |
| DB + Auth | Supabase Postgres + Google OAuth + RLS |

- Cloudflare DNS: separate CNAMEs (`pozz` → Vercel, `api-pozz` → Render DNS-only).
- No Redis in MVP; async work in Postgres `jobs`.
- Render `DATABASE_URL` uses the Supavisor pooler, not `db.<ref>.supabase.co`.
- Health: `GET /api/health` (liveness) and optional `GET /api/health/ready`.
- Frontend implements ApiPulse (5 s waking / 30 s healthy).
- AWS Secrets Manager and cloudflared quick tunnels are **legacy-only**.

## Consequences

Greenfield Task 0 must create `backend/Dockerfile`, FastAPI health, and a Next.js app before any production deploy. Local Streamlit remains available for prompt/flow reference.
