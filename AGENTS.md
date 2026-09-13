# POZZ — agent notes

## Stack (must match deployment standard)

| Layer | Platform |
|---|---|
| Frontend | **Vercel** (Next.js PWA, greenfield) |
| Backend | **Render** (FastAPI / Docker) |
| Database + Auth | **Supabase** (Postgres, Google OAuth, RLS) |

Voice/text: OpenRouter (text) + Groq Whisper STT + patient TTS; Gemini Live when lamp ON (`VOICE_MODE=speech_to_speech`). Langfuse Cloud = prompt SoT. No Redis in MVP. Domains: `pozz` / `api-pozz`.fmazurkiewicz.dev — see `docs/architecture-for-cursor.md`.

**Legacy:** root `app.py` + `modules/` is a Streamlit prototype (AWS Secrets Manager + cloudflared). It is **reference only** — not the production deploy target. New product code lives in `backend/` + `frontend/` after greenfield Task 0.

## Commands

```bash
# Backend
cd backend && python -m pip install -r requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000
cd backend && python -m pytest

# Frontend
cd frontend && npm install && npm run dev
cd frontend && npm run lint && npm test && npm run build

# Health: GET http://localhost:8000/api/health
# Voice: GET http://localhost:8000/api/voice/config
```

Package 0 scaffold lives in `backend/` and `frontend/`. Do not add product features to `app.py`.

Local dev without Supabase (after scaffold): leave `NEXT_PUBLIC_SUPABASE_*` empty so the frontend sends `dev-token`, and set `DEV_AUTH_ENABLED=true` (with empty `SUPABASE_URL`) so the backend accepts it. Production rejects `dev-token`.

## Docs map

| Path | Role |
|---|---|
| `docs/architecture-for-cursor.md` | Business + technical architecture (authoritative for build) |
| `docs/ux/` | UX/UI spec, decisions, screens, design system |
| `docs/business/` | Business-only artifacts (to be filled) |
| `docs/technical/` | Local setup, env names (`configuration.md`), ADRs |

## Graft + Superpowers

- **Superpowers:** process skills before action (global `superpowers.mdc`). Creative work → brainstorming → writing-plans.
- **Graft:** before broad exploration `npx -y @nanonets/graft map` / `graft ask "…" --source` (or MCP). Cache in `/graft/` (gitignored — never commit). Wiring: `.cursor/rules/graft.mdc`, `.cursor/mcp.json`. Native global install may need VS C++ build tools; `npx` is enough. Rebuild with `npx -y @nanonets/graft build` when `graft check` reports no graph.
- **Taste:** skill `design-taste-frontend` (global, not in git). Overlay `.cursor/rules/taste-skill-dials.mdc` — VARIANCE 3 / MOTION 2 / DENSITY 6 (clinical / trust-first).
- **Language:** user may write Polish; agent replies and new code in English (`language.mdc`). Product UI already in Polish is preserve.

## Learned User Preferences

- Keep Superpowers mandatory; Graft and Superpowers belong in global Cursor rules; greenfield FastAPI + Next.js — Streamlit is reference only; no feature code until scaffold + relevant plan Task 0 pass.
- Product UI language is always Polish. Agent chat, identifiers, commits, and docs the agent writes are English.
- Stack must match `deployment-standard.mdc`: Vercel + Render + Supabase. Do not reintroduce AWS EC2, Secrets Manager, or cloudflared quick tunnels as the production path.
- Native language of doctors using the app is Polish; simulated patients speak Polish; LLM prompts force Polish output.
- Chat always has a text input. Simulation lamp: ON = Gemini Live, OFF = chained STT + patient TTS. Mute Listening while the patient speaks. Spec: `docs/superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`.
- Next work after scaffold: schema + auth → text simulation → chained voice → Live → evaluation → Interview tab → admin. SoT: `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md`.
- Simulation modes: doctor asks / patient asks / ask the AI (meta). End interview → doctor writes treatment plan → LLM evaluation vs hidden gold plan.
- Recorded interview (real audio) is a separate surface from the simulated patient chat.
- New public signups wait on `AuthGate` until admin Accept (`users.is_approved`); allowlist / local `dev-token` auto-approved on insert only.
- Monthly spend_cap is admin-configurable and sums STT + gen AI (+ TTS if added); when exceeded, block costly actions for the rest of the month but allow browsing existing sessions.
- Menu hub: Profile, Sessions, Appearance (System/Light/Dark), optional Admin, Sign out.
- Bottom nav is Simulation / Interview / Menu.
- Admin: bulk-generate patient catalog, spend caps, approval queue. Database wipe is admin-only with typed confirmation.
- All API errors surface as Polish JSON `{ "code": "...", "message": "..." }` (e.g. `spend_cap_exceeded`, `provider_error`, `account_pending_approval`); toasts show the Polish message, never raw English.
- POZZ scope extends beyond doctor training to real-time clinical assistance in settings like SOR / izba przyjęć; Wywiad manual mode has an AI-suggested "Zrób badanie" button (recommended investigations / follow-ups / insights) the doctor can accept or ignore mid-interview.

## Learned Workspace Facts

- POZZ is **greenfield** for production (FastAPI + Next.js PWA). The Streamlit app is the working prototype and **reference only**. ADR: `docs/technical/decisions/2026-09-07-greenfield-no-streamlit-deploy.md`.
- Target production: Supabase + Render + Vercel + Cloudflare; Render Root Directory `backend`, Runtime Docker (never `Docker` as root); Cloudflare `api-pozz` CNAME to Render DNS-only, `pozz` CNAME to Vercel — separate records; Google OAuth callback `/auth/callback` with server-side PKCE exchange (not client-side on `/`).
- `CORS_ORIGINS` is a required Render env var (production host `https://pozz.fmazurkiewicz.dev`); default local is `http://localhost:3000,http://127.0.0.1:3000`. Operator runbook: `docs/technical/production-deploy.md`.
- Supabase RLS will enforce per-user access on conversations, messages, transcripts; shared `patients` catalog is readable by approved users; `jobs` admin-only. Render backend bypasses RLS via direct SQLAlchemy (defense in depth for direct Supabase client).
- Langfuse Cloud is the runtime prompt SoT (tracing, cost, prompt management); promptfoo suites gate CI (`npm run promptfoo`, no API keys) once backend exists.
- Default new-user `spend_cap_usd` is 10 per calendar month (Europe/Warsaw).
- Default `TEXT_MODEL` is `google/gemini-2.5-flash-lite` on OpenRouter (matches the prototype). Deprecated slugs return 404 — set explicitly in Render env after deploy.
- STT default: Groq Whisper (`whisper-large-v3`), fallback OpenAI / OpenRouter. No durable audio blobs in MVP (`audio_ref` null); temp files only.
- Patient catalog is shared; conversations and evaluations are per-user. “Next patient” means an unused-by-this-user catalog row, or generate on demand from keywords.
- First-time patients (~20%): card shows name + age only; chronic / ops / allergies / family history must be gathered in the interview.
- Hidden scenario gold (full HPI, traps, treatment plan) is never shown until after End interview evaluation.
- Privacy: `GET /api/privacy/export` + `DELETE /api/privacy/content` (typed confirmation `USUŃ MOJE DANE`); UI at `/menu/privacy`, informational page `/privacy`. Content erase keeps the auth account.
- `diarization_test/` is experimental Gradio work — out of MVP deploy.
- Graft `/graft/` remains gitignored. Native `graft build` may fail without tree-sitter build tools; MCP + rule wiring is enough until then.
- Wywiad "Nagranie" tab is a live recorder (mic → in-memory `Blob` → `POST /api/interviews/recordings`); no file upload, no durable audio. Implementation: `frontend/src/components/interview/RecordedRecorder.tsx`. Same backend endpoint as the legacy upload form — STT + speaker diarization.
- Wywiad "Opis i plan" card (after generation) has an "Ukryj opis" / "Pokaż opis" toggle; collapse state is in-memory only (no localStorage). Sim header has both "Następny pacjent" (random catalog) and "Wygeneruj pacjenta" with a keyword input — non-empty keywords generate a private patient on `POST /api/patients/next`. Escape clears the keyword input.
