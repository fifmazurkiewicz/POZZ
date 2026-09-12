# ADR — Architecture Decision Records

Short dated decisions. Living detail often lives in `docs/architecture-for-cursor.md`; extract here when a decision needs a stable ID.

## 2026-09-07 — constitution hygiene in-repo

- Taste + language rules copied into `.cursor/rules/`; POZZ dials VARIANCE 3 / MOTION 2 / DENSITY 6
- Env names: `docs/technical/configuration.md`
- Dependabot: pip at `/`, GitHub Actions at `/`; frontend/backend ecosystems added when Task 0 lands
- CI: Gitleaks; backend pytest + promptfoo + frontend lint/test/build after scaffold
- ADR: `docs/technical/decisions/2026-09-07-constitution-hygiene.md`

## 2026-09-07 — stack: Vercel + Render + Supabase

- Production hosting matches Langy deployment standard
- Domains: `pozz.fmazurkiewicz.dev` / `api-pozz.fmazurkiewicz.dev`
- No Redis in MVP; Postgres jobs
- ADR: `docs/technical/decisions/2026-09-07-stack-vercel-render-supabase.md`

## 2026-09-07 — greenfield; Streamlit reference only

- Production = own `backend/` + `frontend/` from specs
- Streamlit `app.py` is the prototype; do not deploy it to Vercel/Render
- Task 0 = greenfield MVP skeleton, not a Streamlit lift-and-shift
- ADR: `docs/technical/decisions/2026-09-07-greenfield-no-streamlit-deploy.md`

## 2026-09-07 — domain / user journey

- Shared patient catalog; per-user conversations + RLS
- Simulation modes + end-interview evaluation; recorded + manual interview
- Approval gate + monthly spend cap ($10, Europe/Warsaw)
- Spec: `docs/superpowers/specs/2026-09-07-domain-user-journey-design.md`

## 2026-09-07 — chained STT only (superseded 2026-09-08)

- Originally: no patient TTS / no Live in MVP
- **Superseded** by 2026-09-08 Live/TTS lamp

## 2026-09-08 — Simulation Live vs TTS lamp (superseded 2026-09-11)

- Lamp ON = Gemini Live; OFF = chained STT + patient TTS
- `localStorage` `pozz-sim-live-gemini`; env `VOICE_MODE` still caps Live
- Mute Listening while the patient speaks
- Spec: `docs/superpowers/specs/2026-09-08-simulation-live-tts-lamp-design.md`
- Build order: `docs/superpowers/specs/2026-09-08-refactor-build-order-design.md`

## 2026-09-11 — Interview modal error/loading states and STT dictation

- `useAbortableAction` distinguishes user cancel (`signal.aborted` short-circuits the
  catch) from transport abort / network failure; new `cancelled`, `reset`, `getError()`
  exports
- `InterviewActions` modal: separate "Anuluj" label, inline "Trwa…" status, microphone
  button that calls `/api/voice/transcribe` and appends the result to the active
  textarea (examination ≤ 2000 chars, plan ≤ 8000 chars)
- Suppresses the modal error display whenever `cancelled === true` so Cancel never shows
  "Spróbuj ponownie"
- Added `vitest.config.ts` (jsdom env + `@` alias) and dev deps
  `@testing-library/react`, `jsdom`

## 2026-09-11 — Unhandled exception shapes for interview endpoints

- Backend `app.exception_handler(Exception)` returns 502 for upstream provider errors
  (httpx) and 503 for everything else, with `{"detail":{"code":"provider_error",
  "message":"Nie udało się wykonać operacji. Spróbuj ponownie."}}`
- Leaves `HTTPException` and validation errors untouched so existing spend-cap /
  conversation-completed codes still flow through
- Test `test_failed_evaluation_leaves_conversation_open` updated to assert 502 body
  instead of raised `RuntimeError`
- ADR: `docs/technical/decisions/2026-09-11-unhandled-exception-shapes.md`

## 2026-09-11 — Interview controls and voice settings

- Simulation and manual Interview share Stop, Examination and End interview controls
- Completed conversations persist evaluation + `ended_at` and become read-only
- Voice-mode and local ElevenLabs voice ID settings live in Menu
- Current client supports chained TTS; Live is displayed as unavailable
- ADR: `docs/technical/decisions/2026-09-11-interview-controls-and-voice-settings.md`

## 2026-09-11 — production deploy runbook

- Operator SoT: `docs/technical/production-deploy.md` (Supabase → schema → Google OAuth → Render → `api-pozz` DNS-only → Vercel → `pozz` CNAME)
- `CORS_ORIGINS` is required on Render (`https://pozz.fmazurkiewicz.dev`); documented in `configuration.md`
- Hosting choices unchanged (Vercel / Render Docker / Supabase)

## 2026-09-07 — user approval gate

- `users.is_approved`; new signups wait until admin Accept
- Allowlist / local `dev-token` auto-approved on insert only
- Spec: `docs/superpowers/specs/2026-09-07-user-approval-gate-design.md`
