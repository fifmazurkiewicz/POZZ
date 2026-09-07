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

## 2026-09-07 — chained STT, no patient TTS in MVP

- Doctor mic optional; text always available
- Groq Whisper default; LLM text replies
- Gemini Live / patient TTS = Phase 2
- Spec: `docs/superpowers/specs/2026-09-07-stack-deploy-design.md`

## 2026-09-07 — user approval gate

- `users.is_approved`; new signups wait until admin Accept
- Allowlist / local `dev-token` auto-approved on insert only
- Spec: `docs/superpowers/specs/2026-09-07-user-approval-gate-design.md`
