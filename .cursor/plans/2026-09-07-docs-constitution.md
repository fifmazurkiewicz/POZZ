---
name: 2026-09-07 docs constitution
overview: Land Langy-standard docs, Cursor rules, specs, and hygiene so POZZ can be built and deployed on Vercel / Render / Supabase.
todos:
  - id: rules
    content: Copy constitution into .cursor/rules + mcp.json
    status: completed
  - id: docs
    content: architecture, UX, technical, ADRs, superpowers specs
    status: completed
  - id: hygiene
    content: AGENTS.md, env examples, gitignore, Gitleaks CI, Dependabot
    status: completed
  - id: plans
    content: greenfield scaffold plan + keep/adapt/drop
    status: completed
---

# Plan — docs + constitution (2026-09-07)

## Goal

POZZ repo matches Langy’s documentation/process standard, with product content for the POZ simulator.

## Decisions

| Decision | Why |
|---|---|
| Copy Langy rules, not Langy product screens | Same hosting/process; different domain |
| Streamlit stays at root | Reference until Task 0; no silent delete |
| CI = Gitleaks only until backend tests exist | Avoid red pytest on empty `backend/` |
| Domains `pozz` / `api-pozz` | Matches deployment-standard pattern |

## Delta

- **ADDED:** `.cursor/rules/**`, `docs/**`, `AGENTS.md`, `.env.example`, CI, Dependabot, placeholder `backend/` + `frontend/` notes
- **MODIFIED:** `.gitignore`, `README.md`
- **REMOVED:** none of the Streamlit app
