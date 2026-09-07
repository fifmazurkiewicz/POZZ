# ADR — Constitution hygiene in-repo

**Date:** 2026-09-07  
**Status:** accepted

## Context

Langy’s in-repo constitution (Graft, Superpowers, spec-driven docs, Taste, English agent replies, deployment standard, secrets rules) is the house standard. POZZ had none of that: Streamlit + AWS scripts, no `docs/`, no Cursor rules, no CI secret scan.

## Decision

- Commit the same rule set under `.cursor/rules/` (including `taste-skill.mdc`, `language.mdc`, `deployment-standard.mdc`).
- Taste dials for POZZ: VARIANCE 3 / MOTION 2 / DENSITY 6 (clinical / trust-first).
- Document environment **names** in `docs/technical/configuration.md`; never commit secret values.
- Dependabot: pip at repo root (legacy `pyproject.toml`) and GitHub Actions at `/`. Add npm/pip for `frontend/` / `backend/` when those package manifests exist.
- CI: Gitleaks secret scan (pinned image + allowlist for env templates / empty `*_API_KEY=` placeholders). Python 3.12 and Node 22 jobs land with Task 0.
- Nested `backend/AGENTS.md` and `frontend/AGENTS.md` exist as placeholders until the scaffold. Graft `/graft/` remains gitignored.

## Consequences

Agents cloning only this repo get the full constitution. Taste SKILL.md stays global and uncommitted. Graft MCP still needs Cursor to start the server from `.cursor/mcp.json`.
