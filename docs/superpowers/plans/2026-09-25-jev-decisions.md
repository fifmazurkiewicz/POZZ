# POZZ Jev Decisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach non-clinical Jev review signals to existing POZZ evaluations and recorded-interview drafts.

**Architecture:** Reuse the current OpenRouter HTTP conventions in a dedicated decision client. The evaluation service obtains optional signals after its normal LLM work; unavailable signals never alter completion or clinical text.

**Tech Stack:** Python, FastAPI, httpx, SQLAlchemy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-jev-decisions-design.md`

## Global Constraints

- Endpoint `/api/alpha/decisions`, model `typesafe/jev-1.13`, server-only key.
- Jev is advisory training UI only; it cannot diagnose, triage, prescribe, or block a draft.
- Retain RLS, transcript privacy, and existing clinician-review label.

## Review Focus

- API failure must not fail an interview evaluation.
- Raw transcripts must not be copied into a new audit/event store.
- Review labels must remain non-clinical and Polish in UI.
- A high signal cannot change the gold-plan evaluation.
- 429/5xx retry must be bounded.

---

### Task 1: OpenRouter Decisions client

**Files:** Create `backend/app/llm/jev.py`, `backend/tests/llm/test_jev.py`; modify `backend/app/settings.py` if configuration is absent.

**Interfaces:** Produce `evaluate_training_signals(state: dict) -> dict | None`.

- [ ] Write mocked HTTP tests for response parsing, retry exhaustion, invalid JSON, and no raw-state event payload.
- [ ] Run `cd backend && python -m pytest tests/llm/test_jev.py -q`; expect failure.
- [ ] Implement the pinned Decisions API client with the existing OpenRouter headers and bounded backoff.
- [ ] Run focused tests and expect pass.
- [ ] Commit `feat: add POZZ Jev decision client`.

### Task 2: Advisory evaluation signal

**Files:** Modify the existing evaluation/recorded-interview service and response schema identified by `tests/patients/test_interview_plan.py`; modify/add focused tests.

**Interfaces:** Existing evaluation output gains optional `decision_signals`; absence is valid.

- [ ] Write a test that a successful signal is included, while an unavailable client leaves current output unchanged.
- [ ] Implement post-evaluation invocation only; never alter clinical draft content or success status.
- [ ] Run patient and recorded-interview tests.
- [ ] Commit `feat: add POZZ advisory Jev signals`.
