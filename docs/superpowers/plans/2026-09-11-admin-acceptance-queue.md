# Admin acceptance queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admins can list signups and Accept them so AuthGate waiting users can enter Simulation.

**Architecture:** FastAPI `/api/admin/users` behind `get_admin_user`. Next.js `/menu/admin` uses existing Classical tokens and `apiFetch`.

**Tech Stack:** FastAPI, SQLAlchemy, Next.js App Router, Vitest, pytest.

## Global Constraints

- Product UI copy is Polish; identifiers and agent docs stay English.
- Sole allowlist default: `fifmazurkiewicz@gmail.com`.
- Taste overlay: VARIANCE 3 / MOTION 2 / DENSITY 6. Existing Classical CSS only.
- TDD: failing behavior test before implementation.
- No spend-cap editor, bulk generate, or Sessions list in this slice.

---

### Task 1: Admin API

**Files:**
- Create: `backend/tests/admin/test_users.py`
- Create: `backend/app/api/routes/admin.py`
- Modify: `backend/app/auth/deps.py`
- Modify: `backend/app/api/router.py`

**Interfaces:**
- Consumes: `get_approved_user`, `User`
- Produces: `get_admin_user`, `GET /api/admin/users`, `PATCH /api/admin/users/{user_id}`

- [x] **Step 1: Write the failing tests** in `backend/tests/admin/test_users.py`
- [x] **Step 2: Run** `cd backend && python -m pytest tests/admin/test_users.py -v` — expect FAIL (404 / missing routes)
- [x] **Step 3: Implement** `get_admin_user`, list + patch
- [x] **Step 4: Re-run pytest** — expect PASS

### Task 2: Admin screen

**Files:**
- Create: `frontend/src/lib/admin/access.ts`, `access.test.ts`, `api.ts`
- Create: `frontend/src/app/menu/admin/page.tsx`
- Modify: `frontend/src/app/menu/page.tsx`
- Modify: `frontend/src/lib/auth/routePolicy.ts` + test

**Interfaces:**
- Consumes: `GET/PATCH /api/admin/users`, `useAuth().isAdmin`
- Produces: Menu → Admin; `/menu/admin` queue

- [x] **Step 1: Failing Vitest** for `canOpenAdmin` and pending-first sort
- [x] **Step 2: Implement** access helpers, API client, page, Menu link
- [x] **Step 3:** `cd frontend && npm test && npm run lint`

### Task 3: Docs + verify

- [x] Sync architecture §7.7, UX spec §8, ADR, working plan
- [x] `cd backend && python -m pytest` and `cd frontend && npm test && npm run lint && npm run build`
- [ ] Browser: Menu → Admin as admin; non-admin hidden (pending manual browser check)
