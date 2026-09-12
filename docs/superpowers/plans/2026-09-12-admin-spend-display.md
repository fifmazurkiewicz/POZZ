# Admin spend display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admins see each user's current monthly spend next to the editable cap, and know when a user is at cap.

**Architecture:** `GET /api/admin/users` gains `monthly_spend_usd` + `at_cap` (single `GROUP BY` query over `usage_ledger`). Next.js Admin screen renders a read-only spend line per user.

**Tech Stack:** FastAPI, SQLAlchemy, Next.js App Router, Vitest, pytest.

## Global Constraints

- Product UI copy is Polish; identifiers and agent docs stay English.
- TDD: failing behavior test before implementation.
- No cap-mechanism changes, no progress bars, no period picker.
- Taste overlay: VARIANCE 3 / MOTION 2 / DENSITY 6. Existing Classical CSS only.

---

### Task 1: Admin API returns spend

**Files:**
- Modify: `backend/app/api/routes/admin.py`
- Modify: `backend/tests/admin/test_users.py`

**Interfaces:**
- Consumes: `monthly_spend_usd` semantics (sum `usage_ledger.cost_usd` for current month)
- Produces: `monthly_spend_usd` + `at_cap` on every admin-listed user

- [x] **Step 1: Failing tests** — add `UsageLedger` rows, assert list returns `monthly_spend_usd` and `at_cap`; assert users without ledger return `0.0` / `false`
- [x] **Step 2: Run** `cd backend && python -m pytest tests/admin/test_users.py -v` — expect FAIL (missing fields)
- [x] **Step 3: Implement** one aggregated spend query in `admin.py`
- [x] **Step 4: Re-run pytest** — expect PASS

### Task 2: Admin screen shows spend

**Files:**
- Modify: `frontend/src/lib/admin/api.ts`
- Modify: `frontend/src/app/menu/admin/page.tsx`

**Interfaces:**
- Consumes: extended `AdminUser` payload
- Produces: read-only spend line per user row

- [x] **Step 1:** Extend `AdminUser` type
- [x] **Step 2:** Render `Wydano: … / limit …` (or `· bez limitu`), with `wyczerpany` note when `at_cap`
- [x] **Step 3:** `cd frontend && npm test && npm run lint`

### Task 3: Docs + verify

- [x] Sync working plan decisions + canonical plan
- [x] `cd backend && python -m pytest` and `cd frontend && npm test && npm run lint && npm run build`