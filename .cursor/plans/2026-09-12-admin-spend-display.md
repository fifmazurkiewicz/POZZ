---
name: 2026-09-12 admin spend display
overview: Show current monthly spend per user in the Admin panel next to the editable cap.
todos:
  - id: api
    content: TDD admin list returns monthly_spend_usd + at_cap (one grouped query)
    status: completed
  - id: ui
    content: Admin screen shows spend + remaining vs cap per user
    status: completed
  - id: verify
    content: pytest, frontend lint/test/build
    status: completed
---

Canonical: [`docs/superpowers/plans/2026-09-12-admin-spend-display.md`](../../docs/superpowers/plans/2026-09-12-admin-spend-display.md).

## Decisions

- **2026-09-12 — spend is read-only in the Admin list.** Cap stays the only editable field; spend always derives from `usage_ledger`.
- **2026-09-12 — one GROUP BY query** for all users' monthly spend, not N queries.
- **2026-09-12 — at_cap mirrors existing semantics** (`cap > 0 and spent >= cap`).