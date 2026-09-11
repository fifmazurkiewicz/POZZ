---
name: 2026-09-11 admin acceptance queue
overview: Ship GET/PATCH /api/admin/users and Menu → Admin Accept queue. Sole admin fifmazurkiewicz@gmail.com.
todos:
  - id: api
    content: TDD admin list + Accept + hide self-revoke
    status: in_progress
  - id: ui
    content: Admin screen + Menu link
    status: pending
  - id: verify
    content: pytest, frontend lint/test/build, browser
    status: pending
---

Canonical: [`docs/superpowers/plans/2026-09-11-admin-acceptance-queue.md`](../../docs/superpowers/plans/2026-09-11-admin-acceptance-queue.md).

## Decisions

- **2026-09-11 — ship approval queue now.** Cap + bulk generate stay later (Package 7 remainder).
- **2026-09-11 — revoke others allowed; self-revoke forbidden.**
