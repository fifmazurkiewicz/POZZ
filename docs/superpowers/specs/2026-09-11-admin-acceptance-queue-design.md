# Admin acceptance queue — POZZ

**Date:** 2026-09-11  
**Status:** approved (ships the missing Admin half of [`2026-09-07-user-approval-gate-design.md`](./2026-09-07-user-approval-gate-design.md))

Sole production admin: `fifmazurkiewicz@gmail.com` via `ALLOWED_ADMIN_EMAILS`. AuthGate already waits. This change adds the Accept API and screen.

## Out of scope

Sessions history, spend-cap editor, bulk patient generate, email notify, product-UI wipe.

## Requirements (Given / When / Then)

1. **Given** an approved admin, **When** they `GET /api/admin/users`, **Then** they receive every user with `id`, `email`, `display_name`, `is_admin`, `is_approved`, `created_at`. Pending users are listed first.
2. **Given** a non-admin (or unapproved) caller, **When** they hit any `/api/admin/*` route, **Then** the API returns 403 (`admin_required` or `account_pending_approval`).
3. **Given** an approved admin, **When** they `PATCH /api/admin/users/{id}` with `{ "is_approved": true }`, **Then** that user’s `is_approved` becomes true and `spend_cap_usd` is unchanged.
4. **Given** an approved admin, **When** they `PATCH` their own id with `{ "is_approved": false }`, **Then** the API returns 400 `cannot_self_revoke` and the row is unchanged.
5. **Given** an approved admin, **When** they open Menu → Admin, **Then** they see the Polish queue and can Accept a pending account.
6. **Given** an approved non-admin, **When** they open `/menu/admin`, **Then** they are sent back to Menu (no queue).

## API

| Method | Path | Auth | Body / result |
|---|---|---|---|
| GET | `/api/admin/users` | approved + `is_admin` | `{ "users": AdminUser[] }` |
| PATCH | `/api/admin/users/{id}` | approved + `is_admin` | `{ "is_approved": bool }` → `AdminUser` |

`AdminUser`: `id`, `email`, `display_name`, `is_admin`, `is_approved`, `created_at` (ISO or null).

## UI

- Menu shows **Admin** only when `is_admin`.
- Screen: `/menu/admin`. Classical tokens. Pending block first (**Zaakceptuj**). Approved block second; **Cofnij** hidden on the admin’s own row.
- Polish copy. Loading / empty / error states. No email send.

## Decisions

- **2026-09-11 — ship queue now, rest of Package 7 later.** Approval is the production blocker. Cap + bulk generate stay later.
- **2026-09-11 — revoke others allowed; self-revoke forbidden.** Matches the 2026-09-07 gate spec.
