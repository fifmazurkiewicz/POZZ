# User approval gate — POZZ

**Date:** 2026-09-07  
**Status:** approved  

Same product table as Langy / TeacherHelper. Do not invent a second policy.

## POZZ mapping

| Shared rule | POZZ |
|---|---|
| `/me` ungated | `GET /api/auth/me` returns `is_approved` |
| Feature 403 | `require_approved` on patients, simulation turns, STT, evaluation, interviews, jobs, admin except auth/me |
| $10 grant | `spend_cap_usd` default 10 (Europe/Warsaw). Accept does not overwrite it. |
| Waiting UI | `AuthGate` after a valid session; Polish copy; poll `/api/auth/me` every 15s |
| Admin | `GET /api/admin/users` includes `is_approved`; `PATCH /api/admin/users/{id}` `{is_approved}`; hide self-revoke |
| Migration | `supabase/migrations/` + SQLAlchemy `User.is_approved` (Task 0+) |
| Allowlist | `ALLOWED_ADMIN_EMAILS` auto-approved **and** `is_admin` on insert |
| Dev | `dev-token` auto-approved on insert only; never on Render |

No email notifications in v1.
