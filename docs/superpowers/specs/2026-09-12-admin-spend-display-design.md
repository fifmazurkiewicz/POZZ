# Admin spend display — POZZ

**Date:** 2026-09-12
**Status:** approved

Shows the current monthly spend for each user in the Admin panel, next to the existing editable monthly cap. Spend data already exists in `usage_ledger` and is already surfaced to admins on `GET /api/me`; this change exposes it in the admin user list and UI.

## Out of scope

Changing the cap mechanism, billing, invoice data, exporting spend history, period pickers (current calendar month only), progress bars.

## Requirements (Given / When / Then)

1. **Given** an approved admin, **When** they `GET /api/admin/users`, **Then** each user row includes `monthly_spend_usd` (float ≥ 0, current calendar month in `spend_cap_tz`) and `at_cap` (`spend_cap_usd > 0 and monthly_spend_usd >= spend_cap_usd`). Users with no ledger rows report `0.0` / `false`.
2. **Given** an approved admin, **When** they open Menu → Admin, **Then** each user row shows the monthly spend and (unless cap is 0) the remaining amount vs the cap, in Polish.
3. **Given** a non-admin or unapproved caller, **When** they hit `GET /api/admin/users`, **Then** the API still returns 403 (`admin_required` / `account_pending_approval`) and no spend data leaks.

## API

`AdminUser` (GET `/api/admin/users` response) gains two read-only fields:

- `monthly_spend_usd: float`
- `at_cap: bool`

`PATCH` behavior is unchanged and never accepts these fields.

## UI

- Each row in `/menu/admin` shows, under the cap input, a read-only line:
  - cap > 0: `Wydano: 21,43 USD / limit 25,00 USD`
  - cap = 0: `Wydano: 21,43 USD · bez limitu`
  - `at_cap`: marked with an inline "wyczerpany" note (text only, no new tokens).
- Display-only, driven by server-fetched data; no edit affordance for the spend amount.

## Decisions

- **2026-09-12 — single aggregated GROUP BY for the whole list.** Reuse `monthly_spend_usd(db, user_id)` semantics (sum `usage_ledger.cost_usd` for current month) but compute all users in one query instead of N queries.
- **2026-09-12 — at_cap mirrors existing semantics** (`cap > 0 and spent >= cap`), identical to `GET /api/me` and `assert_under_cap`.