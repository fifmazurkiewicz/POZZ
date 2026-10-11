# POZZ — approval notification design

## Goal

Notify `fifmazurkiewicz@gmail.com` once when a newly created, non-admin POZZ account enters the existing approval queue. The administrator can set the existing monthly USD spend cap and accept the account from the existing admin page.

## Scope and configuration

- This design is POZZ-only; it does not call another application or shared notification service.
- Add backend-only `RESEND_API_KEY`, `RESEND_MAIL=fifmazurkiewicz@gmail.com`, and `RESEND_FROM=Powiadomienia <powiadomienia@fmazurkiewicz.dev>` settings. The Resend key is a POZZ-specific sending-only key.
- Retain `ALLOWED_ADMIN_EMAILS=fifmazurkiewicz@gmail.com`. The allowlisted admin remains auto-approved on first profile creation.

## Flow

`get_current_user` is the existing profile-creation boundary. When it inserts a non-admin `User(is_approved=false)`, it records one pending approval notification and sends one transactional email through Resend. Later logins must not create another notification. The mail contains only the application name, new account email, creation time, current default USD cap, and the existing `/menu/admin` URL.

Resend failure never blocks login, profile creation, the waiting screen, or the 403 feature gate. Store `pending`, `sent`, or `failed`, a sent timestamp, and a safe error summary. The admin users response exposes that status. The existing admin page adds a retry action for `failed` or unsent notifications; retry is admin-only and updates the same record. No automatic retry is added.

The existing pending screen adds an accessible `mailto:fifmazurkiewicz@gmail.com` link with an app-specific subject. It does not expose feature navigation.

## Data, security, and UI

- Add the notification state to the local user record through a migration, rather than storing a Resend key, message body, or raw provider response.
- Do not send admin-created/allowlisted accounts, approved accounts, or previously-notified accounts.
- Keep Resend calls in a small backend adapter with an injectable transport for tests; never import it in the frontend or commit a key.
- Reuse the current `spend_cap_usd` admin field and approval action. No token-count limit is introduced.

## Tests and rollout

Test new non-admin creation sends exactly one mail; a later login sends none; a provider error leaves the account pending and records `failed`; an admin retry sends once; non-admin retry is forbidden; and the pending screen contains the contact link. Configure the three environment variables in POZZ deployment before enabling the release, then create a test account and confirm one mail and manual acceptance.
