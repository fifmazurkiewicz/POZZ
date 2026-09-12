# Privacy and security foundation

Status: accepted for local review; production privacy notice remains blocked on legal/operator decisions.

## Decision

POZZ is synthetic-data-only by default. Users must not enter identifiable real-patient data. Manual cases and cases generated from user keywords are private to their creator; only non-private system-generated catalog cases may be shared.

The application provides an authenticated JSON export and typed-confirmation deletion of user-owned application content. Content deletion retains the Supabase/Google authentication account and does not claim immediate erasure from provider systems or backups.

Direct Supabase clients may update only `users.display_name`. Approval, administrator status and spend caps are backend-admin operations.

## Testable requirements

- Given a signed-in user, when they request an export, then it includes their account profile, conversations, messages, private cases, transcripts, suggestions, usage records, patient state and jobs, and excludes another user's records.
- Given a user who enters generation keywords, when the patient is stored, then it is private and cannot be selected by another user.
- Given an authenticated Supabase client, when it attempts to update `is_admin`, `is_approved` or `spend_cap_usd`, then database privileges deny the update.
- Given a user who confirms `USUŃ MOJE DANE`, when deletion succeeds, then their conversations, messages, transcripts, suggestions, patient state, jobs, usage and private cases are no longer normally queryable.
- Given an anonymous visitor, when they open `/privacy`, then the AI and draft privacy notice is available before sign-in.

## Production gates not satisfied by this package

- Controller identity/contact details, legal bases, Article 9 condition if real clinical data is introduced, final retention periods and age rules.
- DPIA decision and qualified Polish/EU legal review.
- Processor agreements, regions, subprocessor register and international-transfer safeguards for every configured provider.
- Full authentication-account erasure, backup expiry/erasure procedure and provider-side deletion workflows.
- Automated retention enforcement, consent/policy-version ledger, privacy-request workflow and incident-response evidence.
- Central data-minimising provider gateway and redacted Langfuse tracing.

Real-patient recordings or identifiable clinical cases must not be enabled until those gates are approved and implemented.
