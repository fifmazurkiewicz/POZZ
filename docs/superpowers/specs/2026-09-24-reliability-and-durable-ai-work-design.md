# Reliability and durable AI work — POZZ

**Status:** approved design; implementation plan pending review.

## Goal

Make POZZ reliable on slow networks and small mobile screens, while ensuring a
reload or route change does not lose a recorded-interview transcription or
manual-interview case-plan request.

## Scope

1. Cache CORS preflights; expose accurate loading/error/retry states; make the
   Interview screen independently scrollable; normalize response DTOs; retain
   newly created sessions in the smallest shared client store needed by Sessions.
2. Persist the two long-running operations currently performed synchronously:
   recorded-interview transcription with speaker assignment and manual case-plan
   generation.

Product text remains Polish. No Redis, durable audio blob, browser reload, or
new client data-cache dependency is introduced.

## Reliability pass

### CORS

`CORSMiddleware` receives `max_age=86_400`. Authenticated requests still
preflight where browsers require it, but compatible browsers cache the result
for up to one day. A backend test asserts the preflight response header.

### Response contracts

Conversation, user, job, and privacy routes use Pydantic response models. DTO
factories stay the single place that converts ORM UUIDs, decimals, JSON values,
and nullable timestamps into JSON-safe values. Existing field names remain
unchanged.

### Client state and feedback

A tiny React context keeps known `SimulationSession` summaries by
`conversation_id`. Successful creation and fetches upsert their returned
session. Sessions reads it first, then soft-refreshes and merges the API
result. This is not an optimistic object or a new query-cache dependency.

Sessions and Admin explicitly represent `loading`, `ready`, and `error` and
provide a Polish retry action. Privacy export and erasure have busy and Polish
error states; download failures are handled instead of escaping as unhandled
rejections. Existing action buttons remain the retry path for action failures.

### Mobile layout

The global body stays `h-dvh` and non-scrolling. Every route below an
`overflow-hidden` parent owns a `min-h-0 flex-1 overflow-y-auto` content
region. The Interview setup form and active interview share that region.
Interactive content receives bottom safe-area padding; no confirmation control
may sit below an unscrollable fold.

## Durable jobs

### Data model and execution

Reuse the existing Postgres `jobs` table. `kind` gains `transcribe_interview`
and `generate_case_plan`; `payload` contains only JSON-safe metadata and
result references, never audio bytes. Jobs belong to the requesting user and
identify the target conversation once it exists.

Supported states are `queued`, `running`, `done`, and `failed`. Failed jobs
store a Polish public `{ code, message }` error, never raw provider failures.
Privacy export and erasure keep their existing job behavior.

Because MVP forbids durable audio and has no independent worker, recording
upload commits a job then starts in-process work immediately. The client gets
a job ID after upload succeeds. If a Render restart loses the in-memory audio,
the job fails with a Polish re-record instruction. This is the strongest
guarantee compatible with no audio persistence. Case-plan jobs are fully
restart-resumable because their inputs already persist.

### API

* `POST /api/conversations/{id}/plan` creates or returns an active
  `generate_case_plan` job and returns `202 Accepted`. Closed, foreign, and
  spend-capped conversations retain current behavior. An explicit later
  regenerate creates a new job.
* `POST /api/interviews/recordings` creates its private conversation and a
  `transcribe_interview` job, then returns `202 Accepted` with the job DTO.
* `GET /api/jobs/{job_id}` returns only an owned job. A done job includes the
  current conversation DTO; a failed job includes the Polish error DTO. A
  foreign ID returns 404.
* A job service owns creation, atomic claim, completion, and failure mapping.
  It checks spend before queueing and again before claim. Concurrent polls can
  claim a job only once.

### Frontend behavior

`useDurableJob` polls `GET /api/jobs/{id}` every two seconds while active and
stores active IDs in `sessionStorage`. On mount, compatible active IDs resume
polling. Route changes and `router.refresh()` do not cancel the server job.

Interview shows Polish queued/running status. Completion upserts the returned
conversation and renders it. Failure displays the Polish error with a
contextual retry: regenerate the plan or record/upload again. Users can leave
the route while work continues.

## Error handling and accessibility

New API errors retain the existing FastAPI `detail` wrapper around
`{ "code": "…", "message": "…" }`. UI states use Polish copy,
`role="status"`/`role="alert"`, disabled busy controls, and keyboard-accessible
retry buttons. Failed loads never render as empty lists.

## Out of scope

Redis, Celery, a third-party queue, a new query-cache package, durable audio,
live streaming, Gemini Live, and changes to patient generation or simulation
turn requests.

## Verification

Test CORS max-age; DTO UUID/decimal/null serialization; job ownership,
transitions, provider failures, and idempotent claim; plan persistence after
status reload; loading/empty/error/retry states; privacy errors; immediate
session upserts; job polling/resumption and retries; and independently
scrollable Interview content on mobile.
