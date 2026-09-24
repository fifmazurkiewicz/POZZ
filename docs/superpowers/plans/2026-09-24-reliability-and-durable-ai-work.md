# Reliability and Durable AI Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve perceived reliability and make manual-plan and recorded-interview work observable and recoverable through durable Postgres job records.

**Architecture:** Keep current FastAPI/SQLAlchemy and React state patterns. Add JSON-safe response DTOs and a small session context, then put job creation/state transitions in one backend service and poll owned jobs from a sessionStorage-backed frontend hook. Recording execution remains best-effort across a Render restart because raw audio is intentionally not durable; plan jobs are resumable from persisted conversation data.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy, Postgres/SQLite tests, Next.js, React, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-24-reliability-and-durable-ai-work-design.md`

## Global Constraints

- Product copy is Polish; API errors retain `{ "code": "…", "message": "…" }` inside FastAPI `detail`.
- Use Postgres `jobs`; no Redis, queue dependency, durable audio storage, or client cache package.
- Preserve existing public wire field names and ownership semantics.
- Do not add production work to legacy Streamlit files.
- Keep long-running work independent from route changes and soft refreshes.

## Review Focus

- An initial fetch that fails must render a Polish error and retry action, never an empty successful list.
- An owned active job must be resumed after a route remount without duplicate provider execution.
- A foreign job ID must return 404 and reveal neither status nor payload.
- A completed plan job must expose no private scenario or gold treatment plan.
- A short mobile viewport must scroll to every Interview setup and action control.

## File structure

- `backend/app/main.py`: CORS max-age configuration.
- `backend/app/api/dtos.py`: JSON-safe Pydantic response models and factories.
- `backend/app/jobs/service.py`: job lifecycle, atomic claim, worker functions, and safe error mapping.
- `backend/app/api/routes/jobs.py`: owned job status endpoint.
- `backend/app/api/routes/conversations.py`, `backend/app/api/routes/interviews.py`: queue plan/recording work instead of blocking on it.
- `backend/app/api/router.py`: jobs router registration.
- `backend/tests/test_cors.py`, `backend/tests/jobs/test_jobs.py`: CORS and job behavior tests.
- `frontend/src/components/SessionStore.tsx`: minimal upsert/merge session context.
- `frontend/src/lib/jobs/api.ts`, `frontend/src/lib/jobs/useDurableJob.ts`: job DTO client and polling/resumption hook.
- `frontend/src/app/layout.tsx`: session-store provider.
- `frontend/src/app/interview/page.tsx`, `frontend/src/components/interview/RecordedRecorder.tsx`: job UI and retries.
- `frontend/src/app/menu/sessions/page.tsx`, `frontend/src/app/menu/admin/page.tsx`, `frontend/src/app/menu/privacy/page.tsx`: explicit load/error/retry UX.
- `frontend/src/components/ViewportFrame.tsx`, `frontend/src/app/interview/page.tsx`, `frontend/src/app/globals.css`: independently scrollable, safe-area-aware Interview layout.

### Task 1: CORS and DTO contract foundation

**Files:**
- Modify: `backend/app/main.py:82-88`
- Create: `backend/app/api/dtos.py`
- Create: `backend/tests/test_cors.py`
- Test: `backend/tests/test_cors.py`

**Interfaces:**
- Produces: `ConversationResponse`, `JobResponse`, and `error_detail(code, message)` for routes and frontend-compatible JSON.

- [ ] **Step 1: Write the failing CORS test**

```python
def test_preflight_is_cached(sqlite_client):
    response = sqlite_client.options("/api/conversations", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    assert response.headers["access-control-max-age"] == "86400"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && python -m pytest tests/test_cors.py -v`

Expected: FAIL because Starlette returns its default max age.

- [ ] **Step 3: Add the minimal CORS setting and DTOs**

```python
# backend/app/main.py
app.add_middleware(CORSMiddleware, ..., max_age=86_400)

# backend/app/api/dtos.py
class JobResponse(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "failed"]
    conversation: ConversationResponse | None = None
    error: ErrorDetail | None = None
```

Normalize UUIDs with `str`, decimal costs with `float`, datetimes with
`isoformat()`, and nullable ORM fields to `None`. Convert the existing
conversation payload into `ConversationResponse.model_validate(...)` rather
than renaming response fields.

- [ ] **Step 4: Apply response models to existing conversation/admin/privacy routes**

Use `response_model=` on JSON endpoints and route all currently handwritten
payloads through DTO factories. Keep download privacy export as its existing
JSON object but validate nested UUID/decimal/null fields before return.

- [ ] **Step 5: Run focused backend tests**

Run: `cd backend && python -m pytest tests/test_cors.py tests/admin tests/privacy tests/patients -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/app/api/dtos.py backend/app/api/routes backend/tests/test_cors.py
git commit -m "feat: add API response contracts and CORS caching"
```

### Task 2: Durable backend jobs

**Files:**
- Create: `backend/app/jobs/__init__.py`
- Create: `backend/app/jobs/service.py`
- Create: `backend/app/api/routes/jobs.py`
- Modify: `backend/app/api/router.py`
- Modify: `backend/app/api/routes/conversations.py:239-275`
- Modify: `backend/app/api/routes/interviews.py:30-70`
- Create: `backend/tests/jobs/test_jobs.py`

**Interfaces:**
- Consumes: `ConversationResponse`, `JobResponse`, `Conversation`, `Job`, and provider/transcription functions.
- Produces: `create_job`, `claim_job`, `run_case_plan_job`, `run_recording_job`, and `GET /api/jobs/{job_id}`.

- [ ] **Step 1: Write failing lifecycle and ownership tests**

```python
def test_plan_post_returns_owned_queued_job(sqlite_client):
    response = sqlite_client.post(f"/api/conversations/{conversation_id}/plan", headers=AUTH)
    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "running", "done"}

def test_other_user_cannot_read_job(sqlite_client):
    response = sqlite_client.get(f"/api/jobs/{job_id}", headers=OTHER_AUTH)
    assert response.status_code == 404
```

- [ ] **Step 2: Run lifecycle tests to verify they fail**

Run: `cd backend && python -m pytest tests/jobs/test_jobs.py -v`

Expected: FAIL because jobs routes/services do not exist.

- [ ] **Step 3: Implement a single job service**

```python
def claim_job(db: Session, job_id: UUID) -> Job | None:
    job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
    if job is None or job.status != "queued":
        return None
    job.status = "running"
    db.commit()
    return job
```

Implement `create_job`, `complete_job`, and `fail_job` in the same service.
`fail_job` stores only the Polish public code/message. Call `assert_under_cap`
before creation and before claim. The plan worker reloads the conversation,
generates and persists `interview_summary`, then returns its DTO. Use a
FastAPI `BackgroundTasks` callback after the transaction commits.

- [ ] **Step 4: Queue plan generation and expose status**

Change `POST /plan` to create/reuse an active job and return `202`. Add the
owned `GET /api/jobs/{id}` status route. A done status includes a freshly
loaded conversation DTO; it must omit hidden scenario and gold plan.

- [ ] **Step 5: Queue recording work without persisting audio**

Create conversation + job after upload validation. Copy only the upload to a
bounded temporary file for the in-process task, delete it in `finally`, and
never place it in DB/job payload/export. On restart or missing temp input,
mark failed with Polish re-record guidance. On success persist transcript and
return it through done-job conversation status.

- [ ] **Step 6: Add race, failure, and completion tests**

Test single successful claim under two calls, provider failure mapping, spend
cap at queue/claim, done-plan persistence after status reload, transcript
completion, no audio in DB, and 404 job ownership.

- [ ] **Step 7: Run backend verification**

Run: `cd backend && python -m pytest tests/jobs tests/interviews/test_recorded_interview.py tests/patients/test_interview_plan.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/jobs backend/app/api/routes backend/app/api/router.py backend/tests/jobs backend/tests/interviews backend/tests/patients
git commit -m "feat: persist AI work as owned jobs"
```

### Task 3: Session store and durable job client

**Files:**
- Create: `frontend/src/components/SessionStore.tsx`
- Create: `frontend/src/lib/jobs/api.ts`
- Create: `frontend/src/lib/jobs/useDurableJob.ts`
- Modify: `frontend/src/app/layout.tsx:28-42`
- Test: `frontend/src/lib/jobs/useDurableJob.test.tsx`

**Interfaces:**
- Consumes: `SimulationSession`, `JobResponse`, `apiFetch`.
- Produces: `useSessions().upsert`, `useSessions().merge`, and `useDurableJob(job, token)`.

- [ ] **Step 1: Write failing hook tests**

```tsx
it("restores an active job from sessionStorage and polls it", async () => {
  sessionStorage.setItem("pozz:jobs", JSON.stringify(["job-1"]));
  renderHook(() => useDurableJob(token));
  await waitFor(() => expect(fetchJob).toHaveBeenCalledWith(token, "job-1"));
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- useDurableJob.test.tsx`

Expected: FAIL because the job client and hook do not exist.

- [ ] **Step 3: Implement minimal shared session storage**

Use a context with a `Map<string, SimulationSession>` represented as immutable
state. `upsert` replaces by ID; `merge` upserts all server sessions. Wrap it
inside `AuthProvider` in the root layout. Do not prepopulate records before a
successful API response.

- [ ] **Step 4: Implement polling and restore**

`useDurableJob` stores only active IDs in `sessionStorage`, polls every 2,000
ms, stops/removes an ID on done or failed, and calls `upsert` on done. After
two transient fetch errors, use a 5,000-ms interval until a successful poll.
Do not issue a second POST while a matching active job is known.

- [ ] **Step 5: Run frontend hook tests**

Run: `cd frontend && npm test -- useDurableJob.test.tsx`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/SessionStore.tsx frontend/src/lib/jobs frontend/src/app/layout.tsx
git commit -m "feat: poll durable AI jobs from the client"
```

### Task 4: Wire durable job UX and instant Sessions updates

**Files:**
- Modify: `frontend/src/app/interview/page.tsx`
- Modify: `frontend/src/components/interview/RecordedRecorder.tsx`
- Modify: `frontend/src/lib/interview/api.ts`
- Modify: `frontend/src/lib/simulation/api.ts`
- Modify: `frontend/src/components/simulation/SimulationClient.tsx`
- Modify: `frontend/src/app/menu/sessions/page.tsx`
- Test: `frontend/src/app/interview/page.test.tsx`
- Test: `frontend/src/components/interview/RecordedRecorder.test.tsx`
- Test: `frontend/src/app/menu/sessions/page.test.tsx`

**Interfaces:**
- Consumes: `useDurableJob`, `useSessions`, `JobResponse`.
- Produces: persistent in-UI recording/plan states and session list updates from returned server objects.

- [ ] **Step 1: Write failing UI tests**

```tsx
it("shows the queued plan status then renders the completed plan", async () => {
  vi.mocked(createCasePlanJob).mockResolvedValue({ id: "job-1", status: "queued" });
  render(<InterviewPage />);
  await user.click(screen.getByRole("button", { name: "Generuj opis i plan" }));
  expect(screen.getByText("Generowanie opisu i planu…")).toBeTruthy();
});
```

- [ ] **Step 2: Run UI tests to verify they fail**

Run: `cd frontend && npm test -- interview/page.test.tsx RecordedRecorder.test.tsx menu/sessions/page.test.tsx`

Expected: FAIL because these flows expect synchronous conversation responses.

- [ ] **Step 3: Replace synchronous UI waits with job submission/polling**

The plan action submits a job and displays Polish queued/running state. The
recording flow displays its existing submission state until the upload returns
a job, then delegates to polling. Done results set `session` and clear active
job storage. Failures show `role="alert"` plus a retry button that safely
creates a new job/upload.

- [ ] **Step 4: Upsert all created/fetched sessions**

Use `upsert` in simulation next-patient, manual case creation, recording/job
completion, and conversation restore. Sessions renders store data immediately,
then invokes one reload function which merges API results.

- [ ] **Step 5: Add job completion/failure and session-visibility tests**

Assert no hidden scenario/gold text is rendered, session appears before the
Sessions network refresh resolves, an error has a retry, and a re-mounted page
continues polling the saved active job.

- [ ] **Step 6: Run frontend behavior tests**

Run: `cd frontend && npm test -- interview/page.test.tsx SimulationClient.test.tsx RecordedRecorder.test.tsx menu/sessions/page.test.tsx`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/app/interview frontend/src/components/interview frontend/src/components/simulation frontend/src/lib frontend/src/app/menu/sessions
git commit -m "feat: show durable AI work in the UI"
```

### Task 5: Explicit feedback and mobile layout hardening

**Files:**
- Modify: `frontend/src/app/menu/sessions/page.tsx`
- Modify: `frontend/src/app/menu/admin/page.tsx`
- Modify: `frontend/src/app/menu/privacy/page.tsx`
- Modify: `frontend/src/components/ViewportFrame.tsx`
- Modify: `frontend/src/app/interview/page.tsx`
- Modify: `frontend/src/app/globals.css`
- Test: `frontend/src/app/menu/sessions/page.test.tsx`
- Test: `frontend/src/app/menu/admin/page.test.tsx`
- Test: `frontend/src/app/menu/privacy/page.test.tsx`

**Interfaces:**
- Produces: reusable local `load` callbacks with explicit feedback and a scroll-safe Interview root.

- [ ] **Step 1: Write failing loading/retry tests**

```tsx
it("shows retry instead of an empty history after a failed load", async () => {
  vi.mocked(apiFetch).mockRejectedValueOnce(new ApiError("Nie udało się pobrać historii.", 503));
  render(<SessionsPage />);
  expect(await screen.findByRole("button", { name: "Spróbuj ponownie" })).toBeTruthy();
});
```

- [ ] **Step 2: Run feedback tests to verify they fail**

Run: `cd frontend && npm test -- menu/sessions/page.test.tsx menu/admin/page.test.tsx menu/privacy/page.test.tsx`

Expected: FAIL because initial loads have no explicit state or retry.

- [ ] **Step 3: Implement local load states and privacy failure handling**

Extract one `load` callback per Sessions/Admin screen, set `loading` before
fetch, and render loading, error + `Spróbuj ponownie`, ready-empty, or ready
list distinctly. Privacy catches `ApiError`, disables active controls, and
reports export/erase success or failure in Polish.

- [ ] **Step 4: Make Interview independently scrollable**

Keep `/simulation` behavior unchanged. For `/interview`, make its root and
its content child `min-h-0 flex-1 overflow-y-auto`, including the no-session
setup form. Add `pb-[calc(1.5rem+env(safe-area-inset-bottom))]` to the
scrolling content. Do not rely on body scroll.

- [ ] **Step 5: Run feedback/layout tests and lint**

Run: `cd frontend && npm test -- menu/sessions/page.test.tsx menu/admin/page.test.tsx menu/privacy/page.test.tsx interview/page.test.tsx && npm run lint`

Expected: PASS with no ESLint warnings.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/menu frontend/src/components/ViewportFrame.tsx frontend/src/app/interview/page.tsx frontend/src/app/globals.css frontend/src/app/menu
git commit -m "fix: expose retry states and mobile scrolling"
```

### Task 6: Full verification

**Files:**
- Modify only if a test reveals an in-scope defect.

- [ ] **Step 1: Run backend suite**

Run: `cd backend && python -m pytest`

Expected: PASS.

- [ ] **Step 2: Run frontend suite and production build**

Run: `cd frontend && npm test && npm run lint && npm run build`

Expected: PASS.

- [ ] **Step 3: Review the final diff**

Run: `git diff main...HEAD --check && git status --short`

Expected: no whitespace errors and no generated/cache files.

- [ ] **Step 4: Commit only a correction created by Steps 1-3**

Stage the exact files changed by a failing verification command and commit with
`test: correct durable AI work regression`. Do not create an empty commit.
