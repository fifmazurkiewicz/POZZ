# ADR — Unhandled exception shapes for interview endpoints

**Date:** 2026-09-11
**Status:** accepted
**Supersedes:** none

## Context

`POST /api/conversations/{id}/examinations` and `POST /api/conversations/{id}/finish` call
`get_text_provider().complete(...)` synchronously inside a SQLAlchemy session. Any provider
exception (`httpx.HTTPStatusError`, `httpx.TimeoutException`, `RuntimeError`, plain
`Exception`) bubbles out of the route handler, FastAPI's default exception handler converts
it to a `500 Internal Server Error` plain-text body, and the frontend's
`apiFetch` parses the non-JSON body and surfaces `"Internal Server Error"` to the user.

Reported failures (Sep 11):

- "Zrób badanie" → user typed `pomiar ciśnienia`, request failed with the same generic
  message in two consecutive attempts.
- "Zakończ wywiad" → user typed a short and a longer diagnosis, both failed identically.
- The frontend fallback string `Nie udało się wykonać operacji. Spróbuj ponownie.` is also
  reachable when the browser aborts the request (cold start, Cancel, network drop), but
  the screenshots point at a server-side response, not a transport abort.

The contract is broken on three axes:

1. **English body** in a Polish product.
2. **No diagnostic code** the frontend can branch on (e.g. to retry on cold start vs.
   surface a hard error).
3. **No traceback logging** at the access layer — Uvicorn logs the traceback but the API
   consumer sees only a 500.

## Decision

Add a single `app.exception_handler(Exception)` in `backend/app/main.py` that:

- Returns structured JSON `{"detail": {"code": ..., "message": "Nie udało się wykonać
  operacji. Spróbuj ponownie."}}` for any uncaught exception.
- Status code:
  - `502 Bad Gateway` for `httpx.HTTPStatusError` (provider 4xx/5xx) and any
    `httpx.HTTPError` (timeouts, network) — caller's fault is the upstream.
  - `503 Service Unavailable` for everything else — temporary backend failure.
- Logs the original exception via `logging.exception` so tracebacks stay visible in the
  Render logs.
- Leaves `HTTPException`, `RequestValidationError` and FastAPI's own handlers untouched —
  those already return Polish `{"code", "message"}` payloads (`spend_cap_exceeded`,
  `conversation_completed`, etc.).

The contract change is purely additive on the wire:

| Before | After |
|---|---|
| `500` text/plain `Internal Server Error` | `502/503` JSON `{"detail":{"code":"provider_error","message":"Nie udało się wykonać operacji. Spróbuj ponownie."}}` |

The frontend already displays `ApiError.message`, so it will now show the Polish string
without a code change. The `code` is preserved in `ApiError.code` for future branching.

The `parseApiError` helper already extracts `detail.message` from this shape.

## Consequences

- Provider 4xx/5xx becomes a single visible failure mode the user can retry, instead of
  surfacing as opaque English text.
- The existing tests (`test_failed_evaluation_leaves_conversation_open`) assert the
  exception is re-raised inside the route; the handler must still allow the test to
  observe `RuntimeError` (it uses `TestClient`, which propagates handler-added exceptions
  only when the test asserts on the raise). Update that test to assert on the JSON 502
  body instead.
- Future retries should remain manual; the cold-start banner is the right home for
  automatic wake-up signaling.