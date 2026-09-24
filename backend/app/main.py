import logging

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.router import api_router
from app.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

PROVIDER_ERROR_MESSAGE = "Nie udało się wykonać operacji. Spróbuj ponownie."


def _provider_error_detail(exc: httpx.HTTPError) -> tuple[str, str]:
    if not isinstance(exc, httpx.HTTPStatusError):
        return "provider_error", PROVIDER_ERROR_MESSAGE
    upstream_status = exc.response.status_code
    if upstream_status in {401, 403}:
        return "provider_auth_error", "Usługa AI nie jest poprawnie skonfigurowana."
    if upstream_status == 402:
        return "provider_credit_exhausted", "Brak środków na koncie usługi AI."
    if upstream_status == 404:
        return "provider_model_unavailable", "Skonfigurowany model AI jest niedostępny."
    if upstream_status == 429:
        return (
            "provider_rate_limited",
            "Usługa AI jest przeciążona. Spróbuj ponownie za chwilę.",
        )
    return "provider_error", PROVIDER_ERROR_MESSAGE


class UnhandledExceptionMiddleware(BaseHTTPMiddleware):
    """Translate uncaught failures before the response passes through CORS."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)
        # This is the deliberate last-resort application boundary. Specific
        # HTTP errors have already been handled by FastAPI inside this layer.
        except Exception as exc:  # noqa: BLE001
            return _error_response(request, exc)


def _error_response(request: Request, exc: Exception) -> JSONResponse:
    """Map uncaught provider/network failures to a Polish JSON error.

    - `httpx.HTTPError` (timeouts, connection errors, upstream 4xx/5xx) → 502
      "provider_error" — the upstream is the failing party.
    - Anything else (DB, programming errors) → 503 — temporary backend failure.

    `HTTPException` / `RequestValidationError` keep their existing handlers
    so existing structured codes (`spend_cap_exceeded`, `conversation_completed`)
    continue to flow through unchanged.
    """
    if isinstance(exc, httpx.HTTPError):
        status_code = status.HTTP_502_BAD_GATEWAY
        error_code, error_message = _provider_error_detail(exc)
        logger.warning(
            "Provider failure on %s %s: %s", request.method, request.url.path, exc
        )
    else:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error_code, error_message = "provider_error", PROVIDER_ERROR_MESSAGE
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": error_code, "message": error_message}},
    )


app = FastAPI(title="POZZ API", version="0.1.0")
# Middleware added later is outermost. CORS must wrap the error translator so
# even an uncaught provider failure remains readable by the browser.
app.add_middleware(UnhandledExceptionMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86_400,
)
app.include_router(api_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "pozz", "docs": "/docs"}
