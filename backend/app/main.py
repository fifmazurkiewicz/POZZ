import logging

import httpx
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title="POZZ API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")


PROVIDER_ERROR_MESSAGE = "Nie udało się wykonać operacji. Spróbuj ponownie."


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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
        logger.warning("Provider failure on %s %s: %s", request.method, request.url.path, exc)
    else:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": "provider_error", "message": PROVIDER_ERROR_MESSAGE}},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "pozz", "docs": "/docs"}
