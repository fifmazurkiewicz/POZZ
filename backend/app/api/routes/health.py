from fastapi.responses import JSONResponse

from fastapi import APIRouter

from app.db import ping_database
from app.settings import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness — fast, no dependencies. Render health check hits this."""
    return {"status": "ok", "service": "pozz"}


@router.get("/health/ready", response_model=None)
def ready():
    """Readiness — requires DATABASE_URL in Package 0 (no live ping yet)."""
    settings = get_settings()
    if not settings.database_url or not ping_database():
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "checks": {"database": "error"}},
        )
    return {"status": "ok", "checks": {"database": "ok"}}
