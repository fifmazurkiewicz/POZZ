from fastapi import APIRouter

from app.api.routes import (
    admin,
    auth,
    conversations,
    health,
    interviews,
    patients,
    privacy,
    voice,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(interviews.router, prefix="/interviews", tags=["interviews"])
api_router.include_router(
    conversations.router, prefix="/conversations", tags=["conversations"]
)
api_router.include_router(privacy.router, prefix="/privacy", tags=["privacy"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
