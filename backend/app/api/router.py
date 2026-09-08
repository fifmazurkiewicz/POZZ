from fastapi import APIRouter

from app.api.routes import auth, conversations, health, patients, voice

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
