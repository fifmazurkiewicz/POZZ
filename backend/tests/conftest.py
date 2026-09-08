from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.pop("DATABASE_URL", None)
os.environ["VOICE_MODE"] = "speech_to_speech"
os.environ["TTS_PROVIDER"] = "elevenlabs"
os.environ["DEV_AUTH_ENABLED"] = "true"
os.environ["SUPABASE_URL"] = ""
os.environ["OPENROUTER_API_KEY"] = ""


@pytest.fixture
def db_engine():
    from app.db import Base
    import app.models  # noqa: F401

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Session:
    session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sqlite_client(db_engine) -> TestClient:
    from app.db import get_db
    from app.main import app

    SessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)

    def _get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
