import uuid

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from app.auth.deps import get_current_user, require_approved
from app.db import get_db
from app.main import app
from app.models import User
from app.settings import get_settings

ADMIN_EMAIL = "fifmazurkiewicz@gmail.com"


class FakeSession:
    def __init__(self, existing: User | None = None) -> None:
        self.existing = existing
        self.added: User | None = None
        self.commits = 0

    def get(self, _model, _pk):  # noqa: ANN001
        return self.existing

    def add(self, obj: User) -> None:
        self.added = obj
        self.existing = obj

    def commit(self) -> None:
        self.commits += 1

    def refresh(self, _obj: User) -> None:
        return None

    def scalar(self, _stmt=None):  # noqa: ANN001
        return 0


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _unsigned_token(user_id: uuid.UUID, email: str) -> str:
    return jwt.encode(
        {"sub": str(user_id), "email": email, "user_metadata": {"full_name": "Test"}},
        key="",
        algorithm="none",
    )


def test_new_non_admin_insert_is_not_approved():
    db = FakeSession()
    token = _unsigned_token(uuid.uuid4(), "doctor@example.com")
    user = get_current_user(_creds(token), db)
    assert user.is_approved is False
    assert db.added is not None
    assert db.added.is_approved is False
    assert db.added.spend_cap_usd == 10


def test_admin_allowlist_insert_is_approved():
    db = FakeSession()
    token = _unsigned_token(uuid.uuid4(), ADMIN_EMAIL)
    user = get_current_user(_creds(token), db)
    assert user.is_admin is True
    assert user.is_approved is True


def test_dev_token_insert_is_approved_admin():
    db = FakeSession()
    user = get_current_user(_creds("dev-token"), db)
    assert user.is_admin is True
    assert user.is_approved is True
    assert str(user.id) == "00000000-0000-4000-8000-000000000001"


def test_later_request_does_not_flip_is_approved_for_allowlist_email():
    existing = User(
        id=uuid.uuid4(),
        email=ADMIN_EMAIL,
        is_admin=True,
        is_approved=False,
        spend_cap_usd=10,
    )
    db = FakeSession(existing=existing)
    token = _unsigned_token(existing.id, ADMIN_EMAIL)
    user = get_current_user(_creds(token), db)
    assert user.is_approved is False
    assert db.added is None


def test_require_approved_returns_403_account_pending_approval():
    pending = User(id=uuid.uuid4(), email="wait@example.com", is_approved=False)
    with pytest.raises(HTTPException) as exc:
        require_approved(pending)
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "account_pending_approval"
    assert "akceptacj" in exc.value.detail["message"].lower()


def test_require_approved_passes_when_approved():
    approved = User(id=uuid.uuid4(), email="ok@example.com", is_approved=True)
    assert require_approved(approved) is approved


def test_me_allows_unapproved_user():
    pending = User(
        id=uuid.uuid4(),
        email="wait@example.com",
        is_admin=False,
        is_approved=False,
        spend_cap_usd=10,
    )

    app.dependency_overrides[get_current_user] = lambda: pending
    app.dependency_overrides[get_db] = lambda: FakeSession(existing=pending)
    try:
        response = TestClient(app).get("/api/auth/me")
        assert response.status_code == 200
        body = response.json()
        assert body["is_approved"] is False
        assert body["email"] == "wait@example.com"
        assert body["spend_cap_usd"] == 10
    finally:
        app.dependency_overrides.clear()


def test_me_requires_authorization(sqlite_client: TestClient):
    response = sqlite_client.get("/api/auth/me")
    assert response.status_code == 401


def test_feature_route_forbidden_when_unapproved():
    pending = User(id=uuid.uuid4(), email="wait@example.com", is_admin=False, is_approved=False)

    app.dependency_overrides[get_current_user] = lambda: pending
    app.dependency_overrides[get_db] = lambda: FakeSession(existing=pending)
    try:
        response = TestClient(app).post("/api/patients/next", json={})
        assert response.status_code == 403
        body = response.json()
        detail = body.get("detail", body)
        assert detail["code"] == "account_pending_approval"
    finally:
        app.dependency_overrides.clear()


def test_dev_auth_allowed_only_without_supabase_url(monkeypatch):
    monkeypatch.setenv("DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    get_settings.cache_clear()
    try:
        assert get_settings().dev_auth_allowed is False
    finally:
        monkeypatch.setenv("SUPABASE_URL", "")
        get_settings.cache_clear()
