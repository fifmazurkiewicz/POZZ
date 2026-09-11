import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.models import User

ADMIN_EMAIL = "fifmazurkiewicz@gmail.com"


def _user(
    *,
    email: str,
    is_admin: bool = False,
    is_approved: bool = False,
    spend_cap_usd: float = 10,
) -> User:
    return User(
        id=uuid.uuid4(),
        email=email,
        display_name=email.split("@")[0],
        is_admin=is_admin,
        is_approved=is_approved,
        spend_cap_usd=spend_cap_usd,
    )


def _override(client: TestClient, user: User) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: user


def test_admin_lists_users_pending_first(sqlite_client: TestClient, db_session: Session):
    admin = _user(email=ADMIN_EMAIL, is_admin=True, is_approved=True)
    pending = _user(email="wait@example.com")
    approved = _user(email="ok@example.com", is_approved=True)
    db_session.add_all([admin, pending, approved])
    db_session.commit()
    _override(sqlite_client, admin)

    response = sqlite_client.get("/api/admin/users")
    assert response.status_code == 200, response.text
    users = response.json()["users"]
    emails = [row["email"] for row in users]
    assert "wait@example.com" in emails
    assert emails.index("wait@example.com") < emails.index("ok@example.com")
    first_pending = next(row for row in users if row["email"] == "wait@example.com")
    assert first_pending["is_approved"] is False
    assert first_pending["is_admin"] is False
    assert "id" in first_pending
    assert "display_name" in first_pending
    assert "created_at" in first_pending


def test_non_admin_cannot_list_users(sqlite_client: TestClient, db_session: Session):
    doctor = _user(email="doctor@example.com", is_approved=True)
    db_session.add(doctor)
    db_session.commit()
    _override(sqlite_client, doctor)

    response = sqlite_client.get("/api/admin/users")
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["code"] == "admin_required"


def test_unapproved_cannot_list_users(sqlite_client: TestClient):
    pending = _user(email="wait@example.com")
    _override(sqlite_client, pending)

    response = sqlite_client.get("/api/admin/users")
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["code"] == "account_pending_approval"


def test_admin_accepts_user_without_changing_cap(sqlite_client: TestClient, db_session: Session):
    admin = _user(email=ADMIN_EMAIL, is_admin=True, is_approved=True)
    pending = _user(email="wait@example.com", spend_cap_usd=10)
    db_session.add_all([admin, pending])
    db_session.commit()
    _override(sqlite_client, admin)

    response = sqlite_client.patch(
        f"/api/admin/users/{pending.id}",
        json={"is_approved": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_approved"] is True
    assert body["id"] == str(pending.id)

    db_session.expire_all()
    refreshed = db_session.get(User, pending.id)
    assert refreshed is not None
    assert refreshed.is_approved is True
    assert float(refreshed.spend_cap_usd) == 10


def test_admin_can_update_user_monthly_spend_cap(sqlite_client: TestClient, db_session: Session):
    admin = _user(email=ADMIN_EMAIL, is_admin=True, is_approved=True)
    user = _user(email="cap@example.com", is_approved=True)
    db_session.add_all([admin, user])
    db_session.commit()
    _override(sqlite_client, admin)

    response = sqlite_client.patch(f"/api/admin/users/{user.id}", json={"spend_cap_usd": 25.5})
    assert response.status_code == 200, response.text
    assert response.json()["spend_cap_usd"] == 25.5
    db_session.refresh(user)
    assert float(user.spend_cap_usd) == 25.5


def test_admin_cannot_self_revoke(sqlite_client: TestClient, db_session: Session):
    admin = _user(email=ADMIN_EMAIL, is_admin=True, is_approved=True)
    db_session.add(admin)
    db_session.commit()
    _override(sqlite_client, admin)

    response = sqlite_client.patch(
        f"/api/admin/users/{admin.id}",
        json={"is_approved": False},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "cannot_self_revoke"

    db_session.expire_all()
    refreshed = db_session.get(User, admin.id)
    assert refreshed is not None
    assert refreshed.is_approved is True


def test_admin_can_revoke_other_user(sqlite_client: TestClient, db_session: Session):
    admin = _user(email=ADMIN_EMAIL, is_admin=True, is_approved=True)
    other = _user(email="ok@example.com", is_approved=True)
    db_session.add_all([admin, other])
    db_session.commit()
    _override(sqlite_client, admin)

    response = sqlite_client.patch(
        f"/api/admin/users/{other.id}",
        json={"is_approved": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["is_approved"] is False


def test_patch_unknown_user_is_404(sqlite_client: TestClient, db_session: Session):
    admin = _user(email=ADMIN_EMAIL, is_admin=True, is_approved=True)
    db_session.add(admin)
    db_session.commit()
    _override(sqlite_client, admin)

    response = sqlite_client.patch(
        f"/api/admin/users/{uuid.uuid4()}",
        json={"is_approved": True},
    )
    assert response.status_code == 404
