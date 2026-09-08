import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.tokens import DEV_TOKEN, decode_access_token
from app.db import get_db
from app.models import User
from app.settings import get_settings

security = HTTPBearer(auto_error=False)

ACCOUNT_PENDING_CODE = "account_pending_approval"
ACCOUNT_PENDING_MESSAGE = "Konto oczekuje na akceptację administratora."


def require_approved(user: User) -> User:
    if not user.is_approved:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": ACCOUNT_PENDING_CODE, "message": ACCOUNT_PENDING_MESSAGE},
        )
    return user


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    settings = get_settings()
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization")

    payload = decode_access_token(creds.credentials, settings)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = uuid.UUID(sub)
    user = db.get(User, user_id)
    if user is None:
        email = payload.get("email")
        is_admin = (email or "").lower() in settings.admin_email_set
        if settings.dev_auth_allowed and creds.credentials == DEV_TOKEN:
            is_admin = True
        meta = payload.get("user_metadata") or {}
        display_name = meta.get("full_name") if isinstance(meta, dict) else None
        user = User(
            id=user_id,
            email=email,
            display_name=display_name,
            is_admin=is_admin,
            is_approved=is_admin,
            spend_cap_usd=10,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif settings.dev_auth_allowed and creds.credentials == DEV_TOKEN and not user.is_admin:
        user.is_admin = True
        db.commit()
        db.refresh(user)
    return user


def get_approved_user(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    return require_approved(user)
