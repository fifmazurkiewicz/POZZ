import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.deps import get_admin_user
from app.db import get_db
from app.models import UsageLedger, User
from app.spend.service import month_start, monthly_spend_usd

router = APIRouter()

CANNOT_SELF_REVOKE_CODE = "cannot_self_revoke"
CANNOT_SELF_REVOKE_MESSAGE = "Nie możesz cofnąć akceptacji własnego konta."


class PatchUserBody(BaseModel):
    is_approved: bool | None = None
    spend_cap_usd: float | None = Field(default=None, ge=0, le=10_000)


def _user_payload(user: User, spent: float) -> dict:
    created = user.created_at.isoformat() if user.created_at else None
    cap = float(user.spend_cap_usd)
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "is_admin": user.is_admin,
        "is_approved": user.is_approved,
        "spend_cap_usd": cap,
        "monthly_spend_usd": spent,
        "at_cap": cap > 0 and spent >= cap,
        "created_at": created,
    }


@router.get("/users")
def list_users(
    _admin: Annotated[User, Depends(get_admin_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    rows = db.scalars(select(User).order_by(User.is_approved.asc(), User.created_at.desc())).all()
    spent_by_user = dict(
        db.execute(
            select(UsageLedger.user_id, func.sum(UsageLedger.cost_usd))
            .where(UsageLedger.created_at >= month_start())
            .group_by(UsageLedger.user_id)
        ).all()
    )
    return {
        "users": [
            _user_payload(row, float(spent_by_user.get(row.id, 0) or 0)) for row in rows
        ]
    }


@router.patch("/users/{user_id}")
def patch_user(
    user_id: uuid.UUID,
    body: PatchUserBody,
    admin: Annotated[User, Depends(get_admin_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if target.id == admin.id and body.is_approved is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": CANNOT_SELF_REVOKE_CODE, "message": CANNOT_SELF_REVOKE_MESSAGE},
        )
    if body.is_approved is not None:
        target.is_approved = body.is_approved
    if body.spend_cap_usd is not None:
        target.spend_cap_usd = body.spend_cap_usd
    db.commit()
    db.refresh(target)
    return _user_payload(target, monthly_spend_usd(db, target.id))
