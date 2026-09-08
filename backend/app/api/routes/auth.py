from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.deps import get_current_user
from app.db import get_db
from app.models import User
from app.spend.service import monthly_spend_usd

router = APIRouter()


@router.get("/me")
def me(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    cap = float(user.spend_cap_usd)
    spent = monthly_spend_usd(db, user.id)
    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "is_admin": user.is_admin,
        "is_approved": user.is_approved,
        "spend_cap_usd": cap,
        "monthly_spend_usd": spent,
        "at_cap": cap > 0 and spent >= cap,
        "onboarding_completed_at": (
            user.onboarding_completed_at.isoformat() if user.onboarding_completed_at else None
        ),
    }
