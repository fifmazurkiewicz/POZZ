from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.deps import get_approved_user
from app.db import get_db
from app.models import User
from app.privacy.service import build_user_export, delete_user_content

router = APIRouter()


class DeleteContentBody(BaseModel):
    confirmation: Literal["USUŃ MOJE DANE"]


@router.get("/export")
def export_personal_data(
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Return a portable JSON copy of content owned by the signed-in user."""
    return build_user_export(db, user)


@router.delete("/content")
def erase_personal_content(
    _body: DeleteContentBody,
    user: Annotated[User, Depends(get_approved_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Erase app content while retaining the authentication/account record."""
    return {
        "status": "deleted",
        "account_retained": True,
        "deleted": delete_user_content(db, user),
    }
