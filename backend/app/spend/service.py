from __future__ import annotations

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import UsageLedger
from app.settings import get_settings


def monthly_spend_usd(db: Session, user_id: uuid.UUID) -> float:
    settings = get_settings()
    tz = ZoneInfo(settings.spend_cap_tz)
    now = datetime.now(tz)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = db.scalar(
        select(func.coalesce(func.sum(UsageLedger.cost_usd), 0)).where(
            UsageLedger.user_id == user_id,
            UsageLedger.created_at >= start,
        )
    )
    return float(total or 0)
