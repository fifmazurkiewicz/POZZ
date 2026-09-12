from __future__ import annotations

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import UsageLedger
from app.settings import get_settings


def month_start() -> datetime:
    settings = get_settings()
    tz = ZoneInfo(settings.spend_cap_tz)
    now = datetime.now(tz)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def monthly_spend_usd(db: Session, user_id: uuid.UUID) -> float:
    start = month_start()
    total = db.scalar(
        select(func.coalesce(func.sum(UsageLedger.cost_usd), 0)).where(
            UsageLedger.user_id == user_id,
            UsageLedger.created_at >= start,
        )
    )
    return float(total or 0)
