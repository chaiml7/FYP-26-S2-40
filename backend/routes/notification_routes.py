"""Administrative endpoints for previewing and dispatching ready-analysis emails."""

import os
from datetime import date

from fastapi import APIRouter, Header, HTTPException, Query

from backend.services.notification_service import (
    admin_key_is_valid,
    dispatch_analysis_ready_emails,
)


router = APIRouter(prefix="/notifications", tags=["notifications"])


def _require_notification_admin_key(value: str | None) -> None:
    if not os.getenv("NOTIFICATION_ADMIN_KEY"):
        raise HTTPException(
            status_code=503,
            detail="NOTIFICATION_ADMIN_KEY is not configured on the backend.",
        )
    if not admin_key_is_valid(value):
        raise HTTPException(status_code=403, detail="Invalid notification admin key.")


@router.post("/analysis-ready/dispatch")
def dispatch_ready_notifications(
    notification_date: date | None = Query(default=None),
    dry_run: bool = Query(
        default=True,
        description="Preview readiness without claiming or sending deliveries.",
    ),
    x_notification_admin_key: str | None = Header(default=None),
):
    _require_notification_admin_key(x_notification_admin_key)
    try:
        return dispatch_analysis_ready_emails(
            notification_date=notification_date,
            dry_run=dry_run,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
