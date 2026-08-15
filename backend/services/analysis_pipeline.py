"""Orchestrate the twice-daily technical and sentiment analysis refresh."""

import logging
import os
from datetime import datetime, timezone
from typing import Callable

from backend.services.sentiment.sentiment_pipeline import run_pipeline
from backend.services.technical.technical_service import (
    generate_all_technical_predictions,
    import_all_technical_prices,
)

logger = logging.getLogger(__name__)


def _run_stage(name: str, operation: Callable[[], dict]) -> dict:
    try:
        result = operation()
    except Exception as exc:
        logger.exception("Scheduled analysis stage %s failed", name)
        return {"status": "error", "error": str(exc)}

    status = "ok"
    if name == "technical_price_import" and any(
        row.get("status") == "error" for row in result.get("results", [])
    ):
        status = "partial"
    elif name == "sentiment_analysis" and any(
        row.get("status") == "error" for row in result.get("results", [])
    ):
        status = "partial"
    elif name == "technical_predictions" and result.get("stocks_failed", 0):
        status = "partial"

    return {"status": status, "result": result}


def run_scheduled_analysis_pipeline() -> dict:
    """Refresh active-stock inputs and then create the latest predictions."""
    started_at = datetime.now(timezone.utc)
    technical_period = os.getenv("TECHNICAL_IMPORT_PERIOD", "2y")

    stages = {
        "technical_price_import": _run_stage(
            "technical_price_import",
            lambda: import_all_technical_prices(
                period=technical_period,
                stock_scope="active",
            ),
        ),
        "sentiment_analysis": _run_stage(
            "sentiment_analysis",
            lambda: run_pipeline(refresh_existing=True),
        ),
        "technical_predictions": _run_stage(
            "technical_predictions",
            generate_all_technical_predictions,
        ),
    }

    stage_statuses = [stage["status"] for stage in stages.values()]
    if all(status == "ok" for status in stage_statuses):
        status = "ok"
    elif all(status == "error" for status in stage_statuses):
        status = "error"
    else:
        status = "partial"

    result = {
        "status": status,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "technical_period": technical_period,
        "stages": stages,
    }
    logger.info("Scheduled analysis pipeline completed with status %s", status)
    return result
