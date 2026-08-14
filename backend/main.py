from contextlib import asynccontextmanager
import os
import logging
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from backend.routes.financial_routes import router as financial_router
from backend.routes.notification_routes import router as notification_router
from backend.routes.billing_routes import router as billing_router
from backend.routes.stock_routes import router as stock_router
from backend.routes.technical_routes import router as technical_router
from backend.routes.user_routes import router as user_router
from backend.services.sentiment.sentiment_pipeline import run_pipeline
from backend.services.notification_service import dispatch_analysis_ready_emails
from backend.services.analysis_pipeline import run_scheduled_analysis_pipeline

scheduler = BackgroundScheduler()
logger = logging.getLogger(__name__)


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() == "true"


def _analysis_timezone() -> ZoneInfo:
    timezone_name = os.getenv("ANALYSIS_SCHEDULER_TIMEZONE", "Asia/Singapore")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"Invalid ANALYSIS_SCHEDULER_TIMEZONE: {timezone_name}"
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs_configured = False
    analysis_scheduler_enabled = _enabled("ENABLE_ANALYSIS_SCHEDULER")
    if analysis_scheduler_enabled:
        scheduler.add_job(
            run_scheduled_analysis_pipeline,
            "cron",
            hour="0,12",
            minute=0,
            timezone=_analysis_timezone(),
            id="analysis-refresh-0000-1200",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        jobs_configured = True
    elif _enabled("ENABLE_SENTIMENT_SCHEDULER"):
        logger.warning(
            "ENABLE_SENTIMENT_SCHEDULER is deprecated; use "
            "ENABLE_ANALYSIS_SCHEDULER for the combined pipeline."
        )
        scheduler.add_job(
            run_pipeline,
            "cron",
            hour="0,12",
            minute=0,
            timezone=_analysis_timezone(),
            id="sentiment-legacy-0000-1200",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        jobs_configured = True

    if _enabled("ENABLE_EMAIL_NOTIFICATION_SCHEDULER"):
        scheduler.add_job(
            dispatch_analysis_ready_emails,
            "interval",
            minutes=15,
            id="analysis-ready-email-dispatch",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        jobs_configured = True

    if jobs_configured and not scheduler.running:
        scheduler.start()

    yield

    if scheduler.running:
        scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Backend REST API is running"}


app.include_router(stock_router, prefix="/api")
app.include_router(user_router, prefix="/api")
app.include_router(financial_router, prefix="/api")
app.include_router(technical_router, prefix="/api")
app.include_router(notification_router, prefix="/api")
app.include_router(billing_router, prefix="/api")
