from contextlib import asynccontextmanager
import os
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

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs_configured = False
    if os.getenv("ENABLE_SENTIMENT_SCHEDULER", "false").lower() == "true":
        scheduler.add_job(run_pipeline, "cron", hour=23, minute=0, id="sentiment-2300", replace_existing=True)
        scheduler.add_job(run_pipeline, "cron", hour=23, minute=30, id="sentiment-2330", replace_existing=True)
        scheduler.add_job(run_pipeline, "cron", hour=1, minute=0, id="sentiment-0100", replace_existing=True)
        scheduler.add_job(run_pipeline, "cron", hour=3, minute=0, id="sentiment-0300", replace_existing=True)
        jobs_configured = True

    if os.getenv("ENABLE_EMAIL_NOTIFICATION_SCHEDULER", "false").lower() == "true":
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
