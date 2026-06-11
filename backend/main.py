from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from routes.financial_routes import router as financial_router
from routes.stock_routes import router as stock_router
from routes.user_routes import router as user_router
from services.sentiment.sentiment_pipeline import run_pipeline

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("ENABLE_SENTIMENT_SCHEDULER", "false").lower() == "true":
        scheduler.add_job(run_pipeline, "cron", hour=23, minute=0)
        scheduler.add_job(run_pipeline, "cron", hour=23, minute=30)
        scheduler.add_job(run_pipeline, "cron", hour=1,  minute=0)
        scheduler.add_job(run_pipeline, "cron", hour=3,  minute=0)
        scheduler.start()

    yield

    if scheduler.running:
        scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
