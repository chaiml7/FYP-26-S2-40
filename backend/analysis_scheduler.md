# Twice-daily analysis scheduler

The backend can refresh active-stock technical prices, pull and score sentiment
news, and generate technical predictions at 00:00 and 12:00 Singapore time.

```dotenv
ENABLE_ANALYSIS_SCHEDULER=true
ANALYSIS_SCHEDULER_TIMEZONE=Asia/Singapore
TECHNICAL_IMPORT_PERIOD=2y
```

Only enable the scheduler on one backend process. APScheduler runs inside that
process, so duplicate API replicas would create duplicate jobs. The job uses
`is_active=true` stocks for both technical and sentiment work. Its stages are
isolated: one failed provider or model is logged without discarding successful
stages.

The default two-year price refresh is enough for a newly activated stock to
produce 252-trading-day features without downloading ten years of data twice
per day. Earlier history already stored in Supabase is retained.

The deployed API must install `backend/requirements-ml.txt` and have enough
memory to load FinBERT and the active XGBoost artifact. An always-on process is
also required; a sleeping Render web service cannot reliably run in-process
midnight/noon jobs.
