# Sentiment Analysis ML Pipeline — Design Spec
**Date:** 2026-05-24  
**Author:** Bali Pratama Tok  
**Scope:** `backend/services/sentiment/` + scheduler + API endpoints

---

## 1. Overview

A nightly batch pipeline that fetches financial news headlines for a fixed watchlist of stocks, scores them with FinBERT, and stores results in Supabase. The frontend reads pre-computed scores via a REST endpoint — FinBERT never runs on a user request.

---

## 2. Architecture

```
APScheduler (nightly cron, inside FastAPI)
        │
        ▼
sentiment_pipeline.py  ← orchestrator, loops over WATCHLIST
        │
        ├──► finnhub_service.py      → FinnHub /company-news API
        ├──► news_scraper_service.py → NewsAPI /everything
        │
        ▼
   [raw headlines: {headline, source, published_at}]
        │
        ▼
finbert_service.py  ← lazy-loads ProsusAI/finbert, scores in batches of 16
                      returns: {label, score} per headline
        │
        ▼
sentiment_aggregator.py  ← upserts rows to sentiment_scores table
                           computes daily avg score + label per symbol
        │
        ▼
Supabase (sentiment_scores table)
        │
        ▼
GET /api/stocks/{symbol}/sentiment  ← reads from Supabase
POST /api/sentiment/run-pipeline    ← manual trigger
```

---

## 3. Database Schema

Table: `sentiment_scores`

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | auto-generated PK |
| `symbol` | `text` | e.g. `AAPL` |
| `headline` | `text` | raw headline text |
| `source` | `text` | `finnhub`, `newsapi`, `rss`, `reddit`, etc. |
| `published_at` | `timestamptz` | article publication time |
| `label` | `text` | `positive`, `negative`, `neutral` |
| `score` | `float4` | FinBERT confidence (0.0–1.0) |
| `model_version` | `text` | e.g. `ProsusAI/finbert` |
| `created_at` | `timestamptz` | auto-generated insert time |

**Upsert key:** `symbol + headline + published_at`  
**Indexes:** `symbol`, `published_at`

---

## 4. Service Components

### `backend/services/sentiment/`

**`finnhub_service.py`**
- `fetch_news(symbol, from_date) → list[dict]`
- Calls FinnHub `/company-news`
- Returns `[{headline, source: "finnhub", published_at}]`
- Retry: exponential backoff, max 3 attempts on 429/timeout
- 0.5s sleep between symbols to respect 60 calls/min limit

**`news_scraper_service.py`**
- `fetch_news(symbol, company_name, from_date) → list[dict]`
- Calls NewsAPI `/everything` with company name as query
- Returns same shape as finnhub_service
- On 429/quota exceeded: skip entirely for this run (daily quota — backoff won't help)

**`finbert_service.py`**
- Module-level `_model = None`, `_tokenizer = None` (lazy load)
- `load_model()` — loads `ProsusAI/finbert` from HuggingFace cache on first call
- `score_headlines(headlines: list[str]) → list[dict]`
- Batch size: 16 headlines per inference call
- On load failure: raises exception immediately — pipeline aborts this run

**`sentiment_aggregator.py`**
- `save_scores(symbol, scored_headlines) → dict` — upserts rows to `sentiment_scores`
- `get_sentiment_summary(symbol, days=7) → dict` — reads from Supabase, returns:
  - `daily_scores`: `[{date, avg_score, label, headline_count}]`
  - `headlines`: raw list of recent scored headlines
- Daily label derivation from avg_score: `> 0.6 → positive`, `< 0.4 → negative`, `else → neutral`

**`sentiment_pipeline.py`** (orchestrator)
- `WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "BABA"]`
- `run_pipeline() → dict` — main entry point
- Idempotent: checks if today's data exists for each symbol before processing — skips if present
- Per-symbol try/except: one failure does not abort the run
- Returns summary: `{symbols_processed, results: [{symbol, headlines_scored, status}]}`

---

## 5. API Endpoints

### `GET /api/stocks/{symbol}/sentiment`

Returns aggregated daily scores + raw headlines for the last 7 days.

**Response:**
```json
{
  "symbol": "AAPL",
  "daily_scores": [
    {
      "date": "2026-05-24",
      "avg_score": 0.74,
      "label": "positive",
      "headline_count": 12
    }
  ],
  "headlines": [
    {
      "headline": "Apple hits record revenue in Q2",
      "source": "finnhub",
      "published_at": "2026-05-24T09:30:00Z",
      "label": "positive",
      "score": 0.91
    }
  ]
}
```

### `POST /api/sentiment/run-pipeline`

Manual trigger. Runs the full pipeline synchronously.

**Response:**
```json
{
  "message": "Pipeline complete",
  "symbols_processed": 10,
  "results": [
    {"symbol": "AAPL", "headlines_scored": 14, "status": "ok"},
    {"symbol": "BABA", "headlines_scored": 0, "status": "no_data"}
  ]
}
```

---

## 6. Error Handling & Retry Strategy

### Level 1 — Per-API-call (inside each service)

Exponential backoff: wait 2s → 4s → give up (3 attempts max).

| Error | Handling |
|---|---|
| FinnHub 429 | Backoff + retry, max 3 attempts |
| NewsAPI 429 / quota exceeded | Skip NewsAPI for this entire run |
| Network timeout | Retry 3× with backoff |
| Empty / malformed response | Treat as `no_data`, do not retry |
| Supabase write failure | Retry 3× with backoff |

### Level 2 — Pipeline-level retry (scheduler)

Pipeline is **idempotent** — checks for existing today's data per symbol before running.

Scheduled attempts:
```
11:00pm — primary run
11:30pm — retry (skips symbols already complete)
01:00am — retry
03:00am — final retry
```

If a symbol still has no data after the 3am run, it is logged as missing for that day. No partial or corrupt writes.

### FinBERT failures
- Load failure → abort entire pipeline run, log error, defer to next scheduled attempt
- Never attempt scoring with a partially loaded model

---

## 7. Scheduler Setup

APScheduler runs inside FastAPI. No separate process required.

```python
# main.py
from apscheduler.schedulers.background import BackgroundScheduler
from services.sentiment.sentiment_pipeline import run_pipeline

scheduler = BackgroundScheduler()
scheduler.add_job(run_pipeline, "cron", hour=23, minute=0)
scheduler.add_job(run_pipeline, "cron", hour=23, minute=30)
scheduler.add_job(run_pipeline, "cron", hour=1,  minute=0)
scheduler.add_job(run_pipeline, "cron", hour=3,  minute=0)
scheduler.start()
```

Shuts down cleanly via FastAPI lifespan handler.

---

## 8. Extensibility

Adding a new news source (Reddit, RSS, etc.) requires:
1. A new `xxx_service.py` returning `[{headline, source: "xxx", published_at}]`
2. One extra call in `sentiment_pipeline.py`
3. No changes to schema, FinBERT, aggregator, or API

The `source` column in `sentiment_scores` is the only coupling point.

---

## 9. Testing Checkpoints

| What | How |
|---|---|
| FinBERT scores correctly | Run `score_headlines(["Apple profits up", "Tesla recall"])` directly, verify labels |
| FinnHub fetcher returns data | Call `fetch_news("AAPL", from_date=yesterday)`, verify non-empty list |
| NewsAPI fetcher returns data | Same for `fetch_news("AAPL", "Apple", from_date=yesterday)` |
| Pipeline end-to-end | `POST /api/sentiment/run-pipeline`, check Supabase rows inserted |
| Read endpoint | `GET /api/stocks/AAPL/sentiment`, verify response shape |
| Retry logic | Use bad API key, confirm pipeline logs errors and completes without crash |
| Idempotency | Run pipeline twice in a row, confirm no duplicate rows in Supabase |

---

## 10. Out of Scope (MVP)

- RSS scrapers (Reuters, MarketWatch) — stretch goal, add via extensibility pattern above
- Reddit sentiment (`praw`) — stretch goal
- Sentiment as ML model input feature (XGBoost/LSTM ensemble) — stretch goal per PRD
- SHAP explainability — out of scope per PRD
