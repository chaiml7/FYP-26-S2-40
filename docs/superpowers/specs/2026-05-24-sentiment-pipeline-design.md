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

## 9. Testing Framework

### Structure

```
backend/
  tests/
    sentiment/
      __init__.py
      conftest.py                   ← shared fixtures and mocks
      test_finbert_service.py
      test_finnhub_service.py
      test_news_scraper_service.py
      test_sentiment_aggregator.py
      test_sentiment_pipeline.py
      test_sentiment_routes.py

scripts/
  test_sentiment_manual.py          ← manual end-to-end test script (runs against real APIs)
```

**Dependencies:** `pytest`, `pytest-mock`, `httpx` (for FastAPI TestClient)

---

### Unit Tests (automated, no real APIs)

All external calls (FinnHub, NewsAPI, Supabase, HuggingFace) are mocked.

**`conftest.py` — shared fixtures:**
- `mock_headlines` — sample list of `{headline, source, published_at}` dicts
- `mock_scored_headlines` — same with `label` and `score` added
- `mock_supabase` — mocked Supabase client (patches `supabase_client.supabase`)
- `finbert_not_loaded` — fixture that ensures `_model` and `_tokenizer` are reset to `None` before each test

---

**`test_finbert_service.py`**

| Test | What it checks |
|---|---|
| `test_lazy_load_on_first_call` | `_model` is None before call, not None after |
| `test_score_returns_label_and_score` | Output has `label` in `{positive, negative, neutral}` and `score` in `[0, 1]` |
| `test_score_empty_list` | Returns `[]`, does not raise |
| `test_score_single_headline` | Works with list of 1 item |
| `test_score_batch_larger_than_16` | 20 headlines processed correctly across two batches |
| `test_headline_over_512_tokens` | Very long headline is truncated, does not raise |
| `test_load_failure_raises` | Patched loader raises → `score_headlines` propagates exception |
| `test_model_not_reloaded_on_second_call` | `load_model()` called twice → model loaded only once (check call count) |

---

**`test_finnhub_service.py`**

| Test | What it checks |
|---|---|
| `test_fetch_returns_headlines` | Mocked 200 response → returns list of dicts with correct keys |
| `test_fetch_empty_response` | API returns `[]` → returns `[]`, no exception |
| `test_fetch_malformed_json` | API returns garbage → returns `[]`, logs error |
| `test_retry_on_429` | First call returns 429, second returns 200 → retries and succeeds |
| `test_retry_exhausted` | All 3 attempts return 429 → raises exception after backoff |
| `test_retry_on_timeout` | `requests.Timeout` on first call → retries, succeeds on second |
| `test_rate_limit_sleep` | Verify 0.5s sleep is called between symbol fetches |
| `test_published_at_format` | `published_at` is a valid ISO 8601 string |

---

**`test_news_scraper_service.py`**

| Test | What it checks |
|---|---|
| `test_fetch_returns_headlines` | Mocked 200 → returns list with correct shape |
| `test_fetch_empty_response` | API returns no articles → returns `[]` |
| `test_quota_exceeded_skips` | 429 response → returns `[]` immediately, does NOT retry |
| `test_quota_exceeded_does_not_retry` | Confirm retry is not attempted on 429 (check call count = 1) |
| `test_malformed_response` | Missing `articles` key → returns `[]`, no crash |
| `test_network_timeout_retries` | Timeout → retries up to 3×, then returns `[]` |

---

**`test_sentiment_aggregator.py`**

| Test | What it checks |
|---|---|
| `test_save_scores_upserts_rows` | Mocked Supabase upsert called with correct data shape |
| `test_save_scores_empty_list` | Empty input → upsert not called, returns gracefully |
| `test_label_positive_threshold` | avg_score = 0.61 → label = `positive` |
| `test_label_negative_threshold` | avg_score = 0.39 → label = `negative` |
| `test_label_neutral_threshold` | avg_score = 0.50 → label = `neutral` |
| `test_label_boundary_exact_06` | avg_score = 0.60 → label = `neutral` (boundary inclusive check) |
| `test_get_summary_returns_correct_shape` | Mocked Supabase select → response has `daily_scores` and `headlines` keys |
| `test_get_summary_empty_db` | No rows in DB → returns `{daily_scores: [], headlines: []}` |
| `test_daily_aggregation_groups_by_date` | 5 headlines on same date → one entry in `daily_scores` with `headline_count: 5` |
| `test_supabase_write_retry` | First upsert fails → retries up to 3×, succeeds on second |
| `test_supabase_write_all_retries_fail` | All 3 upsert attempts fail → raises exception |

---

**`test_sentiment_pipeline.py`**

| Test | What it checks |
|---|---|
| `test_run_pipeline_processes_all_watchlist` | All 10 symbols attempted |
| `test_idempotency_skips_existing` | Symbol already has today's data → fetch/score not called for that symbol |
| `test_idempotency_processes_missing` | Symbol missing today's data → fetch/score called |
| `test_one_symbol_failure_continues` | FinnHub raises for AAPL → pipeline continues to TSLA |
| `test_all_sources_called_per_symbol` | Both FinnHub and NewsAPI called for each symbol |
| `test_newsapi_skip_does_not_abort` | NewsAPI returns `[]` (quota) → FinnHub results still scored and saved |
| `test_finbert_load_failure_aborts_run` | FinBERT raises → pipeline aborts, returns error status |
| `test_no_headlines_returns_no_data_status` | Both fetchers return `[]` → symbol status is `no_data` |
| `test_result_shape` | Return dict has `symbols_processed` and `results` keys |
| `test_pipeline_run_at_midnight` | `from_date` correctly resolves to yesterday across midnight boundary |

---

**`test_sentiment_routes.py`** (FastAPI TestClient)

| Test | What it checks |
|---|---|
| `test_get_sentiment_200` | Valid symbol with data → 200, correct response shape |
| `test_get_sentiment_404_no_data` | Valid symbol, no rows in DB → 404 with detail message |
| `test_get_sentiment_symbol_uppercased` | `aapl` → treated same as `AAPL` |
| `test_run_pipeline_200` | `POST /api/sentiment/run-pipeline` → 200, returns summary |
| `test_run_pipeline_finbert_failure` | FinBERT mock raises → 500 with error detail |

---

### Manual Test Script (`scripts/test_sentiment_manual.py`)

Run directly against real APIs and real Supabase. Prints pass/fail for each step.

```
python scripts/test_sentiment_manual.py
```

**Steps executed:**

1. **FinBERT smoke test** — score 3 known headlines, print labels + scores. Verify labels match expectations.
2. **FinnHub fetch** — fetch AAPL news for yesterday, print headline count and first 3 headlines.
3. **NewsAPI fetch** — fetch AAPL / "Apple" news, print headline count.
4. **Pipeline trigger** — call `run_pipeline()` directly (not via HTTP), print per-symbol results.
5. **Supabase row check** — query `sentiment_scores` for today, print row count per symbol.
6. **Idempotency check** — run pipeline again, confirm row count unchanged.
7. **API endpoint check** — hit `GET /api/stocks/AAPL/sentiment` via `httpx`, print `daily_scores` and first 3 headlines.
8. **Error simulation** — temporarily replace FinnHub API key with invalid value, confirm pipeline logs error for all symbols without crashing, confirm `no_data` status returned.
9. **Rate limit simulation** — mock a 429 from FinnHub mid-run, confirm retry logic fires and logs backoff.

Each step prints `[PASS]` or `[FAIL: <reason>]`. Script exits with code 1 if any step fails.

---

### Running Tests

```bash
# All unit tests
cd backend && pytest tests/sentiment/ -v

# Single service
pytest tests/sentiment/test_finbert_service.py -v

# Manual end-to-end (requires running backend + real API keys)
python scripts/test_sentiment_manual.py
```

---

## 10. Out of Scope (MVP)

- RSS scrapers (Reuters, MarketWatch) — stretch goal, add via extensibility pattern above
- Reddit sentiment (`praw`) — stretch goal
- Sentiment as ML model input feature (XGBoost/LSTM ensemble) — stretch goal per PRD
- SHAP explainability — out of scope per PRD
