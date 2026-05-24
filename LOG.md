# StockLens — Team Activity Log

> Personal log — tracked on `bali` branch only. Not merged to `main`.  
> Format each entry: `### YYYY-MM-DD — [Who] — [What]`

---

## Sprint 1

### 2026-05-24 — Bali — Project Setup

**What we did:**
- Cloned repo from https://github.com/chaiml7/FYP-26-S2-40
- Reviewed existing codebase:
  - Backend: FastAPI with CRUD for `stocks`, yfinance OHLCV import, stock history retrieval
  - Frontend: Minimal React app — just a list of active stocks fetched from backend
  - DB: 3 tables in Supabase — `stocks`, `daily_ohlcv`, `predictions`
- Set up `bali` branch as personal working branch
- Created CLAUDE.md, LOG.md, .claude/settings.json
- Moved PTR and PRD into `docs/` folder
- Added `sentiment_scores` table to planned DB schema
- Read through PRD and Prelim Tech Report — confirmed Bali's scope is sentiment ML pipeline

**What I'm responsible for:**
- Sentiment analysis ML pipeline:
  - FinnHub API — company news fetching
  - FinBERT (ProsusAI/finbert) — financial sentiment scoring
  - NewsAPI / RSS scrapers — supplementary news sources
  - `sentiment_scores` Supabase table
  - `/api/stocks/{symbol}/sentiment` endpoint

**Next steps:**
- Create `sentiment_scores` table in Supabase
- Set up `backend/services/sentiment/` module
- Implement FinnHub news fetcher
- Implement FinBERT inference service
- Wire up API endpoint

**Problems encountered:**
- git clone on Windows mangled the path — fixed by running clone directly via bash
- frontend/.env is committed to the repo with the Supabase anon key (flagged for later cleanup)

**Learnt:**
- Project uses FastAPI not Flask (PRD says Flask but codebase uses FastAPI — going with FastAPI)
- Supabase anon key in .env is public-safe, but secret key should never be committed
- FinBERT model is large (~440MB) — need to plan for model caching / lazy loading

---

## Sprint 2

### 2026-05-24 — Bali — Sentiment Analysis ML Pipeline (full implementation)

**What I did:**
- Designed full sentiment pipeline via brainstorming + spec + implementation plan
- Implemented all sentiment services on branch `feature/bali-sentiment-pipeline` (branched from `bali`)
- Created `sentiment_scores` table in Supabase with RLS enabled
- All 63 unit tests passing (mocked — no real API calls)

**Services built (`backend/services/sentiment/`):**
- `finbert_service.py` — lazy-loads ProsusAI/finbert, batches inference (size 16), atomic model load
- `finnhub_service.py` — FinnHub `/company-news`, exponential backoff, 0.5s rate limit sleep
- `news_scraper_service.py` — NewsAPI `/everything`, quota-aware (429 = skip, no retry)
- `sentiment_aggregator.py` — Supabase upsert, daily avg + label, idempotency check (`has_data_for_today`)
- `sentiment_pipeline.py` — orchestrator, WATCHLIST of 10 symbols, per-symbol error isolation

**Other changes:**
- `routes/stock_routes.py` — added `GET /api/stocks/{symbol}/sentiment` + `POST /api/sentiment/run-pipeline`
- `main.py` — APScheduler nightly cron (11pm, 11:30pm, 1am, 3am) via FastAPI lifespan
- `requirements.txt` — added transformers, torch, APScheduler, requests, pytest, pytest-mock, httpx
- `backend/.env.example` — template with all 4 required keys
- `scripts/test_sentiment_manual.py` — 8-step E2E test script (requires running backend + real API keys)

**Tests (`backend/tests/sentiment/`):**
- `test_finbert_service.py`, `test_finnhub_service.py`, `test_news_scraper_service.py`
- `test_sentiment_aggregator.py`, `test_sentiment_pipeline.py`, `test_sentiment_routes.py`
- `conftest.py` — shared fixtures and mocks

**Key bugs fixed during implementation:**
- FinBERT partial load: atomic assignment (load both locals first, then assign globals)
- FinnHub epoch-0 timestamp: guard `item.get("datetime")` before use
- Pipeline mock mutation: `list()` copy of fetch results prevents cross-iteration mutation
- NewsAPI key name mismatch: `.env` had `NEWS_API_KEY`, code expects `NEWSAPI_KEY` — fixed

**Current state:**
- Branch `feature/bali-sentiment-pipeline` has 14 commits, all LOCAL ONLY (not pushed yet)
- `.env` is complete with all 4 keys (FINNHUB_API_KEY, NEWSAPI_KEY, SUPABASE_URL, SUPABASE_SECRET_KEY)
- Next session: run `scripts/test_sentiment_manual.py` to verify real APIs work, then push branch + open PR to `main`

**Next steps:**
- Start backend server: `cd backend && uvicorn main:app --reload`
- Run manual E2E test: `python scripts/test_sentiment_manual.py`
- If all pass: push `feature/bali-sentiment-pipeline` → PR to `main` (code only, no docs/.claude/LOG.md)
- Then start next phase (ML models: XGBoost/LSTM, or frontend charts)

---

## Issues / Bugs Tracker

| Date | Issue | Status | Resolution |
|---|---|---|---|
| 2026-05-24 | frontend/.env committed with anon key | Open | Add to .gitignore cleanup task |
| 2026-05-24 | PRD says Flask, codebase uses FastAPI | Resolved | Using FastAPI, noted discrepancy |

---

## Key Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-05-24 | Use `bali` branch for personal files, PRs to main for code only | Keep main clean, share context across machines |
| 2026-05-24 | FinBERT over VADER for sentiment | PRD specifies FinBERT as primary; VADER as fallback if compute is an issue |
| 2026-05-24 | FinnHub + NewsAPI + RSS as news sources | Free tier coverage + redundancy |
