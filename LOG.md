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
<!-- Add entries here as work progresses -->

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
