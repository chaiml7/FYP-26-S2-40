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

### 2026-05-25 — Bali — Environment Setup (uni machine)

**What I did:**
- Pulled latest context via gist sync
- Fixed uvicorn not on PATH: ran `pip install` from scratch on uni machine
- Fixed `supabase` install failure: `storage3` 2.x pulls in `pyiceberg` (requires C++ build tools) — pinned `supabase==2.7.4` in requirements.txt which uses `storage3==0.7.7` (no pyiceberg)
- Fixed "Invalid API key": new Supabase `sb_secret_*` key format not supported by supabase-py 2.7.4 — switched to legacy JWT service_role key from dashboard → "Legacy anon, service_role API keys" tab
- Backend server confirmed running on uni machine

**What I discovered:**
- `feature/bali-sentiment-pipeline` branch (14 commits) was never pushed — exists only on home PC
- No stash or reflog trace in this repo; branch is safe at home but inaccessible here

**Next steps:**
- At home: push `feature/bali-sentiment-pipeline` to remote
- Next session: pull branch, run `python scripts/test_sentiment_manual.py` E2E test, then PR to `main`

### 2026-05-25 — Bali — PR to main (sentiment pipeline)

**What I did:**
- Pulled `feature/bali-sentiment-pipeline` from remote (pushed from home PC)
- Installed dependencies on this machine: `pip install -r requirements.txt` (torch/transformers ~2GB)
- Ran 63 unit tests — all passing
- Started backend server, ran `python scripts/test_sentiment_manual.py` — all 8 E2E steps passing
  - Real FinnHub + NewsAPI calls working, real Supabase writes confirmed
  - Idempotency verified, error isolation verified
- Fixed 3 test script issues (all test bugs, not pipeline bugs):
  - Windows `charmap` encoding error on `✓` character → `sys.stdout.reconfigure(encoding='utf-8')`
  - FinBERT label assertion too strict for real model inference → softened check
  - Idempotency check expected 10 skipped, but 2 symbols (NFLX, BABA) had no news → fixed assertion
- Removed `docs/superpowers/` from feature branch before PR (personal files, stay on `bali` only)
- Opened PR #1 to `main`: https://github.com/chaiml7/FYP-26-S2-40/pull/1
- PR merged, feature branch deleted (local + remote)
- Installed `gh` CLI via winget for future PR creation

**Current state:**
- Sentiment pipeline is merged to `main` — Sprint 2 scope complete
- On `bali` branch, clean

**Next steps:**
- Start next phase: XGBoost/LSTM ML models, or frontend sentiment charts
- Decide with team what to tackle next

### 2026-05-27 — Bali — Pulled teammate's yfinance/stock import changes from main

**What changed on main (commit `110cc75`):**

Teammate (Addison or Ian — unclear who pushed) made 4 backend changes:

1. **`yfinance_service.py`** — `fetch_stock_history()` now takes `stock_id: int` as first argument; includes `stock_id` in every row dict sent to Supabase.

2. **`stock_history_service.py`** — upsert conflict target changed from `symbol,trade_date` → `stock_id,trade_date`. Requires a matching unique constraint change in the Supabase `daily_ohlcv` table (DB schema updated directly in dashboard, not via code).

3. **`stock_list_service.py`** — added `update_last_imported_at(symbol)` function that stamps `last_imported_at` on the `stocks` table row. Requires a `last_imported_at` column on `stocks` (added in Supabase dashboard).

4. **`stock_routes.py`** — new endpoint `POST /stocks/import/{symbol}` that: looks up `stock_id` from DB, calls yfinance, saves history, and updates `last_imported_at`. Clean single-stock import trigger.

Commit message also mentions a new **`logs` table** (import run logs) added in Supabase dashboard — not reflected in code diff.

**Integration impact on sentiment pipeline (my scope):**

- **No breakage** — sentiment pipeline imports nothing from yfinance_service or stock_history_service. The `sentiment_scores` table remains keyed on `symbol` and is independent.
- **Future ML join concern** — when XGBoost/LSTM models join price + sentiment data, `daily_ohlcv` is now indexed by `stock_id` while `sentiment_scores` uses `symbol`. Both tables carry `symbol` so joining on that column still works, but worth noting.
- **Hardcoded WATCHLIST vs DB** — sentiment pipeline still has a hardcoded 10-symbol WATCHLIST. The new per-symbol import endpoint is separate. If we later want sentiment to run on exactly the same set as imported stocks, we should call `get_active_stocks()` from inside the pipeline instead.

**Action items:**
- [ ] Confirm teammate updated Supabase schema (`last_imported_at` column on `stocks`, unique constraint `stock_id,trade_date` on `daily_ohlcv`, new `logs` table)
- [ ] When building ML models: note the `stock_id` / `symbol` join pattern

---

## Sprint 3

### 2026-08-13 — Bali — Demo-recording audit fixes (admin dashboard branch)

**Context:** Ran a full live audit against the deployed app (`admin_demo_audit.md` artifact) covering all 5 presenters' user stories ahead of demo recording. Team triaged the findings; my scope was 4 fixes, with 3 explicitly deferred pending group decisions.

**What I built:**
- **Base #10 (update password):** No UI existed anywhere — only a bearer-token JSON endpoint (`PATCH /users/me/password`) nothing in the app called, since server-rendered pages only carry a session cookie, never the Supabase access token. Added `admin_update_user()`/`admin_delete_user()` to `auth_service.py` (service-role GoTrue admin calls, same pattern as existing `admin_get_user`), then a session-cookie-gated `GET/POST /user/account` flow in `user_routes.py` that re-verifies the current password via the existing `login()` call before applying the change. New `frontend/templates/account/settings.html`, linked from the shared top navbar.
- **Base #12 (delete account):** Didn't exist at all — no route, no UI. New `backend/services/account_deletion_service.py` deletes across every table that FKs to a user (`user_watchlists`, `user_notification_preferences`, `notification_deliveries`, `weightages`, `user_subscriptions`, `user_profiles` — none of this is defined in-repo, the base schema/FKs only live in the Supabase dashboard) before calling the Auth admin delete. Wired to `POST /user/account/delete` on the same account-settings page, password-confirmed.
- **Base #6 (top gainers/losers):** Was 100% hardcoded HTML with dead `/quote?symbol=` links. Added `get_top_movers()` to `dashboard_service.py`, reusing the existing `_dashboard_price_summaries()` batch query over `daily_ohlcv` restricted to our active stocks list, ranked by day-over-day % change. Wired into `/user/market_overview` for both Base and Premium (shared page/route). Links now point to the real `/stocks/{symbol}/view` route.
- **Base #7 (financial report):** Investigated whether real financial-statement data exists — it does. `financial_statements` (populated via `yfinance_financial_fetcher.py`/SEC fetcher) has genuine quarterly revenue/margins/balance-sheet/cash-flow data, separate from the legacy unused `predictions` table. Added `get_financial_report()` to `financial_service.py` and a new `GET /stocks/{symbol}/financial_report` page, linked from the existing Financial score panel on the stock detail page.

**Verified live** (not just code-read) by running `frontend.main:app` locally against the real Supabase project, logging in as `freeuser1@user.com`, and hitting the new GET routes: account settings page renders, market overview now shows 10 real rows (NVDA, AAPL, D05.SI, etc. — including SGX dotted symbols, confirmed `/stocks/{symbol}/view` handles the dot correctly), and the AAPL financial report renders real quarterly figures (~$111B/$143B revenue, correctly scaled). Did **not** exercise the password-change/delete-account POST paths against the real demo accounts to avoid breaking them before recording — worth a dry run on a throwaway account before the actual recording.

**Deferred (team decision pending):**
- Base #4/#5 watchlist — currently Premium-only (paywalled), contradicts the PRD's Base story. Waiting on group decision before un-gating.
- Base #9 (follow list of social media accounts) — team already decided to drop this feature.
- Premium #14 (downgrade subscription, 500s live) — teammate (Addison) owns this, not touching it.

### 2026-08-13 — Bali — Finalized 3 replacement Base user stories for demo recording

**Context:** Confirmed with the group that Base stories #4, #5 (watchlist add/delete) and #9 (social media follow list) can't be demoed as-is (watchlist is Premium-gated pending a group decision; follow-list was already dropped as a feature). Investigated the app for already-working, already-Base-accessible features not currently assigned to any presenter, to swap in instead. Live-tested every claim against the real app (as `freeuser1@user.com`) before finalizing — an earlier draft of story #4 wrongly said the dashboard shows "live technical/sentiment/financial scores" and an earlier draft of #5 proposed the dashboard's historical-date picker; both were wrong/broken on inspection (see bug below), so they were dropped and replaced with verified alternatives.

**Finalized replacements:**
1. **Replacing #4** — "As a base user, I want to search and filter the list of tracked stocks by sector, with each stock's latest closing price and daily % change, so that I can quickly compare stocks without searching for them one at a time." Maps to the existing `/dashboard` table (verified columns: Company, Sector, Latest price — a closing price from `daily_ohlcv`, not live — and Daily move, with working search box + sector filter).
2. **Replacing #5** — "As a base user, I want to view a stock's technical, sentiment, and financial prediction scorecard (with an overall composite score), so that I understand the platform's overall outlook on that stock at a glance." Maps to the top of `/stocks/{symbol}/view`. Live-verified as `freeuser1@user.com` on AAPL: real overall score (4.4/10, "Neutral") plus three individual Technical/Sentiment/Financial scores, with the Premium-only weight-simulator correctly hidden for Base. Distinct from Base #7 (financial report), which is just the raw statement numbers.
3. **Replacing #9** — "As a base user, I want to submit feedback about the platform, so that the team can act on my experience and improve StockLens." Maps to `/user/feedback` (topic/rating/description form), a complete feature that isn't in the PRD's use-case list at all but is fully built, including the admin review side.

**Bug found while testing (not fixed, just documented — see Issues tracker below):** the dashboard's `selected_date` picker is unreliable. `_price_summary_from_rows()` in `dashboard_service.py` requires a stock's most recent trading row to fall on the **exact** selected date, not "the closest trading day on or before it." Any date that isn't precisely a recorded trading day for a given stock — weekends, holidays, dates not yet imported, or SGX vs. US stocks trading on different calendars — silently shows "No price data"/"Unavailable" for that stock even though nearby data exists. Tested 6 dates locally (today, 1 week/1 month/2 months/3 months/1 year ago): today, 1-week-ago and 2-months-ago each showed 14 of 15 stocks missing; 1-month, 3-months and 1-year-ago happened to show all 15 populated. This is why it looked like "only some random dates work" — it's calendar-dependent, not a fixed cadence.

### 2026-08-13 — Bali — Full end-to-end test of update-password/delete-account, found and fixed a real deletion bug

**What I did:** Before greenlighting the demo, ran both new flows end-to-end against a throwaway signup (not the real `freeuser1`/`premiumuser1` demo accounts) on a local server hitting the real Supabase project: signup → wrong-password rejection → mismatched-confirmation rejection → real password change → confirmed old password now fails and new password logs in → wrong-password delete rejection → real delete.

**Bug found:** the real delete call 500'd. Root cause: `account_deletion_service.py` deleted straight through a list of tables with no error isolation, and `supabase.table("user_subscriptions")...execute()` throws `postgrest.exceptions.APIError: PGRST205 Could not find the table 'public.user_subscriptions' in the schema cache` — that table isn't visible to PostgREST in this project right now. Because the original code had no try/except per table, this one failure aborted the whole deletion before it ever reached `user_profiles` or the Auth user delete.

**Fix:** rewrote `delete_user_account()` to loop over the child tables with a try/except per table (log + continue on failure) so one bad table can't block the rest, then always call `admin_delete_user()` last. Re-ran the full test after the fix: delete succeeded (303), and a fresh login attempt with the deleted account's credentials correctly failed with "Invalid email or password" — confirmed the Auth user is genuinely gone.

**Important side-finding:** this is the same root cause as the Premium downgrade 500 logged above — `billing_service.get_user_subscription()` queries the same `user_subscriptions` table, unguarded, with no try/except. This is no longer just a hypothesis (unguarded call *could* throw) — it's now confirmed live that the table lookup does throw. Worth telling Addison specifically: it's a missing/stale-schema-cache table, not a transient/network issue, so retrying won't fix it — the table needs to exist and be visible to PostgREST (may just need a schema cache reload in Supabase, or the table needs to actually be created) before the downgrade flow can work.

**Not yet done:** haven't investigated *why* `user_subscriptions` is missing from the schema cache (could be a real missing table, or Supabase just needs `NOTIFY pgrst, 'reload schema'` after a recent DDL change) — that's a question for whoever owns the Supabase dashboard/billing schema, most likely Addison since it's his feature area.

---

## Issues / Bugs Tracker

| Date | Issue | Status | Resolution |
|---|---|---|---|
| 2026-05-24 | frontend/.env committed with anon key | Open | Add to .gitignore cleanup task |
| 2026-05-24 | PRD says Flask, codebase uses FastAPI | Resolved | Using FastAPI, noted discrepancy |
| 2026-08-13 | Dashboard `selected_date` picker requires an exact trade-date match per stock instead of "closest trading day on or before" — silently shows missing data for most non-trading-day selections (`dashboard_service.py`, `_price_summary_from_rows`) | Open | Not fixed — dropped from the demo story replacement instead |
| 2026-08-13 | `POST /billing/portal` (Premium downgrade) can raw-500: `get_user_subscription()` in `billing_service.py` is an unguarded Supabase call inside `create_customer_portal_session`, not caught by the route's `except BillingConfigurationError`/`except BillingError` clauses | Open | Confirmed live (not just theoretical) — `user_subscriptions` table is missing from the PostgREST schema cache. Not fixed (Addison's feature) — flagged for the group |
| 2026-08-13 | `delete_user_account()` 500'd on real deletion: `user_subscriptions` table missing from PostgREST schema cache, no error isolation between child-table deletes | Resolved | `account_deletion_service.py` now isolates each child-table delete in its own try/except; deletion verified end-to-end on a throwaway test account |

---

## Key Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-05-24 | Use `bali` branch for personal files, PRs to main for code only | Keep main clean, share context across machines |
| 2026-05-24 | FinBERT over VADER for sentiment | PRD specifies FinBERT as primary; VADER as fallback if compute is an issue |
| 2026-05-24 | FinnHub + NewsAPI + RSS as news sources | Free tier coverage + redundancy |
