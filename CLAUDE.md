# StockLens — Claude Context

## Project
**StockLens** — Webservice for Stock Market Prediction  
FYP Group: FYP-26-S2-40 | SIM UOWCSIT 321 CSCI321 | Project ID: CSIT-26-S2-13  
Deadline: Final Product & Documentation → **15 Aug 2026** | Final Presentation → **22 Aug 2026**

## Team
| Name | Role | Branch |
|---|---|---|
| Chai Ming Liang (PM) | Backend, Docs | `main` |
| Bali Pratama Tok | **Sentiment ML Pipeline**, Frontend | `bali` |
| Anbu Maithrishree | Frontend | feature branches |
| Chuan Yuhong Addison | Backend | feature branches |
| Chong Yik Siong Ian | Backend | feature branches |

## Repository
- GitHub: https://github.com/chaiml7/FYP-26-S2-40
- Main language: Python (backend), JavaScript/React (frontend)

## Branch Strategy
```
main               ← stable, no personal/doc files
  └── feature/*    ← code-only feature branches, PR → main
bali               ← Bali's working branch; has docs/, .claude/, LOG.md, CLAUDE.md
  └── feature/bali-* ← Bali's feature work, branches off bali, PRs → main (code only)
```
**Never merge `bali` directly into `main`.** PRs to main include only code, not docs/ or .claude/.

## Architecture
```
Frontend  →  FastAPI Backend  →  Supabase (PostgreSQL)
React+Vite   Python              stocks, daily_ohlcv,
(port 5173)  (port 8000)         predictions, sentiment_scores
```

## Tech Stack
| Layer | Stack |
|---|---|
| Frontend | React 19, Vite 8, @supabase/supabase-js |
| Backend | FastAPI, uvicorn, python-dotenv |
| Database | Supabase (PostgreSQL) |
| Price Data | yfinance |
| Sentiment | FinBERT (ProsusAI/finbert), FinnHub API, NewsAPI |
| ML Models | XGBoost (scikit-learn), LSTM (Keras/TensorFlow) |
| Auth | Supabase Auth (role-based: Guest / Base / Premium / Admin) |

## Directory Structure
```
backend/
  database/supabase_client.py     Supabase Python client
  routes/stock_routes.py          REST API routes
  services/                       Business logic layer
    stock_list_service.py
    stock_history_service.py
    stock_service.py
    yfinance_service.py
    sentiment/                    ← Bali's work
      finnhub_service.py
      news_scraper_service.py
      finbert_service.py
      sentiment_aggregator.py
  ml/                             ← ML models
    xgboost_model.py
    lstm_model.py
  main.py
  requirements.txt

frontend/
  src/
    App.jsx                       Root component (currently basic stock list)
    supabaseClient.js

docs/                             ← bali branch only, not merged to main
  PrelimTechReport.md
  PRD.md

.claude/                          ← bali branch only
  settings.json

LOG.md                            ← bali branch only, personal log
CLAUDE.md                         ← bali branch only
```

## Database Schema
| Table | Key Columns |
|---|---|
| `stocks` | id, symbol, company_name, is_active |
| `daily_ohlcv` | symbol, trade_date, open, high, low, close, volume, source (upsert key: symbol+trade_date) |
| `predictions` | symbol, created_at, predicted_price, signal, model_type, rmse, mae, mape, directional_accuracy |
| `sentiment_scores` | symbol, published_at, source, headline, score, label, model_version |

## Supabase
- Dashboard: https://supabase.com → sign in → project `fcpfsdjnryelyqknjfne`
- Direct URL: https://supabase.com/dashboard/project/fcpfsdjnryelyqknjfne
- Frontend env key: VITE_SUPABASE_ANON_KEY (in frontend/.env)
- Backend env key: SUPABASE_SECRET_KEY (in backend/.env, never committed)

## Sentiment Pipeline (Bali's Scope)
Goal: Collect financial news → run FinBERT → store per-stock sentiment score → expose via API

Sources:
- **FinnHub** — company news API (free tier: 60 calls/min)
- **NewsAPI** — general financial news (free tier: 100 req/day)
- **RSS scrapers** — Reuters, MarketWatch (no rate limits)

Pipeline flow:
```
[News Sources] → fetch_news() → [Raw Headlines]
→ finbert_score() → [label: positive/negative/neutral, score: float]
→ aggregate_by_symbol() → [daily avg score per stock]
→ save to sentiment_scores table
→ GET /api/stocks/{symbol}/sentiment → frontend
```

## Dev Commands
```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Key Docs
- `docs/PRD.md` — Full Product Requirements Document
- `docs/PrelimTechReport.md` — Preliminary Technical Report
- `LOG.md` — Team activity log (personal, not merged to main)

## Scope Boundary (from PRD)
| Core | Stretch | Out of Scope |
|---|---|---|
| yfinance + Supabase pipeline | SGX coverage | Real-time/intraday streaming |
| XGBoost + LSTM models | LSTM+XGBoost+Sentiment ensemble | Live trading / order execution |
| Sentiment (VADER/FinBERT) | SHAP explainability | Options, futures, crypto |
| React+Vite frontend | PWA mobile | Native apps |
| RBAC 4-tier auth | | Multi-language |
| MAS disclaimer | | Automated retraining |
| RMSE, MAE, DA, MAPE dashboard | | |
