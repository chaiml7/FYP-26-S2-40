# Financial ML Pipeline

Predicts financial outlook (Positive / Neutral / Negative) for 11 tech stocks
using 3+ years of annual Cash Flow, Balance Sheet, and Income Statement data
from Yahoo Finance. Trained model and predictions are stored in Supabase.

## Structure

```
financial_ml/
├── backend/
│   ├── database/
│   │   └── supabase_client.py       # Supabase connection
│   ├── services/
│   │   ├── financial/
│   │   │   ├── yfinance_fetcher.py  # Fetch from Yahoo Finance
│   │   │   ├── feature_engineering.py  # 16 ratio features + labeling
│   │   │   └── financial_pipeline.py   # Train + predict
│   │   ├── financials_service.py    # financial_statements table I/O
│   │   ├── prediction_service.py    # financial_predictions table I/O
│   │   └── stock_list_service.py    # stocks table queries
│   ├── main.py
│   └── schemas.py
├── models/
│   └── financial_model.joblib       # saved after training
├── scripts/
│   ├── run_pipeline.py              # run this to train + predict
│   └── test_prediction.py           # test on new data
├── .env.example
├── requirements.txt
└── requirements-ml.txt
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and add your SUPABASE_URL and SUPABASE_ANON_KEY

# 4. Create Supabase tables
# Run create_tables.sql in your Supabase SQL editor
```

## Run

```bash
# Full pipeline: fetch → store → train → predict
python scripts/run_pipeline.py

# Test on new financial data (after training)
python scripts/test_prediction.py
```

## Supabase Tables

| Table | Purpose |
|-------|---------|
| `stocks` | Existing — provides stock_id |
| `financial_statements` | Input — raw 3-year financial data |
| `financial_predictions` | Output — prediction + score + confidence |

## Output schema (financial_predictions)

| Column | Type | Description |
|--------|------|-------------|
| stock_id | integer | FK → stocks.id |
| ticker | text | e.g. AAPL |
| prediction | text | positive / neutral / negative |
| score | float | 1.0 / 0.0 / -1.0 |
| confidence | float | 0.0 – 1.0 |
| prob_positive | float | probability for positive class |
| prob_neutral | float | probability for neutral class |
| prob_negative | float | probability for negative class |
| model_version | text | gbm_v1 |
| period | text | fiscal period e.g. 2024-09-30 |
| created_at | timestamptz | when prediction was made |
