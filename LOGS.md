# Session Log 09/06/2026

## Technical Analysis Pipeline

Implemented a separate technical-analysis backend package under:

- `backend/services/technical/`
- `backend/routes/technical.py`
- `technical_analysis/`

The technical-analysis code is separate from the sentiment pipeline. The technical router file exists, but it is not currently included in `backend/main.py` because that router wiring was removed when requested.

## Main Work Completed

- Added yfinance daily OHLCV fetching with a 10-year default lookback.
- Added Supabase upsert support for `stock_prices`.
- Added technical indicator calculation and Supabase upsert support for `technical_indicators`.
- Added next-day direction classification target.
- Added LightGBM as the primary ML model.
- Kept fallbacks to XGBoost and RandomForestClassifier.
- Added chronological walk-forward validation.
- Added baseline comparisons:
  - previous-direction baseline
  - majority-class baseline
- Added technical model evaluation script:
  - `scripts/evaluate_technical_model.py`
- Added manual technical pipeline test script:
  - `scripts/test_technical_manual.py`
- Added SQL table setup script:
  - `technical_analysis/create_technical_tables.sql`
- Updated `.env.example` to remove real-looking Supabase secret values and use placeholders only.

## Features Used

The technical model now uses 61 features, including:

- Daily return
- Log return
- High-low range
- Open-close gap
- Volume change
- Moving averages
- 50 EMA / 200 SMA trend filter
- RSI
- MACD
- Relative volume
- ATR
- Support / resistance distance
- Breakout / breakdown flags
- Lagged returns
- Market context from SPY, QQQ, VIX, and mapped sector ETF data

## Accuracy Improvement Changes

Implemented the requested improvements 1-5:

- LightGBM hyperparameter tuning using chronological validation only.
- Market context features.
- Less noisy target threshold:

```python
target_direction = 1 if next_day_return > 0.002 else 0
```

- Validation-based probability decision-threshold tuning.
- LightGBM feature-importance selection.

## NVDA Evaluation Metrics

Command run:

```powershell
python scripts/evaluate_technical_model.py --symbol NVDA
```

Metrics:

| Metric | Value |
| --- | ---: |
| Best tuning accuracy | 0.5222 |
| Accuracy | 0.5361 |
| Precision | 0.5689 |
| Recall | 0.6975 |
| F1 score | 0.5904 |
| ROC AUC | 0.5514 |
| Previous-direction baseline accuracy | 0.4813 |
| Majority-class baseline accuracy | 0.5222 |
| Difference vs previous-direction baseline | +0.0549 |
| Difference vs majority-class baseline | +0.0139 |

Confusion matrix:

| Actual / Predicted | Pred 0 | Pred 1 |
| --- | ---: | ---: |
| Actual 0, down/equal | 121 | 210 |
| Actual 1, up | 111 | 252 |

Top features:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `return_lag_10` | 66.0000 |
| 2 | `return_lag_2` | 63.0000 |
| 3 | `relative_volume` | 58.0000 |
| 4 | `market_vix_level` | 55.0000 |
| 5 | `market_spy_return_1d` | 52.0000 |
| 6 | `distance_to_support` | 50.0000 |
| 7 | `macd_histogram` | 48.0000 |
| 8 | `volume_change` | 46.0000 |
| 9 | `return_lag_5` | 45.0000 |
| 10 | `return_lag_3` | 44.0000 |
| 11 | `return_lag_1` | 40.0000 |
| 12 | `return_1d` | 39.0000 |
| 13 | `market_sector_return_5d` | 35.0000 |
| 14 | `return_5d` | 34.0000 |
| 15 | `volume_sma_20` | 33.0000 |

## Potential Issues

- yfinance may rate limit requests, return empty data, or temporarily fail, especially when running the pipeline for many tickers.
- yfinance data availability can vary by ticker, market, holiday, or symbol format.
- The model predicts next-day direction, which is naturally noisy. Accuracy should be compared against baselines rather than expected to be very high.
- If the Supabase SQL has already been run before, rerun `technical_analysis/create_technical_tables.sql` so the newer indicator and prediction metadata columns are added.
- The Supabase key that appeared in `.env.example` should be considered exposed and rotated in Supabase.

---

# Session Update 11/06/2026

## Pipeline Updates Completed

- Added raw yfinance storage into `daily_ohlcv`.
- Updated the technical pipeline so data is read back from Supabase before later steps:
  - yfinance data is inserted into `daily_ohlcv`
  - `daily_ohlcv` is read from Supabase
  - market-context columns are added and inserted into `stock_prices`
  - `stock_prices` is read from Supabase
  - technical indicators are calculated and inserted into `technical_indicators`
  - `technical_indicators` is read from Supabase for model training
- Added paginated Supabase reads so 10 years of rows are not cut off by the default 1,000-row response limit.
- Updated `stock_prices`, `technical_indicators`, and `daily_ohlcv` SQL definitions to use UUID primary keys.
- Added `open`, `high`, and `low` to `technical_indicators` because these are model features.
- Kept SPY, QQQ, VIX, and sector ETF values as market-context columns, not separate prediction rows in `stock_prices` or `technical_indicators`.
- Added market-context columns for:
  - SPY returns and 200 SMA flag
  - QQQ returns and 200 SMA flag
  - VIX level and returns
  - mapped sector ETF returns and 200 SMA flag
- Updated `--all` stock selection to skip market-context symbols such as `SPY`, `QQQ`, `^VIX`, `XLK`, `XLC`, and `XLY`.
- Updated target creation so pooled training does not leak next-day labels between tickers.
- Split model workflow into separate train and predict scripts.
- Added model training modes:
  - train on all target tickers
  - train on one specified ticker
- Added local model artifact saving at:

```text
backend/artifacts/technical_direction_model.joblib
```

- Added `.joblib` and `backend/artifacts/` to `.gitignore` so saved models stay local.
- Added prediction script that loads the saved model and writes predictions into `direction_predictions`.
- Added API functions/routes for saved-model prediction:
  - `POST /technical/predict-trends`
  - `POST /technical/predict-trends/{symbol}`

## Current Table Flow

```text
yfinance
-> daily_ohlcv
-> stock_prices
-> technical_indicators
-> saved local model
-> direction_predictions
```

## Table Changes

Run `technical_analysis/create_technical_tables.sql` in Supabase to apply the latest schema.

Important changes:

- `daily_ohlcv.id` uses UUID.
- `stock_prices.id` uses UUID.
- `technical_indicators.id` uses UUID.
- `technical_indicators` includes:
  - `open`
  - `high`
  - `low`
- `stock_prices` and `technical_indicators` include market-context columns for SPY, QQQ, VIX, and sector ETF features.

## Commands To Run

Run these from the project root:

```powershell
cd C:\Users\ianch\Schoolwork\fyp\FYP-26-S2-40
```

Install/update backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

Sync all prediction-target stocks:

```powershell
python scripts\sync_technical_prices.py --all
```

Sync one stock:

```powershell
python scripts\sync_technical_prices.py --symbol NVDA
```

Train one model using all target tickers:

```powershell
python scripts\train_technical_model.py --all
```

Train one model using a single ticker:

```powershell
python scripts\train_technical_model.py --symbol NVDA
```

Sync first, then train on all target tickers:

```powershell
python scripts\train_technical_model.py --all --sync-first
```

Sync first, then train on one ticker:

```powershell
python scripts\train_technical_model.py --symbol NVDA --sync-first
```

Predict trends using the saved model:

```powershell
python scripts\predict_technical_trends.py
```

Predict one symbol using the saved model:

```powershell
python scripts\predict_technical_trends.py --symbol NVDA
```

Evaluate the technical model:

```powershell
python scripts\evaluate_technical_model.py --symbol NVDA
```

Run the manual technical pipeline test:

```powershell
python scripts\test_technical_manual.py
```

## Current Model Behavior

- The model predicts next-day direction, not exact next-day price.
- Target:

```python
target_direction = 1 if next_day_return > 0.002 else 0
```

- `--all` training learns from all prediction-target tickers in `technical_indicators`.
- `--symbol NVDA` training learns only from NVDA rows in `technical_indicators`.
- A single-ticker saved model defaults to predicting that same ticker.
- An all-ticker saved model predicts all tickers unless `--symbol` is provided.
- Metrics saved into `direction_predictions` are model-level validation metrics for the saved model.
