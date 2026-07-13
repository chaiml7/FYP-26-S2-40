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

The LightGBM training script now also supports changing this target threshold
from the command line:

```powershell
python scripts\train_technical_model.py --all --threshold 0.005
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

---

# Session Update 12/06/2026

## Local-Only Prediction Program

Added a local-only prediction script:

- `scripts/predict_technical_trends_local.py`

This script does not read from or write to Supabase. It:

- loads the saved model from `backend/artifacts/technical_direction_model.joblib`
- fetches fresh yfinance data locally
- rebuilds the same technical indicators and market-context features in memory
- predicts next-day direction for the latest completed row or a selected historical `--as-of-date`
- prints the result in the terminal
- writes local results to `technical_analysis/local_predictions.json`

Train a model first:

```powershell
python scripts\train_technical_model.py --symbol NVDA
```

Run a local latest prediction:

```powershell
python scripts\predict_technical_trends_local.py --symbol NVDA
```

Run a local historical next-day direction prediction:

```powershell
python scripts\predict_technical_trends_local.py --symbol NVDA --as-of-date 2026-06-04
```

Use an all-ticker saved model and predict one ticker locally:

```powershell
python scripts\train_technical_model.py --all
python scripts\predict_technical_trends_local.py --symbol NVDA
```

Retrain the all-ticker model using only rows before June 4, 2026:

```powershell
python scripts\train_technical_model.py --all --train-before-date 2026-06-04
```

Then test the June 4 feature row locally:

```powershell
python scripts\predict_technical_trends_local.py --symbol NVDA --as-of-date 2026-06-04
```

The cutoff is exclusive, so `--train-before-date 2026-06-04` trains on rows
dated before `2026-06-04` and excludes the June 4 row itself.

Optional custom output path:

```powershell
python scripts\predict_technical_trends_local.py --symbol NVDA --output technical_analysis\nvda_local_prediction.json
```

Potential issue:

- yfinance can rate limit, return empty data, or temporarily fail. If that happens, rerun the local prediction later or reduce the number of symbols requested.

---

# Session Update 04/07/2026

## XGBoost Technical Model

Added a separate XGBoost model workflow based on the two referenced GitHub examples:

- `MachineLearningIndicator_Classifier/ml_trading_xgboost_St.ipynb`
- `taleblou/XGBoost-Price-Prediction`

Implemented files:

- `backend/services/technical/xgboost_model_service.py`
- `scripts/train_xgboost_technical_model.py`
- `scripts/predict_xgboost_technical_model.py`

The XGBoost workflow does not replace the existing LightGBM artifact. It saves a separate model at:

```text
backend/artifacts/technical_xgboost_model.joblib
```

What it does:

- uses stored `technical_indicators` as model features
- creates a three-class target:
  - `0 = neutral`
  - `1 = down`
  - `2 = up`
- supports configurable lookahead days and direction threshold
- uses chronological `TimeSeriesSplit` validation
- tunes a small XGBoost parameter grid
- tunes down/up probability thresholds
- selects top XGBoost feature-importance features
- trains an `XGBClassifier` for direction
- trains an `XGBRegressor` wrapped in `MultiOutputRegressor` for next OHLC return prediction
- reports classification metrics:
  - accuracy
  - balanced accuracy
  - macro F1
  - macro precision
  - macro recall
  - MCC
- reports regression metrics:
  - MSE
  - MAE
  - R2
  - median absolute error
  - explained variance

Train all tickers:

```powershell
python scripts\train_xgboost_technical_model.py --all
```

Train one ticker:

```powershell
python scripts\train_xgboost_technical_model.py --symbol NVDA
```

Train all tickers before a historical date:

```powershell
python scripts\train_xgboost_technical_model.py --all --train-before-date 2026-06-04
```

Predict one ticker using the saved XGBoost model:

```powershell
python scripts\predict_xgboost_technical_model.py --symbol NVDA
```

Predict from a historical stored feature row:

```powershell
python scripts\predict_xgboost_technical_model.py --symbol NVDA --as-of-date 2026-06-04
```

Prediction output is written locally to:

```text
technical_analysis/xgboost_predictions.json
```

## Binary XGBoost Technical Model

Added a binary XGBoost direction model as a simpler alternative to the three-class XGBoost model.

Implemented files:

- `backend/services/technical/binary_xgboost_model_service.py`
- `scripts/train_binary_xgboost_technical_model.py`
- `scripts/predict_binary_xgboost_technical_model.py`

The binary model saves a separate local artifact at:

```text
backend/artifacts/technical_xgboost_binary_model.joblib
```

Target:

```python
target_direction = 1 if future_close_return > threshold else 0
```

Classes:

- `0 = down_equal`
- `1 = up`

What it does:

- uses stored `technical_indicators` as model features
- supports configurable lookahead days and direction threshold
- uses chronological `TimeSeriesSplit` validation
- tunes a small XGBoost parameter grid
- tunes the binary decision threshold
- selects top XGBoost feature-importance features
- reports accuracy, balanced accuracy, precision, recall, F1, ROC AUC, MCC, and majority baseline accuracy

Train all tickers:

```powershell
python scripts\train_binary_xgboost_technical_model.py --all
```

Train one ticker:

```powershell
python scripts\train_binary_xgboost_technical_model.py --symbol NVDA
```

Train all tickers before a historical date:

```powershell
python scripts\train_binary_xgboost_technical_model.py --all --train-before-date 2026-06-04
```

Predict one ticker:

```powershell
python scripts\predict_binary_xgboost_technical_model.py --symbol NVDA
```

Prediction output is written locally to:

```text
technical_analysis/binary_xgboost_predictions.json
```

## LSTM Technical Model

Added a binary LSTM direction model that uses rolling sequences of stored technical indicators.

Implemented files:

- `backend/services/technical/lstm_model_service.py`
- `scripts/train_lstm_technical_model.py`
- `scripts/predict_lstm_technical_model.py`

The LSTM model saves a separate local PyTorch artifact at:

```text
backend/artifacts/technical_lstm_model.pt
```

Target:

```python
target_direction = 1 if future_close_return > threshold else 0
```

Classes:

- `0 = down_equal`
- `1 = up`

What it does:

- uses rolling feature sequences from `technical_indicators`
- defaults to 30 trading rows per sequence
- supports configurable lookahead days and direction threshold
- uses chronological `TimeSeriesSplit` validation
- standardizes features using training-fold-only scaler during validation
- tunes the binary decision threshold
- reports accuracy, balanced accuracy, precision, recall, F1, ROC AUC, MCC, and majority baseline accuracy

Train all tickers:

```powershell
python scripts\train_lstm_technical_model.py --all
```

Train one ticker:

```powershell
python scripts\train_lstm_technical_model.py --symbol NVDA
```

Train all tickers before a historical date:

```powershell
python scripts\train_lstm_technical_model.py --all --train-before-date 2026-06-04
```

Adjust sequence length or epochs:

```powershell
python scripts\train_lstm_technical_model.py --all --sequence-length 45 --epochs 30
```

Predict one ticker:

```powershell
python scripts\predict_lstm_technical_model.py --symbol NVDA
```

Prediction output is written locally to:

```text
technical_analysis/lstm_predictions.json
```

## ARIMA Technical Model

Added a classical ARIMA close-price forecasting baseline.

Implemented files:

- `backend/services/technical/arima_model_service.py`
- `scripts/train_arima_technical_model.py`
- `scripts/predict_arima_technical_model.py`

The ARIMA model saves a separate local artifact at:

```text
backend/artifacts/technical_arima_model.joblib
```

What it does:

- uses stored `technical_indicators.close`
- fits one ARIMA model per ticker
- supports a fixed `--order p,d,q` or small AIC-based order selection
- performs rolling chronological validation
- forecasts future close
- converts forecast return into binary direction:
  - `0 = down_equal`
  - `1 = up`
- reports accuracy, balanced accuracy, precision, recall, F1, MCC, MAE, RMSE, and confusion matrix

Train all tickers:

```powershell
python scripts\train_arima_technical_model.py --all
```

Train one ticker:

```powershell
python scripts\train_arima_technical_model.py --symbol NVDA
```

Train with a fixed order:

```powershell
python scripts\train_arima_technical_model.py --symbol NVDA --order 5,1,0
```

Train before a historical date:

```powershell
python scripts\train_arima_technical_model.py --symbol NVDA --train-before-date 2026-06-04
```

Predict one ticker:

```powershell
python scripts\predict_arima_technical_model.py --symbol NVDA
```

Prediction output is written locally to:

```text
technical_analysis/arima_predictions.json
```

## Chronos / TimesFM Foundation Forecasting

Added a zero-shot foundation-model forecasting script:

- `backend/services/technical/foundation_forecast_service.py`
- `scripts/predict_foundation_forecast_model.py`
- `backend/requirements-foundation-models.txt`

These are not trained on the local technical dataset. They read stored
`technical_indicators` close history from Supabase, forecast future close, and
convert the forecast return into a binary direction.

Supported models:

- `chronos`
- `chronons` alias for `chronos`
- `timesfm`

Install optional dependencies:

```powershell
pip install -r backend\requirements-foundation-models.txt
```

Run Chronos:

```powershell
python scripts\predict_foundation_forecast_model.py --model chronos --symbol NVDA
```

Chronos stock forecasts pass `freq="B"` by default because trading-day data
skips weekends and holidays:

```powershell
python scripts\predict_foundation_forecast_model.py --model chronos --symbol NVDA --freq B
```

Run TimesFM:

```powershell
python scripts\predict_foundation_forecast_model.py --model timesfm --symbol NVDA
```

Run from a historical date:

```powershell
python scripts\predict_foundation_forecast_model.py --model chronos --symbol NVDA --as-of-date 2026-06-04
```

Prediction output is written locally to:

```text
technical_analysis/foundation_forecast_predictions.json
```

Backtest recent Chronos forecasts:

```powershell
python scripts\backtest_foundation_forecast_model.py --model chronos --symbol NVDA
```

Backtest recent TimesFM forecasts:

```powershell
python scripts\backtest_foundation_forecast_model.py --model timesfm --symbol NVDA
```

Backtest a historical date range:

```powershell
python scripts\backtest_foundation_forecast_model.py --model chronos --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
```

By default, the backtest uses the most recent 30 eligible historical windows.
Use `--max-windows 0` to test every eligible window, though this may take a long
time with foundation models:

```powershell
python scripts\backtest_foundation_forecast_model.py --model chronos --symbol NVDA --max-windows 0
```

Backtest output is written locally to:

```text
technical_analysis/foundation_forecast_backtest.json
```

Backtest metrics:

- accuracy
- balanced accuracy
- precision
- recall
- F1 score
- MCC
- confusion matrix

Notes:

- The scripts do not write to Supabase.
- The first run downloads pretrained model weights and may take a while.
- Chronos defaults to `amazon/chronos-2`.
- TimesFM defaults to `google/timesfm-2.5-200m-pytorch`.

## CatBoost Technical Model

Implemented a separate binary CatBoost model workflow.

Files added:

- `backend/services/technical/catboost_model_service.py`
- `scripts/train_catboost_technical_model.py`
- `scripts/predict_catboost_technical_model.py`

Updated:

- `backend/requirements.txt`
- `technical_analysis/README.md`
- `.gitignore`

The CatBoost model:

- reads stored `technical_indicators` from Supabase
- does not write predictions to Supabase
- predicts next-direction as `0 = down/equal`, `1 = up`
- uses the same configurable direction threshold, default `0.002`
- uses chronological `TimeSeriesSplit` validation
- tunes CatBoost parameters across a small candidate grid
- tunes the probability decision threshold
- selects top features by CatBoost feature importance
- uses `symbol` as a categorical feature when available
- saves a local artifact to `backend/artifacts/technical_catboost_model.joblib`
- writes local predictions to `technical_analysis/catboost_predictions.json`

Install dependency:

```powershell
pip install -r backend\requirements.txt
```

Train all tickers:

```powershell
python scripts\train_catboost_technical_model.py --all
```

Train one ticker:

```powershell
python scripts\train_catboost_technical_model.py --symbol NVDA
```

Train with a different target threshold:

```powershell
python scripts\train_catboost_technical_model.py --symbol NVDA --threshold 0.005
```

Train before a historical date:

```powershell
python scripts\train_catboost_technical_model.py --symbol NVDA --train-before-date 2026-06-04
```

Predict one ticker:

```powershell
python scripts\predict_catboost_technical_model.py --symbol NVDA
```

Predict from a historical stored feature row:

```powershell
python scripts\predict_catboost_technical_model.py --symbol NVDA --as-of-date 2026-06-04
```

Extra data needed:

- No new Supabase tables or columns are required for the current CatBoost workflow.
- The model uses the existing `technical_indicators` data and optional existing `symbol` column.

## Technical Model Visual Backtest Report

Added a local HTML/SVG report script:

- `scripts/plot_technical_model_backtest.py`

The report:

- reads saved local artifacts for `lightgbm`, `catboost`, or `binary-xgboost`
- reads historical feature rows from Supabase `technical_indicators`
- plots actual close price over time
- overlays predicted up/down markers
- marks wrong predictions with a dark outline
- calculates accuracy, balanced accuracy, precision, recall, F1, ROC AUC, MCC, and confusion matrix
- writes a local HTML report and JSON output
- does not write to Supabase
- warns when the saved artifact training range overlaps the plotted period

Clean out-of-sample LightGBM example:

```powershell
python scripts\train_technical_model.py --symbol NVDA --train-before-date 2026-01-01
python scripts\plot_technical_model_backtest.py --model lightgbm --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
```

CatBoost example:

```powershell
python scripts\plot_technical_model_backtest.py --model catboost --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
```

Binary XGBoost example:

```powershell
python scripts\plot_technical_model_backtest.py --model binary-xgboost --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
```

## Same-Window Technical Model Comparison

Added a script to test multiple saved technical-analysis models on the same
symbol and same date window:

- `scripts/compare_technical_models.py`

The script:

- reads saved local artifacts for `lightgbm`, `xgboost`, `binary-xgboost`, `catboost`, `lstm`, and `arima`
- reads historical rows from Supabase `technical_indicators`
- uses the same requested symbol and same requested date window for all models
- aligns the final metrics on the intersection of dates where every available selected model has a prediction
- reports accuracy, balanced accuracy, precision, recall, F1, ROC AUC, MCC, confusion matrix, and correct counts
- skips missing artifacts unless `--require-all` is passed
- writes a local HTML and JSON report
- does not write to Supabase

Run all available saved models:

```powershell
python scripts\compare_technical_models.py --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
```

Run selected models only:

```powershell
python scripts\compare_technical_models.py --symbol NVDA --models lightgbm,catboost,binary-xgboost --rows 120
```

Use a shared target threshold:

```powershell
python scripts\compare_technical_models.py --symbol NVDA --threshold 0.005 --rows 120
```

Output:

```text
technical_analysis/<symbol>_model_comparison.html
technical_analysis/<symbol>_model_comparison.json
```

## Multi-Ticker Technical Model Comparison

Added a combined ticker report script:

- `scripts/compare_technical_models_multi_ticker.py`

The script:

- defaults to the 10 FYP tickers: `AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO`
- defaults to `binary-xgboost`
- reads saved local artifacts and Supabase `technical_indicators`
- writes one HTML file containing all evaluated tickers
- includes overall metrics across all tickers
- includes per-ticker metrics
- explicitly includes balanced accuracy
- does not write to Supabase

Run the 10-ticker binary XGBoost report:

```powershell
python scripts\compare_technical_models_multi_ticker.py --models binary-xgboost --start-date 2026-01-01 --end-date 2026-06-04
```

Output:

```text
technical_analysis/selected_tickers_model_comparison.html
technical_analysis/selected_tickers_model_comparison.json
```

## Retrain And Compare Technical Models

Added an orchestration script:

- `scripts/retrain_and_compare_technical_models.py`

The script:

- retrains selected saved-artifact models using the existing training scripts
- compares only the models that retrained successfully
- runs `scripts/compare_technical_models.py` after retraining
- uses the same symbol, threshold, and test window for every model
- writes a local run log to `technical_analysis/<symbol>_retrain_compare_run.json`
- does not write to Supabase

Clean out-of-sample retrain and test:

```powershell
python scripts\retrain_and_compare_technical_models.py --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
```

This trains each model using rows before `2026-01-01`, then tests the saved
artifacts on `2026-01-01` through `2026-06-04`.

Intentional leakage / overfit check:

```powershell
python scripts\retrain_and_compare_technical_models.py --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04 --mode leaky-window
```

The leaky mode trains through the end of the test window and then tests on the
same window. Metrics from this mode are expected to be inflated and should not
be reported as real out-of-sample accuracy.

Run selected models only:

```powershell
python scripts\retrain_and_compare_technical_models.py --symbol NVDA --models lightgbm,catboost,binary-xgboost --start-date 2026-01-01 --end-date 2026-06-04
```

Added selected-symbol binary XGBoost training support:

```powershell
python scripts\train_binary_xgboost_technical_model.py --symbols AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO --train-before-date 2026-01-01
```

Added `--train-symbols` to the retrain helper for binary XGBoost clean tests:

```powershell
python scripts\retrain_and_compare_technical_models.py --symbol NVDA --models binary-xgboost --train-symbols AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO --start-date 2026-01-01 --end-date 2026-06-04
```

## Per-Ticker Binary XGBoost Accuracy Report

Added:

- `scripts/train_binary_xgboost_per_ticker_report.py`

The script:

- trains one binary XGBoost model per ticker
- uses rows before the test window for training
- tests each ticker-specific model on the same requested window
- saves each ticker-specific artifact under `backend/artifacts/per_ticker_binary_xgboost/`
- writes one Markdown report with accuracy and balanced accuracy
- writes a JSON report with full metrics

Run:

```powershell
python scripts\train_binary_xgboost_per_ticker_report.py --start-date 2026-01-01 --end-date 2026-06-04
```

Output:

```text
technical_analysis/per_ticker_binary_xgboost_report.md
technical_analysis/per_ticker_binary_xgboost_report.json
```

## Binary XGBoost Regularization Improvements

Updated the binary XGBoost workflow with stronger anti-overfitting controls:

- more aggressively regularized hyperparameter candidates
- shallow trees by default in the tuning grid
- lower learning rates with more boosting rounds
- stronger `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, and `gamma`
- purged chronological validation folds
- optional embargo period after validation folds
- optional liquidity/outlier sample weighting
- seeded ensemble averaging by default
- internal early stopping where supported by the installed XGBoost version

Updated files:

- `backend/services/technical/binary_xgboost_model_service.py`
- `scripts/train_binary_xgboost_technical_model.py`
- `scripts/train_binary_xgboost_per_ticker_report.py`
- `technical_analysis/README.md`

Example:

```powershell
python scripts\train_binary_xgboost_technical_model.py --symbol NVDA --ensemble-size 5 --purge-days 2 --early-stopping-rounds 75
```

Disable sample weighting:

```powershell
python scripts\train_binary_xgboost_technical_model.py --symbol NVDA --no-sample-weighting
```

---

# Session Update 11/07/2026

## Independent Technical Model Bundles

Added:

- `scripts/train_all_technical_model_bundles.py`

The script trains the implemented technical-analysis model families across the
10 FYP tickers:

```text
AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO
```

Included models:

- `lightgbm`
- `binary-xgboost`
- `xgboost`
- `catboost`
- `lstm`
- `arima`

Ticker independence is handled by training one ticker-specific submodel per
ticker. Each model family still saves only one local bundle artifact containing
all ticker submodels.

Bundle artifacts:

```text
backend/artifacts/technical_lightgbm_bundle.joblib
backend/artifacts/technical_binary_xgboost_bundle.joblib
backend/artifacts/technical_xgboost_bundle.joblib
backend/artifacts/technical_catboost_bundle.joblib
backend/artifacts/technical_lstm_bundle.joblib
backend/artifacts/technical_arima_bundle.joblib
```

Each bundle stores:

- model name
- selected symbols
- target threshold
- training cutoff
- feature configuration
- ticker-specific submodels
- selected features or preprocessors where applicable
- ticker-specific thresholds where applicable
- ticker-specific evaluation metrics

The script reads existing Supabase `technical_indicators` data and does not
write predictions back to Supabase.

Run the full bundle training:

```powershell
python scripts\train_all_technical_model_bundles.py --threshold 0.002 --train-before-date 2026-01-01 --test-start-date 2026-01-01 --test-end-date 2026-06-04 --continue-on-error
```

Run selected models only:

```powershell
python scripts\train_all_technical_model_bundles.py --models lightgbm binary-xgboost catboost --continue-on-error
```

Run only LSTM with fewer epochs:

```powershell
python scripts\train_all_technical_model_bundles.py --models lstm --lstm-epochs 5 --continue-on-error
```

Report outputs:

```text
technical_analysis/model_bundle_results.md
technical_analysis/model_bundle_results.csv
technical_analysis/model_bundle_results.json
technical_analysis/model_bundle_results.html
```

The reports show one row per model type, each ticker's accuracy and balanced
accuracy, mean accuracy, mean balanced accuracy, the best model per ticker, and
the best overall model by mean balanced accuracy.
