"""
Manual end-to-end test for the technical-analysis pipeline.
Run from repo root: python scripts/test_technical_manual.py
Requires: backend server running at localhost:8000 and backend/.env configured.
"""
import math
import os
import sys

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from database.supabase_client import supabase
from services.technical.indicator_service import add_technical_indicators
from services.technical.model_service import FEATURES, prepare_training_data, walk_forward_validation
from services.technical.price_service import add_market_context_features, fetch_price_history

BASE_URL = "http://localhost:8000/api"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f" - {detail}" if detail else ""))
        failures.append(name)


def step(n, label):
    print(f"\n--- Step {n}: {label} ---")


def is_metric_value(value):
    return value is None or (isinstance(value, (int, float)) and not math.isnan(value))


def is_between_zero_and_one(value):
    return value is None or 0.0 <= float(value) <= 1.0


selected_stock = None
price_df = pd.DataFrame()
enriched_price_df = pd.DataFrame()
indicator_df = pd.DataFrame()
pipeline_response = {}


# Step 1: Supabase stocks fetch
step(1, "Supabase stocks fetch")
try:
    response = supabase.table("stocks").select("id, symbol").execute()
    stocks = [
        {"id": row.get("id"), "symbol": str(row.get("symbol") or "").upper()}
        for row in (response.data or [])
        if row.get("id") is not None and row.get("symbol")
    ]
    selected_stock = next((stock for stock in stocks if stock["symbol"] == "AAPL"), None)
    selected_stock = selected_stock or (stocks[0] if stocks else None)
    check("stocks table returns rows", len(stocks) > 0, f"got {len(stocks)} rows")
    check("each stock has id and symbol", all(stock["id"] and stock["symbol"] for stock in stocks))
    if selected_stock:
        print(f"  Using {selected_stock['symbol']} for pipeline checks")
except Exception as e:
    check("Supabase stocks fetch", False, str(e))


# Step 2: yfinance price fetch
step(2, "yfinance price fetch")
try:
    symbol = selected_stock["symbol"] if selected_stock else "AAPL"
    price_df = fetch_price_history(symbol)
    enriched_price_df = add_market_context_features(price_df, symbol=symbol)
    expected_price_columns = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    check("price data is not empty", not price_df.empty)
    check("price columns exist", expected_price_columns.issubset(set(price_df.columns)))
    check("dates are sorted ascending", price_df["date"].is_monotonic_increasing if not price_df.empty else False)
    print(f"  Got {len(price_df)} daily rows for {symbol}")
except Exception as e:
    check("yfinance price fetch", False, str(e))


# Step 3: Indicator calculation
step(3, "Indicator calculation")
try:
    indicator_df = add_technical_indicators(enriched_price_df)
    expected_indicator_columns = {
        "rsi_14",
        "sma_20",
        "sma_200",
        "ema_20",
        "ema_50",
        "trend_filter_50_200",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "relative_volume",
        "vwap_20",
        "support_20",
        "resistance_20",
        "rolling_volatility_20",
        "return_lag_1",
        "close_lag_1",
        "market_spy_return_1d",
        "market_qqq_return_1d",
        "market_vix_return_1d",
        "market_sector_return_1d",
    }
    valid_rows = indicator_df.dropna(subset=list(expected_indicator_columns)) if not indicator_df.empty else pd.DataFrame()
    check("indicator columns exist", expected_indicator_columns.issubset(set(indicator_df.columns)))
    check("valid numeric indicator rows remain", len(valid_rows) > 0)
    check(
        "indicator values are numeric",
        valid_rows[list(expected_indicator_columns)].apply(pd.to_numeric, errors="coerce").notna().all().all()
        if not valid_rows.empty
        else False,
    )
except Exception as e:
    check("Indicator calculation", False, str(e))


# Step 4: Training target
step(4, "Training target")
try:
    X, y, clean_df = prepare_training_data(indicator_df)
    check("X, y, clean_df are not empty", not X.empty and not y.empty and not clean_df.empty)
    check("target_direction exists", "target_direction" in clean_df.columns)
    check("target_direction is not inside FEATURES", "target_direction" not in FEATURES)
    check("target values are only 0 and 1", set(y.unique()).issubset({0, 1}))
except Exception as e:
    check("Training target", False, str(e))


# Step 5: Walk-forward validation
step(5, "Walk-forward validation")
try:
    metrics = walk_forward_validation(indicator_df)
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    check("metrics exist", all(key in metrics for key in metric_keys))
    check("decision threshold exists", "decision_threshold" in metrics)
    check("target threshold exists", "target_return_threshold" in metrics)
    check("metrics are numeric where possible", all(is_metric_value(metrics[key]) for key in metric_keys))
    check("metrics are between 0 and 1 where available", all(is_between_zero_and_one(metrics[key]) for key in metric_keys))
    print(f"  Metrics: { {key: metrics[key] for key in metric_keys} }")
except Exception as e:
    check("Walk-forward validation", False, str(e))


# Step 6: Single-symbol pipeline API
step(6, "Single-symbol pipeline API")
try:
    symbol = selected_stock["symbol"] if selected_stock else "AAPL"
    response = httpx.post(f"{BASE_URL}/technical/run-pipeline/{symbol}", timeout=300)
    check(f"POST /technical/run-pipeline/{symbol} returns 200", response.status_code == 200, f"got {response.status_code}")
    pipeline_response = response.json() if response.content else {}
    status = pipeline_response.get("status")
    check("response contains symbol", pipeline_response.get("symbol") == symbol)
    check("response contains status", status in {"ok", "skipped", "no_data"})
    if status not in {"skipped", "no_data"}:
        required_prediction_fields = {"latest_date", "predicted_direction", "predicted_probability", "confidence", "decision_threshold"}
        check("response contains prediction fields", required_prediction_fields.issubset(set(pipeline_response.keys())))
    print(f"  Pipeline response: {pipeline_response}")
except Exception as e:
    check("Single-symbol pipeline API", False, str(e))


# Step 7: Supabase prediction row check
step(7, "Supabase storage and prediction row check")
try:
    stock_id = selected_stock["id"] if selected_stock else None
    daily_rows_response = (
        supabase.table("daily_ohlcv")
        .select("id")
        .eq("stock_id", stock_id)
        .limit(1)
        .execute()
    )
    stock_prices_response = (
        supabase.table("stock_prices")
        .select("id")
        .eq("stock_id", stock_id)
        .limit(1)
        .execute()
    )
    response = (
        supabase.table("direction_predictions")
        .select("*")
        .eq("stock_id", stock_id)
        .order("latest_date", desc=True)
        .limit(1)
        .execute()
    )
    prediction_rows = response.data or []
    check("daily_ohlcv has yfinance rows", len(daily_rows_response.data or []) > 0)
    check("stock_prices has technical-ready rows", len(stock_prices_response.data or []) > 0)
    check("direction_predictions has at least one row", len(prediction_rows) > 0)
    if prediction_rows:
        pipeline_response.setdefault("latest_date", prediction_rows[0]["latest_date"])
        print(f"  Latest prediction row: {prediction_rows[0]}")
except Exception as e:
    check("Supabase prediction row check", False, str(e))


# Step 8: Idempotency check
step(8, "Idempotency check")
try:
    symbol = selected_stock["symbol"] if selected_stock else "AAPL"
    stock_id = selected_stock["id"] if selected_stock else None
    latest_date = pipeline_response.get("latest_date")
    before = (
        supabase.table("direction_predictions")
        .select("id")
        .eq("stock_id", stock_id)
        .eq("latest_date", latest_date)
        .execute()
    )
    count_before = len(before.data or [])

    rerun = httpx.post(f"{BASE_URL}/technical/run-pipeline/{symbol}", timeout=300)
    after = (
        supabase.table("direction_predictions")
        .select("id")
        .eq("stock_id", stock_id)
        .eq("latest_date", latest_date)
        .execute()
    )
    count_after = len(after.data or [])
    check("re-run returns cleanly", rerun.status_code == 200, f"got {rerun.status_code}")
    check("row count unchanged for same stock/date", count_before == count_after, f"before={count_before}, after={count_after}")
except Exception as e:
    check("Idempotency check", False, str(e))


# Step 9: Read direction prediction endpoint
step(9, "Read direction prediction endpoint")
try:
    symbol = selected_stock["symbol"] if selected_stock else "AAPL"
    response = httpx.get(f"{BASE_URL}/stocks/{symbol}/direction-prediction", timeout=30)
    check("GET direction prediction returns 200", response.status_code == 200, f"got {response.status_code}")
    data = response.json() if response.content else {}
    required_fields = {"latest_close", "predicted_direction", "predicted_probability", "confidence", "accuracy", "precision", "recall", "f1_score", "decision_threshold", "target_return_threshold"}
    check("response contains prediction and metrics", required_fields.issubset(set(data.keys())))
    print(f"  Direction prediction: {data}")
except Exception as e:
    check("Read direction prediction endpoint", False, str(e))


# Step 10: Read technical indicators endpoint
step(10, "Read technical indicators endpoint")
try:
    symbol = selected_stock["symbol"] if selected_stock else "AAPL"
    response = httpx.get(f"{BASE_URL}/stocks/{symbol}/technical-indicators", timeout=30)
    check("GET technical indicators returns 200", response.status_code == 200, f"got {response.status_code}")
    data = response.json() if response.content else []
    check("response returns a list of indicator rows", isinstance(data, list) and len(data) > 0)
    print(f"  Returned {len(data)} indicator rows")
except Exception as e:
    check("Read technical indicators endpoint", False, str(e))


# Step 11: Invalid symbol handling
step(11, "Invalid symbol handling")
try:
    response = httpx.post(f"{BASE_URL}/technical/run-pipeline/INVALIDTICKERTEST", timeout=120)
    clean_failure = response.status_code in {200, 400, 404}
    if response.status_code == 200:
        clean_failure = response.json().get("status") in {"no_data", "skipped", "error"}
    check("invalid symbol handled cleanly", clean_failure, f"status={response.status_code}, body={response.text[:200]}")
except Exception as e:
    check("Invalid symbol handling", False, str(e))


print(f"\n{'=' * 50}")
if failures:
    print(f"\033[91m{len(failures)} FAILED: {', '.join(failures)}\033[0m")
    sys.exit(1)

print("\033[92mAll steps PASSED\033[0m")
