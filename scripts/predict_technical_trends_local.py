"""
Predict next-day direction locally without reading from or writing to Supabase.

Run from repo root after training a model:
    python scripts/predict_technical_trends_local.py --symbol NVDA
    python scripts/predict_technical_trends_local.py --symbol NVDA --as-of-date 2026-06-04

The script fetches yfinance data locally, calculates the same technical
features, loads backend/artifacts/technical_direction_model.joblib, prints the
prediction, and writes a JSON result file on this machine.
"""
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.technical.model_service import (  # noqa: E402
    FEATURES,
    MODEL_ARTIFACT_PATH,
    TARGET_RETURN_THRESHOLD,
    load_model_artifact,
)

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]
MARKET_CONTEXT_COLUMNS = [
    "market_spy_return_1d",
    "market_spy_return_5d",
    "market_spy_above_sma_200",
    "market_qqq_return_1d",
    "market_qqq_return_5d",
    "market_qqq_above_sma_200",
    "market_vix_level",
    "market_vix_return_1d",
    "market_vix_return_5d",
    "market_sector_return_1d",
    "market_sector_return_5d",
    "market_sector_above_sma_200",
]

SECTOR_ETF_BY_SYMBOL = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMD": "XLK",
    "AVGO": "XLK",
    "ORCL": "XLK",
    "CRM": "XLK",
    "ADBE": "XLK",
    "INTC": "XLK",
    "CSCO": "XLK",
    "GOOG": "XLC",
    "GOOGL": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "HD": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "SBUX": "XLY",
    "JPM": "XLF",
    "BAC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "JNJ": "XLV",
    "PFE": "XLV",
    "MRK": "XLV",
    "UNH": "XLV",
    "XOM": "XLE",
    "CVX": "XLE",
    "BABA": "KWEB",
}


def main() -> int:
    args = parse_args()
    try:
        result = run_local_predictions(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print_result(result, output_path)
    return 0 if result["status"] == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict technical next-day direction locally using yfinance and "
            "the saved model artifact. No Supabase reads or writes."
        )
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help=(
            "Ticker to predict. Can be repeated or comma-separated. If omitted, "
            "uses the symbols saved in the model metadata."
        ),
    )
    parser.add_argument(
        "--as-of-date",
        help=(
            "Use this completed trading day's features to predict the next "
            "trading day. Example: 2026-06-04 predicts 2026-06-05."
        ),
    )
    parser.add_argument(
        "--period",
        default="10y",
        help="yfinance lookback period. Default: 10y",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval. Default: 1d",
    )
    parser.add_argument(
        "--model-path",
        default=str(MODEL_ARTIFACT_PATH),
        help="Path to saved .joblib model artifact.",
    )
    parser.add_argument(
        "--output",
        default="technical_analysis/local_predictions.json",
        help="Local JSON output path. Default: technical_analysis/local_predictions.json",
    )
    return parser.parse_args()


def run_local_predictions(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_model_artifact(args.model_path)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})
    decision_threshold = float(metadata.get("decision_threshold") or 0.5)
    trained_symbol = metadata.get("trained_symbol")
    symbols = requested_symbols(args.symbol, metadata)

    if trained_symbol:
        invalid_symbols = [symbol for symbol in symbols if symbol != trained_symbol]
        if invalid_symbols:
            raise ValueError(
                f"saved model was trained only for {trained_symbol}; "
                f"cannot predict {', '.join(invalid_symbols)} with it"
            )

    predictions = []
    errors = []
    for symbol in symbols:
        try:
            predictions.append(
                predict_one_symbol(
                    symbol=symbol,
                    model=model,
                    metadata=metadata,
                    decision_threshold=decision_threshold,
                    period=args.period,
                    interval=args.interval,
                    as_of_date=args.as_of_date,
                )
            )
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    return {
        "status": "ok" if predictions else "no_data",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "local_yfinance",
        "supabase_used": False,
        "model_artifact_path": str(Path(args.model_path)),
        "model_scope": metadata.get("model_scope"),
        "trained_symbol": trained_symbol,
        "symbols_requested": symbols,
        "as_of_date": normalize_date(args.as_of_date) if args.as_of_date else None,
        "decision_threshold": decision_threshold,
        "target_return_threshold": metadata.get(
            "target_return_threshold",
            TARGET_RETURN_THRESHOLD,
        ),
        "metrics": metadata.get("metrics", {}),
        "predictions": predictions,
        "errors": errors,
    }


def requested_symbols(raw_symbols: list[str] | None, metadata: dict[str, Any]) -> list[str]:
    if raw_symbols:
        symbols = []
        for item in raw_symbols:
            symbols.extend(part.strip().upper() for part in item.split(",") if part.strip())
        return sorted(set(symbols))

    trained_symbol = metadata.get("trained_symbol")
    if trained_symbol:
        return [str(trained_symbol).upper()]

    metadata_symbols = metadata.get("symbols") or []
    symbols = [str(symbol).strip().upper() for symbol in metadata_symbols if symbol]
    if not symbols:
        raise ValueError("no symbols provided and no symbols found in model metadata")

    return sorted(set(symbols))


def predict_one_symbol(
    symbol: str,
    model: Any,
    metadata: dict[str, Any],
    decision_threshold: float,
    period: str,
    interval: str,
    as_of_date: str | None,
) -> dict[str, Any]:
    price_df = fetch_price_history(symbol, period=period, interval=interval)
    if price_df.empty:
        raise ValueError("no yfinance price rows returned")

    enriched_df = add_market_context_features(
        price_df,
        symbol=symbol,
        period=period,
        interval=interval,
    )
    indicator_df = add_technical_indicators(enriched_df)
    feature_ready_df = indicator_df.replace([np.inf, -np.inf], np.nan)
    feature_ready_df = feature_ready_df.dropna(subset=FEATURES).copy()
    if feature_ready_df.empty:
        raise ValueError("not enough complete local indicator rows")

    feature_ready_df["date"] = feature_ready_df["date"].astype(str)
    if as_of_date:
        target_date = normalize_date(as_of_date)
        rows = feature_ready_df[feature_ready_df["date"] == target_date]
        if rows.empty:
            raise ValueError(f"no complete feature row found for {target_date}")
        row = rows.iloc[-1]
    else:
        row = feature_ready_df.sort_values("date", ascending=True).iloc[-1]

    latest_features = pd.DataFrame([row[FEATURES].to_dict()])
    probability_up = probability_for_class(model, latest_features, target_class=1)
    prediction = int(probability_up >= decision_threshold)
    predicted_probability = probability_up if prediction == 1 else 1 - probability_up

    next_row = next_indicator_row(indicator_df, str(row["date"]))
    actual_next_day_return = None
    actual_direction = None
    predicted_for_date = None
    if next_row is not None:
        predicted_for_date = str(next_row["date"])
        actual_next_day_return = float(next_row["close"] / row["close"] - 1)
        actual_direction = (
            "up"
            if actual_next_day_return > TARGET_RETURN_THRESHOLD
            else "down"
        )

    return {
        "symbol": symbol,
        "as_of_date": str(row["date"]),
        "predicted_for_date": predicted_for_date,
        "as_of_close": float(row["close"]),
        "predicted_direction": "up" if prediction == 1 else "down",
        "predicted_probability": float(predicted_probability),
        "confidence": float(predicted_probability),
        "probability_up": float(probability_up),
        "decision_threshold": decision_threshold,
        "actual_direction": actual_direction,
        "actual_next_day_return": actual_next_day_return,
        "model_used": metadata.get("model_used", model.__class__.__name__),
    }


def fetch_price_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    raw_df = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    missing_columns = [column for column in PRICE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{symbol} yfinance data missing columns: {missing_columns}")

    df = df[PRICE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(str)
    return df.sort_values("date", ascending=True).reset_index(drop=True)


def add_market_context_features(
    df: pd.DataFrame,
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date
    result = result.dropna(subset=["date"])
    result["date"] = result["date"].astype(str)
    result = result.sort_values("date", ascending=True).reset_index(drop=True)

    context_sources = [
        ("SPY", "market_spy", False),
        ("QQQ", "market_qqq", False),
        ("^VIX", "market_vix", True),
    ]
    sector_etf = SECTOR_ETF_BY_SYMBOL.get(symbol.upper())
    if sector_etf:
        context_sources.append((sector_etf, "market_sector", False))

    context_period = market_context_period(period)
    for ticker, prefix, include_level in context_sources:
        context_df = build_context_frame(
            ticker=ticker,
            prefix=prefix,
            period=context_period,
            interval=interval,
            include_level=include_level,
        )
        if not context_df.empty:
            result = result.merge(context_df, on="date", how="left")

    for column in MARKET_CONTEXT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce").ffill()

    neutral_defaults = {
        "market_spy_return_1d": 0.0,
        "market_spy_return_5d": 0.0,
        "market_spy_above_sma_200": 0.0,
        "market_qqq_return_1d": 0.0,
        "market_qqq_return_5d": 0.0,
        "market_qqq_above_sma_200": 0.0,
        "market_vix_level": 0.0,
        "market_vix_return_1d": 0.0,
        "market_vix_return_5d": 0.0,
        "market_sector_return_1d": 0.0,
        "market_sector_return_5d": 0.0,
        "market_sector_above_sma_200": 0.0,
    }
    result = result.fillna(value=neutral_defaults)
    for column in [
        "market_spy_above_sma_200",
        "market_qqq_above_sma_200",
        "market_sector_above_sma_200",
    ]:
        result[column] = result[column].astype(int)

    return result


def build_context_frame(
    ticker: str,
    prefix: str,
    period: str,
    interval: str,
    include_level: bool,
) -> pd.DataFrame:
    history = fetch_price_history(ticker, period=period, interval=interval)
    if history.empty:
        return pd.DataFrame()

    close = pd.to_numeric(history["close"], errors="coerce")
    context = pd.DataFrame({"date": history["date"]})
    context[f"{prefix}_return_1d"] = close.pct_change()
    context[f"{prefix}_return_5d"] = close.pct_change(periods=5)

    if include_level:
        context[f"{prefix}_level"] = close
    else:
        sma_200 = close.rolling(window=200, min_periods=200).mean()
        context[f"{prefix}_above_sma_200"] = (close > sma_200).astype(int)

    return context.replace([np.inf, -np.inf], np.nan)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date
    result = result.dropna(subset=["date"])
    result["date"] = result["date"].astype(str)
    result = result.sort_values("date", ascending=True).reset_index(drop=True)

    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    close = result["close"]
    high = result["high"]
    low = result["low"]
    open_price = result["open"]
    volume = result["volume"]

    result["return_1d"] = close.pct_change()
    result["log_return"] = np.log(close / close.shift(1))
    result["return_5d"] = close.pct_change(periods=5)
    result["high_low_range"] = (high - low) / close
    result["open_close_gap"] = (close - open_price) / open_price

    result["sma_5"] = close.rolling(window=5, min_periods=5).mean()
    result["sma_10"] = close.rolling(window=10, min_periods=10).mean()
    result["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    result["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    result["sma_200"] = close.rolling(window=200, min_periods=200).mean()
    result["ema_10"] = close.ewm(span=10, adjust=False, min_periods=10).mean()
    result["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    result["trend_filter_50_200"] = (result["ema_50"] > result["sma_200"]).astype(int)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))
    result.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100

    result["bb_middle"] = result["sma_20"]
    bb_std = close.rolling(window=20, min_periods=20).std()
    result["bb_upper"] = result["bb_middle"] + (2 * bb_std)
    result["bb_lower"] = result["bb_middle"] - (2 * bb_std)
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()

    result["rolling_volatility_5"] = result["return_1d"].rolling(window=5, min_periods=5).std()
    result["rolling_volatility_10"] = result["return_1d"].rolling(window=10, min_periods=10).std()
    result["rolling_volatility_20"] = result["return_1d"].rolling(window=20, min_periods=20).std()
    result["volume_sma_20"] = volume.rolling(window=20, min_periods=20).mean()
    result["volume_change"] = volume.pct_change()
    result["relative_volume"] = volume / result["volume_sma_20"].replace(0, np.nan)

    typical_price = (high + low + close) / 3
    rolling_volume = volume.rolling(window=20, min_periods=20).sum()
    rolling_price_volume = (typical_price * volume).rolling(window=20, min_periods=20).sum()
    result["vwap_20"] = rolling_price_volume / rolling_volume.replace(0, np.nan)

    result["support_20"] = low.rolling(window=20, min_periods=20).min()
    result["resistance_20"] = high.rolling(window=20, min_periods=20).max()
    result["distance_to_support"] = (close - result["support_20"]) / close
    result["distance_to_resistance"] = (result["resistance_20"] - close) / close

    previous_resistance_20 = result["resistance_20"].shift(1)
    previous_support_20 = result["support_20"].shift(1)
    result["breakout_indicator"] = (close > previous_resistance_20).astype(int)
    result["breakdown_indicator"] = (close < previous_support_20).astype(int)

    for lag in [1, 2, 3, 5, 10]:
        result[f"return_lag_{lag}"] = result["return_1d"].shift(lag)
    for lag in [1, 2, 5]:
        result[f"close_lag_{lag}"] = close.shift(lag)

    return result.replace([np.inf, -np.inf], np.nan)


def next_indicator_row(indicator_df: pd.DataFrame, as_of_date: str) -> pd.Series | None:
    rows = indicator_df.copy()
    rows["_date_sort"] = pd.to_datetime(rows["date"], errors="coerce")
    as_of_timestamp = pd.to_datetime(as_of_date, errors="coerce")
    future_rows = rows[rows["_date_sort"] > as_of_timestamp]
    if future_rows.empty:
        return None
    return future_rows.sort_values("_date_sort", ascending=True).iloc[0]


def probability_for_class(model: Any, X: pd.DataFrame, target_class: int) -> float:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if target_class not in classes:
        return 1.0 if classes and classes[0] == target_class else 0.0
    return float(probabilities[0][classes.index(target_class)])


def market_context_period(period: str) -> str:
    period_text = str(period or "").strip().lower()
    if period_text == "max":
        return "max"
    if period_text.endswith("y") and period_text[:-1].isdigit():
        return f"{int(period_text[:-1]) + 1}y"
    if period_text.endswith("mo") and period_text[:-2].isdigit():
        return f"{int(period_text[:-2]) + 12}mo"
    return period


def normalize_date(value: str) -> str:
    return pd.to_datetime(value, errors="raise").date().isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def print_result(result: dict[str, Any], output_path: Path) -> None:
    print(f"status: {result['status']}")
    print(f"source: {result['source']}")
    print(f"supabase_used: {result['supabase_used']}")
    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"output: {output_path}")
    if result["errors"]:
        print(f"errors: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"{error['symbol']}: {error['error']}")

    for prediction in result["predictions"]:
        line = (
            f"{prediction['symbol']} as_of={prediction['as_of_date']} "
            f"predicts={prediction['predicted_direction']} "
            f"confidence={prediction['confidence']:.4f}"
        )
        if prediction["predicted_for_date"]:
            line += f" for={prediction['predicted_for_date']}"
        if prediction["actual_direction"]:
            line += (
                f" actual={prediction['actual_direction']} "
                f"return={prediction['actual_next_day_return']:.4f}"
            )
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
