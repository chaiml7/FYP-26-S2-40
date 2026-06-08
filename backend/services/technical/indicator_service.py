from typing import Any

import numpy as np
import pandas as pd

from database.supabase_client import supabase
from services.technical.price_service import MARKET_CONTEXT_COLUMNS, _to_json_value

INDICATOR_UPSERT_BATCH_SIZE = 500

INDICATOR_COLUMNS = [
    "date",
    "close",
    "volume",
    "return_1d",
    "log_return",
    "return_5d",
    "high_low_range",
    "open_close_gap",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_10",
    "ema_20",
    "ema_50",
    "trend_filter_50_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "volume_sma_20",
    "volume_change",
    "relative_volume",
    "vwap_20",
    "support_20",
    "resistance_20",
    "distance_to_support",
    "distance_to_resistance",
    "breakout_indicator",
    "breakdown_indicator",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "close_lag_1",
    "close_lag_2",
    "close_lag_5",
    *MARKET_CONTEXT_COLUMNS,
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily technical indicators using only current and past rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date
    result = result.dropna(subset=["date"])
    result["date"] = result["date"].astype(str)
    result = result.sort_values("date", ascending=True).reset_index(drop=True)

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if column in result.columns:
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

    # Daily yfinance data has no intraday ticks, so this is a rolling
    # volume-weighted typical-price approximation rather than tick VWAP.
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


def upsert_technical_indicators(stock_id: int, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    """Upsert calculated indicator rows into Supabase."""
    if df is None or df.empty:
        return {"rows_saved": 0}

    clean_symbol = symbol.upper()
    rows = []
    for _, row in df.iterrows():
        payload = {
            "stock_id": _to_json_value(stock_id),
            "symbol": clean_symbol,
        }
        for column in INDICATOR_COLUMNS:
            if column in row:
                payload[column] = _to_json_value(row[column])
        rows.append(payload)

    saved_rows = 0
    for batch in _chunks(rows, INDICATOR_UPSERT_BATCH_SIZE):
        response = (
            supabase.table("technical_indicators")
            .upsert(batch, on_conflict="stock_id,date")
            .execute()
        )
        saved_rows += len(response.data or batch)

    return {"rows_saved": saved_rows}


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]
