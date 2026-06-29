"""Normalized technical features and next-day three-class targets."""

import numpy as np
import pandas as pd


RETURN_THRESHOLD = 0.002
EPSILON = 1e-12

FEATURE_COLUMNS = [
    "return_1d",
    "log_return",
    "return_5d",
    "high_low_range",
    "open_close_gap",
    "close_to_sma_5",
    "close_to_sma_10",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_sma_200",
    "close_to_ema_10",
    "close_to_ema_20",
    "close_to_ema_50",
    "ema_20_to_ema_50",
    "trend_filter_50_200",
    "rsi_14_scaled",
    "macd_to_close",
    "macd_signal_to_close",
    "macd_histogram_to_close",
    "bb_width",
    "atr_to_close",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "volume_change",
    "relative_volume",
    "distance_to_support",
    "distance_to_resistance",
    "breakout_indicator",
    "breakdown_indicator",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "market_spy_return_1d",
    "market_spy_return_5d",
    "market_spy_above_sma_200",
    "market_qqq_return_1d",
    "market_qqq_return_5d",
    "market_qqq_above_sma_200",
    "market_vix_level_scaled",
    "market_vix_return_1d",
    "market_vix_return_5d",
    "market_sector_return_1d",
    "market_sector_return_5d",
    "market_sector_above_sma_200",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.abs() > EPSILON))


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Technical indicator data is missing columns: " + ", ".join(missing)
        )


def engineer_model_features(indicators: pd.DataFrame) -> pd.DataFrame:
    """Create price-independent features suitable for a shared stock model."""
    required = [
        "stock_id",
        "symbol",
        "date",
        "close",
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_10",
        "ema_20",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        *[
            column
            for column in FEATURE_COLUMNS
            if column
            not in {
                "close_to_sma_5",
                "close_to_sma_10",
                "close_to_sma_20",
                "close_to_sma_50",
                "close_to_sma_200",
                "close_to_ema_10",
                "close_to_ema_20",
                "close_to_ema_50",
                "ema_20_to_ema_50",
                "rsi_14_scaled",
                "macd_to_close",
                "macd_signal_to_close",
                "macd_histogram_to_close",
                "atr_to_close",
                "market_vix_level_scaled",
            }
        ],
        "market_vix_level",
    ]
    _require_columns(indicators, list(dict.fromkeys(required)))

    df = indicators.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_columns = set(required) - {"symbol", "date"}
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    close = df["close"]
    for period in (5, 10, 20, 50, 200):
        df[f"close_to_sma_{period}"] = _safe_ratio(
            close,
            df[f"sma_{period}"],
        ) - 1
    for period in (10, 20, 50):
        df[f"close_to_ema_{period}"] = _safe_ratio(
            close,
            df[f"ema_{period}"],
        ) - 1

    df["ema_20_to_ema_50"] = _safe_ratio(df["ema_20"], df["ema_50"]) - 1
    df["rsi_14_scaled"] = (df["rsi_14"] - 50) / 50
    df["macd_to_close"] = _safe_ratio(df["macd"], close)
    df["macd_signal_to_close"] = _safe_ratio(df["macd_signal"], close)
    df["macd_histogram_to_close"] = _safe_ratio(df["macd_histogram"], close)
    df["atr_to_close"] = _safe_ratio(df["atr_14"], close)
    df["market_vix_level_scaled"] = df["market_vix_level"] / 100

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return (
        df.dropna(subset=["stock_id", "symbol", "date", "close"])
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )


def build_training_dataset(
    indicators: pd.DataFrame,
    return_threshold: float = RETURN_THRESHOLD,
) -> pd.DataFrame:
    """Label each row bearish, neutral, or bullish from the next trading day."""
    df = engineer_model_features(indicators)
    grouped = df.groupby("stock_id", sort=False)
    next_close = grouped["close"].shift(-1)
    df["next_day_return"] = next_close.div(df["close"]) - 1
    df["target_label"] = np.select(
        [
            df["next_day_return"].lt(-return_threshold),
            df["next_day_return"].gt(return_threshold),
        ],
        ["bearish", "bullish"],
        default="neutral",
    )
    return (
        df.dropna(subset=[*FEATURE_COLUMNS, "next_day_return"])
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )
