import pandas as pd

from backend.services.technical.feature_engineering import (
    FEATURE_COLUMNS,
    build_training_dataset,
    engineer_model_features,
)


def _indicator(date: str, close: float) -> dict:
    return {
        "stock_id": 1,
        "symbol": "TEST",
        "date": date,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": 1_000_000,
        "return_1d": 0.01,
        "log_return": 0.00995,
        "return_5d": 0.02,
        "high_low_range": 0.03,
        "open_close_gap": 0.01,
        "sma_5": close * 0.99,
        "sma_10": close * 0.98,
        "sma_20": close * 0.97,
        "sma_50": close * 0.96,
        "sma_200": close * 0.90,
        "ema_10": close * 0.985,
        "ema_20": close * 0.975,
        "ema_50": close * 0.95,
        "trend_filter_50_200": 1,
        "rsi_14": 60,
        "macd": close * 0.01,
        "macd_signal": close * 0.008,
        "macd_histogram": close * 0.002,
        "bb_width": 0.08,
        "atr_14": close * 0.025,
        "rolling_volatility_5": 0.015,
        "rolling_volatility_10": 0.018,
        "rolling_volatility_20": 0.02,
        "volume_change": 0.10,
        "relative_volume": 1.2,
        "distance_to_support": 0.05,
        "distance_to_resistance": 0.04,
        "breakout_indicator": 0,
        "breakdown_indicator": 0,
        "return_lag_1": 0.01,
        "return_lag_2": -0.01,
        "return_lag_3": 0.005,
        "return_lag_5": 0.02,
        "return_lag_10": -0.02,
        "market_spy_return_1d": 0.003,
        "market_spy_return_5d": 0.01,
        "market_spy_above_sma_200": 1,
        "market_qqq_return_1d": 0.004,
        "market_qqq_return_5d": 0.012,
        "market_qqq_above_sma_200": 1,
        "market_vix_level": 18,
        "market_vix_return_1d": -0.02,
        "market_vix_return_5d": 0.01,
        "market_sector_return_1d": 0.005,
        "market_sector_return_5d": 0.015,
        "market_sector_above_sma_200": 1,
    }


def test_model_features_are_normalized():
    result = engineer_model_features(
        pd.DataFrame([_indicator("2026-01-02", 100)])
    )

    assert all(column in result.columns for column in FEATURE_COLUMNS)
    assert round(result.iloc[0]["close_to_sma_20"], 4) == 0.0309
    assert result.iloc[0]["rsi_14_scaled"] == 0.2
    assert result.iloc[0]["atr_to_close"] == 0.025
    assert "close" not in FEATURE_COLUMNS
    assert "volume" not in FEATURE_COLUMNS


def test_three_class_target_uses_next_day_return():
    rows = [
        _indicator("2026-01-02", 100.0),
        _indicator("2026-01-05", 101.0),
        _indicator("2026-01-06", 101.1),
        _indicator("2026-01-07", 100.0),
    ]

    result = build_training_dataset(pd.DataFrame(rows))

    assert result["target_label"].tolist() == [
        "bullish",
        "neutral",
        "bearish",
    ]
