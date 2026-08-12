import pandas as pd
import pytest

from backend.services.technical import binary_xgboost_model as binary_model
from backend.services.technical.feature_engineering import FEATURE_COLUMNS


def _indicator_rows():
    rows = []
    for index, close in enumerate((100.0, 101.0, 103.0, 104.0)):
        rows.append({
            "stock_id": 1,
            "symbol": "TEST",
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
            "close": close,
            **{column: 0.0 for column in FEATURE_COLUMNS},
        })
    return rows


def test_binary_dataset_uses_requested_trading_day_horizon(monkeypatch):
    monkeypatch.setattr(binary_model, "engineer_model_features", lambda frame: frame)

    dataset = binary_model.build_binary_dataset(
        _indicator_rows(),
        return_threshold=0.0298,
        prediction_horizon_days=2,
    )

    assert len(dataset) == 2
    assert dataset.iloc[0]["target_return"] == pytest.approx(0.03)
    assert dataset.iloc[0]["target_direction"] == 1
    assert dataset.iloc[1]["target_direction"] == 0


def test_sentiment_features_keep_no_news_rows_as_zero():
    indicators = pd.DataFrame([
        {"stock_id": 1, "date": "2025-01-01", "close": 100},
        {"stock_id": 2, "date": "2025-01-01", "close": 200},
    ])
    scores = [{
        "stock_id": 1,
        "score_date": "2025-01-01",
        "raw_sentiment": 0.5,
        "bullish_score": 7.5,
        "article_count": 2,
        "positive_count": 2,
        "negative_count": 0,
    }]

    result = binary_model._add_sentiment_features(indicators, scores)

    assert result.loc[result["stock_id"] == 1, "sentiment_available"].iloc[0] == 1.0
    assert result.loc[result["stock_id"] == 2, "sentiment_available"].iloc[0] == 0.0
    assert result.loc[result["stock_id"] == 2, "sentiment_raw"].iloc[0] == 0.0
