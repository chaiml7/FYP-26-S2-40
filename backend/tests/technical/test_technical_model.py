import pandas as pd

from backend.services.technical.technical_model import (
    calculate_technical_score,
    split_dataset_by_date,
)


def test_technical_score_maps_bearish_neutral_and_bullish():
    assert calculate_technical_score({
        "bearish": 1,
        "neutral": 0,
        "bullish": 0,
    }) == (-1.0, 1.0)
    assert calculate_technical_score({
        "bearish": 0,
        "neutral": 1,
        "bullish": 0,
    }) == (0.0, 5.0)
    assert calculate_technical_score({
        "bearish": 0,
        "neutral": 0,
        "bullish": 1,
    }) == (1.0, 10.0)
    assert calculate_technical_score({
        "bearish": 0.2,
        "neutral": 0.3,
        "bullish": 0.5,
    }) == (0.3, 6.5)


def test_split_keeps_whole_dates_in_one_partition_with_embargo():
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    dataset = pd.DataFrame([
        {
            "date": date,
            "stock_id": stock_id,
            "target_label": ["bearish", "neutral", "bullish"][stock_id - 1],
        }
        for date in dates
        for stock_id in (1, 2, 3)
    ])

    train, validation, test = split_dataset_by_date(dataset)

    train_dates = set(train["date"])
    validation_dates = set(validation["date"])
    test_dates = set(test["date"])
    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)
    assert max(train_dates) < min(validation_dates)
    assert max(validation_dates) < min(test_dates)
    assert (min(validation_dates) - max(train_dates)).days == 2
    assert (min(test_dates) - max(validation_dates)).days == 2
