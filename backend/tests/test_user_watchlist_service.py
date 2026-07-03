import pytest
from unittest.mock import patch

from backend.services.user_watchlist_service import (
    add_watchlist_by_symbol,
    get_user_watchlist_symbols,
    remove_watchlist_by_symbol,
)

MODULE = "backend.services.user_watchlist_service"


@patch(f"{MODULE}.add_user_watchlist_stock")
@patch(f"{MODULE}.get_stock_by_symbol")
def test_add_watchlist_by_symbol_resolves_symbol_to_stock_id(mock_get_stock, mock_add):
    mock_get_stock.return_value = [{"id": 5, "symbol": "AAPL"}]
    mock_add.return_value = [{"user_id": "u1", "stock_id": 5}]

    result = add_watchlist_by_symbol("u1", "aapl")

    mock_get_stock.assert_called_once_with("aapl")
    mock_add.assert_called_once_with("u1", 5)
    assert result == {"user_id": "u1", "stock_id": 5}


@patch(f"{MODULE}.get_stock_by_symbol")
def test_add_watchlist_by_symbol_raises_when_symbol_not_found(mock_get_stock):
    mock_get_stock.return_value = []

    with pytest.raises(ValueError):
        add_watchlist_by_symbol("u1", "ZZZZ")


@patch(f"{MODULE}.remove_user_watchlist_stock")
@patch(f"{MODULE}.get_stock_by_symbol")
def test_remove_watchlist_by_symbol_resolves_symbol_to_stock_id(mock_get_stock, mock_remove):
    mock_get_stock.return_value = [{"id": 5, "symbol": "AAPL"}]
    mock_remove.return_value = [{"id": 1}]

    result = remove_watchlist_by_symbol("u1", "aapl")

    mock_remove.assert_called_once_with("u1", 5)
    assert result == {"stock_id": 5, "rows_deleted": 1}


@patch(f"{MODULE}.get_stock_by_symbol")
def test_remove_watchlist_by_symbol_raises_when_symbol_not_found(mock_get_stock):
    mock_get_stock.return_value = []

    with pytest.raises(ValueError):
        remove_watchlist_by_symbol("u1", "ZZZZ")


@patch(f"{MODULE}.get_user_watchlist")
def test_get_user_watchlist_symbols_extracts_symbols(mock_get_watchlist):
    mock_get_watchlist.return_value = [
        {"id": 1, "stock_id": 5, "stocks": {"symbol": "AAPL"}},
        {"id": 2, "stock_id": 6, "stocks": {"symbol": "NVDA"}},
    ]

    result = get_user_watchlist_symbols("u1")

    assert result == ["AAPL", "NVDA"]


@patch(f"{MODULE}.get_user_watchlist")
def test_get_user_watchlist_symbols_skips_rows_missing_stock_join(mock_get_watchlist):
    mock_get_watchlist.return_value = [
        {"id": 1, "stock_id": 5, "stocks": None},
        {"id": 2, "stock_id": 6, "stocks": {"symbol": "NVDA"}},
    ]

    result = get_user_watchlist_symbols("u1")

    assert result == ["NVDA"]


from backend.services.user_watchlist_service import get_user_watchlist_summary


@patch(f"{MODULE}.get_sentiment_summary")
@patch(f"{MODULE}.get_latest_prediction_by_symbol")
@patch(f"{MODULE}._price_summary")
@patch(f"{MODULE}.get_user_watchlist")
def test_get_user_watchlist_summary_builds_row_from_weighted_sentiment(
    mock_get_watchlist, mock_price, mock_prediction, mock_sentiment
):
    mock_get_watchlist.return_value = [
        {
            "id": 10,
            "stock_id": 5,
            "created_at": "2026-06-01T00:00:00Z",
            "stocks": {"symbol": "AAPL", "company_name": "Apple Inc.", "sector": "Tech"},
        }
    ]
    mock_price.return_value = {
        "price": 192.45, "change": 1.23, "change_percent": 0.64, "trade_date": "2026-07-01",
    }
    mock_prediction.return_value = [{"signal": "buy"}]
    mock_sentiment.return_value = {
        "weighted_scores": [{"bullish_score": 7.5, "sentiment_label": "bullish"}],
        "daily_scores": [],
    }

    result = get_user_watchlist_summary("u1")

    assert result == [{
        "watchlist_id": 10,
        "stock_id": 5,
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Tech",
        "price": 192.45,
        "change": 1.23,
        "change_percent": 0.64,
        "trade_date": "2026-07-01",
        "prediction_signal": "buy",
        "sentiment_label": "bullish",
        "sentiment_score": 7.5,
        "added_at": "2026-06-01T00:00:00Z",
    }]
    mock_price.assert_called_once_with("AAPL")
    mock_prediction.assert_called_once_with("AAPL")


@patch(f"{MODULE}.get_sentiment_summary")
@patch(f"{MODULE}.get_latest_prediction_by_symbol")
@patch(f"{MODULE}._price_summary")
@patch(f"{MODULE}.get_user_watchlist")
def test_get_user_watchlist_summary_falls_back_to_legacy_daily_scores(
    mock_get_watchlist, mock_price, mock_prediction, mock_sentiment
):
    mock_get_watchlist.return_value = [
        {
            "id": 11,
            "stock_id": 6,
            "created_at": "2026-06-02T00:00:00Z",
            "stocks": {"symbol": "NVDA", "company_name": "NVIDIA Corp.", "sector": "Tech"},
        }
    ]
    mock_price.return_value = {"price": None, "change": None, "change_percent": None, "trade_date": None}
    mock_prediction.return_value = []
    mock_sentiment.return_value = {
        "weighted_scores": [],
        "daily_scores": [{"avg_score": 0.8, "label": "positive"}],
    }

    result = get_user_watchlist_summary("u1")

    assert result[0]["prediction_signal"] is None
    assert result[0]["sentiment_label"] == "positive"
    assert result[0]["sentiment_score"] == 0.8


@patch(f"{MODULE}.get_sentiment_summary", side_effect=RuntimeError("sentiment service down"))
@patch(f"{MODULE}.get_latest_prediction_by_symbol", return_value=[])
@patch(f"{MODULE}._price_summary")
@patch(f"{MODULE}.get_user_watchlist")
def test_get_user_watchlist_summary_treats_sentiment_failure_as_no_data(
    mock_get_watchlist, mock_price, _mock_prediction, _mock_sentiment
):
    mock_get_watchlist.return_value = [
        {
            "id": 12,
            "stock_id": 7,
            "created_at": "2026-06-03T00:00:00Z",
            "stocks": {"symbol": "PLTR", "company_name": "Palantir", "sector": "Tech"},
        }
    ]
    mock_price.return_value = {"price": 21.4, "change": None, "change_percent": None, "trade_date": None}

    result = get_user_watchlist_summary("u1")

    assert result[0]["sentiment_label"] is None
    assert result[0]["sentiment_score"] is None
