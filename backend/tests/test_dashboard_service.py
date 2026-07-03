from datetime import date
from unittest.mock import patch

from backend.services.dashboard_service import (
    _combined_score,
    _format_volume,
    _model_performance_row,
    _price_summary,
    _score_tone,
    get_dashboard_stocks,
    get_stock_dashboard,
)


def test_public_market_score_uses_product_weights():
    assert _combined_score(8, 6, 4) == 6.6
    assert _combined_score(8, None, 4) is None


def test_public_market_volume_is_compact():
    assert _format_volume(1_250_000) == "1.2M"
    assert _format_volume(None) == "--"


def test_public_model_metrics_formats_registry_metrics():
    result = _model_performance_row(
        "Financial",
        "financial_binary_v1",
        {
            "accuracy": 0.5635,
            "balanced_accuracy": 0.5071,
            "macro_f1": 0.506,
            "log_loss": 0.6923,
        },
        "holdout",
    )

    assert result["accuracy"] == 56.4
    assert result["balanced_accuracy"] == 50.7
    assert result["macro_f1"] == 50.6
    assert result["log_loss"] == 0.6923
    assert result["evaluation_status"] == "Held-out test"


def test_score_tone_distinguishes_missing_and_outlook_ranges():
    assert _score_tone(None) == "unavailable"
    assert _score_tone(3.99) == "bearish"
    assert _score_tone(5) == "neutral"
    assert _score_tone(6) == "bullish"


@patch("backend.services.dashboard_service._recent_prices")
def test_selected_date_requires_price_on_that_exact_day(mock_prices):
    mock_prices.return_value = [
        {"trade_date": "2026-06-10", "close": 100},
        {"trade_date": "2026-06-09", "close": 99},
    ]

    result = _price_summary("AAPL", date(2026, 6, 11))

    assert result["price"] is None
    mock_prices.assert_called_once_with(
        "AAPL",
        selected_date=date(2026, 6, 11),
    )


@patch("backend.services.dashboard_service._dashboard_price_summaries")
@patch("backend.services.dashboard_service.get_active_stocks")
def test_dashboard_stocks_are_sorted_and_use_company_name(
    mock_stocks,
    mock_prices,
):
    mock_stocks.return_value = [
        {"id": 2, "symbol": "MSFT", "company_name": "Microsoft"},
        {"id": 1, "symbol": "AAPL", "company_name": "Apple"},
    ]
    mock_prices.return_value = {
        symbol: {
            "price": 100,
            "change": 1,
            "change_percent": 1,
            "trade_date": "2026-06-11",
        }
        for symbol in ("AAPL", "MSFT")
    }

    result = get_dashboard_stocks()

    assert [stock["symbol"] for stock in result] == ["AAPL", "MSFT"]
    assert result[0]["company_name"] == "Apple"


@patch("backend.services.dashboard_service._dashboard_price_summaries")
@patch("backend.services.dashboard_service.get_active_stocks")
def test_failed_price_lookup_does_not_hide_stock(mock_stocks, mock_prices):
    mock_stocks.return_value = [
        {"id": 1, "symbol": "AAPL", "company_name": "Apple"},
    ]
    mock_prices.side_effect = RuntimeError("temporary database error")

    result = get_dashboard_stocks()

    assert len(result) == 1
    assert result[0]["symbol"] == "AAPL"
    assert result[0]["price"] is None


@patch("backend.services.dashboard_service._recent_prices", return_value=[])
@patch("backend.services.dashboard_service._price_summary")
@patch("backend.services.dashboard_service._financial_prediction")
@patch("backend.services.dashboard_service._sentiment_prediction")
@patch("backend.services.dashboard_service._technical_prediction")
@patch("backend.services.dashboard_service.get_stock_by_symbol")
def test_stock_dashboard_keeps_missing_score_separate_from_bearish(
    mock_stock,
    mock_technical,
    mock_sentiment,
    mock_financial,
    mock_price,
    _mock_history,
):
    mock_stock.return_value = [{
        "id": 1,
        "symbol": "AAPL",
        "company_name": "Apple",
        "is_active": True,
    }]
    mock_technical.return_value = {"technical_score": 3.5}
    mock_sentiment.return_value = None
    mock_financial.return_value = {"fundamental_score": 6.5}
    mock_price.return_value = {
        "price": 100,
        "change": 1,
        "change_percent": 1,
        "trade_date": "2026-06-11",
    }

    selected_date = date(2026, 6, 11)
    result = get_stock_dashboard("aapl", selected_date)

    assert [score["tone"] for score in result["scores"]] == [
        "bearish",
        "unavailable",
        "bullish",
    ]
    assert result["chart_history"] == []
    assert result["price_history"] == []
    mock_technical.assert_called_once_with("AAPL", selected_date)
    mock_sentiment.assert_called_once_with("AAPL", selected_date)
    mock_financial.assert_called_once_with("AAPL", selected_date)
    mock_price.assert_called_once_with("AAPL", selected_date)
