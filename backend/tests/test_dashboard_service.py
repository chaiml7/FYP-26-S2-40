from unittest.mock import patch

from backend.services.dashboard_service import (
    _score_tone,
    get_dashboard_stocks,
    get_stock_dashboard,
)


def test_score_tone_distinguishes_missing_and_outlook_ranges():
    assert _score_tone(None) == "unavailable"
    assert _score_tone(3.99) == "bearish"
    assert _score_tone(5) == "neutral"
    assert _score_tone(6) == "bullish"


@patch("backend.services.dashboard_service._price_summary")
@patch("backend.services.dashboard_service.get_active_stocks")
def test_dashboard_stocks_are_sorted_and_use_company_name(
    mock_stocks,
    mock_price,
):
    mock_stocks.return_value = [
        {"id": 2, "symbol": "MSFT", "company_name": "Microsoft"},
        {"id": 1, "symbol": "AAPL", "company_name": "Apple"},
    ]
    mock_price.return_value = {
        "price": 100,
        "change": 1,
        "change_percent": 1,
        "trade_date": "2026-06-11",
    }

    result = get_dashboard_stocks()

    assert [stock["symbol"] for stock in result] == ["AAPL", "MSFT"]
    assert result[0]["company_name"] == "Apple"


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

    result = get_stock_dashboard("aapl")

    assert [score["tone"] for score in result["scores"]] == [
        "bearish",
        "unavailable",
        "bullish",
    ]
