from unittest.mock import patch

from backend.services.stock_list_service import search_active_stocks

MODULE = "backend.services.stock_list_service"

SAMPLE_STOCKS = [
    {"symbol": "AAPL", "company_name": "Apple Inc.", "is_active": True},
    {"symbol": "AMD", "company_name": "Advanced Micro Devices", "is_active": True},
    {"symbol": "AMZN", "company_name": "Amazon.com Inc.", "is_active": True},
    {"symbol": "TSLA", "company_name": "Tesla Inc.", "is_active": True},
]


@patch(f"{MODULE}.get_active_stocks", return_value=SAMPLE_STOCKS)
def test_search_active_stocks_matches_symbol_prefix(mock_stocks):
    result = search_active_stocks("am")
    symbols = [s["symbol"] for s in result]
    assert "AMD" in symbols
    assert "AMZN" in symbols
    assert "AAPL" not in symbols


@patch(f"{MODULE}.get_active_stocks", return_value=SAMPLE_STOCKS)
def test_search_active_stocks_matches_company_name(mock_stocks):
    result = search_active_stocks("tesla")
    assert [s["symbol"] for s in result] == ["TSLA"]


@patch(f"{MODULE}.get_active_stocks", return_value=SAMPLE_STOCKS)
def test_search_active_stocks_symbol_matches_rank_before_name_matches(mock_stocks):
    result = search_active_stocks("a")
    symbols = [s["symbol"] for s in result]
    # AAPL/AMD/AMZN match on symbol prefix; TSLA only matches via company name
    # ("Tesla Inc." contains "a") and must be ranked after all symbol matches.
    assert symbols.index("TSLA") > max(symbols.index(s) for s in ("AAPL", "AMD", "AMZN"))


@patch(f"{MODULE}.get_active_stocks", return_value=SAMPLE_STOCKS)
def test_search_active_stocks_case_insensitive(mock_stocks):
    result = search_active_stocks("AAPL")
    assert [s["symbol"] for s in result] == ["AAPL"]


@patch(f"{MODULE}.get_active_stocks", return_value=SAMPLE_STOCKS)
def test_search_active_stocks_empty_query_returns_empty(mock_stocks):
    assert search_active_stocks("") == []
    assert search_active_stocks("   ") == []


@patch(f"{MODULE}.get_active_stocks", return_value=SAMPLE_STOCKS)
def test_search_active_stocks_respects_limit(mock_stocks):
    result = search_active_stocks("a", limit=2)
    assert len(result) == 2
