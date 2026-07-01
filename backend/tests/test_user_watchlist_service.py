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
