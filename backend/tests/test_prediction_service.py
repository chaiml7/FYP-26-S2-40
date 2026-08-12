from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.services.prediction_service import get_financial_score
from backend.services.prediction_service import get_technical_score


@patch("backend.services.prediction_service.supabase")
def test_get_financial_score_returns_stored_fundamental_score(mock_supabase):
    query = MagicMock()
    mock_supabase.table.return_value = query
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.execute.return_value = SimpleNamespace(
        data=[{"fundamental_score": 7.83}]
    )

    result = get_financial_score("aapl", date(2026, 3, 31))

    assert result == 7.83
    query.select.assert_called_once_with("fundamental_score")
    query.eq.assert_any_call("ticker", "AAPL")
    query.eq.assert_any_call("period", "2026-03-31")


@patch("backend.services.prediction_service.supabase")
def test_get_financial_score_returns_zero_when_score_is_missing(mock_supabase):
    query = MagicMock()
    mock_supabase.table.return_value = query
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.execute.return_value = SimpleNamespace(
        data=[{"fundamental_score": None}]
    )

    assert get_financial_score("AAPL", date(2026, 3, 31)) == 0


@patch("backend.services.prediction_service.supabase")
def test_get_technical_score_returns_stored_score(mock_supabase):
    query = MagicMock()
    mock_supabase.table.return_value = query
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.execute.return_value = SimpleNamespace(
        data=[{"technical_score": 8.25}]
    )

    result = get_technical_score("nvda")

    assert result == 8.25
    query.select.assert_called_once_with("technical_score")
    query.eq.assert_called_once_with("symbol", "NVDA")
