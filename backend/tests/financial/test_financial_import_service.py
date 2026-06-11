from unittest.mock import patch

from services.financial.financial_service import import_financial_statements


@patch("services.financial.financial_service.save_financial_statements")
@patch("services.financial.financial_service.fetch_quarterly_financial_statements")
@patch("services.financial.financial_service.get_stock_by_symbol")
def test_import_financial_statements_saves_yfinance_rows(
    mock_stock,
    mock_fetch,
    mock_save,
):
    mock_stock.return_value = [{"id": 1, "symbol": "AAPL"}]
    rows = [{
        "stock_id": 1,
        "ticker": "AAPL",
        "period": "2025-12-31",
        "period_type": "quarterly",
    }]
    mock_fetch.return_value = {
        "symbol": "AAPL",
        "yfinance_symbol": "AAPL",
        "rows": rows,
        "skipped_periods": [],
    }
    mock_save.return_value = rows

    result = import_financial_statements("aapl")

    mock_fetch.assert_called_once_with("AAPL", 1)
    mock_save.assert_called_once_with(rows)
    assert result["rows_saved"] == 1
    assert result["symbol"] == "AAPL"
