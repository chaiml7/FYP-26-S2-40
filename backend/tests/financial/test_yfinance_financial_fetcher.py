from unittest.mock import MagicMock, patch

import pandas as pd

from backend.services.financial.yfinance_financial_fetcher import (
    fetch_quarterly_financial_statements,
    get_yfinance_symbol,
)


PERIOD = pd.Timestamp("2025-12-31")


def _statement(values: dict) -> pd.DataFrame:
    return pd.DataFrame({PERIOD: values})


def _ticker_with_statements(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
):
    ticker = MagicMock()
    ticker.get_income_stmt.return_value = income
    ticker.get_balance_sheet.return_value = balance
    ticker.get_cash_flow.return_value = cashflow
    return ticker


def test_yfinance_symbol_uses_dash_for_share_class():
    assert get_yfinance_symbol("BRK.B") == "BRK-B"


@patch("backend.services.financial.yfinance_financial_fetcher.yf.Ticker")
def test_fetches_aligned_quarter_and_derives_free_cashflow(mock_ticker):
    income = _statement({
        "TotalRevenue": 1000,
        "GrossProfit": 400,
        "OperatingIncome": 150,
        "NetIncome": 100,
    })
    balance = _statement({
        "TotalAssets": 2000,
        "TotalLiabilitiesNetMinorityInterest": 1100,
        "StockholdersEquity": 900,
        "CurrentAssets": 700,
        "CurrentLiabilities": 500,
        "TotalDebt": 300,
    })
    cashflow = _statement({
        "OperatingCashFlow": 180,
        "CapitalExpenditure": -40,
    })
    mock_ticker.return_value = _ticker_with_statements(
        income,
        balance,
        cashflow,
    )

    result = fetch_quarterly_financial_statements("TEST", 7)

    assert result["symbol"] == "TEST"
    assert result["skipped_periods"] == []
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["stock_id"] == 7
    assert row["period"] == "2025-12-31"
    assert row["total_revenue"] == 1000
    assert row["total_liabilities"] == 1100
    assert row["free_cashflow"] == 140
    assert row["research_development"] is None


@patch("backend.services.financial.yfinance_financial_fetcher.yf.Ticker")
def test_skips_period_when_core_field_is_missing(mock_ticker):
    income = _statement({"TotalRevenue": 1000})
    balance = _statement({
        "TotalAssets": 2000,
        "StockholdersEquity": 900,
    })
    cashflow = _statement({"OperatingCashFlow": 180})
    mock_ticker.return_value = _ticker_with_statements(
        income,
        balance,
        cashflow,
    )

    result = fetch_quarterly_financial_statements("TEST", 7)

    assert result["rows"] == []
    assert result["skipped_periods"] == [{
        "period": "2025-12-31",
        "missing_core_fields": ["current_liabilities"],
    }]


@patch("backend.services.financial.yfinance_financial_fetcher.yf.Ticker")
def test_requests_quarterly_unformatted_statements(mock_ticker):
    empty = pd.DataFrame()
    ticker = _ticker_with_statements(empty, empty, empty)
    mock_ticker.return_value = ticker

    fetch_quarterly_financial_statements("AAPL", 1)

    ticker.get_income_stmt.assert_called_once_with(
        freq="quarterly",
        pretty=False,
    )
    ticker.get_balance_sheet.assert_called_once_with(
        freq="quarterly",
        pretty=False,
    )
    ticker.get_cash_flow.assert_called_once_with(
        freq="quarterly",
        pretty=False,
    )
