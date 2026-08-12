"""Fetch and normalize quarterly company statements from yfinance."""

from datetime import datetime, timezone
from numbers import Number

import pandas as pd
import yfinance as yf


STATEMENT_FIELDS = {
    "total_revenue": (
        "TotalRevenue",
        "OperatingRevenue",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": (
        "OperatingIncome",
        "TotalOperatingIncomeAsReported",
    ),
    "net_income": (
        "NetIncome",
        "NetIncomeCommonStockholders",
    ),
    "ebitda": (
        "EBITDA",
        "NormalizedEBITDA",
    ),
    "research_development": (
        "ResearchAndDevelopment",
        "ResearchDevelopment",
    ),
    "interest_expense": (
        "InterestExpense",
        "InterestExpenseNonOperating",
    ),
    "total_assets": ("TotalAssets",),
    "total_liabilities": (
        "TotalLiabilitiesNetMinorityInterest",
        "TotalLiabilities",
    ),
    "total_equity": (
        "StockholdersEquity",
        "TotalEquityGrossMinorityInterest",
    ),
    "cash_and_equivalents": (
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalents",
        "Cash",
    ),
    "total_debt": ("TotalDebt",),
    "current_assets": ("CurrentAssets",),
    "current_liabilities": ("CurrentLiabilities",),
    "operating_cashflow": (
        "OperatingCashFlow",
        "TotalCashFromOperatingActivities",
    ),
    "capex": (
        "CapitalExpenditure",
        "CapitalExpenditures",
    ),
    "free_cashflow": ("FreeCashFlow",),
    "investing_cashflow": (
        "InvestingCashFlow",
        "TotalCashflowsFromInvestingActivities",
    ),
    "financing_cashflow": (
        "FinancingCashFlow",
        "TotalCashFromFinancingActivities",
    ),
}

INCOME_FIELDS = {
    "total_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "research_development",
    "interest_expense",
}
BALANCE_FIELDS = {
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_equivalents",
    "total_debt",
    "current_assets",
    "current_liabilities",
}
CASHFLOW_FIELDS = {
    "operating_cashflow",
    "capex",
    "free_cashflow",
    "investing_cashflow",
    "financing_cashflow",
}
CORE_FIELDS = (
    "total_revenue",
    "total_assets",
    "total_equity",
    "current_liabilities",
)


def get_yfinance_symbol(symbol: str) -> str:
    """Convert database symbols to Yahoo's ticker convention."""
    return symbol.upper().replace(".", "-")


def _quarterly_statement(ticker, method_name: str) -> pd.DataFrame:
    method = getattr(ticker, method_name)
    statement = method(freq="quarterly", pretty=False)
    if statement is None:
        return pd.DataFrame()
    return statement


def _normalize_index(statement: pd.DataFrame) -> pd.DataFrame:
    if statement.empty:
        return statement
    normalized = statement.copy()
    normalized.index = [
        "".join(character for character in str(value) if character.isalnum())
        for value in normalized.index
    ]
    return normalized


def _numeric_value(statement: pd.DataFrame, period, aliases: tuple[str, ...]):
    if statement.empty or period not in statement.columns:
        return None

    for alias in aliases:
        normalized_alias = "".join(
            character for character in alias if character.isalnum()
        )
        if normalized_alias not in statement.index:
            continue

        value = statement.at[normalized_alias, period]
        if pd.isna(value) or not isinstance(value, Number):
            return None
        return float(value)

    return None


def _periods(*statements: pd.DataFrame) -> list:
    periods = {
        pd.Timestamp(period)
        for statement in statements
        for period in statement.columns
        if not pd.isna(period)
    }
    return sorted(periods)


def _field_value(
    field: str,
    period,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
):
    if field in INCOME_FIELDS:
        statement = income
    elif field in BALANCE_FIELDS:
        statement = balance
    else:
        statement = cashflow
    return _numeric_value(statement, period, STATEMENT_FIELDS[field])


def _derive_free_cashflow(row: dict) -> None:
    if row["free_cashflow"] is not None:
        return
    operating_cashflow = row["operating_cashflow"]
    capex = row["capex"]
    if operating_cashflow is not None and capex is not None:
        row["free_cashflow"] = operating_cashflow + capex


def _invalid_core_fields(row: dict) -> list:
    return [
        field
        for field in CORE_FIELDS
        if row[field] is None or row[field] == 0
    ]


def fetch_quarterly_financial_statements(
    symbol: str,
    stock_id: int,
) -> dict:
    """
    Fetch all quarterly periods returned by Yahoo Finance.

    Yahoo controls the available history; this importer cannot request an
    arbitrary start date for statement data.
    """
    yahoo_symbol = get_yfinance_symbol(symbol)
    ticker = yf.Ticker(yahoo_symbol)

    income = _normalize_index(_quarterly_statement(ticker, "get_income_stmt"))
    balance = _normalize_index(_quarterly_statement(ticker, "get_balance_sheet"))
    cashflow = _normalize_index(_quarterly_statement(ticker, "get_cash_flow"))

    periods = _periods(income, balance, cashflow)
    if not periods:
        return {
            "symbol": symbol.upper(),
            "yfinance_symbol": yahoo_symbol,
            "rows": [],
            "skipped_periods": [],
        }

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    skipped_periods = []

    for period in periods:
        row = {
            "stock_id": stock_id,
            "ticker": symbol.upper(),
            "period": period.date().isoformat(),
            "period_type": "quarterly",
            "fetched_at": fetched_at,
        }
        for field in STATEMENT_FIELDS:
            row[field] = _field_value(
                field,
                period,
                income,
                balance,
                cashflow,
            )

        _derive_free_cashflow(row)
        missing_core_fields = _invalid_core_fields(row)
        if missing_core_fields:
            skipped_periods.append({
                "period": row["period"],
                "missing_core_fields": missing_core_fields,
            })
            continue

        rows.append(row)

    return {
        "symbol": symbol.upper(),
        "yfinance_symbol": yahoo_symbol,
        "rows": rows,
        "skipped_periods": skipped_periods,
    }
