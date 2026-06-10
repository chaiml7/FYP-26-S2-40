"""
backend/services/financial/yfinance_fetcher.py
Fetches QUARTERLY Cash Flow, Balance Sheet, and Income Statement
from Yahoo Finance. Returns last 12 quarters (~3 years) per ticker.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


def _safe_get(col_series, key: str, default: float = 0.0) -> float:
    try:
        if col_series is None:
            return default
        df = col_series if isinstance(col_series, pd.DataFrame) else col_series.to_frame()
        if df.empty:
            return default
        if key in df.index:
            vals = df.loc[key].dropna()
            return float(vals.iloc[0]) if len(vals) else default
        matches = [i for i in df.index if key.lower() in str(i).lower()]
        if matches:
            vals = df.loc[matches[0]].dropna()
            return float(vals.iloc[0]) if len(vals) else default
    except Exception:
        pass
    return default


def fetch_quarterly_statements(ticker: str, stock_id: int) -> list[dict]:
    """
    Fetch quarterly financial statements only.
    Yahoo Finance returns last 12 quarters (Q1 2024 → Q1 2026).
    """
    print(f"    [{ticker}] Fetching quarterly statements from Yahoo Finance ...")
    t = yf.Ticker(ticker)
    records = []

    try:
        inc = t.quarterly_financials
        bal = t.quarterly_balance_sheet
        cf  = t.quarterly_cashflow

        if inc is None or inc.empty:
            print(f"    [{ticker}] No quarterly data — skipping.")
            return records

        for col in inc.columns:
            period_str = str(col)[:10]

            def g_inc(k): return _safe_get(inc[col].to_frame(), k)
            def g_bal(k): return _safe_get(
                bal[col].to_frame() if bal is not None and col in bal.columns else pd.DataFrame(), k
            )
            def g_cf(k): return _safe_get(
                cf[col].to_frame() if cf is not None and col in cf.columns else pd.DataFrame(), k
            )

            records.append({
                "stock_id":             stock_id,
                "ticker":               ticker,
                "period":               period_str,
                "period_type":          "quarterly",
                "fetched_at":           datetime.utcnow().isoformat(),
                "total_revenue":        g_inc("Total Revenue"),
                "gross_profit":         g_inc("Gross Profit"),
                "operating_income":     g_inc("Operating Income"),
                "net_income":           g_inc("Net Income"),
                "ebitda":               g_inc("EBITDA"),
                "research_development": g_inc("Research Development"),
                "interest_expense":     g_inc("Interest Expense"),
                "total_assets":         g_bal("Total Assets"),
                "total_liabilities":    g_bal("Total Liabilities"),
                "total_equity":         g_bal("Stockholders Equity"),
                "cash_and_equivalents": g_bal("Cash And Cash Equivalents"),
                "total_debt":           g_bal("Total Debt"),
                "current_assets":       g_bal("Current Assets"),
                "current_liabilities":  g_bal("Current Liabilities"),
                "operating_cashflow":   g_cf("Operating Cash Flow"),
                "capex":                g_cf("Capital Expenditure"),
                "free_cashflow":        g_cf("Free Cash Flow"),
                "investing_cashflow":   g_cf("Investing Cash Flow"),
                "financing_cashflow":   g_cf("Financing Cash Flow"),
            })

        records.sort(key=lambda x: x["period"])
        print(f"    [{ticker}] {len(records)} quarters fetched ({records[0]['period']} → {records[-1]['period']})")

    except Exception as e:
        print(f"    [{ticker}] Error during fetch: {e}")

    return records
