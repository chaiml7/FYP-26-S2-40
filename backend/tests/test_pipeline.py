"""
backend/tests/test_pipeline.py
Basic tests for the financial ML pipeline.

Usage:
    cd financial_ml
    python -m pytest backend/tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import pandas as pd
from backend.services.financial.feature_engineering import engineer_features, label_outlook, FEATURE_COLS
from backend.services.financial.financial_pipeline import predict_new
from backend.services.financials_service import SYMBOL_TO_YFINANCE, get_stock_id_map


def make_sample_row():
    return {
        "stock_id": 1, "ticker": "TEST", "period": "2024-03-31",
        "total_revenue": 100e9, "gross_profit": 43e9, "operating_income": 29e9,
        "net_income": 24e9, "ebitda": 35e9, "research_development": 8e9,
        "interest_expense": 1e9, "total_assets": 200e9, "total_liabilities": 140e9,
        "total_equity": 60e9, "cash_and_equivalents": 30e9, "total_debt": 50e9,
        "current_assets": 60e9, "current_liabilities": 40e9,
        "operating_cashflow": 35e9, "capex": -5e9, "free_cashflow": 30e9,
        "investing_cashflow": -8e9, "financing_cashflow": -20e9,
    }


def test_feature_engineering():
    """Features are computed correctly from raw financials."""
    df  = pd.DataFrame([make_sample_row(), make_sample_row()])
    df["period"] = ["2023-03-31", "2024-03-31"]
    out = engineer_features(df)
    assert "gross_margin" in out.columns
    assert "net_margin" in out.columns
    assert len(out) == 2
    print("  test_feature_engineering PASSED")


def test_label_outlook():
    """Label function returns valid class."""
    row = pd.Series({
        "revenue_growth": 0.15, "net_margin": 0.24, "fcf_margin": 0.30,
        "current_ratio": 1.5, "debt_to_equity": 0.8, "roe": 0.40,
        "net_income_growth": 0.10, "ocf_to_net_income": 1.2,
    })
    label = label_outlook(row)
    assert label in ("positive", "neutral", "negative")
    print(f"  test_label_outlook PASSED → {label}")


def test_feature_cols_complete():
    """All 16 feature columns are defined."""
    assert len(FEATURE_COLS) == 16
    print(f"  test_feature_cols_complete PASSED → {len(FEATURE_COLS)} features")


def test_symbol_mapping():
    """PLTTR maps to PLTR for yfinance."""
    assert SYMBOL_TO_YFINANCE.get("PLTTR") == "PLTR"
    print("  test_symbol_mapping PASSED")


if __name__ == "__main__":
    test_feature_engineering()
    test_label_outlook()
    test_feature_cols_complete()
    test_symbol_mapping()
    print("\n  All tests passed!")
