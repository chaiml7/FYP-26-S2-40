import pandas as pd

from services.financial.feature_engineering import (
    FEATURE_COLUMNS,
    build_training_dataset,
    engineer_features,
)
from services.financial.financial_model import _chronological_holdout


def _statement(stock_id: int, ticker: str, period: str, revenue: float, quality: float):
    return {
        "stock_id": stock_id,
        "ticker": ticker,
        "period": period,
        "period_type": "quarterly",
        "total_revenue": revenue,
        "gross_profit": revenue * (0.40 + quality * 0.04),
        "operating_income": revenue * (0.10 + quality * 0.04),
        "net_income": revenue * (0.08 + quality * 0.03),
        "research_development": revenue * 0.05,
        "total_assets": revenue * 2,
        "total_liabilities": revenue * (1.1 - quality * 0.10),
        "total_equity": revenue * (0.9 + quality * 0.10),
        "total_debt": revenue * (0.5 - quality * 0.05),
        "current_assets": revenue * (0.8 + quality * 0.05),
        "current_liabilities": revenue * 0.5,
        "operating_cashflow": revenue * (0.12 + quality * 0.04),
        "capex": revenue * -0.03,
        "free_cashflow": revenue * (0.07 + quality * 0.05),
    }


def test_engineer_features_computes_expected_ratios():
    statements = pd.DataFrame([
        _statement(1, "TEST", "2025-03-31", 100, 0),
        _statement(1, "TEST", "2025-06-30", 110, 0.5),
    ])

    result = engineer_features(statements)

    assert len(result) == 2
    assert all(column in result.columns for column in FEATURE_COLUMNS)
    assert result.iloc[0]["gross_margin"] == 0.40
    assert round(result.iloc[1]["revenue_growth"], 4) == 0.10


def test_training_labels_use_following_quarter_outcome():
    statements = pd.DataFrame([
        _statement(1, "TEST", "2025-03-31", 100, 0),
        _statement(1, "TEST", "2025-06-30", 125, 1),
        _statement(1, "TEST", "2025-09-30", 90, -1),
    ])

    result = build_training_dataset(statements)

    assert len(result) == 2
    assert result.iloc[0]["target_label"] == "positive"
    assert result.iloc[1]["target_label"] == "negative"
    assert result.iloc[0]["period"].date().isoformat() == "2025-03-31"


def test_invalid_zero_statement_is_excluded():
    invalid = _statement(1, "TEST", "2025-03-31", 100, 0)
    invalid["total_revenue"] = 0
    valid = _statement(1, "TEST", "2025-06-30", 110, 0)

    result = engineer_features(pd.DataFrame([invalid, valid]))

    assert len(result) == 1
    assert result.iloc[0]["period"].date().isoformat() == "2025-06-30"


def test_chronological_holdout_uses_latest_labeled_row_per_stock():
    rows = []
    for stock_id, ticker in [(1, "AAA"), (2, "BBB")]:
        rows.extend([
            _statement(stock_id, ticker, "2025-03-31", 100, 0),
            _statement(stock_id, ticker, "2025-06-30", 110, 0.2),
            _statement(stock_id, ticker, "2025-09-30", 120, 0.4),
        ])
    dataset = build_training_dataset(pd.DataFrame(rows))

    train, holdout = _chronological_holdout(dataset)

    assert len(train) == 2
    assert len(holdout) == 2
    assert set(holdout["period"].dt.date.astype(str)) == {"2025-06-30"}
