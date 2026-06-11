"""Feature and next-quarter target construction for financial statements."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "current_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "asset_turnover",
    "roe",
    "roa",
    "ocf_to_net_income",
    "capex_intensity",
    "rd_intensity",
    "revenue_growth",
    "net_income_growth",
    "fcf_growth",
]

FINANCIAL_COLUMNS = [
    "total_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "research_development",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "operating_cashflow",
    "capex",
    "free_cashflow",
]

REQUIRED_COLUMNS = ["stock_id", "ticker", "period", *FINANCIAL_COLUMNS]
TARGET_POSITIVE_THRESHOLD = 0.15
TARGET_NEGATIVE_THRESHOLD = -0.15
EPSILON = 1e-9


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.abs() > EPSILON
    return numerator.div(denominator.where(valid_denominator))


def _growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    valid_previous = previous.abs() > EPSILON
    return current.sub(previous).div(previous.abs().where(valid_previous))


def _require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Financial statements are missing columns: {', '.join(missing)}")


def engineer_features(statements: pd.DataFrame) -> pd.DataFrame:
    """Return valid quarterly statements with ratio and trend features."""
    _require_columns(statements)
    df = statements.copy()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")

    for column in FINANCIAL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "period_type" in df.columns:
        df = df[df["period_type"].fillna("quarterly").eq("quarterly")]

    valid = (
        df["period"].notna()
        & df["total_revenue"].gt(0)
        & df["total_assets"].gt(0)
        & df["current_liabilities"].gt(0)
        & df["total_equity"].abs().gt(EPSILON)
    )
    df = df.loc[valid].sort_values(["stock_id", "period"]).copy()

    df["gross_margin"] = _safe_ratio(df["gross_profit"], df["total_revenue"])
    df["operating_margin"] = _safe_ratio(df["operating_income"], df["total_revenue"])
    df["net_margin"] = _safe_ratio(df["net_income"], df["total_revenue"])
    df["fcf_margin"] = _safe_ratio(df["free_cashflow"], df["total_revenue"])
    df["current_ratio"] = _safe_ratio(df["current_assets"], df["current_liabilities"])
    df["debt_to_equity"] = _safe_ratio(df["total_debt"], df["total_equity"])
    df["debt_to_assets"] = _safe_ratio(df["total_liabilities"], df["total_assets"])
    df["asset_turnover"] = _safe_ratio(df["total_revenue"], df["total_assets"])
    df["roe"] = _safe_ratio(df["net_income"], df["total_equity"])
    df["roa"] = _safe_ratio(df["net_income"], df["total_assets"])
    df["ocf_to_net_income"] = _safe_ratio(df["operating_cashflow"], df["net_income"])
    df["capex_intensity"] = _safe_ratio(df["capex"].abs(), df["total_revenue"])
    df["rd_intensity"] = _safe_ratio(df["research_development"], df["total_revenue"])

    grouped = df.groupby("stock_id", sort=False)
    df["revenue_growth"] = _growth(
        df["total_revenue"],
        grouped["total_revenue"].shift(1),
    )
    df["net_income_growth"] = _growth(
        df["net_income"],
        grouped["net_income"].shift(1),
    )
    df["fcf_growth"] = _growth(
        df["free_cashflow"],
        grouped["free_cashflow"].shift(1),
    )

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


def _scaled_change(change: pd.Series, meaningful_change: float) -> pd.Series:
    return change.div(meaningful_change).clip(-1.0, 1.0)


def build_training_dataset(statements: pd.DataFrame) -> pd.DataFrame:
    """
    Label quarter Q from quarter Q+1 outcomes.

    This avoids training the model to reproduce rules calculated from the same
    quarter's input features.
    """
    df = engineer_features(statements)
    if df.empty:
        return df

    grouped = df.groupby("stock_id", sort=False)
    next_revenue = grouped["total_revenue"].shift(-1)
    next_operating_margin = grouped["operating_margin"].shift(-1)
    next_net_margin = grouped["net_margin"].shift(-1)
    next_fcf_margin = grouped["fcf_margin"].shift(-1)
    next_debt_to_assets = grouped["debt_to_assets"].shift(-1)

    next_revenue_growth = _growth(next_revenue, df["total_revenue"])
    operating_margin_change = next_operating_margin - df["operating_margin"]
    net_margin_change = next_net_margin - df["net_margin"]
    fcf_margin_change = next_fcf_margin - df["fcf_margin"]
    debt_ratio_improvement = df["debt_to_assets"] - next_debt_to_assets

    df["target_score"] = (
        0.30 * _scaled_change(next_revenue_growth, 0.10)
        + 0.25 * _scaled_change(operating_margin_change, 0.03)
        + 0.20 * _scaled_change(net_margin_change, 0.03)
        + 0.15 * _scaled_change(fcf_margin_change, 0.05)
        + 0.10 * _scaled_change(debt_ratio_improvement, 0.03)
    )

    df["target_label"] = np.select(
        [
            df["target_score"].ge(TARGET_POSITIVE_THRESHOLD),
            df["target_score"].le(TARGET_NEGATIVE_THRESHOLD),
        ],
        ["positive", "negative"],
        default="neutral",
    )

    return df[df["target_score"].notna()].reset_index(drop=True)
