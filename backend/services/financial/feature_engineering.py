"""
backend/services/financial/feature_engineering.py
Derives 16 financial health ratio features from raw statement data.
Uses probabilistic soft labeling to avoid overfit on small datasets.
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
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

SCORE_MAP = {
    "positive":  1.0,
    "neutral":   0.0,
    "negative": -1.0,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio-based features from raw financial statement columns."""
    e = 1e-9
    df = df.copy()

    df["gross_margin"]      = df["gross_profit"]         / (df["total_revenue"]       + e)
    df["operating_margin"]  = df["operating_income"]     / (df["total_revenue"]       + e)
    df["net_margin"]        = df["net_income"]           / (df["total_revenue"]       + e)
    df["fcf_margin"]        = df["free_cashflow"]        / (df["total_revenue"]       + e)
    df["current_ratio"]     = df["current_assets"]       / (df["current_liabilities"] + e)
    df["debt_to_equity"]    = df["total_debt"]           / (df["total_equity"]        + e)
    df["debt_to_assets"]    = df["total_liabilities"]    / (df["total_assets"]        + e)
    df["asset_turnover"]    = df["total_revenue"]        / (df["total_assets"]        + e)
    df["roe"]               = df["net_income"]           / (df["total_equity"]        + e)
    df["roa"]               = df["net_income"]           / (df["total_assets"]        + e)
    df["ocf_to_net_income"] = df["operating_cashflow"]   / (df["net_income"]          + e)
    df["capex_intensity"]   = abs(df["capex"])           / (df["total_revenue"]       + e)
    df["rd_intensity"]      = df["research_development"] / (df["total_revenue"]       + e)

    df = df.sort_values(["stock_id", "period"])
    df["revenue_growth"]    = df.groupby("stock_id")["total_revenue"].pct_change()
    df["net_income_growth"] = df.groupby("stock_id")["net_income"].pct_change()
    df["fcf_growth"]        = df.groupby("stock_id")["free_cashflow"].pct_change()

    return df


def label_outlook(row: pd.Series) -> str:
    """
    Nuanced labeling using a continuous score across 8 dimensions.
    Wider neutral band ensures realistic class distribution.
    """
    score = 0.0

    # 1. Revenue growth (weighted x2)
    rg = row.get("revenue_growth") or 0
    if   rg >  0.20: score += 2.0
    elif rg >  0.08: score += 1.0
    elif rg >  0.00: score += 0.5
    elif rg > -0.05: score -= 0.5
    else:            score -= 2.0

    # 2. Net margin
    nm = row.get("net_margin") or 0
    if   nm > 0.20: score += 2.0
    elif nm > 0.10: score += 1.0
    elif nm > 0.03: score += 0.5
    elif nm > 0.00: score -= 0.5
    else:           score -= 2.0

    # 3. FCF margin
    fm = row.get("fcf_margin") or 0
    if   fm > 0.15: score += 2.0
    elif fm > 0.08: score += 1.0
    elif fm > 0.00: score += 0.5
    elif fm > -0.05: score -= 0.5
    else:            score -= 2.0

    # 4. Current ratio
    cr = row.get("current_ratio") or 1
    if   cr > 2.5: score += 1.0
    elif cr > 1.5: score += 0.5
    elif cr > 1.0: score += 0.0
    elif cr > 0.8: score -= 1.0
    else:          score -= 2.0

    # 5. Debt to equity
    de = row.get("debt_to_equity") or 1
    if   de < 0.3:  score += 1.0
    elif de < 0.8:  score += 0.5
    elif de < 1.5:  score += 0.0
    elif de < 3.0:  score -= 1.0
    else:           score -= 2.0

    # 6. ROE
    roe = row.get("roe") or 0
    if   roe > 0.30: score += 1.0
    elif roe > 0.15: score += 0.5
    elif roe > 0.00: score += 0.0
    else:            score -= 1.0

    # 7. Net income growth
    ng = row.get("net_income_growth") or 0
    if   ng >  0.20: score += 1.0
    elif ng >  0.05: score += 0.5
    elif ng > -0.10: score += 0.0
    else:            score -= 1.0

    # 8. OCF to net income (cash quality)
    ocf = row.get("ocf_to_net_income") or 0
    if   ocf > 1.5: score += 1.0
    elif ocf > 1.0: score += 0.5
    elif ocf > 0.5: score += 0.0
    else:           score -= 1.0

    # Wider neutral band → more realistic distribution
    if   score >= 5.0:  return "positive"
    elif score <= -2.0: return "negative"
    else:               return "neutral"


def prepare_training_data(df: pd.DataFrame):
    """Apply feature engineering + labeling. Returns (X, y, df_with_features)."""
    df = engineer_features(df)
    df["label"] = df.apply(label_outlook, axis=1)

    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    df_clean = df.dropna(subset=FEATURE_COLS)
    X = df_clean[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    y = df_clean["label"]

    return X, y, df_clean


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix for inference."""
    df = engineer_features(df)
    return df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
