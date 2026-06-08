import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from services.financials_service import fetch_company_quarterly_statements

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "financials_ml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LAGS = 4


def _sanitize_filename(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _quarter_features(period: pd.Period) -> np.ndarray:
    quarter = period.quarter
    angle = 2 * np.pi * (quarter - 1) / 4
    return np.array([np.sin(angle), np.cos(angle)], dtype=float)


def _series_to_samples(series: pd.Series, num_lags: int = DEFAULT_LAGS):
    series = series.dropna().astype(float)
    if len(series) <= num_lags:
        return None, None

    X = []
    y = []

    for t in range(num_lags, len(series)):
        lags = series.iloc[t - num_lags:t].to_numpy(dtype=float)
        if np.any(np.isnan(lags)):
            continue

        growth = np.diff(lags) / np.where(lags[:-1] == 0, np.nan, lags[:-1])
        growth_last = float(growth[-1]) if len(growth) > 0 and np.isfinite(growth[-1]) else 0.0
        avg_growth = float(np.nanmean(growth)) if np.any(np.isfinite(growth)) else 0.0

        current_period = pd.Period(series.index[t], freq="Q")
        quarter_feats = _quarter_features(current_period)

        features = np.concatenate([lags, [growth_last, avg_growth], quarter_feats])
        X.append(features)
        y.append(float(series.iloc[t]))

    if len(y) == 0:
        return None, None

    return np.vstack(X), np.array(y, dtype=float)


def _next_quarter_period(series: pd.Series) -> pd.Period:
    last_period = pd.Period(series.index[-1], freq="Q")
    candidate = last_period + 1
    current_upcoming = pd.Period(pd.Timestamp.today(), freq="Q") + 1
    return current_upcoming if current_upcoming > candidate else candidate


def _build_prediction_features(series: pd.Series, num_lags: int = DEFAULT_LAGS):
    series = series.dropna().astype(float)
    if len(series) < num_lags:
        return None

    lags = series.iloc[-num_lags:].to_numpy(dtype=float)
    if np.any(np.isnan(lags)):
        return None

    growth = np.diff(lags) / np.where(lags[:-1] == 0, np.nan, lags[:-1])
    growth_last = float(growth[-1]) if len(growth) > 0 and np.isfinite(growth[-1]) else 0.0
    avg_growth = float(np.nanmean(growth)) if np.any(np.isfinite(growth)) else 0.0

    next_period = _next_quarter_period(series)
    quarter_feats = _quarter_features(next_period)

    return np.concatenate([lags, [growth_last, avg_growth], quarter_feats]), str(next_period)


def _model_path(symbol: str, category: str, item: str) -> Path:
    file_name = _sanitize_filename(f"{symbol}_{category}_{item}.pkl")
    return MODEL_DIR / file_name


def train_item_model(symbol: str, category: str, item: str, num_lags: int = DEFAULT_LAGS, test_size: float = 0.2):
    statements = fetch_company_quarterly_statements(symbol)
    df = statements.get(category)
    if df is None or df.empty or item not in df.columns:
        return None

    series = df[item].dropna()
    X, y = _series_to_samples(series, num_lags)
    if X is None or y is None or len(y) == 0:
        return None

    pipeline = Pipeline([
        ("model", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbosity=0
        ))
    ])

    if len(y) == 1:
        pipeline.fit(X, y)
        train_score = None
        test_score = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        pipeline.fit(X_train, y_train)
        train_score = float(pipeline.score(X_train, y_train))
        test_score = float(pipeline.score(X_test, y_test))

    model_data = {
        "symbol": symbol,
        "category": category,
        "item": item,
        "num_lags": num_lags,
        "pipeline": pipeline,
    }

    model_path = _model_path(symbol, category, item)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    return {
        "symbol": symbol,
        "category": category,
        "item": item,
        "model_path": str(model_path),
        "train_score": train_score,
        "test_score": test_score,
        "samples": len(y),
    }


def load_item_model(symbol: str, category: str, item: str):
    model_path = _model_path(symbol, category, item)
    if not model_path.exists():
        return None

    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_item_next_quarter(symbol: str, category: str, item: str, num_lags: int = DEFAULT_LAGS):
    model_data = load_item_model(symbol, category, item)
    if model_data is None:
        return None

    statements = fetch_company_quarterly_statements(symbol)
    df = statements.get(category)
    if df is None or df.empty or item not in df.columns:
        return None

    series = df[item].dropna()
    features_quarter = _build_prediction_features(series, num_lags)
    if features_quarter is None:
        return None

    features, next_quarter = features_quarter
    prediction = float(model_data["pipeline"].predict(features.reshape(1, -1))[0])
    return {
        "symbol": symbol,
        "category": category,
        "item": item,
        "next_quarter": next_quarter,
        "predicted_value": prediction,
        "model_path": str(_model_path(symbol, category, item)),
    }
