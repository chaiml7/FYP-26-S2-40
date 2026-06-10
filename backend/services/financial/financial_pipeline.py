"""
backend/services/financial/financial_pipeline.py
- predict_from_db      → latest quarter per ticker → financial_predictions
- predict_all_quarters → all quarters per ticker   → financial_predictions_history
"""

import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .feature_engineering import (
    FEATURE_COLS,
    SCORE_MAP,
    engineer_features,
    prepare_training_data,
)

MODEL_PATH    = Path(__file__).resolve().parents[3] / "models" / "financial_model.joblib"
MODEL_VERSION = "gbm_v1"


def train(df):
    X, y, df_clean = prepare_training_data(df)
    print(f"  Training on {len(df_clean)} rows")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    base_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=2)
    pipeline   = Pipeline([("scaler", StandardScaler()), ("model", calibrated)])

    if False:  # cross-val disabled — small dataset
        cv_scores = cross_val_score(pipeline, X, y_enc, cv=2, scoring="accuracy")
        print(f"  Cross-val accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    pipeline.fit(X, y_enc)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline":      pipeline,
        "label_encoder": le,
        "feature_cols":  FEATURE_COLS,
        "model_version": MODEL_VERSION,
        "trained_at":    datetime.utcnow().isoformat(),
    }, MODEL_PATH)

    print(f"  Model saved → {MODEL_PATH}")
    return pipeline, le


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the pipeline first.")
    return joblib.load(MODEL_PATH)


def _build_prediction(row, pred, proba, classes) -> dict:
    """Build a prediction dict from a row and model output."""
    raw_score  = float(round(max(proba), 4))
    confidence = float(round(raw_score * 100, 2))
    return {
        "stock_id":      int(row["stock_id"]),
        "ticker":        row["ticker"],
        "prediction":    pred,
        "score":         raw_score,
        "confidence":    confidence,
        "model_version": MODEL_VERSION,
        "period":        row["period"],
        "created_at":    datetime.utcnow().isoformat(),
    }


def predict_from_db(df):
    """
    Predict on the LATEST quarter per ticker.
    → stored in financial_predictions (current outlook, 11 rows)
    """
    artifact = _load_model()
    pipeline = artifact["pipeline"]
    le       = artifact["label_encoder"]
    classes  = list(le.classes_)

    df_feat = engineer_features(df)
    latest  = df_feat.sort_values("period").groupby("stock_id").last().reset_index()

    X      = latest[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    proba  = pipeline.predict_proba(X)
    labels = le.inverse_transform(pipeline.predict(X))

    predictions = []
    for i, row in latest.iterrows():
        idx  = list(latest.index).index(i)
        pred = labels[idx]
        p    = _build_prediction(row, pred, proba[idx], classes)
        predictions.append(p)
        print(f"  {row['ticker']:6s} | {pred:8s} | score {p['score']:.4f} | confidence {p['confidence']:.2f}%")

    return predictions


def predict_all_quarters(df):
    """
    Predict on EVERY quarter per ticker.
    → stored in financial_predictions_history (full history)
    """
    artifact = _load_model()
    pipeline = artifact["pipeline"]
    le       = artifact["label_encoder"]
    classes  = list(le.classes_)

    df_feat = engineer_features(df)
    # Drop rows with NaN features (first row per ticker has no growth metrics)
    df_clean = df_feat.dropna(subset=FEATURE_COLS)

    X      = df_clean[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    proba  = pipeline.predict_proba(X)
    labels = le.inverse_transform(pipeline.predict(X))

    predictions = []
    for i, (_, row) in enumerate(df_clean.iterrows()):
        pred = labels[i]
        p    = _build_prediction(row, pred, proba[i], classes)
        predictions.append(p)

    print(f"  Generated {len(predictions)} historical predictions across all quarters")
    return predictions


def predict_new(income: dict, balance: dict, cashflow: dict, ticker: str = "NEW") -> dict:
    import pandas as pd

    artifact   = _load_model()
    pipeline   = artifact["pipeline"]
    le         = artifact["label_encoder"]

    row        = {**income, **balance, **cashflow, "stock_id": 0, "ticker": ticker, "period": "new"}
    df         = engineer_features(pd.DataFrame([row]))
    X          = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    proba      = pipeline.predict_proba(X)[0]
    pred       = le.inverse_transform(pipeline.predict(X))[0]
    raw_score  = float(round(max(proba), 4))
    confidence = float(round(raw_score * 100, 2))

    result = {
        "ticker":     ticker,
        "prediction": pred,
        "score":      raw_score,
        "confidence": confidence,
    }
    print(f"\n  Prediction → {pred.upper()}  score={raw_score:.4f}  confidence={confidence:.2f}%")
    return result
