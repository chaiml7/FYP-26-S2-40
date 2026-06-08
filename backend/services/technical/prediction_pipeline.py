import logging
from typing import Any

import numpy as np
import pandas as pd

from database.supabase_client import supabase
from services.technical.indicator_service import (
    add_technical_indicators,
    upsert_technical_indicators,
)
from services.technical.model_service import (
    FEATURES,
    TARGET_RETURN_THRESHOLD,
    DEFAULT_MAX_FEATURES,
    get_feature_importance,
    train_final_model,
    tune_lightgbm_params,
    walk_forward_validation,
)
from services.technical.price_service import (
    add_market_context_features,
    fetch_price_history,
    get_stocks_from_supabase,
    upsert_stock_prices,
)

logger = logging.getLogger(__name__)


def run_technical_pipeline_for_stock(stock_id: int, symbol: str) -> dict[str, Any]:
    """Run the full technical-analysis classification pipeline for one ticker."""
    clean_symbol = symbol.upper()
    price_df = fetch_price_history(clean_symbol)

    if price_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No yfinance price history returned",
        }

    price_result = upsert_stock_prices(stock_id, clean_symbol, price_df)

    enriched_price_df = add_market_context_features(price_df, symbol=clean_symbol)
    indicator_df = add_technical_indicators(enriched_price_df)
    indicator_result = upsert_technical_indicators(stock_id, clean_symbol, indicator_df)

    feature_ready_df = indicator_df.replace([np.inf, -np.inf], np.nan)
    feature_ready_df = feature_ready_df.dropna(subset=FEATURES).copy()
    if feature_ready_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "Not enough rows to calculate complete technical indicators",
            "prices_saved": price_result["rows_saved"],
            "indicators_saved": indicator_result["rows_saved"],
        }

    latest_row = feature_ready_df.iloc[-1]
    latest_date = str(latest_row["date"])
    latest_close = float(latest_row["close"])

    if _prediction_exists(stock_id, latest_date):
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "skipped",
            "latest_date": latest_date,
            "reason": "Prediction already exists for this stock and latest trading day",
            "prices_saved": price_result["rows_saved"],
            "indicators_saved": indicator_result["rows_saved"],
        }

    tuning_result = tune_lightgbm_params(
        indicator_df,
        target_return_threshold=TARGET_RETURN_THRESHOLD,
    )
    best_params = tuning_result.get("best_params", {})
    validation_metrics = walk_forward_validation(
        indicator_df,
        model_params=best_params,
        tune_threshold=True,
        use_feature_selection=True,
        max_features=DEFAULT_MAX_FEATURES,
        target_return_threshold=TARGET_RETURN_THRESHOLD,
    )
    decision_threshold = float(validation_metrics.get("decision_threshold") or 0.5)
    model, clean_df, model_used = train_final_model(
        indicator_df,
        model_params=best_params,
        target_return_threshold=TARGET_RETURN_THRESHOLD,
        use_feature_selection=True,
        max_features=DEFAULT_MAX_FEATURES,
    )

    latest_features = pd.DataFrame([latest_row[FEATURES].to_dict()])
    probability_up = _probability_for_class(model, latest_features, target_class=1)
    prediction = int(probability_up >= decision_threshold)
    predicted_probability = probability_up if prediction == 1 else 1 - probability_up
    feature_importance = get_feature_importance(model, limit=15)
    selected_features = getattr(model, "selected_features_", FEATURES)

    prediction_row = {
        "stock_id": stock_id,
        "symbol": clean_symbol,
        "latest_date": latest_date,
        "latest_close": latest_close,
        "predicted_direction": "up" if prediction == 1 else "down",
        "predicted_probability": predicted_probability,
        "confidence": predicted_probability,
        "model_used": model_used,
        "accuracy": validation_metrics.get("accuracy"),
        "precision": validation_metrics.get("precision"),
        "recall": validation_metrics.get("recall"),
        "f1_score": validation_metrics.get("f1_score"),
        "roc_auc": validation_metrics.get("roc_auc"),
        "baseline_accuracy": validation_metrics.get("baseline_accuracy"),
        "majority_baseline_accuracy": validation_metrics.get("majority_baseline_accuracy"),
        "target_return_threshold": TARGET_RETURN_THRESHOLD,
        "decision_threshold": decision_threshold,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_features": feature_importance,
        "tuned_params": best_params,
    }

    response = supabase.table("direction_predictions").insert(prediction_row).execute()
    saved_prediction = (response.data or [prediction_row])[0]

    return {
        "stock_id": stock_id,
        "symbol": clean_symbol,
        "status": "ok",
        "latest_date": latest_date,
        "latest_close": latest_close,
        "predicted_direction": prediction_row["predicted_direction"],
        "predicted_probability": predicted_probability,
        "confidence": predicted_probability,
        "model_used": model_used,
        "accuracy": validation_metrics.get("accuracy"),
        "precision": validation_metrics.get("precision"),
        "recall": validation_metrics.get("recall"),
        "f1_score": validation_metrics.get("f1_score"),
        "roc_auc": validation_metrics.get("roc_auc"),
        "baseline_accuracy": validation_metrics.get("baseline_accuracy"),
        "majority_baseline_accuracy": validation_metrics.get("majority_baseline_accuracy"),
        "target_return_threshold": TARGET_RETURN_THRESHOLD,
        "decision_threshold": decision_threshold,
        "selected_feature_count": len(selected_features),
        "top_features": feature_importance,
        "tuned_params": best_params,
        "training_rows": int(len(clean_df)),
        "validation_windows": len(validation_metrics.get("windows", [])),
        "prices_saved": price_result["rows_saved"],
        "indicators_saved": indicator_result["rows_saved"],
        "prediction_id": saved_prediction.get("id"),
    }


def run_daily_technical_pipeline() -> dict[str, Any]:
    """Run the technical pipeline for every stock in Supabase."""
    stocks = get_stocks_from_supabase()
    results = []

    for stock in stocks:
        stock_id = stock["id"]
        symbol = stock["symbol"]
        try:
            results.append(run_technical_pipeline_for_stock(stock_id, symbol))
        except Exception as exc:
            logger.exception("Technical pipeline failed for %s", symbol)
            results.append(
                {
                    "stock_id": stock_id,
                    "symbol": symbol,
                    "status": "error",
                    "reason": str(exc),
                }
            )

    return {
        "message": "Technical pipeline complete",
        "symbols_processed": len(stocks),
        "results": results,
    }


def _prediction_exists(stock_id: int, latest_date: str) -> bool:
    response = (
        supabase.table("direction_predictions")
        .select("id")
        .eq("stock_id", stock_id)
        .eq("latest_date", latest_date)
        .limit(1)
        .execute()
    )
    return bool(response.data)


def _probability_for_class(model: Any, X: pd.DataFrame, target_class: int) -> float:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if target_class not in classes:
        return 1.0 if classes and classes[0] == target_class else 0.0

    class_index = classes.index(target_class)
    return float(probabilities[0][class_index])
