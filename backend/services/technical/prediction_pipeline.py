import logging
from typing import Any

import numpy as np
import pandas as pd

from database.supabase_client import supabase
from services.technical.indicator_service import (
    add_technical_indicators,
    get_all_technical_indicators_from_supabase,
    get_technical_indicators_from_supabase,
    upsert_technical_indicators,
)
from services.technical.model_service import (
    DEFAULT_MAX_FEATURES,
    FEATURES,
    MODEL_ARTIFACT_PATH,
    TARGET_RETURN_THRESHOLD,
    get_feature_importance,
    load_model_artifact,
    prepare_training_data,
    save_model_artifact,
    train_final_model,
    tune_lightgbm_params,
    walk_forward_validation,
)
from services.technical.price_service import (
    add_market_context_features,
    fetch_price_history,
    get_daily_ohlcv_from_supabase,
    get_stock_by_symbol,
    get_stock_prices_from_supabase,
    get_stocks_from_supabase,
    upsert_daily_ohlcv,
    upsert_stock_prices,
)

logger = logging.getLogger(__name__)


def run_technical_pipeline_for_stock(stock_id: int, symbol: str) -> dict[str, Any]:
    """Sync one ticker, then train a single-ticker model for it."""
    return train_global_technical_model(
        symbol=symbol,
        sync_first=True,
        sync_stocks=[{"id": stock_id, "symbol": symbol}],
    )


def run_daily_technical_pipeline() -> dict[str, Any]:
    """Sync every target ticker, then train one shared model."""
    return train_global_technical_model(sync_first=True)


def train_global_technical_model(
    symbol: str | None = None,
    sync_first: bool = False,
    sync_stocks: list[dict[str, Any]] | None = None,
    period: str = "10y",
    interval: str = "1d",
) -> dict[str, Any]:
    """Train/evaluate one LightGBM model across all stored ticker histories."""
    clean_symbol = symbol.upper() if symbol else None
    sync_results = []
    if sync_first:
        if sync_stocks is not None:
            stocks_to_sync = sync_stocks
        elif clean_symbol:
            stock = get_stock_by_symbol(clean_symbol)
            if stock is None:
                return {
                    "status": "no_data",
                    "reason": f"{clean_symbol} was not found in the stocks table",
                }
            stocks_to_sync = [stock]
        else:
            stocks_to_sync = get_stocks_from_supabase()

        for stock in stocks_to_sync:
            try:
                sync_results.append(
                    sync_technical_tables_for_stock(
                        stock_id=stock["id"],
                        symbol=stock["symbol"],
                        period=period,
                        interval=interval,
                    )
                )
            except Exception as exc:
                logger.exception("Technical sync failed for %s", stock.get("symbol"))
                sync_results.append(
                    {
                        "stock_id": stock.get("id"),
                        "symbol": stock.get("symbol"),
                        "status": "error",
                        "reason": str(exc),
                    }
                )

    if clean_symbol:
        stock = get_stock_by_symbol(clean_symbol)
        if stock is None:
            return {
                "status": "no_data",
                "reason": f"{clean_symbol} was not found in the stocks table",
                "sync_results": sync_results,
            }
        indicator_df = get_technical_indicators_from_supabase(stock["id"], stock["symbol"])
    else:
        indicator_df = get_all_technical_indicators_from_supabase()

    if indicator_df.empty:
        return {
            "status": "no_data",
            "reason": (
                f"No technical_indicators rows found for {clean_symbol}"
                if clean_symbol
                else "No technical_indicators rows found for model training"
            ),
            "sync_results": sync_results,
        }

    try:
        _, y, clean_df = prepare_training_data(
            indicator_df,
            target_return_threshold=TARGET_RETURN_THRESHOLD,
        )
    except ValueError as exc:
        return {
            "status": "no_data",
            "reason": str(exc),
            "indicator_rows": len(indicator_df),
            "sync_results": sync_results,
        }

    if clean_df.empty or y.empty:
        return {
            "status": "no_data",
            "reason": "Not enough complete indicator rows to create ML targets",
            "indicator_rows": len(indicator_df),
            "sync_results": sync_results,
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
    model, final_clean_df, model_used = train_final_model(
        indicator_df,
        model_params=best_params,
        target_return_threshold=TARGET_RETURN_THRESHOLD,
        use_feature_selection=True,
        max_features=DEFAULT_MAX_FEATURES,
    )

    selected_features = getattr(model, "selected_features_", FEATURES)
    feature_importance = get_feature_importance(model, limit=15)
    symbols = sorted(str(value) for value in indicator_df["symbol"].dropna().unique())
    model_metadata = {
        "model_used": model_used,
        "model_scope": "single_ticker" if clean_symbol else "global_all_tickers",
        "trained_symbol": clean_symbol,
        "symbols": symbols,
        "feature_count": len(FEATURES),
        "training_rows": int(len(final_clean_df)),
        "target_return_threshold": TARGET_RETURN_THRESHOLD,
        "decision_threshold": decision_threshold,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_features": feature_importance,
        "tuned_params": best_params,
        "metrics": _metrics_summary(validation_metrics),
    }
    artifact_path = save_model_artifact(model, model_metadata, MODEL_ARTIFACT_PATH)

    return {
        "status": "ok",
        "model_scope": "single_ticker" if clean_symbol else "global_all_tickers",
        "trained_symbol": clean_symbol,
        "symbols_trained": symbols,
        "symbols_trained_count": len(symbols),
        "indicator_rows": len(indicator_df),
        "clean_training_rows": int(len(final_clean_df)),
        "target_return_threshold": TARGET_RETURN_THRESHOLD,
        "decision_threshold": decision_threshold,
        "model_used": model_used,
        "model_artifact_path": str(artifact_path),
        "accuracy": validation_metrics.get("accuracy"),
        "precision": validation_metrics.get("precision"),
        "recall": validation_metrics.get("recall"),
        "f1_score": validation_metrics.get("f1_score"),
        "roc_auc": validation_metrics.get("roc_auc"),
        "baseline_accuracy": validation_metrics.get("baseline_accuracy"),
        "majority_baseline_accuracy": validation_metrics.get("majority_baseline_accuracy"),
        "selected_feature_count": len(selected_features),
        "top_features": feature_importance,
        "tuned_params": best_params,
        "sync_results": sync_results,
    }


def run_global_technical_pipeline(
    sync_first: bool = False,
    sync_stocks: list[dict[str, Any]] | None = None,
    period: str = "10y",
    interval: str = "1d",
) -> dict[str, Any]:
    """Backward-compatible alias for training the shared technical model."""
    return train_global_technical_model(
        sync_first=sync_first,
        sync_stocks=sync_stocks,
        period=period,
        interval=interval,
    )


def predict_trends_with_saved_model(symbol: str | None = None) -> dict[str, Any]:
    """Load the locally saved model and save latest trend predictions."""
    try:
        artifact = load_model_artifact(MODEL_ARTIFACT_PATH)
    except FileNotFoundError:
        return {
            "status": "no_model",
            "reason": f"Saved model was not found at {MODEL_ARTIFACT_PATH}",
        }

    model = artifact["model"]
    metadata = artifact.get("metadata", {})
    trained_symbol = metadata.get("trained_symbol")
    if trained_symbol and symbol and symbol.upper() != str(trained_symbol).upper():
        return {
            "status": "wrong_model_scope",
            "reason": (
                f"Saved model was trained for {trained_symbol}; "
                f"train an all-ticker model or a {symbol.upper()} model first."
            ),
            "model_artifact_path": str(MODEL_ARTIFACT_PATH),
        }

    effective_symbol = symbol or trained_symbol
    if effective_symbol:
        stock = get_stock_by_symbol(str(effective_symbol))
        if stock is None:
            return {
                "status": "no_data",
                "reason": f"{str(effective_symbol).upper()} was not found in the stocks table",
                "model_artifact_path": str(MODEL_ARTIFACT_PATH),
            }
        indicator_df = get_technical_indicators_from_supabase(stock["id"], stock["symbol"])
    else:
        indicator_df = get_all_technical_indicators_from_supabase()

    if indicator_df.empty:
        return {
            "status": "no_data",
            "reason": (
                f"No technical_indicators rows found for {str(effective_symbol).upper()}"
                if effective_symbol
                else "No technical_indicators rows found for prediction"
            ),
            "model_artifact_path": str(MODEL_ARTIFACT_PATH),
        }

    metrics = metadata.get("metrics", {})
    selected_features = metadata.get(
        "selected_features",
        getattr(model, "selected_features_", FEATURES),
    )
    feature_importance = metadata.get("top_features", get_feature_importance(model, limit=15))
    tuned_params = metadata.get("tuned_params", {})
    decision_threshold = float(
        metadata.get("decision_threshold")
        or metrics.get("decision_threshold")
        or 0.5
    )
    model_used = metadata.get("model_used", model.__class__.__name__)

    predictions = _build_latest_predictions(
        indicator_df=indicator_df,
        model=model,
        model_used=model_used,
        validation_metrics=metrics,
        decision_threshold=decision_threshold,
        selected_features=selected_features,
        feature_importance=feature_importance,
        tuned_params=tuned_params,
        symbol=effective_symbol,
    )
    saved_predictions = _upsert_direction_predictions(predictions)

    return {
        "status": "ok",
        "model_artifact_path": str(MODEL_ARTIFACT_PATH),
        "model_used": model_used,
        "decision_threshold": decision_threshold,
        "target_return_threshold": metadata.get(
            "target_return_threshold",
            TARGET_RETURN_THRESHOLD,
        ),
        "symbols_trained": metadata.get("symbols", []),
        "model_scope": metadata.get("model_scope"),
        "trained_symbol": trained_symbol,
        "requested_symbol": symbol,
        "prediction_rows_saved": len(saved_predictions),
        "predictions": saved_predictions,
    }


def sync_technical_tables_for_stock(
    stock_id: int,
    symbol: str,
    period: str = "10y",
    interval: str = "1d",
) -> dict[str, Any]:
    """Sync yfinance data through daily_ohlcv, stock_prices, and indicators."""
    clean_symbol = symbol.upper()
    price_df = fetch_price_history(clean_symbol, period=period, interval=interval)
    if price_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No yfinance price history returned",
        }

    daily_ohlcv_result = upsert_daily_ohlcv(stock_id, clean_symbol, price_df)
    supabase_price_df = get_daily_ohlcv_from_supabase(stock_id, clean_symbol)
    if supabase_price_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No daily_ohlcv rows were available after yfinance upsert",
            "daily_ohlcv_saved": daily_ohlcv_result["rows_saved"],
        }

    enriched_price_df = add_market_context_features(
        supabase_price_df,
        symbol=clean_symbol,
        period=period,
        interval=interval,
    )
    stock_prices_result = upsert_stock_prices(stock_id, clean_symbol, enriched_price_df)
    training_price_df = get_stock_prices_from_supabase(stock_id, clean_symbol)
    if training_price_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No stock_prices rows were available for indicators",
            "daily_ohlcv_saved": daily_ohlcv_result["rows_saved"],
            "stock_prices_saved": stock_prices_result["rows_saved"],
        }

    indicator_df = add_technical_indicators(training_price_df)
    indicator_result = upsert_technical_indicators(stock_id, clean_symbol, indicator_df)
    stored_indicator_df = get_technical_indicators_from_supabase(stock_id, clean_symbol)

    return {
        "stock_id": stock_id,
        "symbol": clean_symbol,
        "status": "ok",
        "period": period,
        "interval": interval,
        "yfinance_rows": len(price_df),
        "daily_ohlcv_saved": daily_ohlcv_result["rows_saved"],
        "daily_ohlcv_read": len(supabase_price_df),
        "stock_prices_saved": stock_prices_result["rows_saved"],
        "stock_prices_read": len(training_price_df),
        "technical_indicator_rows_saved": indicator_result["rows_saved"],
        "technical_indicator_rows_read": len(stored_indicator_df),
    }


def _build_latest_predictions(
    indicator_df: pd.DataFrame,
    model: Any,
    model_used: str,
    validation_metrics: dict[str, Any],
    decision_threshold: float,
    selected_features: list[str],
    feature_importance: list[dict[str, float]],
    tuned_params: dict[str, Any],
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    feature_ready_df = indicator_df.replace([np.inf, -np.inf], np.nan)
    feature_ready_df = feature_ready_df.dropna(subset=FEATURES).copy()
    if feature_ready_df.empty:
        return []

    feature_ready_df["date"] = feature_ready_df["date"].astype(str)
    latest_rows = (
        feature_ready_df.sort_values(["stock_id", "date"], ascending=True)
        .groupby("stock_id", as_index=False, sort=False)
        .tail(1)
        .copy()
    )

    if symbol:
        latest_rows = latest_rows[latest_rows["symbol"].str.upper() == symbol.upper()]

    predictions = []
    for _, row in latest_rows.iterrows():
        latest_features = pd.DataFrame([row[FEATURES].to_dict()])
        probability_up = _probability_for_class(model, latest_features, target_class=1)
        prediction = int(probability_up >= decision_threshold)
        predicted_probability = probability_up if prediction == 1 else 1 - probability_up
        predictions.append(
            {
                "stock_id": int(row["stock_id"]),
                "symbol": str(row["symbol"]).upper(),
                "latest_date": str(row["date"]),
                "latest_close": float(row["close"]),
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
                "majority_baseline_accuracy": validation_metrics.get(
                    "majority_baseline_accuracy"
                ),
                "target_return_threshold": TARGET_RETURN_THRESHOLD,
                "decision_threshold": decision_threshold,
                "selected_feature_count": len(selected_features),
                "selected_features": selected_features,
                "top_features": feature_importance,
                "tuned_params": tuned_params,
            }
        )

    return predictions


def _upsert_direction_predictions(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not prediction_rows:
        return []

    response = (
        supabase.table("direction_predictions")
        .upsert(prediction_rows, on_conflict="stock_id,latest_date")
        .execute()
    )
    return response.data or prediction_rows


def _probability_for_class(model: Any, X: pd.DataFrame, target_class: int) -> float:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if target_class not in classes:
        return 1.0 if classes and classes[0] == target_class else 0.0

    class_index = classes.index(target_class)
    return float(probabilities[0][class_index])


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "baseline_accuracy": metrics.get("baseline_accuracy"),
        "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
        "validation_windows": len(metrics.get("windows", [])),
    }
