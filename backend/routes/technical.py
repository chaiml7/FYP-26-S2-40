from fastapi import APIRouter, HTTPException, Query

from database.supabase_client import supabase
from services.technical.prediction_pipeline import (
    run_daily_technical_pipeline,
    run_technical_pipeline_for_stock,
)
from services.technical.price_service import get_stock_by_symbol

router = APIRouter()


@router.post("/technical/run-pipeline")
def trigger_daily_technical_pipeline():
    try:
        return run_daily_technical_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/technical/run-pipeline/{symbol}")
def trigger_single_symbol_technical_pipeline(symbol: str):
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table",
        )

    try:
        return run_technical_pipeline_for_stock(stock["id"], stock["symbol"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stocks/{symbol}/direction-prediction")
def get_latest_direction_prediction(symbol: str):
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table",
        )

    response = (
        supabase.table("direction_predictions")
        .select(
            "stock_id, symbol, latest_date, latest_close, predicted_direction, "
            "predicted_probability, confidence, model_used, accuracy, precision, "
            "recall, f1_score, roc_auc, baseline_accuracy, "
            "majority_baseline_accuracy, target_return_threshold, "
            "decision_threshold, selected_feature_count, selected_features, "
            "top_features, tuned_params, created_at"
        )
        .eq("stock_id", stock["id"])
        .order("latest_date", desc=True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No direction prediction found for {symbol.upper()}",
        )

    return rows[0]


@router.get("/stocks/{symbol}/technical-indicators")
def get_latest_technical_indicators(
    symbol: str,
    limit: int = Query(default=30, ge=1, le=500),
):
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table",
        )

    response = (
        supabase.table("technical_indicators")
        .select("*")
        .eq("stock_id", stock["id"])
        .order("date", desc=True)
        .limit(limit)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No technical indicators found for {symbol.upper()}",
        )

    return rows
