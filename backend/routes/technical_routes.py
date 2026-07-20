"""Manual technical data, model, and prediction endpoints."""

from fastapi import APIRouter, HTTPException, Query

from backend.services.technical.indicator_service import (
    get_technical_indicators_from_supabase,
)
from backend.services.technical.price_service import get_stock_by_symbol
from backend.services.technical.technical_service import (
    backtest_binary_xgboost_technical_model,
    evaluate_binary_xgboost_walk_forward,
    generate_all_technical_predictions,
    generate_technical_prediction,
    import_all_technical_prices,
    import_technical_prices,
    read_latest_prediction,
    read_model_version,
    read_model_versions,
    read_prediction_history,
    set_active_technical_model,
    train_binary_xgboost_technical_model,
    train_technical_model,
)


router = APIRouter(prefix="/technical", tags=["technical"])


def _raise_technical_error(exc: Exception):
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prices/import")
def import_all_prices(
    period: str = Query(default="10y"),
    stock_scope: str = Query(
        default="active",
        description="active or all",
    ),
):
    try:
        return import_all_technical_prices(period, stock_scope=stock_scope)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/prices/import/{symbol}")
def import_prices(
    symbol: str,
    period: str = Query(default="10y"),
):
    try:
        return import_technical_prices(symbol, period)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/model/train")
def train_model():
    try:
        return train_technical_model()
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/model/train/binary-xgboost")
def train_binary_xgboost_model(
    return_threshold: float = Query(default=0.01, ge=0.0, le=0.10),
    train_before_date: str = Query(default=None),
    prediction_horizon_days: int = Query(default=5, ge=1, le=20),
    use_sentiment_features: bool = Query(default=False),
    stock_scope: str = Query(
        default="active",
        description="active or all",
    ),
    activate: bool = Query(default=True),
):
    try:
        return train_binary_xgboost_technical_model(
            return_threshold=return_threshold,
            train_before_date=train_before_date,
            prediction_horizon_days=prediction_horizon_days,
            use_sentiment_features=use_sentiment_features,
            stock_scope=stock_scope,
            activate=activate,
        )
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/model/evaluate/binary-xgboost/walk-forward")
def evaluate_binary_xgboost_model_walk_forward(
    return_threshold: float = Query(default=0.01, ge=0.0, le=0.10),
    prediction_horizon_days: int = Query(default=5, ge=1, le=20),
    use_sentiment_features: bool = Query(default=False),
    stock_scope: str = Query(
        default="active",
        description="active or all",
    ),
    test_window_dates: int = Query(default=63, ge=20, le=252),
    max_folds: int = Query(default=4, ge=1, le=10),
):
    try:
        return evaluate_binary_xgboost_walk_forward(
            return_threshold=return_threshold,
            prediction_horizon_days=prediction_horizon_days,
            use_sentiment_features=use_sentiment_features,
            stock_scope=stock_scope,
            test_window_dates=test_window_dates,
            max_folds=max_folds,
        )
    except Exception as exc:
        _raise_technical_error(exc)


@router.get("/model/versions")
def view_model_versions():
    return read_model_versions()


@router.get("/model/versions/{model_version}")
def view_model_version(model_version: str):
    version = read_model_version(model_version)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail="Technical model version not found.",
        )
    return version


@router.post("/model/versions/{model_version}/activate")
def activate_model(
    model_version: str,
):
    try:
        return set_active_technical_model(model_version)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/predictions")
def create_all_predictions(
    model_version: str = None,
):
    try:
        return generate_all_technical_predictions(model_version)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/predictions/{symbol}")
def create_prediction(
    symbol: str,
    model_version: str = None,
):
    try:
        return generate_technical_prediction(symbol, model_version)
    except Exception as exc:
        _raise_technical_error(exc)


@router.get("/predictions/{symbol}/latest")
def view_latest_prediction(symbol: str):
    prediction = read_latest_prediction(symbol)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"No technical prediction found for {symbol.upper()}.",
        )
    return prediction


@router.get("/predictions/{symbol}")
def view_prediction_history(symbol: str):
    return read_prediction_history(symbol)


@router.get("/backtests/binary-xgboost/{symbol}")
def view_binary_xgboost_backtest(
    symbol: str,
    model_version: str = Query(default=None),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
    confidence_threshold: float = Query(default=0.60, ge=0.0, le=1.0),
    transaction_cost_bps: float = Query(default=10.0, ge=0.0, le=500.0),
):
    try:
        return backtest_binary_xgboost_technical_model(
            symbol,
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=confidence_threshold,
            transaction_cost_bps=transaction_cost_bps,
        )
    except Exception as exc:
        _raise_technical_error(exc)


@router.get("/indicators/{symbol}")
def view_indicators(
    symbol: str,
    limit: int = Query(default=30, ge=1, le=500),
):
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the active stocks table.",
        )
    indicators = get_technical_indicators_from_supabase(
        stock["id"],
        stock["symbol"],
    )
    if indicators.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No technical indicators found for {symbol.upper()}.",
        )
    return indicators.tail(limit).iloc[::-1].to_dict(orient="records")
