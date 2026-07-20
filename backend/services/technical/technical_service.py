"""Application service for technical data, models, and predictions."""

from backend.services.technical.indicator_service import (
    add_technical_indicators,
    get_all_technical_indicators_from_supabase,
    get_technical_indicators_from_supabase,
    upsert_technical_indicators,
)
from backend.services.technical.price_service import (
    add_market_context_features,
    fetch_price_history,
    get_daily_ohlcv_from_supabase,
    get_stock_by_symbol,
    get_stock_prices_from_supabase,
    get_stocks_from_supabase,
    upsert_daily_ohlcv,
    upsert_stock_prices,
)
from backend.services.technical.technical_model import (
    activate_local_model,
    predict_latest,
    train_model,
)
from backend.services.technical.binary_xgboost_model import (
    activate_local_model as activate_binary_xgboost_local_model,
    backtest_model as backtest_binary_xgboost_model,
    is_binary_xgboost_model_version,
    load_model_metadata as load_binary_xgboost_model_metadata,
    predict_latest as predict_binary_xgboost_latest,
    train_model as train_binary_xgboost_model,
    walk_forward_evaluate as walk_forward_evaluate_binary_xgboost_model,
)
from backend.services.sentiment.sentiment_aggregator import get_all_daily_sentiment_scores
from backend.services.technical.technical_repository import (
    activate_model_version,
    get_active_model_version,
    get_latest_prediction,
    get_model_version,
    get_prediction_history,
    list_model_versions,
    save_model_version,
    save_technical_prediction,
)


def import_technical_prices(symbol: str, period: str = "10y") -> dict:
    symbol = symbol.upper()
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise ValueError(f"{symbol} is not in the active stocks table.")

    price_df = fetch_price_history(symbol, period=period, interval="1d")
    if price_df.empty:
        raise ValueError(f"No daily yfinance history was returned for {symbol}.")

    raw_result = upsert_daily_ohlcv(stock["id"], symbol, price_df)
    raw_prices = get_daily_ohlcv_from_supabase(stock["id"], symbol)
    enriched = add_market_context_features(
        raw_prices,
        symbol=symbol,
        period=period,
        interval="1d",
    )
    price_result = upsert_stock_prices(stock["id"], symbol, enriched)
    stored_prices = get_stock_prices_from_supabase(stock["id"], symbol)
    indicators = add_technical_indicators(stored_prices)
    indicator_result = upsert_technical_indicators(
        stock["id"],
        symbol,
        indicators,
    )

    return {
        "symbol": symbol,
        "period": period,
        "yfinance_rows": len(price_df),
        "daily_ohlcv_rows_saved": raw_result["rows_saved"],
        "stock_price_rows_saved": price_result["rows_saved"],
        "technical_indicator_rows_saved": indicator_result["rows_saved"],
    }


def import_all_technical_prices(
    period: str = "10y",
    stock_scope: str = "active",
) -> dict:
    stocks = get_stocks_from_supabase(stock_scope=stock_scope)
    results = []
    for stock in stocks:
        try:
            results.append({
                "status": "imported",
                **import_technical_prices(stock["symbol"], period),
            })
        except Exception as exc:
            results.append({
                "symbol": stock["symbol"],
                "status": "error",
                "error": str(exc),
            })
    return {
        "stock_scope": stock_scope,
        "stocks_processed": len(stocks),
        "stocks_imported": sum(
            result["status"] == "imported"
            for result in results
        ),
        "results": results,
    }


def train_technical_model() -> dict:
    indicators = get_all_technical_indicators_from_supabase()
    if indicators.empty:
        raise ValueError("No technical indicator rows were found.")

    metadata = train_model(indicators)
    save_model_version(metadata)
    activated = activate_model_version(metadata["model_version"])
    if activated is None:
        raise RuntimeError(
            "Technical model was saved but could not be activated."
        )
    activate_local_model(metadata["model_version"])
    return metadata


def train_binary_xgboost_technical_model(
    return_threshold: float = 0.01,
    train_before_date: str = None,
    prediction_horizon_days: int = 5,
    use_sentiment_features: bool = False,
    stock_scope: str = "active",
    activate: bool = True,
) -> dict:
    indicators = get_all_technical_indicators_from_supabase(stock_scope=stock_scope)
    if indicators.empty:
        raise ValueError("No technical indicator rows were found.")

    kwargs = {}
    if return_threshold is not None:
        kwargs["return_threshold"] = return_threshold
    if train_before_date:
        kwargs["train_before_date"] = train_before_date
    kwargs["prediction_horizon_days"] = prediction_horizon_days
    kwargs["use_sentiment_features"] = use_sentiment_features
    if use_sentiment_features:
        kwargs["sentiment_scores"] = get_all_daily_sentiment_scores()

    metadata = train_binary_xgboost_model(indicators, **kwargs)
    metadata["training_stock_scope"] = stock_scope
    metadata["training_stock_count"] = int(indicators["stock_id"].nunique())
    save_model_version(metadata)
    if activate:
        activation_eligibility = metadata.get("activation_eligibility", {})
        if not activation_eligibility.get("passed", False):
            metadata["activated"] = False
            metadata["activation_reason"] = (
                "Model was saved but not activated because its holdout accuracy "
                "was not above 50 percent."
            )
            return metadata
        activated = activate_model_version(metadata["model_version"])
        if activated is None:
            raise RuntimeError(
                "Binary XGBoost technical model was saved but could not be activated."
            )
        activate_binary_xgboost_local_model(metadata["model_version"])
        metadata["activated"] = True
    return metadata


def _resolve_model_version(model_version: str = None) -> str:
    if model_version:
        if get_model_version(model_version) is None:
            raise ValueError(f"Unknown technical model version: {model_version}")
        return model_version

    active = get_active_model_version()
    if active is None:
        raise FileNotFoundError(
            "No active technical model. Train or activate one first."
        )
    return active["model_version"]


def _activate_local_technical_model(model_version: str) -> dict:
    if is_binary_xgboost_model_version(model_version):
        return activate_binary_xgboost_local_model(model_version)
    return activate_local_model(model_version)


def _ensure_binary_xgboost_model_is_eligible(model_version: str) -> None:
    metadata = load_binary_xgboost_model_metadata(model_version)
    test_accuracy = metadata.get("test_metrics", {}).get("accuracy")
    if test_accuracy is None or float(test_accuracy) <= 0.50:
        raise ValueError(
            "This binary XGBoost model has holdout accuracy at or below 50 percent "
            "and cannot be used for predictions or activation. Retrain it first."
        )


def _predict_latest_for_model(indicators, model_version: str) -> list[dict]:
    if is_binary_xgboost_model_version(model_version):
        _ensure_binary_xgboost_model_is_eligible(model_version)
        metadata = load_binary_xgboost_model_metadata(model_version)
        sentiment_scores = (
            get_all_daily_sentiment_scores()
            if metadata.get("use_sentiment_features", False)
            else None
        )
        return predict_binary_xgboost_latest(
            indicators,
            model_version,
            sentiment_scores=sentiment_scores,
        )
    return predict_latest(indicators, model_version)


def generate_technical_prediction(
    symbol: str,
    model_version: str = None,
) -> dict:
    symbol = symbol.upper()
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise ValueError(f"{symbol} is not in the active stocks table.")

    indicators = get_technical_indicators_from_supabase(
        stock["id"],
        symbol,
    )
    if indicators.empty:
        raise ValueError(f"No technical indicators were found for {symbol}.")

    selected_version = _resolve_model_version(model_version)
    predictions = _predict_latest_for_model(indicators, selected_version)
    if not predictions:
        raise ValueError(f"No complete prediction row was available for {symbol}.")
    return save_technical_prediction(predictions[0])


def generate_all_technical_predictions(
    model_version: str = None,
) -> dict:
    indicators = get_all_technical_indicators_from_supabase()
    if indicators.empty:
        raise ValueError("No technical indicator rows were found.")

    selected_version = _resolve_model_version(model_version)
    predictions = _predict_latest_for_model(indicators, selected_version)
    saved = []
    errors = []
    for prediction in predictions:
        try:
            saved.append(save_technical_prediction(prediction))
        except RuntimeError as exc:
            errors.append({
                "symbol": prediction["symbol"],
                "error": str(exc),
            })
    return {
        "model_version": selected_version,
        "stocks_processed": len(saved),
        "stocks_failed": len(errors),
        "predictions": saved,
        "errors": errors,
    }


def set_active_technical_model(model_version: str) -> dict:
    if get_model_version(model_version) is None:
        raise ValueError(f"Unknown technical model version: {model_version}")
    if is_binary_xgboost_model_version(model_version):
        _ensure_binary_xgboost_model_is_eligible(model_version)
    _activate_local_technical_model(model_version)
    activated = activate_model_version(model_version)
    if activated is None:
        raise RuntimeError(
            f"Could not activate technical model {model_version}."
        )
    return activated


def read_model_versions() -> list:
    return list_model_versions()


def read_model_version(model_version: str) -> dict:
    return get_model_version(model_version)


def read_latest_prediction(symbol: str) -> dict:
    active = get_active_model_version()
    return get_latest_prediction(
        symbol,
        active["model_version"] if active else None,
    )


def read_prediction_history(symbol: str) -> list:
    return get_prediction_history(symbol)


def backtest_binary_xgboost_technical_model(
    symbol: str,
    model_version: str = None,
    start_date: str = None,
    end_date: str = None,
    confidence_threshold: float = 0.60,
    transaction_cost_bps: float = 10.0,
) -> dict:
    symbol = symbol.upper()
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise ValueError(f"{symbol} is not in the active stocks table.")

    selected_version = _resolve_model_version(model_version)
    if not is_binary_xgboost_model_version(selected_version):
        raise ValueError(
            f"{selected_version} is not a binary XGBoost technical model."
        )

    indicators = get_technical_indicators_from_supabase(
        stock["id"],
        symbol,
    )
    if indicators.empty:
        raise ValueError(f"No technical indicators were found for {symbol}.")

    metadata = load_binary_xgboost_model_metadata(selected_version)
    sentiment_scores = (
        get_all_daily_sentiment_scores()
        if metadata.get("use_sentiment_features", False)
        else None
    )
    return backtest_binary_xgboost_model(
        indicators,
        selected_version,
        start_date=start_date,
        end_date=end_date,
        confidence_threshold=confidence_threshold,
        transaction_cost_bps=transaction_cost_bps,
        sentiment_scores=sentiment_scores,
    )


def evaluate_binary_xgboost_walk_forward(
    return_threshold: float = 0.01,
    prediction_horizon_days: int = 5,
    use_sentiment_features: bool = False,
    stock_scope: str = "active",
    test_window_dates: int = 63,
    max_folds: int = 4,
) -> dict:
    indicators = get_all_technical_indicators_from_supabase(stock_scope=stock_scope)
    if indicators.empty:
        raise ValueError("No technical indicator rows were found.")
    sentiment_scores = get_all_daily_sentiment_scores() if use_sentiment_features else None
    result = walk_forward_evaluate_binary_xgboost_model(
        indicators,
        return_threshold=return_threshold,
        prediction_horizon_days=prediction_horizon_days,
        use_sentiment_features=use_sentiment_features,
        sentiment_scores=sentiment_scores,
        test_window_dates=test_window_dates,
        max_folds=max_folds,
    )
    result["training_stock_scope"] = stock_scope
    result["training_stock_count"] = int(indicators["stock_id"].nunique())
    return result
