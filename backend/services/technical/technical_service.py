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


def import_all_technical_prices(period: str = "10y") -> dict:
    stocks = get_stocks_from_supabase()
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
    predictions = predict_latest(indicators, selected_version)
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
    predictions = predict_latest(indicators, selected_version)
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
    activate_local_model(model_version)
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
