"""Application service for versioned financial model operations."""

from backend.services.financial.financial_model import (
    activate_local_model,
    predict_latest,
    train_model,
)
from backend.services.financial.financial_repository import (
    activate_model_version,
    get_active_model_version,
    get_financial_prediction_history,
    get_latest_financial_prediction,
    get_model_version,
    list_model_versions,
    load_financial_statements,
    save_financial_prediction,
    save_financial_statements,
    save_model_version,
)
from backend.services.financial.yfinance_financial_fetcher import (
    fetch_quarterly_financial_statements,
)
from backend.services.stock_list_service import get_active_stocks, get_stock_by_symbol


MODEL_REGISTRY_COLUMNS = {
    "model_version",
    "parent_version",
    "training_mode",
    "model_path",
    "metadata_path",
    "trained_at",
    "training_rows",
    "cumulative_training_rows",
    "holdout_rows",
    "dataset_start",
    "dataset_end",
    "class_distribution",
    "hyperparameters",
    "metrics",
    "feature_columns",
    "labels",
    "evaluation_mode",
}


def _registry_payload(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if key in MODEL_REGISTRY_COLUMNS
    }


def import_financial_statements(symbol: str) -> dict:
    symbol = symbol.upper()
    stocks = get_stock_by_symbol(symbol)
    if not stocks:
        raise ValueError(f"{symbol} is not in the stocks table.")

    stock = stocks[0]
    fetched = fetch_quarterly_financial_statements(symbol, stock["id"])
    saved = save_financial_statements(fetched["rows"])
    return {
        "symbol": symbol,
        "yfinance_symbol": fetched["yfinance_symbol"],
        "periods_received": len(fetched["rows"]),
        "rows_saved": len(saved),
        "skipped_periods": fetched["skipped_periods"],
        "message": (
            "Quarterly financial statements imported."
            if saved
            else "No valid quarterly financial statements were returned."
        ),
    }


def import_all_financial_statements() -> dict:
    stocks = get_active_stocks()
    results = []

    for stock in stocks:
        symbol = stock["symbol"].upper()
        try:
            fetched = fetch_quarterly_financial_statements(symbol, stock["id"])
            saved = save_financial_statements(fetched["rows"])
            results.append({
                "symbol": symbol,
                "status": "imported" if saved else "no_valid_data",
                "periods_received": len(fetched["rows"]),
                "rows_saved": len(saved),
                "skipped_periods": fetched["skipped_periods"],
            })
        except Exception as exc:
            results.append({
                "symbol": symbol,
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


def train_financial_model(
    training_mode: str = "fresh",
    base_version: str = None,
) -> dict:
    statements = load_financial_statements()
    if not statements:
        raise ValueError("No quarterly financial statements were found.")

    if training_mode == "continue" and not base_version:
        active = get_active_model_version()
        if active is None:
            raise ValueError(
                "base_version is required because there is no active model."
            )
        base_version = active["model_version"]

    metadata = train_model(statements, training_mode, base_version)
    save_model_version(_registry_payload(metadata))
    activated = activate_model_version(metadata["model_version"])
    if activated is None:
        raise RuntimeError("Model was saved but could not be activated in the registry.")
    activate_local_model(metadata["model_version"])
    return metadata


def _resolve_model_version(model_version: str = None) -> str:
    if model_version:
        registry_entry = get_model_version(model_version)
        if registry_entry is None:
            raise ValueError(f"Unknown financial model version: {model_version}")
        return model_version

    active = get_active_model_version()
    if active is None:
        raise FileNotFoundError("No active financial model. Train or activate one first.")
    return active["model_version"]


def generate_financial_prediction(
    symbol: str,
    model_version: str = None,
) -> dict:
    symbol = symbol.upper()
    statements = load_financial_statements(symbol)
    if not statements:
        raise ValueError(f"No quarterly financial statements found for {symbol}.")

    selected_version = _resolve_model_version(model_version)
    prediction = predict_latest(statements, selected_version)
    stored = save_financial_prediction(prediction)
    return {
        **stored,
        "probabilities": prediction["probabilities"],
        "prediction_horizon": "next_quarter",
    }


def generate_all_financial_predictions(model_version: str = None) -> dict:
    statements = load_financial_statements()
    if not statements:
        raise ValueError("No quarterly financial statements were found.")

    selected_version = _resolve_model_version(model_version)
    statements_by_symbol = {}
    for row in statements:
        statements_by_symbol.setdefault(row["ticker"].upper(), []).append(row)

    results = []
    errors = []
    for symbol, symbol_statements in statements_by_symbol.items():
        try:
            prediction = predict_latest(symbol_statements, selected_version)
            stored = save_financial_prediction(prediction)
            results.append({
                **stored,
                "probabilities": prediction["probabilities"],
                "prediction_horizon": "next_quarter",
            })
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    return {
        "model_version": selected_version,
        "stocks_processed": len(results),
        "stocks_failed": len(errors),
        "predictions": results,
        "errors": errors,
    }


def set_active_financial_model(model_version: str) -> dict:
    registry_entry = get_model_version(model_version)
    if registry_entry is None:
        raise ValueError(f"Unknown financial model version: {model_version}")
    activate_local_model(model_version)
    activated = activate_model_version(model_version)
    if activated is None:
        raise RuntimeError(f"Could not activate financial model {model_version}.")
    return activated


def read_financial_model_versions() -> list:
    return list_model_versions()


def read_financial_model_version(model_version: str) -> dict:
    return get_model_version(model_version)


def read_latest_financial_prediction(symbol: str) -> dict:
    active = get_active_model_version()
    if active is None:
        return get_latest_financial_prediction(symbol)
    return get_latest_financial_prediction(
        symbol,
        active["model_version"],
    )


def read_financial_prediction_history(symbol: str) -> list:
    return get_financial_prediction_history(symbol)
