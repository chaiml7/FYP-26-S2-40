"""Application service for versioned financial model operations."""

from backend.services.financial.financial_model import (
    BINARY_MODEL_FAMILY,
    activate_local_model,
    predict_latest_binary,
    predict_latest,
    train_binary_model,
    train_model,
    tune_binary_model,
    tune_model,
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
from backend.services.financial.sec_financial_fetcher import (
    fetch_sec_quarterly_financial_statements,
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


def import_sec_financial_statements(symbol: str) -> dict:
    symbol = symbol.upper()
    stocks = get_stock_by_symbol(symbol)
    if not stocks:
        raise ValueError(f"{symbol} is not in the stocks table.")

    stock = stocks[0]
    fetched = fetch_sec_quarterly_financial_statements(symbol, stock["id"])
    saved = save_financial_statements(fetched["rows"])
    return {
        "symbol": symbol,
        "cik": fetched["cik"],
        "periods_received": len(fetched["rows"]),
        "rows_saved": len(saved),
        "skipped_periods": fetched["skipped_periods"],
        "message": (
            "SEC quarterly financial statements imported."
            if saved
            else fetched.get("message", "No valid SEC quarterly statements were returned.")
        ),
    }


def import_all_sec_financial_statements() -> dict:
    stocks = get_active_stocks()
    results = []

    for stock in stocks:
        symbol = stock["symbol"].upper()
        try:
            fetched = fetch_sec_quarterly_financial_statements(symbol, stock["id"])
            saved = save_financial_statements(fetched["rows"])
            results.append({
                "symbol": symbol,
                "status": "imported" if saved else "no_valid_data",
                "cik": fetched["cik"],
                "periods_received": len(fetched["rows"]),
                "rows_saved": len(saved),
                "skipped_periods": fetched["skipped_periods"],
                "message": fetched.get("message"),
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


def tune_financial_model(
    save_best: bool = False,
    activate_best: bool = False,
    top_n: int = 5,
) -> dict:
    statements = load_financial_statements()
    if not statements:
        raise ValueError("No quarterly financial statements were found.")

    result = tune_model(
        statements,
        save_best=save_best or activate_best,
        top_n=top_n,
    )
    saved_model = result.get("saved_model")
    if not saved_model:
        return result

    save_model_version(_registry_payload(saved_model))
    if activate_best:
        activated = activate_model_version(saved_model["model_version"])
        if activated is None:
            raise RuntimeError(
                "Tuned model was saved but could not be activated in the registry."
            )
        activate_local_model(saved_model["model_version"])
        result["activated_model"] = activated
    else:
        result["activated_model"] = None

    return result


def tune_binary_financial_model(top_n: int = 5) -> dict:
    statements = load_financial_statements()
    if not statements:
        raise ValueError("No quarterly financial statements were found.")
    return tune_binary_model(statements, top_n=top_n)


def train_binary_financial_model(top_n: int = 5) -> dict:
    statements = load_financial_statements()
    if not statements:
        raise ValueError("No quarterly financial statements were found.")

    result = train_binary_model(statements, top_n=top_n)
    save_model_version(_registry_payload(result["saved_model"]))
    result["activated_model"] = None
    return result


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


def _resolve_binary_model_version(model_version: str = None) -> str:
    if model_version:
        registry_entry = get_model_version(model_version)
        if registry_entry is None:
            raise ValueError(f"Unknown financial model version: {model_version}")
        if not model_version.startswith(f"{BINARY_MODEL_FAMILY}_"):
            raise ValueError(f"{model_version} is not a binary financial model.")
        return model_version

    binary_versions = [
        version
        for version in list_model_versions()
        if str(version.get("model_version", "")).startswith(f"{BINARY_MODEL_FAMILY}_")
    ]
    if not binary_versions:
        raise FileNotFoundError(
            "No saved binary financial model. Run /api/financial/model/train-binary first."
        )
    return binary_versions[0]["model_version"]


def _is_binary_model_version(model_version: str) -> bool:
    return str(model_version).startswith(f"{BINARY_MODEL_FAMILY}_")


def _predict_latest_for_model(statements: list, model_version: str) -> tuple[dict, str]:
    if _is_binary_model_version(model_version):
        return predict_latest_binary(statements, model_version), "binary_financial"
    return predict_latest(statements, model_version), "three_class_financial"


def _prediction_response(stored: dict, prediction: dict, model_type: str) -> dict:
    response = {
        **stored,
        "probabilities": prediction["probabilities"],
        "prediction_horizon": "next_quarter",
        "model_type": model_type,
    }
    if model_type == "binary_financial":
        response.update({
            "binary_prediction": prediction["binary_prediction"],
            "bullish_probability": prediction["bullish_probability"],
            "bearish_probability": prediction["bearish_probability"],
            "display_thresholds": prediction["display_thresholds"],
        })
    return response


def _stored_prediction_response(prediction: dict, model_type: str) -> dict:
    response = {
        **prediction,
        "prediction_horizon": "next_quarter",
        "model_type": model_type,
    }
    if model_type != "binary_financial":
        return response

    probabilities = prediction.get("probabilities") or {}
    bullish_probability = probabilities.get("bullish")
    bearish_probability = probabilities.get("bearish")
    binary_prediction = None
    if bullish_probability is not None and bearish_probability is not None:
        binary_prediction = (
            "bullish"
            if float(bullish_probability) >= float(bearish_probability)
            else "bearish"
        )
    return {
        **response,
        "binary_prediction": binary_prediction,
        "bullish_probability": bullish_probability,
        "bearish_probability": bearish_probability,
    }


def generate_financial_prediction(
    symbol: str,
    model_version: str = None,
) -> dict:
    symbol = symbol.upper()
    statements = load_financial_statements(symbol)
    if not statements:
        raise ValueError(f"No quarterly financial statements found for {symbol}.")

    selected_version = _resolve_model_version(model_version)
    prediction, model_type = _predict_latest_for_model(statements, selected_version)
    stored = save_financial_prediction(prediction)
    return _prediction_response(stored, prediction, model_type)


def generate_binary_financial_prediction(
    symbol: str,
    model_version: str = None,
) -> dict:
    symbol = symbol.upper()
    statements = load_financial_statements(symbol)
    if not statements:
        raise ValueError(f"No quarterly financial statements found for {symbol}.")

    selected_version = _resolve_binary_model_version(model_version)
    prediction = predict_latest_binary(statements, selected_version)
    stored = save_financial_prediction(prediction)
    return _prediction_response(stored, prediction, "binary_financial")


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
            prediction, model_type = _predict_latest_for_model(
                symbol_statements,
                selected_version,
            )
            stored = save_financial_prediction(prediction)
            results.append(_prediction_response(stored, prediction, model_type))
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    return {
        "model_version": selected_version,
        "model_type": (
            "binary_financial"
            if _is_binary_model_version(selected_version)
            else "three_class_financial"
        ),
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
    prediction = get_latest_financial_prediction(
        symbol,
        active["model_version"],
    )
    if prediction is None:
        return None
    model_type = (
        "binary_financial"
        if _is_binary_model_version(active["model_version"])
        else "three_class_financial"
    )
    return _stored_prediction_response(prediction, model_type)


def read_latest_binary_financial_prediction(symbol: str) -> dict:
    history = get_financial_prediction_history(symbol)
    for prediction in history:
        if str(prediction.get("model_version", "")).startswith(f"{BINARY_MODEL_FAMILY}_"):
            return _stored_prediction_response(prediction, "binary_financial")
    return None


def read_financial_prediction_history(symbol: str) -> list:
    return get_financial_prediction_history(symbol)
