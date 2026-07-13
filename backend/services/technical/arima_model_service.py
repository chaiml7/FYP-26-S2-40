from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from services.technical.model_service import TARGET_RETURN_THRESHOLD
from services.technical.xgboost_model_service import (
    date_range_summary,
    filter_training_history,
    normalize_optional_date,
)

ARIMA_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "technical_arima_model.joblib"
)

ARIMA_CLASS_LABELS = {
    0: "down_equal",
    1: "up",
}

DEFAULT_ARIMA_ORDER = (5, 1, 0)
DEFAULT_ARIMA_CANDIDATE_ORDERS = [
    (1, 1, 0),
    (2, 1, 0),
    (3, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (5, 1, 0),
]
DEFAULT_ARIMA_PREDICTION_LENGTH = 1
DEFAULT_ARIMA_VALIDATION_WINDOWS = 30
DEFAULT_ARIMA_MIN_TRAIN_ROWS = 200


def train_arima_artifact(
    indicator_df: pd.DataFrame,
    model_scope: str,
    trained_symbol: str | None = None,
    order: tuple[int, int, int] | None = None,
    prediction_length: int = DEFAULT_ARIMA_PREDICTION_LENGTH,
    threshold: float = TARGET_RETURN_THRESHOLD,
    train_before_date: str | None = None,
    validation_windows: int = DEFAULT_ARIMA_VALIDATION_WINDOWS,
    min_train_rows: int = DEFAULT_ARIMA_MIN_TRAIN_ROWS,
    artifact_path: Path | str = ARIMA_ARTIFACT_PATH,
) -> dict[str, Any]:
    filtered_df = filter_training_history(indicator_df, train_before_date)
    histories = prepare_arima_histories(filtered_df)
    if not histories:
        return {
            "status": "no_data",
            "reason": "No close-price histories were available for ARIMA training",
        }

    models = {}
    symbol_results = []
    all_windows = []
    for symbol, history in histories.items():
        if len(history) < min_train_rows + prediction_length:
            symbol_results.append(
                {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "Not enough rows for ARIMA training",
                    "rows": int(len(history)),
                }
            )
            continue

        selected_order = order or select_arima_order_by_aic(
            history["close"].to_numpy(dtype=float),
            DEFAULT_ARIMA_CANDIDATE_ORDERS,
        )
        windows = walk_forward_arima_validation(
            history=history,
            order=selected_order,
            prediction_length=prediction_length,
            threshold=threshold,
            validation_windows=validation_windows,
            min_train_rows=min_train_rows,
        )
        fitted_model = fit_arima_model(
            history["close"].to_numpy(dtype=float),
            selected_order,
        )
        models[symbol] = {
            "order": selected_order,
            "model": fitted_model,
            "last_date": str(history.iloc[-1]["date"]),
            "last_close": float(history.iloc[-1]["close"]),
            "rows": int(len(history)),
        }
        all_windows.extend(windows)
        symbol_results.append(
            {
                "symbol": symbol,
                "status": "ok",
                "rows": int(len(history)),
                "order": selected_order,
                "window_count": len(windows),
                "metrics": aggregate_arima_metrics(windows),
            }
        )

    if not models:
        return {
            "status": "no_data",
            "reason": "No ARIMA models could be trained",
            "symbols": symbol_results,
        }

    training_date_range = date_range_summary(filtered_df)
    metadata = {
        "model_family": "arima",
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols": sorted(models.keys()),
        "prediction_length": int(prediction_length),
        "direction_threshold": float(threshold),
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "validation_windows": int(validation_windows),
        "min_train_rows": int(min_train_rows),
        "class_labels": ARIMA_CLASS_LABELS,
        "metrics": aggregate_arima_metrics(all_windows),
        "symbols_detail": symbol_results,
    }
    saved_path = save_arima_artifact(models=models, metadata=metadata, path=artifact_path)

    return {
        "status": "ok",
        "model_artifact_path": str(saved_path),
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols_trained": sorted(models.keys()),
        "symbols_trained_count": len(models),
        "prediction_length": int(prediction_length),
        "direction_threshold": float(threshold),
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "validation_windows": int(validation_windows),
        "min_train_rows": int(min_train_rows),
        "metrics": metadata["metrics"],
        "symbols_detail": symbol_results,
    }


def prepare_arima_histories(indicator_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if indicator_df is None or indicator_df.empty:
        return {}

    result = indicator_df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    if "symbol" not in result.columns:
        result["symbol"] = "SERIES"
    result["symbol"] = result["symbol"].astype(str).str.upper()
    if "stock_id" not in result.columns:
        result["stock_id"] = None

    result = result.dropna(subset=["date", "close", "symbol"])
    result["date"] = result["date"].astype(str)
    histories = {}
    for symbol, group in result.groupby("symbol", sort=False):
        clean_group = (
            group[["stock_id", "symbol", "date", "close"]]
            .sort_values("date", ascending=True)
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        if not clean_group.empty:
            histories[symbol] = clean_group
    return histories


def walk_forward_arima_validation(
    history: pd.DataFrame,
    order: tuple[int, int, int],
    prediction_length: int,
    threshold: float,
    validation_windows: int,
    min_train_rows: int,
) -> list[dict[str, Any]]:
    close_values = history["close"].to_numpy(dtype=float)
    last_index = len(history) - prediction_length - 1
    if last_index < min_train_rows:
        return []

    start_index = max(min_train_rows - 1, last_index - max(1, validation_windows) + 1)
    windows = []
    for index in range(start_index, last_index + 1):
        train_values = close_values[: index + 1]
        as_of_close = float(close_values[index])
        actual_close = float(close_values[index + prediction_length])
        try:
            model = fit_arima_model(train_values, order)
            forecast_values = np.asarray(model.forecast(steps=prediction_length), dtype=float)
            predicted_close = float(forecast_values[-1])
        except Exception as exc:
            windows.append(
                {
                    "symbol": str(history.iloc[index]["symbol"]),
                    "as_of_date": str(history.iloc[index]["date"]),
                    "actual_date": str(history.iloc[index + prediction_length]["date"]),
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        predicted_return = predicted_close / as_of_close - 1
        actual_return = actual_close / as_of_close - 1
        predicted_class = int(predicted_return > threshold)
        actual_class = int(actual_return > threshold)
        windows.append(
            {
                "symbol": str(history.iloc[index]["symbol"]),
                "as_of_date": str(history.iloc[index]["date"]),
                "actual_date": str(history.iloc[index + prediction_length]["date"]),
                "status": "ok",
                "order": order,
                "as_of_close": as_of_close,
                "predicted_close": predicted_close,
                "actual_close": actual_close,
                "predicted_return": float(predicted_return),
                "actual_return": float(actual_return),
                "predicted_class": predicted_class,
                "actual_class": actual_class,
                "predicted_direction": ARIMA_CLASS_LABELS[predicted_class],
                "actual_direction": ARIMA_CLASS_LABELS[actual_class],
                "correct": bool(predicted_class == actual_class),
            }
        )
    return windows


def select_arima_order_by_aic(
    close_values: np.ndarray,
    candidate_orders: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    best_order = DEFAULT_ARIMA_ORDER
    best_aic = np.inf
    for candidate_order in candidate_orders:
        try:
            model = fit_arima_model(close_values, candidate_order)
            aic = float(getattr(model, "aic", np.inf))
            if aic < best_aic:
                best_aic = aic
                best_order = candidate_order
        except Exception:
            continue
    return best_order


def fit_arima_model(close_values: np.ndarray, order: tuple[int, int, int]) -> Any:
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(
            close_values,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()


def predict_with_arima_artifact(
    artifact: dict[str, Any],
    indicator_df: pd.DataFrame,
    symbol: str | None = None,
    as_of_date: str | None = None,
) -> list[dict[str, Any]]:
    metadata = artifact.get("metadata", {})
    prediction_length = int(metadata.get("prediction_length", DEFAULT_ARIMA_PREDICTION_LENGTH))
    threshold = float(metadata.get("direction_threshold", TARGET_RETURN_THRESHOLD))
    histories = prepare_arima_histories(indicator_df)
    trained_models = artifact.get("models", {})
    requested_symbols = [symbol.upper()] if symbol else sorted(trained_models.keys())

    predictions = []
    for requested_symbol in requested_symbols:
        if requested_symbol not in histories:
            continue
        if requested_symbol not in trained_models:
            continue
        history = histories[requested_symbol]
        model_payload = trained_models[requested_symbol]
        order = tuple(model_payload["order"])
        if as_of_date:
            cutoff = pd.to_datetime(as_of_date, errors="raise").date().isoformat()
            history_for_prediction = history[history["date"] <= cutoff].copy()
            if history_for_prediction.empty or history_for_prediction.iloc[-1]["date"] != cutoff:
                continue
            model = fit_arima_model(
                history_for_prediction["close"].to_numpy(dtype=float),
                order,
            )
        else:
            history_for_prediction = history.copy()
            model = model_payload["model"]

        if len(history_for_prediction) < max(order[0] + order[1] + order[2] + 2, 10):
            continue

        as_of_row = history_for_prediction.iloc[-1]
        as_of_close = float(as_of_row["close"])
        forecast_values = np.asarray(model.forecast(steps=prediction_length), dtype=float)
        predicted_close = float(forecast_values[-1])
        predicted_return = predicted_close / as_of_close - 1
        predicted_class = int(predicted_return > threshold)
        prediction = {
            "stock_id": (
                int(as_of_row["stock_id"])
                if not pd.isna(as_of_row["stock_id"])
                else None
            ),
            "symbol": requested_symbol,
            "as_of_date": str(as_of_row["date"]),
            "as_of_close": as_of_close,
            "predicted_close": predicted_close,
            "predicted_return": float(predicted_return),
            "predicted_class": predicted_class,
            "predicted_direction": ARIMA_CLASS_LABELS[predicted_class],
            "order": order,
            "prediction_length": prediction_length,
            "direction_threshold": threshold,
        }
        actual_payload = actual_future_payload(
            history=history,
            as_of_date=str(as_of_row["date"]),
            as_of_close=as_of_close,
            prediction_length=prediction_length,
            threshold=threshold,
        )
        if actual_payload.get("actual_class") is not None:
            actual_payload["correct"] = bool(
                predicted_class == actual_payload["actual_class"]
            )
        prediction.update(actual_payload)
        predictions.append(prediction)

    return predictions


def actual_future_payload(
    history: pd.DataFrame,
    as_of_date: str,
    as_of_close: float,
    prediction_length: int,
    threshold: float,
) -> dict[str, Any]:
    future_rows = history[history["date"] > as_of_date].sort_values("date", ascending=True)
    if len(future_rows) < prediction_length:
        return {
            "actual_date": None,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
            "correct": None,
        }

    actual_row = future_rows.iloc[prediction_length - 1]
    actual_close = float(actual_row["close"])
    actual_return = actual_close / as_of_close - 1
    actual_class = int(actual_return > threshold)
    return {
        "actual_date": str(actual_row["date"]),
        "actual_close": actual_close,
        "actual_return": float(actual_return),
        "actual_class": actual_class,
        "actual_direction": ARIMA_CLASS_LABELS[actual_class],
    }


def aggregate_arima_metrics(windows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_windows = [window for window in windows if window.get("status") == "ok"]
    if not clean_windows:
        return empty_arima_metrics("no_valid_windows")

    y_true = np.asarray([window["actual_class"] for window in clean_windows], dtype=int)
    y_pred = np.asarray([window["predicted_class"] for window in clean_windows], dtype=int)
    actual_close = np.asarray([window["actual_close"] for window in clean_windows], dtype=float)
    predicted_close = np.asarray([window["predicted_close"] for window in clean_windows], dtype=float)
    return {
        "window_count": len(clean_windows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "mae": float(mean_absolute_error(actual_close, predicted_close)),
        "rmse": float(np.sqrt(mean_squared_error(actual_close, predicted_close))),
        "support": {
            "down_equal": int((y_true == 0).sum()),
            "up": int((y_true == 1).sum()),
        },
        "predicted_support": {
            "down_equal": int((y_pred == 0).sum()),
            "up": int((y_pred == 1).sum()),
        },
    }


def empty_arima_metrics(reason: str) -> dict[str, Any]:
    return {
        "window_count": 0,
        "accuracy": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "mcc": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "mae": None,
        "rmse": None,
        "reason": reason,
    }


def save_arima_artifact(
    models: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    path: Path | str = ARIMA_ARTIFACT_PATH,
) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": models,
        "metadata": {
            **metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        },
    }
    joblib.dump(payload, artifact_path)
    return artifact_path


def load_arima_artifact(path: Path | str = ARIMA_ARTIFACT_PATH) -> dict[str, Any]:
    return joblib.load(Path(path))
