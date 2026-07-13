from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from services.technical.model_service import TARGET_RETURN_THRESHOLD

DEFAULT_FOUNDATION_MODEL = "chronos"
DEFAULT_CHRONOS_MODEL_ID = "amazon/chronos-2"
DEFAULT_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_PREDICTION_LENGTH = 1
DEFAULT_QUANTILE_LEVELS = [0.1, 0.5, 0.9]
DEFAULT_FORECAST_FREQ = "B"
DEFAULT_BACKTEST_WINDOWS = 30
DEFAULT_BACKTEST_STRIDE = 1
DEFAULT_MIN_CONTEXT_ROWS = 128


def run_foundation_forecast(
    indicator_df: pd.DataFrame,
    model_type: str = DEFAULT_FOUNDATION_MODEL,
    model_id: str | None = None,
    prediction_length: int = DEFAULT_PREDICTION_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    threshold: float = TARGET_RETURN_THRESHOLD,
    as_of_date: str | None = None,
    device: str = "cpu",
    freq: str = DEFAULT_FORECAST_FREQ,
) -> dict[str, Any]:
    """Forecast close prices with Chronos or TimesFM and convert to direction."""
    price_df = prepare_close_history(indicator_df, as_of_date=as_of_date)
    if price_df.empty:
        return {
            "status": "no_data",
            "reason": "No close-price history available for foundation forecast",
        }

    model_type = normalize_model_type(model_type)
    model_id = model_id or default_model_id(model_type)
    prediction_length = max(1, int(prediction_length))
    context_length = max(8, int(context_length))
    context_df = price_df.tail(context_length).copy()
    symbol = str(context_df["symbol"].dropna().iloc[-1]) if "symbol" in context_df else None

    if model_type == "chronos":
        model_bundle = load_foundation_model(
            model_type=model_type,
            model_id=model_id,
            prediction_length=prediction_length,
            context_length=context_length,
            device=device,
        )
        forecast_rows = forecast_with_chronos(
            context_df=context_df,
            model_id=model_id,
            prediction_length=prediction_length,
            device=device,
            freq=freq,
            pipeline=model_bundle["pipeline"],
        )
    elif model_type == "timesfm":
        model_bundle = load_foundation_model(
            model_type=model_type,
            model_id=model_id,
            prediction_length=prediction_length,
            context_length=context_length,
            device=device,
        )
        forecast_rows = forecast_with_timesfm(
            context_df=context_df,
            model_id=model_id,
            prediction_length=prediction_length,
            model=model_bundle["model"],
        )
    else:
        raise ValueError("model_type must be 'chronos' or 'timesfm'")

    if not forecast_rows:
        return {
            "status": "no_data",
            "reason": f"{model_type} returned no forecast rows",
        }

    as_of_row = context_df.iloc[-1]
    as_of_close = float(as_of_row["close"])
    target_step = forecast_rows[min(prediction_length, len(forecast_rows)) - 1]
    predicted_close = float(target_step["prediction"])
    predicted_return = predicted_close / as_of_close - 1
    predicted_class = int(predicted_return > threshold)
    actual = actual_future_result(
        indicator_df,
        as_of_date=str(as_of_row["date"]),
        prediction_length=prediction_length,
        threshold=threshold,
    )

    return {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_type": model_type,
        "model_id": model_id,
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "symbol": symbol,
        "as_of_date": str(as_of_row["date"]),
        "as_of_close": as_of_close,
        "prediction_length": prediction_length,
        "context_length": int(len(context_df)),
        "freq": freq if model_type == "chronos" else None,
        "direction_threshold": float(threshold),
        "predicted_close": predicted_close,
        "predicted_return": float(predicted_return),
        "predicted_class": predicted_class,
        "predicted_direction": "up" if predicted_class == 1 else "down_equal",
        "forecast": forecast_rows,
        **actual,
    }


def run_foundation_backtest(
    indicator_df: pd.DataFrame,
    model_type: str = DEFAULT_FOUNDATION_MODEL,
    model_id: str | None = None,
    prediction_length: int = DEFAULT_PREDICTION_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    threshold: float = TARGET_RETURN_THRESHOLD,
    start_date: str | None = None,
    end_date: str | None = None,
    max_windows: int | None = DEFAULT_BACKTEST_WINDOWS,
    stride: int = DEFAULT_BACKTEST_STRIDE,
    min_context_rows: int = DEFAULT_MIN_CONTEXT_ROWS,
    device: str = "cpu",
    freq: str = DEFAULT_FORECAST_FREQ,
) -> dict[str, Any]:
    """Backtest Chronos or TimesFM forecasts against historical actual closes."""
    price_df = prepare_close_history(indicator_df)
    if price_df.empty:
        return {
            "status": "no_data",
            "reason": "No close-price history available for foundation backtest",
        }

    model_type = normalize_model_type(model_type)
    model_id = model_id or default_model_id(model_type)
    prediction_length = max(1, int(prediction_length))
    context_length = max(8, int(context_length))
    min_context_rows = max(2, min(int(min_context_rows), context_length))
    stride = max(1, int(stride))
    effective_max_windows = None if max_windows is None or int(max_windows) <= 0 else int(max_windows)

    evaluation_indices = backtest_evaluation_indices(
        price_df=price_df,
        prediction_length=prediction_length,
        start_date=start_date,
        end_date=end_date,
        min_context_rows=min_context_rows,
        stride=stride,
        max_windows=effective_max_windows,
    )
    if not evaluation_indices:
        return {
            "status": "no_data",
            "reason": "No historical windows were available for backtesting",
            "available_rows": int(len(price_df)),
            "min_context_rows": min_context_rows,
            "prediction_length": prediction_length,
        }

    model_bundle = load_foundation_model(
        model_type=model_type,
        model_id=model_id,
        prediction_length=prediction_length,
        context_length=context_length,
        device=device,
    )

    rows = []
    errors = []
    for index in evaluation_indices:
        try:
            rows.append(
                backtest_one_window(
                    price_df=price_df,
                    index=index,
                    model_type=model_type,
                    model_bundle=model_bundle,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    threshold=threshold,
                    freq=freq,
                )
            )
        except Exception as exc:
            row = price_df.iloc[index]
            errors.append(
                {
                    "as_of_date": str(row["date"]),
                    "error": str(exc),
                }
            )

    if not rows:
        return {
            "status": "no_data",
            "reason": "All backtest windows failed",
            "errors": errors,
        }

    y_true = np.asarray([row["actual_class"] for row in rows], dtype=int)
    y_pred = np.asarray([row["predicted_class"] for row in rows], dtype=int)
    metrics = foundation_backtest_metrics(y_true, y_pred)
    symbol = str(price_df["symbol"].dropna().iloc[-1]) if "symbol" in price_df else None

    return {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_type": model_type,
        "model_id": model_id,
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "symbol": symbol,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "min_context_rows": min_context_rows,
        "stride": stride,
        "max_windows": effective_max_windows,
        "freq": freq if model_type == "chronos" else None,
        "direction_threshold": float(threshold),
        "window_count": len(rows),
        "error_count": len(errors),
        "date_range": {
            "start": rows[0]["as_of_date"],
            "end": rows[-1]["as_of_date"],
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "class_labels": {
            0: "down_equal",
            1: "up",
        },
        "windows": rows,
        "errors": errors,
    }


def backtest_one_window(
    price_df: pd.DataFrame,
    index: int,
    model_type: str,
    model_bundle: dict[str, Any],
    prediction_length: int,
    context_length: int,
    threshold: float,
    freq: str,
) -> dict[str, Any]:
    context_df = price_df.iloc[: index + 1].tail(context_length).copy()
    if model_type == "chronos":
        forecast_rows = forecast_with_chronos_pipeline(
            pipeline=model_bundle["pipeline"],
            context_df=context_df,
            prediction_length=prediction_length,
            freq=freq,
        )
    else:
        forecast_rows = forecast_with_timesfm_model(
            model=model_bundle["model"],
            context_df=context_df,
            prediction_length=prediction_length,
        )

    if not forecast_rows:
        raise ValueError("No forecast rows returned")

    as_of_row = price_df.iloc[index]
    actual_row = price_df.iloc[index + prediction_length]
    as_of_close = float(as_of_row["close"])
    actual_close = float(actual_row["close"])
    predicted_close = float(forecast_rows[min(prediction_length, len(forecast_rows)) - 1]["prediction"])
    predicted_return = predicted_close / as_of_close - 1
    actual_return = actual_close / as_of_close - 1
    predicted_class = int(predicted_return > threshold)
    actual_class = int(actual_return > threshold)

    return {
        "as_of_date": str(as_of_row["date"]),
        "actual_date": str(actual_row["date"]),
        "as_of_close": as_of_close,
        "predicted_close": predicted_close,
        "actual_close": actual_close,
        "predicted_return": float(predicted_return),
        "actual_return": float(actual_return),
        "predicted_class": predicted_class,
        "actual_class": actual_class,
        "predicted_direction": "up" if predicted_class == 1 else "down_equal",
        "actual_direction": "up" if actual_class == 1 else "down_equal",
        "correct": bool(predicted_class == actual_class),
        "forecast": forecast_rows,
    }


def backtest_evaluation_indices(
    price_df: pd.DataFrame,
    prediction_length: int,
    start_date: str | None,
    end_date: str | None,
    min_context_rows: int,
    stride: int,
    max_windows: int | None,
) -> list[int]:
    dates = pd.to_datetime(price_df["date"], errors="coerce").dt.date
    start_cutoff = pd.to_datetime(start_date, errors="raise").date() if start_date else None
    end_cutoff = pd.to_datetime(end_date, errors="raise").date() if end_date else None
    last_index = len(price_df) - prediction_length - 1
    indices = []
    for index in range(min_context_rows - 1, last_index + 1, stride):
        current_date = dates.iloc[index]
        if start_cutoff and current_date < start_cutoff:
            continue
        if end_cutoff and current_date > end_cutoff:
            continue
        indices.append(index)

    if max_windows is not None and len(indices) > max_windows:
        indices = indices[-max_windows:]
    return indices


def load_foundation_model(
    model_type: str,
    model_id: str,
    prediction_length: int,
    context_length: int,
    device: str = "cpu",
) -> dict[str, Any]:
    if model_type == "chronos":
        return {
            "pipeline": load_chronos_pipeline(model_id=model_id, device=device),
        }
    if model_type == "timesfm":
        return {
            "model": load_timesfm_model(
                model_id=model_id,
                prediction_length=prediction_length,
                context_length=context_length,
            ),
        }
    raise ValueError("model_type must be 'chronos' or 'timesfm'")


def load_chronos_pipeline(model_id: str, device: str = "cpu") -> Any:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos support requires the optional package. Install with: "
            "pip install -r backend/requirements-foundation-models.txt"
        ) from exc

    return Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device,
    )


def load_timesfm_model(
    model_id: str,
    prediction_length: int,
    context_length: int,
) -> Any:
    try:
        import timesfm
        import torch
    except ImportError as exc:
        raise ImportError(
            "TimesFM support requires the optional package. Install with: "
            "pip install -r backend/requirements-foundation-models.txt"
        ) from exc

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=max(context_length, 32),
            max_horizon=max(prediction_length, 1),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def prepare_close_history(
    indicator_df: pd.DataFrame,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    if indicator_df is None or indicator_df.empty:
        return pd.DataFrame(columns=["date", "close"])

    result = indicator_df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"])
    if "symbol" in result.columns:
        result["symbol"] = result["symbol"].astype(str).str.upper()

    if as_of_date:
        cutoff = pd.to_datetime(as_of_date, errors="raise").date()
        result = result[result["date"] <= cutoff]

    result["date"] = result["date"].astype(str)
    return result.sort_values("date", ascending=True).reset_index(drop=True)


def forecast_with_chronos(
    context_df: pd.DataFrame,
    model_id: str,
    prediction_length: int,
    device: str = "cpu",
    freq: str = DEFAULT_FORECAST_FREQ,
    pipeline: Any | None = None,
) -> list[dict[str, Any]]:
    pipeline = pipeline or load_chronos_pipeline(model_id=model_id, device=device)
    return forecast_with_chronos_pipeline(
        pipeline=pipeline,
        context_df=context_df,
        prediction_length=prediction_length,
        freq=freq,
    )


def forecast_with_chronos_pipeline(
    pipeline: Any,
    context_df: pd.DataFrame,
    prediction_length: int,
    freq: str = DEFAULT_FORECAST_FREQ,
) -> list[dict[str, Any]]:
    forecast_input = pd.DataFrame(
        {
            "id": context_df.get("symbol", pd.Series(["series"] * len(context_df))).fillna("series"),
            "timestamp": pd.to_datetime(context_df["date"], errors="coerce"),
            "target": pd.to_numeric(context_df["close"], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "target"])
    forecast_df = pipeline.predict_df(
        forecast_input,
        prediction_length=prediction_length,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        freq=freq,
    )
    return normalize_chronos_forecast(forecast_df)


def normalize_chronos_forecast(forecast_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if forecast_df is None or forecast_df.empty:
        return rows

    for step, (_, row) in enumerate(forecast_df.iterrows(), start=1):
        row_dict = row.to_dict()
        prediction = first_available_value(row_dict, ["predictions", "prediction", "mean", "0.5"])
        if prediction is None:
            continue
        rows.append(
            {
                "step": step,
                "timestamp": str(first_available_value(row_dict, ["timestamp", "date"])),
                "prediction": float(prediction),
                "q10": optional_float(first_available_value(row_dict, ["0.1", "q10", "p10"])),
                "q50": optional_float(first_available_value(row_dict, ["0.5", "q50", "p50"])),
                "q90": optional_float(first_available_value(row_dict, ["0.9", "q90", "p90"])),
            }
        )
    return rows


def forecast_with_timesfm(
    context_df: pd.DataFrame,
    model_id: str,
    prediction_length: int,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    model = model or load_timesfm_model(
        model_id=model_id,
        prediction_length=prediction_length,
        context_length=len(context_df),
    )
    return forecast_with_timesfm_model(
        model=model,
        context_df=context_df,
        prediction_length=prediction_length,
    )


def forecast_with_timesfm_model(
    model: Any,
    context_df: pd.DataFrame,
    prediction_length: int,
) -> list[dict[str, Any]]:
    values = pd.to_numeric(context_df["close"], errors="coerce").dropna().to_numpy(dtype=float)
    point_forecast, quantile_forecast = model.forecast(
        horizon=prediction_length,
        inputs=[values],
    )
    return normalize_timesfm_forecast(point_forecast[0], quantile_forecast[0])


def normalize_timesfm_forecast(
    point_forecast: np.ndarray,
    quantile_forecast: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = []
    for index, prediction in enumerate(point_forecast, start=1):
        quantiles = quantile_forecast[index - 1] if quantile_forecast is not None else None
        rows.append(
            {
                "step": index,
                "timestamp": None,
                "prediction": float(prediction),
                "q10": optional_quantile(quantiles, 1),
                "q50": optional_quantile(quantiles, 5),
                "q90": optional_quantile(quantiles, 9),
            }
        )
    return rows


def actual_future_result(
    indicator_df: pd.DataFrame,
    as_of_date: str,
    prediction_length: int,
    threshold: float,
) -> dict[str, Any]:
    rows = prepare_close_history(indicator_df)
    if rows.empty:
        return {
            "actual_date": None,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
        }

    as_of_timestamp = pd.to_datetime(as_of_date, errors="coerce")
    rows["_date_sort"] = pd.to_datetime(rows["date"], errors="coerce")
    future_rows = rows[rows["_date_sort"] > as_of_timestamp].sort_values("_date_sort")
    if len(future_rows) < prediction_length:
        return {
            "actual_date": None,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
        }

    as_of_rows = rows[rows["date"] == as_of_date]
    if as_of_rows.empty:
        return {
            "actual_date": None,
            "actual_close": None,
            "actual_return": None,
            "actual_direction": None,
        }

    as_of_close = float(as_of_rows.iloc[-1]["close"])
    actual_row = future_rows.iloc[prediction_length - 1]
    actual_close = float(actual_row["close"])
    actual_return = actual_close / as_of_close - 1
    return {
        "actual_date": str(actual_row["date"]),
        "actual_close": actual_close,
        "actual_return": float(actual_return),
        "actual_direction": "up" if actual_return > threshold else "down_equal",
    }


def foundation_backtest_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "support": {
            "down_equal": int((y_true == 0).sum()),
            "up": int((y_true == 1).sum()),
        },
        "predicted_support": {
            "down_equal": int((y_pred == 0).sum()),
            "up": int((y_pred == 1).sum()),
        },
    }


def normalize_model_type(model_type: str) -> str:
    clean_type = str(model_type or "").strip().lower()
    aliases = {
        "chronon": "chronos",
        "chronons": "chronos",
        "chronos": "chronos",
        "timesfm": "timesfm",
        "times_fm": "timesfm",
    }
    if clean_type not in aliases:
        raise ValueError("model_type must be 'chronos' or 'timesfm'")
    return aliases[clean_type]


def default_model_id(model_type: str) -> str:
    if model_type == "timesfm":
        return DEFAULT_TIMESFM_MODEL_ID
    return DEFAULT_CHRONOS_MODEL_ID


def first_available_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None and not pd.isna(row[key]):
            return row[key]
    return None


def optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def optional_quantile(quantiles: np.ndarray | None, index: int) -> float | None:
    if quantiles is None or len(quantiles) <= index:
        return None
    return float(quantiles[index])
