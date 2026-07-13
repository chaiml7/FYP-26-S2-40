"""
Compare saved technical-analysis models on the same historical window.

Examples:
    python scripts/compare_technical_models.py --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
    python scripts/compare_technical_models.py --symbol NVDA --rows 120
    python scripts/compare_technical_models.py --symbol NVDA --models lightgbm,catboost,binary-xgboost

The script reads saved local artifacts and Supabase technical_indicators. It
does not train models and does not write to Supabase.
"""
import argparse
import html
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.arima_model_service import (  # noqa: E402
    ARIMA_ARTIFACT_PATH,
    fit_arima_model,
)
from services.technical.binary_xgboost_model_service import (  # noqa: E402
    BINARY_XGBOOST_ARTIFACT_PATH,
)
from services.technical.catboost_model_service import (  # noqa: E402
    CATBOOST_ARTIFACT_PATH,
)
from services.technical.indicator_service import (  # noqa: E402
    get_technical_indicators_from_supabase,
)
from services.technical.lstm_model_service import (  # noqa: E402
    FEATURES,
    LSTM_ARTIFACT_PATH,
    model_from_artifact,
    predict_lstm_probabilities,
    transform_sequences,
)
from services.technical.model_service import (  # noqa: E402
    MODEL_ARTIFACT_PATH,
    TARGET_RETURN_THRESHOLD,
)
from services.technical.price_service import get_stock_by_symbol  # noqa: E402
from services.technical.xgboost_model_service import (  # noqa: E402
    XGBOOST_ARTIFACT_PATH,
    selective_class_predictions,
)

MODEL_PATHS = {
    "lightgbm": MODEL_ARTIFACT_PATH,
    "xgboost": XGBOOST_ARTIFACT_PATH,
    "binary-xgboost": BINARY_XGBOOST_ARTIFACT_PATH,
    "catboost": CATBOOST_ARTIFACT_PATH,
    "lstm": LSTM_ARTIFACT_PATH,
    "arima": ARIMA_ARTIFACT_PATH,
}

DEFAULT_MODELS = [
    "lightgbm",
    "xgboost",
    "binary-xgboost",
    "catboost",
    "lstm",
    "arima",
]


def main() -> int:
    args = parse_args()
    try:
        result = run_comparison(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    output_html = Path(
        args.output_html
        or default_output_path(args.symbol, suffix=".html")
    )
    output_json = Path(
        args.output_json
        or default_output_path(args.symbol, suffix=".json")
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_html(result), encoding="utf-8")
    output_json.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print_result(result, output_html, output_json)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved technical-analysis models on the same symbol and "
            "same historical as-of date window."
        )
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol to test.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated models to compare. Options: "
            f"{', '.join(MODEL_PATHS)}. Default: all saved-artifact models."
        ),
    )
    parser.add_argument(
        "--start-date",
        help="First as-of feature date to test. Example: 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        help="Last as-of feature date to test. Example: 2026-06-04",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=120,
        help="Most recent eligible rows to test when --start-date is omitted. Default: 120",
    )
    parser.add_argument(
        "--threshold",
        "--target-threshold",
        dest="threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Common actual next-day return threshold for class 1/up. Default: 0.002",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested model artifact cannot be evaluated.",
    )
    parser.add_argument(
        "--output-html",
        help="Local HTML comparison report path.",
    )
    parser.add_argument(
        "--output-json",
        help="Local JSON comparison report path.",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=180,
        help="Maximum common prediction rows shown in the HTML table. Default: 180",
    )
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    symbol = args.symbol.upper()
    selected_models = parse_model_names(args.models)
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        return {"status": "no_data", "reason": f"{symbol} was not found in stocks"}

    indicator_df = get_technical_indicators_from_supabase(stock["id"], stock["symbol"])
    if indicator_df.empty:
        return {
            "status": "no_data",
            "reason": f"No technical_indicators rows found for {symbol}",
        }

    candidate_rows = prepare_candidate_rows(
        indicator_df=indicator_df,
        symbol=symbol,
        threshold=float(args.threshold),
        start_date=args.start_date,
        end_date=args.end_date,
        rows=args.rows,
    )
    if candidate_rows.empty:
        return {
            "status": "no_data",
            "reason": "No complete historical rows were available for that window",
        }

    model_results = {}
    skipped = {}
    for model_name in selected_models:
        try:
            model_result = evaluate_model(
                model_name=model_name,
                symbol=symbol,
                indicator_df=indicator_df,
                candidate_rows=candidate_rows,
                comparison_threshold=float(args.threshold),
            )
        except Exception as exc:
            model_result = {
                "status": "error",
                "reason": str(exc),
                "model_artifact_path": str(MODEL_PATHS[model_name]),
            }

        if model_result.get("status") == "ok":
            model_results[model_name] = model_result
        else:
            skipped[model_name] = model_result

    if args.require_all and skipped:
        return {
            "status": "error",
            "reason": "At least one requested model could not be evaluated",
            "requested_models": selected_models,
            "skipped_models": skipped,
        }

    if not model_results:
        return {
            "status": "no_models",
            "reason": "No requested saved models could be evaluated",
            "requested_models": selected_models,
            "skipped_models": skipped,
        }

    common_dates = common_prediction_dates(model_results)
    if not common_dates:
        return {
            "status": "no_common_window",
            "reason": "Available models did not produce predictions for any shared dates",
            "requested_models": selected_models,
            "available_models": sorted(model_results),
            "skipped_models": skipped,
        }

    aligned_models = align_models_to_dates(model_results, common_dates)
    summary = {
        model_name: calculate_metrics(model_payload["predictions"])
        for model_name, model_payload in aligned_models.items()
    }
    common_rows = common_rows_for_dates(candidate_rows, common_dates)

    return {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "symbol": symbol,
        "requested_models": selected_models,
        "available_models": sorted(aligned_models),
        "skipped_models": skipped,
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "requested_rows": int(args.rows),
        "comparison_threshold": float(args.threshold),
        "candidate_window_start": str(candidate_rows.iloc[0]["date"]),
        "candidate_window_end": str(candidate_rows.iloc[-1]["date"]),
        "candidate_rows": int(len(candidate_rows)),
        "common_window_start": common_dates[0],
        "common_window_end": common_dates[-1],
        "common_rows": int(len(common_dates)),
        "summary": summary,
        "models": aligned_models,
        "common_price_rows": common_rows,
        "max_table_rows": int(args.max_table_rows),
    }


def parse_model_names(value: str) -> list[str]:
    names = []
    for item in value.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in MODEL_PATHS:
            valid = ", ".join(MODEL_PATHS)
            raise ValueError(f"Unknown model '{name}'. Valid models: {valid}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one model must be selected")
    return names


def prepare_candidate_rows(
    indicator_df: pd.DataFrame,
    symbol: str,
    threshold: float,
    start_date: str | None,
    end_date: str | None,
    rows: int,
) -> pd.DataFrame:
    df = indicator_df.copy()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    df["next_date"] = df["date"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    df["actual_next_return"] = df["next_close"] / df["close"] - 1
    df["actual_class"] = (df["actual_next_return"] > threshold).astype(int)
    df.loc[df["actual_next_return"].isna(), "actual_class"] = pd.NA

    if start_date:
        start = pd.to_datetime(start_date, errors="raise").date()
        df = df[df["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date, errors="raise").date()
        df = df[df["date"] <= end]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["actual_class", "actual_next_return", "next_date", "next_close"])
    if not start_date:
        df = df.tail(max(1, int(rows)))
    return df.reset_index(drop=True)


def evaluate_model(
    model_name: str,
    symbol: str,
    indicator_df: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    comparison_threshold: float,
) -> dict[str, Any]:
    artifact_path = Path(MODEL_PATHS[model_name])
    if not artifact_path.exists():
        return {
            "status": "missing_artifact",
            "reason": f"Saved artifact not found at {artifact_path}",
            "model_artifact_path": str(artifact_path),
        }

    if model_name == "lstm":
        artifact = load_lstm_artifact(artifact_path)
    else:
        artifact = joblib.load(artifact_path)

    metadata = artifact.get("metadata", {})
    trained_symbol = metadata.get("trained_symbol")
    if trained_symbol and str(trained_symbol).upper() != symbol.upper():
        return {
            "status": "wrong_symbol",
            "reason": f"Artifact was trained for {trained_symbol}, not {symbol}",
            "model_artifact_path": str(artifact_path),
        }

    if model_name == "lightgbm":
        predictions = predict_binary_probability_artifact(
            model=artifact["model"],
            metadata=metadata,
            candidate_rows=candidate_rows,
            model_name=model_name,
        )
    elif model_name == "binary-xgboost":
        predictions = predict_binary_probability_artifact(
            model=artifact["classifier"],
            metadata=metadata,
            candidate_rows=candidate_rows,
            model_name=model_name,
        )
    elif model_name == "catboost":
        predictions = predict_binary_probability_artifact(
            model=artifact["classifier"],
            metadata=metadata,
            candidate_rows=candidate_rows,
            model_name=model_name,
        )
    elif model_name == "xgboost":
        predictions = predict_three_class_xgboost(
            artifact=artifact,
            candidate_rows=candidate_rows,
        )
    elif model_name == "lstm":
        predictions = predict_lstm_artifact_rows(
            artifact=artifact,
            indicator_df=indicator_df,
            candidate_rows=candidate_rows,
        )
    elif model_name == "arima":
        predictions = predict_arima_artifact_rows(
            artifact=artifact,
            indicator_df=indicator_df,
            candidate_rows=candidate_rows,
            symbol=symbol,
            comparison_threshold=comparison_threshold,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not predictions:
        return {
            "status": "no_predictions",
            "reason": "Model artifact could not produce predictions for the requested window",
            "model_artifact_path": str(artifact_path),
        }

    return {
        "status": "ok",
        "model_artifact_path": str(artifact_path),
        "metadata": summarize_metadata(metadata),
        "artifact_direction_threshold": artifact_threshold(metadata),
        "predictions": predictions,
        "raw_prediction_rows": len(predictions),
    }


def predict_binary_probability_artifact(
    model: Any,
    metadata: dict[str, Any],
    candidate_rows: pd.DataFrame,
    model_name: str,
) -> list[dict[str, Any]]:
    selected_features = metadata.get(
        "selected_features",
        getattr(model, "selected_features_", FEATURES),
    )
    categorical_features = metadata.get("categorical_features", [])
    decision_threshold = float(metadata.get("decision_threshold") or 0.5)
    ready_rows = candidate_rows.replace([np.inf, -np.inf], np.nan)
    ready_rows = ready_rows.dropna(subset=selected_features).copy()
    if ready_rows.empty:
        return []

    X = ready_rows[selected_features].copy()
    if model_name == "catboost":
        for feature in categorical_features:
            if feature in X.columns:
                X[feature] = X[feature].fillna("UNKNOWN").astype(str)

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return []
    probability_up = probabilities[:, classes.index(1)]
    predicted_classes = (probability_up >= decision_threshold).astype(int)
    return prediction_payloads(
        rows=ready_rows,
        predicted_classes=predicted_classes,
        probability_up=probability_up,
        confidence=np.where(
            predicted_classes == 1,
            probability_up,
            1 - probability_up,
        ),
    )


def predict_three_class_xgboost(
    artifact: dict[str, Any],
    candidate_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    classifier = artifact["classifier"]
    metadata = artifact.get("metadata", {})
    selected_features = metadata.get("selected_features", FEATURES)
    thresholds = metadata.get("class_probability_thresholds", {})
    down_threshold = float(thresholds.get("down", 0.55))
    up_threshold = float(thresholds.get("up", 0.55))

    ready_rows = candidate_rows.replace([np.inf, -np.inf], np.nan)
    ready_rows = ready_rows.dropna(subset=selected_features).copy()
    if ready_rows.empty:
        return []

    probabilities = classifier.predict_proba(ready_rows[selected_features])
    raw_classes = selective_class_predictions(
        probabilities,
        down_threshold=down_threshold,
        up_threshold=up_threshold,
    )
    predicted_classes = (raw_classes == 2).astype(int)
    probability_up = probabilities[:, 2] if probabilities.shape[1] > 2 else np.zeros(len(ready_rows))
    selected_confidence = np.max(probabilities, axis=1)
    predictions = prediction_payloads(
        rows=ready_rows,
        predicted_classes=predicted_classes,
        probability_up=probability_up,
        confidence=selected_confidence,
    )
    for index, prediction in enumerate(predictions):
        prediction["raw_predicted_class"] = int(raw_classes[index])
        prediction["raw_predicted_direction"] = {
            0: "neutral",
            1: "down",
            2: "up",
        }.get(int(raw_classes[index]), "unknown")
        prediction["class_probabilities"] = {
            "neutral": float(probabilities[index][0]) if probabilities.shape[1] > 0 else None,
            "down": float(probabilities[index][1]) if probabilities.shape[1] > 1 else None,
            "up": float(probabilities[index][2]) if probabilities.shape[1] > 2 else None,
        }
    return predictions


def predict_lstm_artifact_rows(
    artifact: dict[str, Any],
    indicator_df: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    metadata = artifact.get("metadata", {})
    feature_names = metadata.get("features", FEATURES)
    sequence_length = int(metadata.get("sequence_length", 30))
    decision_threshold = float(metadata.get("decision_threshold") or 0.5)
    target_dates = set(candidate_rows["date"].astype(str))

    rows = indicator_df.copy()
    if "symbol" in candidate_rows.columns and "symbol" in rows.columns:
        symbol = str(candidate_rows.iloc[0]["symbol"]).upper()
        rows = rows[rows["symbol"].astype(str).str.upper() == symbol]
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce", utc=True).dt.date.astype(str)
    rows = rows.replace([np.inf, -np.inf], np.nan)
    rows = rows.dropna(subset=["date", *feature_names]).sort_values("date").reset_index(drop=True)
    if rows.empty:
        return []

    model = model_from_artifact(artifact)
    scaler = artifact["scaler"]
    candidate_by_date = {
        str(row["date"]): row for _, row in candidate_rows.iterrows()
    }
    sequences = []
    meta_rows = []
    for end_index, row in rows.iterrows():
        as_of_date = str(row["date"])
        if as_of_date not in target_dates:
            continue
        start_index = end_index - sequence_length + 1
        if start_index < 0:
            continue
        window = rows.iloc[start_index : end_index + 1][feature_names].to_numpy(
            dtype=np.float32
        )
        if len(window) != sequence_length or np.isnan(window).any():
            continue
        sequences.append(window)
        meta_rows.append(candidate_by_date[as_of_date])

    if not sequences:
        return []

    sequence_array = np.asarray(sequences, dtype=np.float32)
    scaled_sequences = transform_sequences(sequence_array, scaler)
    probability_up = predict_lstm_probabilities(model, scaled_sequences)
    predicted_classes = (probability_up >= decision_threshold).astype(int)
    confidence = np.where(predicted_classes == 1, probability_up, 1 - probability_up)
    return prediction_payloads(
        rows=pd.DataFrame(meta_rows),
        predicted_classes=predicted_classes,
        probability_up=probability_up,
        confidence=confidence,
    )


def predict_arima_artifact_rows(
    artifact: dict[str, Any],
    indicator_df: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    symbol: str,
    comparison_threshold: float,
) -> list[dict[str, Any]]:
    trained_models = artifact.get("models", {})
    if symbol not in trained_models:
        return []
    metadata = artifact.get("metadata", {})
    prediction_length = int(metadata.get("prediction_length", 1))
    order = tuple(trained_models[symbol]["order"])
    min_rows = max(sum(order) + 10, int(metadata.get("min_train_rows", 30)))

    predictions = []
    history_rows = indicator_df.copy()
    if "symbol" in history_rows.columns:
        history_rows = history_rows[
            history_rows["symbol"].astype(str).str.upper() == symbol.upper()
        ]
    history_rows["date"] = pd.to_datetime(
        history_rows["date"],
        errors="coerce",
        utc=True,
    ).dt.date
    history_rows["close"] = pd.to_numeric(history_rows["close"], errors="coerce")
    history_rows = (
        history_rows.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    candidate_by_date = {
        str(row["date"]): row for _, row in candidate_rows.iterrows()
    }
    for as_of_date in sorted(candidate_by_date):
        row = candidate_by_date[as_of_date]
        cutoff = pd.to_datetime(as_of_date, errors="raise").date()
        history = history_rows[history_rows["date"] <= cutoff]
        if len(history) < min_rows:
            continue
        try:
            model = fit_arima_model(history["close"].to_numpy(dtype=float), order)
            forecast_values = np.asarray(
                model.forecast(steps=prediction_length),
                dtype=float,
            )
        except Exception:
            continue
        predicted_close = float(forecast_values[-1])
        as_of_close = float(row["close"])
        predicted_return = predicted_close / as_of_close - 1
        predicted_class = int(predicted_return > comparison_threshold)
        probability_up = 1.0 if predicted_class == 1 else 0.0
        payload = prediction_payloads(
            rows=pd.DataFrame([row]),
            predicted_classes=np.asarray([predicted_class]),
            probability_up=np.asarray([probability_up]),
            confidence=np.asarray([abs(predicted_return)]),
        )[0]
        payload["predicted_close"] = predicted_close
        payload["predicted_return"] = float(predicted_return)
        payload["order"] = list(order)
        predictions.append(payload)
    return predictions


def prediction_payloads(
    rows: pd.DataFrame,
    predicted_classes: np.ndarray,
    probability_up: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, Any]]:
    predictions = []
    for index, (_, row) in enumerate(rows.iterrows()):
        predicted_class = int(predicted_classes[index])
        actual_class = int(row["actual_class"])
        predictions.append(
            {
                "stock_id": int(row["stock_id"]) if "stock_id" in row and not pd.isna(row["stock_id"]) else None,
                "symbol": str(row["symbol"]).upper() if "symbol" in row else None,
                "as_of_date": str(row["date"]),
                "as_of_close": float(row["close"]),
                "next_date": str(row["next_date"]),
                "next_close": float(row["next_close"]),
                "actual_next_return": float(row["actual_next_return"]),
                "actual_class": actual_class,
                "actual_direction": "up" if actual_class == 1 else "down_equal",
                "predicted_class": predicted_class,
                "predicted_direction": "up" if predicted_class == 1 else "down_equal",
                "probability_up": float(probability_up[index]),
                "confidence": float(confidence[index]),
                "correct": bool(predicted_class == actual_class),
            }
        )
    return predictions


def common_prediction_dates(model_results: dict[str, dict[str, Any]]) -> list[str]:
    date_sets = []
    for model_payload in model_results.values():
        date_sets.append(
            {prediction["as_of_date"] for prediction in model_payload["predictions"]}
        )
    common = set.intersection(*date_sets) if date_sets else set()
    return sorted(common)


def align_models_to_dates(
    model_results: dict[str, dict[str, Any]],
    common_dates: list[str],
) -> dict[str, dict[str, Any]]:
    common_set = set(common_dates)
    aligned = {}
    for model_name, model_payload in model_results.items():
        predictions = [
            prediction
            for prediction in model_payload["predictions"]
            if prediction["as_of_date"] in common_set
        ]
        predictions = sorted(predictions, key=lambda item: item["as_of_date"])
        aligned[model_name] = {
            **model_payload,
            "predictions": predictions,
            "common_prediction_rows": len(predictions),
        }
    return aligned


def common_rows_for_dates(
    candidate_rows: pd.DataFrame,
    common_dates: list[str],
) -> list[dict[str, Any]]:
    common_set = set(common_dates)
    rows = candidate_rows[candidate_rows["date"].astype(str).isin(common_set)].copy()
    rows = rows.sort_values("date")
    return [
        {
            "as_of_date": str(row["date"]),
            "as_of_close": float(row["close"]),
            "next_date": str(row["next_date"]),
            "next_close": float(row["next_close"]),
            "actual_next_return": float(row["actual_next_return"]),
            "actual_class": int(row["actual_class"]),
            "actual_direction": "up" if int(row["actual_class"]) == 1 else "down_equal",
        }
        for _, row in rows.iterrows()
    ]


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = np.asarray([item["actual_class"] for item in predictions], dtype=int)
    y_pred = np.asarray([item["predicted_class"] for item in predictions], dtype=int)
    probability_up = np.asarray([item["probability_up"] for item in predictions], dtype=float)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "correct_predictions": int((y_true == y_pred).sum()),
        "incorrect_predictions": int((y_true != y_pred).sum()),
        "total_predictions": int(len(y_true)),
        "predicted_up_count": int((y_pred == 1).sum()),
        "actual_up_count": int((y_true == 1).sum()),
    }
    if len(set(y_true.tolist())) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probability_up))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def summarize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_family": metadata.get("model_family"),
        "model_scope": metadata.get("model_scope"),
        "trained_symbol": metadata.get("trained_symbol"),
        "training_start_date": metadata.get("training_start_date"),
        "training_end_date": metadata.get("training_end_date"),
        "decision_threshold": metadata.get("decision_threshold"),
        "direction_threshold": artifact_threshold(metadata),
        "selected_feature_count": metadata.get("selected_feature_count"),
    }


def artifact_threshold(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("target_return_threshold")
    if value is None:
        value = metadata.get("direction_threshold")
    return None if value is None else float(value)


def load_lstm_artifact(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def render_html(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return minimal_html(
            "Technical Model Comparison",
            f"<p>{escape_text(result.get('reason', 'No comparison generated'))}</p>",
        )

    summary_table = render_summary_table(result)
    skipped = render_skipped_models(result.get("skipped_models", {}))
    prediction_table = render_prediction_table(result)
    body = f"""
    <header>
      <h1>{escape_text(result["symbol"])} Model Comparison</h1>
      <p class="muted">
        Common window {escape_text(result["common_window_start"])} to {escape_text(result["common_window_end"])}
        | {result["common_rows"]} shared rows
        | target threshold {result["comparison_threshold"]:.4f}
      </p>
    </header>
    <section class="panel">
      <h2>Same-Window Metrics</h2>
      {summary_table}
    </section>
    {skipped}
    <section class="panel">
      <h2>Common Prediction Rows</h2>
      {prediction_table}
    </section>
    """
    return minimal_html("Technical Model Comparison", body)


def render_summary_table(result: dict[str, Any]) -> str:
    rows = []
    for model_name, metrics in sorted(
        result["summary"].items(),
        key=lambda item: (
            item[1].get("accuracy") if item[1].get("accuracy") is not None else -1
        ),
        reverse=True,
    ):
        metadata = result["models"][model_name].get("metadata", {})
        warning = ""
        artifact_threshold_value = result["models"][model_name].get(
            "artifact_direction_threshold"
        )
        if artifact_threshold_value is not None and not np.isclose(
            artifact_threshold_value,
            result["comparison_threshold"],
        ):
            warning = f"trained threshold {artifact_threshold_value:.4f}"
        rows.append(
            "<tr>"
            f"<th>{escape_text(model_name)}</th>"
            f"<td>{format_metric(metrics.get('accuracy'))}</td>"
            f"<td>{format_metric(metrics.get('balanced_accuracy'))}</td>"
            f"<td>{format_metric(metrics.get('precision'))}</td>"
            f"<td>{format_metric(metrics.get('recall'))}</td>"
            f"<td>{format_metric(metrics.get('f1_score'))}</td>"
            f"<td>{format_metric(metrics.get('roc_auc'))}</td>"
            f"<td>{metrics.get('correct_predictions')}/{metrics.get('total_predictions')}</td>"
            f"<td>{escape_text(metadata.get('training_end_date'))}</td>"
            f"<td>{escape_text(warning)}</td>"
            "</tr>"
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>Balanced</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>ROC AUC</th>
          <th>Correct</th>
          <th>Trained through</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_skipped_models(skipped_models: dict[str, Any]) -> str:
    if not skipped_models:
        return ""
    rows = []
    for model_name, payload in skipped_models.items():
        rows.append(
            "<tr>"
            f"<th>{escape_text(model_name)}</th>"
            f"<td>{escape_text(payload.get('status'))}</td>"
            f"<td>{escape_text(payload.get('reason'))}</td>"
            "</tr>"
        )
    return f"""
    <section class="panel">
      <h2>Skipped Models</h2>
      <table>
        <thead><tr><th>Model</th><th>Status</th><th>Reason</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def render_prediction_table(result: dict[str, Any]) -> str:
    model_names = sorted(result["models"])
    rows_by_date = {row["as_of_date"]: row for row in result["common_price_rows"]}
    model_predictions = {
        model_name: {
            prediction["as_of_date"]: prediction
            for prediction in model_payload["predictions"]
        }
        for model_name, model_payload in result["models"].items()
    }
    dates = sorted(rows_by_date)
    visible_dates = dates[: max(0, int(result.get("max_table_rows", 180)))]
    header_cells = "".join(f"<th>{escape_text(model)}</th>" for model in model_names)
    body_rows = []
    for date in visible_dates:
        row = rows_by_date[date]
        model_cells = []
        for model_name in model_names:
            prediction = model_predictions[model_name][date]
            status = "ok" if prediction["correct"] else "bad"
            label = "U" if prediction["predicted_class"] == 1 else "D"
            model_cells.append(
                f'<td><span class="{status}">{label}</span> '
                f'{prediction["probability_up"]:.2f}</td>'
            )
        body_rows.append(
            "<tr>"
            f"<td>{escape_text(date)}</td>"
            f"<td>${row['as_of_close']:.2f}</td>"
            f"<td>{escape_text(row['actual_direction'])}</td>"
            f"<td>{row['actual_next_return']:.2%}</td>"
            f"{''.join(model_cells)}"
            "</tr>"
        )
    hidden_count = max(0, len(dates) - len(visible_dates))
    note = (
        f'<p class="muted">{hidden_count} more rows are in the JSON output.</p>'
        if hidden_count
        else ""
    )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Close</th>
          <th>Actual next</th>
          <th>Next return</th>
          {header_cells}
        </tr>
      </thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
    {note}
    """


def minimal_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape_text(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #172033;
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header, section {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      padding: 28px 20px 8px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .muted {{
      color: #667085;
    }}
    .panel {{
      margin: 16px auto;
      padding: 18px;
      background: #ffffff;
      border: 1px solid #d7dde8;
      border-radius: 8px;
      box-sizing: border-box;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #d7dde8;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f8fafc;
      color: #344054;
      font-weight: 700;
    }}
    .ok {{
      color: #15803d;
      font-weight: 700;
    }}
    .bad {{
      color: #b42318;
      font-weight: 700;
    }}
    @media (max-width: 780px) {{
      header, .panel {{
        margin-left: 14px;
        margin-right: 14px;
      }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def default_output_path(symbol: str, suffix: str) -> str:
    return f"technical_analysis/{symbol.upper()}_model_comparison{suffix}"


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def escape_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def print_result(result: dict[str, Any], output_html: Path, output_json: Path) -> None:
    print(f"status: {result['status']}")
    if result["status"] != "ok":
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    print(f"symbol: {result['symbol']}")
    print(f"common_window: {result['common_window_start']} to {result['common_window_end']}")
    print(f"common_rows: {result['common_rows']}")
    print(f"comparison_threshold: {result['comparison_threshold']:.4f}")
    print("")
    print("Model metrics on the same rows:")
    for model_name, metrics in sorted(
        result["summary"].items(),
        key=lambda item: item[1].get("accuracy") or -1,
        reverse=True,
    ):
        print(
            f"{model_name}: "
            f"accuracy={format_metric(metrics.get('accuracy'))}, "
            f"f1={format_metric(metrics.get('f1_score'))}, "
            f"roc_auc={format_metric(metrics.get('roc_auc'))}, "
            f"correct={metrics.get('correct_predictions')}/{metrics.get('total_predictions')}"
        )
    if result.get("skipped_models"):
        print("")
        print("Skipped models:")
        for model_name, payload in result["skipped_models"].items():
            print(f"{model_name}: {payload.get('status')} - {payload.get('reason')}")
    print("")
    print(f"html_report: {output_html}")
    print(f"json_report: {output_json}")


if __name__ == "__main__":
    raise SystemExit(main())
