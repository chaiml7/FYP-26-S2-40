"""
Train independent ticker submodels and save one bundle artifact per model type.

The script reads stored Supabase technical_indicators data, trains one
ticker-specific submodel for each selected ticker, stores those submodels
inside one local bundle per model family, and writes local accuracy reports.

Example:
    python scripts/train_all_technical_model_bundles.py --continue-on-error
"""

import argparse
import csv
import html
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "technical_analysis"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(BACKEND_DIR / ".env")

import compare_technical_models as comparison  # noqa: E402
from services.technical.arima_model_service import (  # noqa: E402
    DEFAULT_ARIMA_MIN_TRAIN_ROWS,
    DEFAULT_ARIMA_PREDICTION_LENGTH,
    DEFAULT_ARIMA_VALIDATION_WINDOWS,
    load_arima_artifact,
    train_arima_artifact,
)
from services.technical.binary_xgboost_model_service import (  # noqa: E402
    DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
    DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
    DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE,
    DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    DEFAULT_BINARY_XGBOOST_SAMPLE_WEIGHTING,
    train_binary_xgboost_artifact,
)
from services.technical.catboost_model_service import (  # noqa: E402
    train_catboost_artifact,
)
from services.technical.indicator_service import (  # noqa: E402
    get_technical_indicators_from_supabase,
)
from services.technical.lstm_model_service import (  # noqa: E402
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_DROPOUT,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_LSTM_HIDDEN_SIZE,
    DEFAULT_LSTM_LEARNING_RATE,
    DEFAULT_LSTM_NUM_LAYERS,
    DEFAULT_SEQUENCE_LENGTH,
    load_lstm_artifact,
    train_lstm_artifact,
)
from services.technical.model_service import (  # noqa: E402
    DEFAULT_MAX_FEATURES,
    FEATURES,
    TARGET_RETURN_THRESHOLD,
    get_feature_importance,
    prepare_training_data,
    train_final_model,
    tune_lightgbm_params,
    walk_forward_validation,
)
from services.technical.price_service import get_stock_by_symbol  # noqa: E402
from services.technical.xgboost_model_service import (  # noqa: E402
    DEFAULT_LOOKAHEAD_DAYS,
    DEFAULT_XGBOOST_MAX_FEATURES,
    date_range_summary,
    filter_training_history,
    normalize_optional_date,
    train_xgboost_artifact,
)


DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "TSLA",
    "AMD",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "PLTR",
    "AVGO",
]

DEFAULT_MODELS = [
    "lightgbm",
    "binary-xgboost",
    "xgboost",
    "catboost",
    "lstm",
    "arima",
]

BUNDLE_FILENAMES = {
    "lightgbm": "technical_lightgbm_bundle.joblib",
    "binary-xgboost": "technical_binary_xgboost_bundle.joblib",
    "xgboost": "technical_xgboost_bundle.joblib",
    "catboost": "technical_catboost_bundle.joblib",
    "lstm": "technical_lstm_bundle.joblib",
    "arima": "technical_arima_bundle.joblib",
}

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "binary-xgboost": "Binary XGBoost",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lstm": "LSTM",
    "arima": "ARIMA",
}


def main() -> int:
    args = parse_args()
    symbols = parse_list_argument(args.symbols, valid_values=None, label="symbol")
    models = parse_list_argument(args.models, valid_values=DEFAULT_MODELS, label="model")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Supabase technical indicator rows...")
    indicator_frames, load_errors = load_indicator_frames(
        symbols=symbols,
        continue_on_error=args.continue_on_error,
    )
    if load_errors and not args.continue_on_error:
        first_symbol, first_error = next(iter(load_errors.items()))
        raise RuntimeError(f"{first_symbol}: {first_error}")
    if not indicator_frames:
        raise RuntimeError("No selected symbols had technical_indicators rows")

    run_result = {
        "status": "running",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "symbols": symbols,
        "available_symbols": sorted(indicator_frames),
        "models": models,
        "threshold": float(args.threshold),
        "train_before_date": normalize_optional_date(args.train_before_date),
        "test_start_date": normalize_optional_date(args.test_start_date),
        "test_end_date": normalize_optional_date(args.test_end_date),
        "load_errors": load_errors,
        "model_bundles": {},
    }

    with tempfile.TemporaryDirectory(prefix="technical_bundle_training_") as temp_dir:
        temp_path = Path(temp_dir)
        for model_name in models:
            print(f"\n=== Training {MODEL_LABELS[model_name]} bundle ===")
            bundle_result = train_model_bundle(
                model_name=model_name,
                symbols=symbols,
                indicator_frames=indicator_frames,
                args=args,
                temp_dir=temp_path,
            )
            run_result["model_bundles"][model_name] = bundle_result
            if has_bundle_errors(bundle_result) and not args.continue_on_error:
                for failed_symbol, failed_payload in bundle_result.get(
                    "symbol_results",
                    {},
                ).items():
                    if failed_payload.get("status") != "ok":
                        raise RuntimeError(
                            f"{model_name}/{failed_symbol}: "
                            f"{failed_payload.get('reason')}"
                        )

    run_result["status"] = status_for_run(run_result)
    report_paths = write_reports(run_result, output_dir)
    print_summary(run_result, report_paths)
    return 0 if run_result["status"] in {"ok", "partial"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent ticker-specific submodels for every selected "
            "technical-analysis model type, while saving exactly one bundle "
            "artifact per model type."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=(
            "Symbols to train. Accepts space-separated or comma-separated values. "
            "Default: AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to train. Accepts space-separated or comma-separated values. "
            f"Options: {', '.join(DEFAULT_MODELS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Return threshold for class 1/up. Default: 0.002",
    )
    parser.add_argument(
        "--train-before-date",
        default="2026-01-01",
        help=(
            "Only train on rows before this date. Default: 2026-01-01. "
            "Use an empty value to train through all available labeled rows."
        ),
    )
    parser.add_argument(
        "--test-start-date",
        default="2026-01-01",
        help="First as-of feature date to evaluate. Default: 2026-01-01",
    )
    parser.add_argument(
        "--test-end-date",
        default="2026-06-04",
        help="Last as-of feature date to evaluate. Default: 2026-06-04",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record model/ticker errors in the reports and keep going.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)),
        help="Directory for report files. Default: technical_analysis",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Chronological validation split count for models that expose it. Default: 5",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_XGBOOST_MAX_FEATURES,
        help="Max selected features for XGBoost-style models. Default: 45",
    )
    parser.add_argument(
        "--lstm-sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Past rows per LSTM sample. Default: 30",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=DEFAULT_LSTM_EPOCHS,
        help="LSTM epochs per validation fold and final model. Default: 20",
    )
    parser.add_argument(
        "--lstm-batch-size",
        type=int,
        default=DEFAULT_LSTM_BATCH_SIZE,
        help="LSTM batch size. Default: 64",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=DEFAULT_LSTM_HIDDEN_SIZE,
        help="LSTM hidden size. Default: 64",
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=DEFAULT_LSTM_NUM_LAYERS,
        help="Number of LSTM layers. Default: 1",
    )
    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=DEFAULT_LSTM_DROPOUT,
        help="LSTM dropout. Default: 0.10",
    )
    parser.add_argument(
        "--lstm-learning-rate",
        type=float,
        default=DEFAULT_LSTM_LEARNING_RATE,
        help="LSTM AdamW learning rate. Default: 0.001",
    )
    parser.add_argument(
        "--binary-ensemble-size",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE,
        help="Seeded binary XGBoost models to average. Default: 3",
    )
    parser.add_argument(
        "--binary-no-sample-weighting",
        action="store_true",
        help="Disable binary XGBoost sample weighting.",
    )
    parser.add_argument(
        "--binary-purge-days",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
        help="Binary XGBoost purged validation days. Default: 1",
    )
    parser.add_argument(
        "--binary-embargo-days",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
        help="Binary XGBoost embargo days after validation folds. Default: 0",
    )
    parser.add_argument(
        "--binary-early-stopping-rounds",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
        help="Binary XGBoost early stopping rounds where supported. Default: 50",
    )
    parser.add_argument(
        "--arima-validation-windows",
        type=int,
        default=DEFAULT_ARIMA_VALIDATION_WINDOWS,
        help="ARIMA rolling validation windows. Default: 30",
    )
    parser.add_argument(
        "--arima-min-train-rows",
        type=int,
        default=DEFAULT_ARIMA_MIN_TRAIN_ROWS,
        help="Minimum ARIMA rows before validation starts. Default: 200",
    )
    return parser.parse_args()


def parse_list_argument(
    raw_values: list[str],
    valid_values: list[str] | None,
    label: str,
) -> list[str]:
    values = []
    for item in raw_values:
        for part in item.split(","):
            clean = part.strip()
            if not clean:
                continue
            clean = clean.lower() if valid_values else clean.upper()
            if valid_values and clean not in valid_values:
                valid_text = ", ".join(valid_values)
                raise argparse.ArgumentTypeError(
                    f"Unknown {label} '{clean}'. Valid values: {valid_text}"
                )
            if clean not in values:
                values.append(clean)
    if not values:
        raise argparse.ArgumentTypeError(f"At least one {label} is required")
    return values


def load_indicator_frames(
    symbols: list[str],
    continue_on_error: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames = {}
    errors = {}
    for symbol in symbols:
        try:
            print(f"[data/{symbol}] reading technical_indicators...")
            stock = get_stock_by_symbol(symbol)
            if stock is None:
                raise ValueError(f"{symbol} was not found in the stocks table")
            indicator_df = get_technical_indicators_from_supabase(
                stock["id"],
                stock["symbol"],
            )
            if indicator_df.empty:
                raise ValueError(f"No technical_indicators rows found for {symbol}")
            frames[symbol] = indicator_df
        except Exception as exc:
            errors[symbol] = str(exc)
            print(f"[data/{symbol}] error: {exc}")
            if not continue_on_error:
                break
    return frames, errors


def train_model_bundle(
    model_name: str,
    symbols: list[str],
    indicator_frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    temp_dir: Path,
) -> dict[str, Any]:
    bundle_path = ARTIFACT_DIR / BUNDLE_FILENAMES[model_name]
    submodels = {}
    symbol_results = {}

    for symbol in symbols:
        if symbol not in indicator_frames:
            symbol_results[symbol] = {
                "status": "skipped",
                "reason": "No loaded technical_indicators rows for symbol",
            }
            continue

        print(f"[{model_name}/{symbol}] training...")
        try:
            submodel_artifact, training_result = train_symbol_submodel(
                model_name=model_name,
                symbol=symbol,
                indicator_df=indicator_frames[symbol],
                args=args,
                temp_dir=temp_dir,
            )
            training_result = normalize_training_result(training_result)
            print(f"[{model_name}/{symbol}] evaluating...")
            evaluation_result = evaluate_symbol_submodel(
                model_name=model_name,
                symbol=symbol,
                artifact=submodel_artifact,
                indicator_df=indicator_frames[symbol],
                args=args,
            )
            submodels[symbol] = submodel_artifact
            symbol_results[symbol] = {
                "status": "ok",
                "training": training_result,
                "evaluation": evaluation_result,
            }
            metrics = evaluation_result.get("metrics", {})
            print(
                f"[{model_name}/{symbol}] done "
                f"accuracy={format_metric(metrics.get('accuracy'))} "
                f"balanced_accuracy={format_metric(metrics.get('balanced_accuracy'))}"
            )
        except Exception as exc:
            symbol_results[symbol] = {"status": "error", "reason": str(exc)}
            print(f"[{model_name}/{symbol}] error: {exc}")
            if not args.continue_on_error:
                break

    bundle_payload = build_bundle_payload(
        model_name=model_name,
        symbols=symbols,
        submodels=submodels,
        symbol_results=symbol_results,
        args=args,
    )
    joblib.dump(bundle_payload, bundle_path)
    print(f"[{model_name}] bundle saved: {bundle_path}")

    return {
        "status": status_for_bundle(symbol_results),
        "artifact_path": str(bundle_path),
        "symbol_results": strip_artifacts_from_results(symbol_results),
        "mean_accuracy": mean_metric(symbol_results, "accuracy"),
        "mean_balanced_accuracy": mean_metric(symbol_results, "balanced_accuracy"),
        "successful_symbols": sorted(submodels),
        "failed_symbols": sorted(
            symbol
            for symbol, payload in symbol_results.items()
            if payload.get("status") not in {"ok"}
        ),
    }


def train_symbol_submodel(
    model_name: str,
    symbol: str,
    indicator_df: pd.DataFrame,
    args: argparse.Namespace,
    temp_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_name == "lightgbm":
        return train_lightgbm_submodel(
            indicator_df=indicator_df,
            symbol=symbol,
            threshold=float(args.threshold),
            train_before_date=args.train_before_date,
        )

    if model_name == "binary-xgboost":
        temp_artifact = temp_dir / f"binary_xgboost_{symbol}.joblib"
        result = train_binary_xgboost_artifact(
            indicator_df=indicator_df,
            model_scope="single_ticker_bundle_submodel",
            trained_symbol=symbol,
            lookahead_days=DEFAULT_LOOKAHEAD_DAYS,
            threshold=float(args.threshold),
            train_before_date=args.train_before_date,
            n_splits=args.n_splits,
            max_features=args.max_features,
            ensemble_size=args.binary_ensemble_size,
            use_sample_weighting=(
                DEFAULT_BINARY_XGBOOST_SAMPLE_WEIGHTING
                and not args.binary_no_sample_weighting
            ),
            purge_days=args.binary_purge_days,
            embargo_days=args.binary_embargo_days,
            early_stopping_rounds=args.binary_early_stopping_rounds,
            artifact_path=temp_artifact,
        )
        return load_temp_joblib_artifact(temp_artifact, result)

    if model_name == "xgboost":
        temp_artifact = temp_dir / f"xgboost_{symbol}.joblib"
        result = train_xgboost_artifact(
            indicator_df=indicator_df,
            model_scope="single_ticker_bundle_submodel",
            trained_symbol=symbol,
            lookahead_days=DEFAULT_LOOKAHEAD_DAYS,
            threshold=float(args.threshold),
            train_before_date=args.train_before_date,
            n_splits=args.n_splits,
            max_features=args.max_features,
            artifact_path=temp_artifact,
        )
        return load_temp_joblib_artifact(temp_artifact, result)

    if model_name == "catboost":
        temp_artifact = temp_dir / f"catboost_{symbol}.joblib"
        result = train_catboost_artifact(
            indicator_df=indicator_df,
            model_scope="single_ticker_bundle_submodel",
            trained_symbol=symbol,
            lookahead_days=DEFAULT_LOOKAHEAD_DAYS,
            threshold=float(args.threshold),
            train_before_date=args.train_before_date,
            n_splits=args.n_splits,
            max_features=args.max_features,
            artifact_path=temp_artifact,
        )
        return load_temp_joblib_artifact(temp_artifact, result)

    if model_name == "lstm":
        temp_artifact = temp_dir / f"lstm_{symbol}.pt"
        result = train_lstm_artifact(
            indicator_df=indicator_df,
            model_scope="single_ticker_bundle_submodel",
            trained_symbol=symbol,
            sequence_length=args.lstm_sequence_length,
            lookahead_days=DEFAULT_LOOKAHEAD_DAYS,
            threshold=float(args.threshold),
            train_before_date=args.train_before_date,
            epochs=args.lstm_epochs,
            batch_size=args.lstm_batch_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_dropout,
            learning_rate=args.lstm_learning_rate,
            n_splits=args.n_splits,
            artifact_path=temp_artifact,
        )
        if result.get("status") != "ok":
            raise RuntimeError(result.get("reason", "LSTM training failed"))
        artifact = load_lstm_artifact(temp_artifact)
        return artifact, result

    if model_name == "arima":
        temp_artifact = temp_dir / f"arima_{symbol}.joblib"
        result = train_arima_artifact(
            indicator_df=indicator_df,
            model_scope="single_ticker_bundle_submodel",
            trained_symbol=symbol,
            order=None,
            prediction_length=DEFAULT_ARIMA_PREDICTION_LENGTH,
            threshold=float(args.threshold),
            train_before_date=args.train_before_date,
            validation_windows=args.arima_validation_windows,
            min_train_rows=args.arima_min_train_rows,
            artifact_path=temp_artifact,
        )
        if result.get("status") != "ok":
            raise RuntimeError(result.get("reason", "ARIMA training failed"))
        artifact = load_arima_artifact(temp_artifact)
        return artifact, result

    raise ValueError(f"Unsupported model: {model_name}")


def train_lightgbm_submodel(
    indicator_df: pd.DataFrame,
    symbol: str,
    threshold: float,
    train_before_date: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_indicator_rows = len(indicator_df)
    filtered_df = filter_training_history(indicator_df, train_before_date)
    if filtered_df.empty:
        raise RuntimeError(f"No technical_indicators rows found before {train_before_date}")

    try:
        _, y, clean_df = prepare_training_data(
            filtered_df,
            target_return_threshold=threshold,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    if clean_df.empty or y.empty:
        raise RuntimeError("Not enough complete indicator rows to create ML targets")

    tuning_result = tune_lightgbm_params(
        filtered_df,
        target_return_threshold=threshold,
    )
    best_params = tuning_result.get("best_params", {})
    validation_metrics = walk_forward_validation(
        filtered_df,
        model_params=best_params,
        tune_threshold=True,
        use_feature_selection=True,
        max_features=DEFAULT_MAX_FEATURES,
        target_return_threshold=threshold,
    )
    decision_threshold = float(validation_metrics.get("decision_threshold") or 0.5)
    model, final_clean_df, model_used = train_final_model(
        filtered_df,
        model_params=best_params,
        target_return_threshold=threshold,
        use_feature_selection=True,
        max_features=DEFAULT_MAX_FEATURES,
    )

    selected_features = getattr(model, "selected_features_", FEATURES)
    feature_importance = get_feature_importance(model, limit=15)
    training_date_range = date_range_summary(filtered_df)
    metadata = {
        "model_family": "lightgbm",
        "model_used": model_used,
        "model_scope": "single_ticker_bundle_submodel",
        "trained_symbol": symbol,
        "symbols": [symbol],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "feature_count": len(FEATURES),
        "training_rows": int(len(final_clean_df)),
        "target_return_threshold": float(threshold),
        "direction_threshold": float(threshold),
        "decision_threshold": decision_threshold,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_features": feature_importance,
        "tuned_params": best_params,
        "metrics": metrics_summary(validation_metrics),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    artifact = {"model": model, "metadata": metadata}
    result = {
        "status": "ok",
        "model_scope": "single_ticker_bundle_submodel",
        "trained_symbol": symbol,
        "symbols_trained": [symbol],
        "symbols_trained_count": 1,
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "raw_indicator_rows": raw_indicator_rows,
        "indicator_rows": len(filtered_df),
        "clean_training_rows": int(len(final_clean_df)),
        "target_return_threshold": float(threshold),
        "decision_threshold": decision_threshold,
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
        "selected_feature_count": len(selected_features),
        "top_features": feature_importance,
        "tuned_params": best_params,
    }
    return artifact, result


def load_temp_joblib_artifact(
    path: Path,
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if result.get("status") != "ok":
        raise RuntimeError(result.get("reason", "training failed"))
    if not path.exists():
        raise RuntimeError(f"Expected temporary artifact was not created: {path}")
    return joblib.load(path), result


def evaluate_symbol_submodel(
    model_name: str,
    symbol: str,
    artifact: dict[str, Any],
    indicator_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_rows = comparison.prepare_candidate_rows(
        indicator_df=indicator_df,
        symbol=symbol,
        threshold=float(args.threshold),
        start_date=args.test_start_date,
        end_date=args.test_end_date,
        rows=120,
    )
    if candidate_rows.empty:
        raise RuntimeError("No complete historical rows were available for test window")

    metadata = artifact.get("metadata", {})
    if model_name == "lightgbm":
        predictions = comparison.predict_binary_probability_artifact(
            model=artifact["model"],
            metadata=metadata,
            candidate_rows=candidate_rows,
            model_name=model_name,
        )
    elif model_name == "binary-xgboost":
        predictions = comparison.predict_binary_probability_artifact(
            model=artifact["classifier"],
            metadata=metadata,
            candidate_rows=candidate_rows,
            model_name=model_name,
        )
    elif model_name == "catboost":
        predictions = comparison.predict_binary_probability_artifact(
            model=artifact["classifier"],
            metadata=metadata,
            candidate_rows=candidate_rows,
            model_name=model_name,
        )
    elif model_name == "xgboost":
        predictions = comparison.predict_three_class_xgboost(
            artifact=artifact,
            candidate_rows=candidate_rows,
        )
    elif model_name == "lstm":
        predictions = comparison.predict_lstm_artifact_rows(
            artifact=artifact,
            indicator_df=indicator_df,
            candidate_rows=candidate_rows,
        )
    elif model_name == "arima":
        predictions = comparison.predict_arima_artifact_rows(
            artifact=artifact,
            indicator_df=indicator_df,
            candidate_rows=candidate_rows,
            symbol=symbol,
            comparison_threshold=float(args.threshold),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not predictions:
        raise RuntimeError("Model could not produce predictions for the test window")

    metrics = comparison.calculate_metrics(predictions)
    return {
        "status": "ok",
        "metrics": metrics,
        "prediction_rows": len(predictions),
        "candidate_rows": int(len(candidate_rows)),
        "window_start": str(predictions[0]["as_of_date"]),
        "window_end": str(predictions[-1]["as_of_date"]),
    }


def build_bundle_payload(
    model_name: str,
    symbols: list[str],
    submodels: dict[str, dict[str, Any]],
    symbol_results: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "artifact_type": "technical_model_bundle",
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
        "saved_at": datetime.now(UTC).isoformat(),
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "symbols": symbols,
        "trained_symbols": sorted(submodels),
        "threshold": float(args.threshold),
        "target_definition": (
            "target_direction = 1 if next_day_return > threshold else 0"
        ),
        "train_before_date": normalize_optional_date(args.train_before_date),
        "test_start_date": normalize_optional_date(args.test_start_date),
        "test_end_date": normalize_optional_date(args.test_end_date),
        "feature_configuration": {
            "features": FEATURES,
            "feature_count": len(FEATURES),
            "ticker_independence": (
                "Each symbol has its own trained submodel inside this one bundle."
            ),
        },
        "submodels": submodels,
        "metrics": {
            symbol: payload.get("evaluation", {}).get("metrics")
            for symbol, payload in symbol_results.items()
            if payload.get("status") == "ok"
        },
        "training_results": {
            symbol: payload.get("training")
            for symbol, payload in symbol_results.items()
            if payload.get("status") == "ok"
        },
        "errors": {
            symbol: payload.get("reason")
            for symbol, payload in symbol_results.items()
            if payload.get("status") != "ok"
        },
    }


def strip_artifacts_from_results(
    symbol_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    clean_results = {}
    for symbol, payload in symbol_results.items():
        clean_results[symbol] = comparison.json_ready(payload)
    return clean_results


def normalize_training_result(result: dict[str, Any]) -> dict[str, Any]:
    clean_result = dict(result)
    if "model_artifact_path" in clean_result:
        clean_result["model_artifact_path"] = "stored_in_model_bundle"
    return clean_result


def metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "baseline_accuracy": metrics.get("baseline_accuracy"),
        "majority_baseline_accuracy": metrics.get("majority_baseline_accuracy"),
    }


def write_reports(run_result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "markdown": output_dir / "model_bundle_results.md",
        "csv": output_dir / "model_bundle_results.csv",
        "json": output_dir / "model_bundle_results.json",
        "html": output_dir / "model_bundle_results.html",
    }
    paths["markdown"].write_text(render_markdown(run_result), encoding="utf-8")
    write_csv(run_result, paths["csv"])
    paths["json"].write_text(
        json.dumps(comparison.json_ready(run_result), indent=2),
        encoding="utf-8",
    )
    paths["html"].write_text(render_html(run_result), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def write_csv(run_result: dict[str, Any], path: Path) -> None:
    symbols = run_result["symbols"]
    headers = ["model"]
    for symbol in symbols:
        headers.extend(
            [
                f"{symbol}_accuracy",
                f"{symbol}_balanced_accuracy",
                f"{symbol}_correct_predictions",
                f"{symbol}_total_predictions",
            ]
        )
    headers.extend(["mean_accuracy", "mean_balanced_accuracy", "artifact_path"])

    rows = []
    for model_name, bundle in run_result["model_bundles"].items():
        row = {"model": model_name, "artifact_path": bundle.get("artifact_path")}
        for symbol in symbols:
            metrics = ticker_metrics(bundle, symbol)
            row[f"{symbol}_accuracy"] = metric_value(metrics, "accuracy")
            row[f"{symbol}_balanced_accuracy"] = metric_value(
                metrics,
                "balanced_accuracy",
            )
            row[f"{symbol}_correct_predictions"] = metric_value(
                metrics,
                "correct_predictions",
            )
            row[f"{symbol}_total_predictions"] = metric_value(
                metrics,
                "total_predictions",
            )
        row["mean_accuracy"] = bundle.get("mean_accuracy")
        row["mean_balanced_accuracy"] = bundle.get("mean_balanced_accuracy")
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(run_result: dict[str, Any]) -> str:
    symbols = run_result["symbols"]
    lines = [
        "# Technical Model Bundle Results",
        "",
        f"Generated at: `{run_result['generated_at']}`",
        "",
        f"Threshold: `{run_result['threshold']}`",
        f"Train before date: `{run_result.get('train_before_date')}`",
        f"Test window: `{run_result.get('test_start_date')}` to `{run_result.get('test_end_date')}`",
        "",
        (
            "Each model type has one saved bundle artifact. Inside that bundle, "
            "each ticker has its own ticker-specific submodel."
        ),
        "",
        "## Summary Table",
        "",
    ]
    header = ["Model"]
    for symbol in symbols:
        header.extend([f"{symbol} acc", f"{symbol} bal acc"])
    header.extend(["Mean acc", "Mean bal acc"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")

    for model_name, bundle in run_result["model_bundles"].items():
        row = [MODEL_LABELS.get(model_name, model_name)]
        for symbol in symbols:
            metrics = ticker_metrics(bundle, symbol)
            row.extend(
                [
                    format_metric(metrics.get("accuracy") if metrics else None),
                    format_metric(metrics.get("balanced_accuracy") if metrics else None),
                ]
            )
        row.extend(
            [
                format_metric(bundle.get("mean_accuracy")),
                format_metric(bundle.get("mean_balanced_accuracy")),
            ]
        )
        lines.append("| " + " | ".join(row) + " |")

    best_by_ticker = best_model_by_ticker(run_result)
    lines.extend(["", "## Best Model Per Ticker", ""])
    lines.append("| Ticker | Best model | Balanced accuracy | Accuracy |")
    lines.append("| --- | --- | ---: | ---: |")
    for symbol in symbols:
        best = best_by_ticker.get(symbol)
        if best:
            lines.append(
                "| "
                + " | ".join(
                    [
                        symbol,
                        MODEL_LABELS.get(best["model"], best["model"]),
                        format_metric(best["balanced_accuracy"]),
                        format_metric(best["accuracy"]),
                    ]
                )
                + " |"
            )
        else:
            lines.append(f"| {symbol} | n/a | n/a | n/a |")

    best_overall = best_overall_model(run_result)
    lines.extend(["", "## Best Overall Model", ""])
    if best_overall:
        lines.append(
            f"`{MODEL_LABELS.get(best_overall['model'], best_overall['model'])}` "
            f"by mean balanced accuracy "
            f"`{format_metric(best_overall['mean_balanced_accuracy'])}`."
        )
    else:
        lines.append("No model produced valid metrics.")

    lines.extend(["", "## Bundle Artifacts", ""])
    for model_name, bundle in run_result["model_bundles"].items():
        lines.append(
            f"- `{MODEL_LABELS.get(model_name, model_name)}`: `{bundle.get('artifact_path')}`"
        )

    error_lines = render_error_lines(run_result)
    if error_lines:
        lines.extend(["", "## Errors", "", *error_lines])

    return "\n".join(lines) + "\n"


def render_html(run_result: dict[str, Any]) -> str:
    symbols = run_result["symbols"]
    summary_head = "<th>Model</th>" + "".join(
        f"<th>{symbol}<br>acc</th><th>{symbol}<br>bal acc</th>"
        for symbol in symbols
    )
    summary_head += "<th>Mean acc</th><th>Mean bal acc</th>"

    rows = []
    for model_name, bundle in run_result["model_bundles"].items():
        cells = [f"<td>{escape(MODEL_LABELS.get(model_name, model_name))}</td>"]
        for symbol in symbols:
            metrics = ticker_metrics(bundle, symbol)
            cells.append(
                f"<td>{format_metric(metrics.get('accuracy') if metrics else None)}</td>"
            )
            cells.append(
                "<td>"
                f"{format_metric(metrics.get('balanced_accuracy') if metrics else None)}"
                "</td>"
            )
        cells.append(f"<td>{format_metric(bundle.get('mean_accuracy'))}</td>")
        cells.append(
            f"<td>{format_metric(bundle.get('mean_balanced_accuracy'))}</td>"
        )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    best_ticker_rows = []
    for symbol, best in best_model_by_ticker(run_result).items():
        best_ticker_rows.append(
            "<tr>"
            f"<td>{escape(symbol)}</td>"
            f"<td>{escape(MODEL_LABELS.get(best['model'], best['model']))}</td>"
            f"<td>{format_metric(best['balanced_accuracy'])}</td>"
            f"<td>{format_metric(best['accuracy'])}</td>"
            "</tr>"
        )

    best_overall = best_overall_model(run_result)
    best_overall_text = (
        "No model produced valid metrics."
        if not best_overall
        else (
            f"{escape(MODEL_LABELS.get(best_overall['model'], best_overall['model']))} "
            "by mean balanced accuracy "
            f"{format_metric(best_overall['mean_balanced_accuracy'])}."
        )
    )

    error_lines = render_error_lines(run_result)
    error_html = ""
    if error_lines:
        error_html = "<h2>Errors</h2><ul>" + "".join(
            f"<li>{escape(line[2:])}</li>" if line.startswith("- ") else f"<li>{escape(line)}</li>"
            for line in error_lines
        ) + "</ul>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Technical Model Bundle Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }}
    .meta {{ color: #52606d; }}
  </style>
</head>
<body>
  <h1>Technical Model Bundle Results</h1>
  <p class="meta">Generated at <code>{escape(run_result['generated_at'])}</code></p>
  <p>
    Threshold <code>{run_result['threshold']}</code>.
    Train before <code>{escape(str(run_result.get('train_before_date')))}</code>.
    Test window <code>{escape(str(run_result.get('test_start_date')))}</code>
    to <code>{escape(str(run_result.get('test_end_date')))}</code>.
  </p>
  <p>
    Each model type has one saved bundle artifact. Inside that bundle, each
    ticker has its own ticker-specific submodel.
  </p>

  <h2>Summary Table</h2>
  <table>
    <thead><tr>{summary_head}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>Best Model Per Ticker</h2>
  <table>
    <thead>
      <tr><th>Ticker</th><th>Best model</th><th>Balanced accuracy</th><th>Accuracy</th></tr>
    </thead>
    <tbody>{''.join(best_ticker_rows)}</tbody>
  </table>

  <h2>Best Overall Model</h2>
  <p>{best_overall_text}</p>
  {error_html}
</body>
</html>
"""


def render_error_lines(run_result: dict[str, Any]) -> list[str]:
    lines = []
    for symbol, reason in run_result.get("load_errors", {}).items():
        lines.append(f"- data/{symbol}: {reason}")
    for model_name, bundle in run_result.get("model_bundles", {}).items():
        for symbol, payload in bundle.get("symbol_results", {}).items():
            if payload.get("status") != "ok":
                lines.append(f"- {model_name}/{symbol}: {payload.get('reason')}")
    return lines


def best_model_by_ticker(run_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    best = {}
    for symbol in run_result["symbols"]:
        for model_name, bundle in run_result["model_bundles"].items():
            metrics = ticker_metrics(bundle, symbol)
            if not metrics or metrics.get("balanced_accuracy") is None:
                continue
            candidate = {
                "model": model_name,
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "accuracy": metrics.get("accuracy"),
            }
            previous = best.get(symbol)
            if previous is None or (
                candidate["balanced_accuracy"],
                candidate["accuracy"] or -1,
            ) > (
                previous["balanced_accuracy"],
                previous["accuracy"] or -1,
            ):
                best[symbol] = candidate
    return best


def best_overall_model(run_result: dict[str, Any]) -> dict[str, Any] | None:
    best = None
    for model_name, bundle in run_result["model_bundles"].items():
        value = bundle.get("mean_balanced_accuracy")
        if value is None:
            continue
        candidate = {"model": model_name, "mean_balanced_accuracy": value}
        if best is None or value > best["mean_balanced_accuracy"]:
            best = candidate
    return best


def ticker_metrics(bundle: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    payload = bundle.get("symbol_results", {}).get(symbol, {})
    return payload.get("evaluation", {}).get("metrics")


def mean_metric(symbol_results: dict[str, dict[str, Any]], key: str) -> float | None:
    values = []
    for payload in symbol_results.values():
        metrics = payload.get("evaluation", {}).get("metrics", {})
        value = metrics.get(key)
        if value is not None:
            values.append(float(value))
    return None if not values else float(np.mean(values))


def metric_value(metrics: dict[str, Any] | None, key: str) -> Any:
    if not metrics:
        return None
    return metrics.get(key)


def status_for_bundle(symbol_results: dict[str, dict[str, Any]]) -> str:
    if not symbol_results:
        return "no_data"
    statuses = [payload.get("status") for payload in symbol_results.values()]
    if statuses and all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "ok" for status in statuses):
        return "partial"
    return "error"


def status_for_run(run_result: dict[str, Any]) -> str:
    bundles = run_result.get("model_bundles", {})
    if not bundles:
        return "no_models"
    statuses = [bundle.get("status") for bundle in bundles.values()]
    if all(status == "ok" for status in statuses) and not run_result.get("load_errors"):
        return "ok"
    if any(status in {"ok", "partial"} for status in statuses):
        return "partial"
    return "error"


def has_bundle_errors(bundle_result: dict[str, Any]) -> bool:
    return bundle_result.get("status") != "ok"


def print_summary(run_result: dict[str, Any], report_paths: dict[str, str]) -> None:
    print("\nTechnical model bundle training complete")
    print(f"status: {run_result['status']}")
    print(f"symbols: {', '.join(run_result['symbols'])}")
    print(f"models: {', '.join(run_result['models'])}")
    print("Bundle artifacts:")
    for model_name, bundle in run_result["model_bundles"].items():
        print(f"  {model_name}: {bundle.get('artifact_path')}")
    print("Reports:")
    for key, value in report_paths.items():
        print(f"  {key}: {value}")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
