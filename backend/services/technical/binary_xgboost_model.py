"""Binary XGBoost technical model for next-trading-day direction.

This model is intentionally separate from the existing three-class LightGBM
model. It uses the same normalized technical feature engineering pipeline, but
solves the simpler production question: is the next trading day likely to move
above the configured return threshold?
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.services.technical.feature_engineering import (
    FEATURE_COLUMNS,
    RETURN_THRESHOLD,
    engineer_model_features,
)


MODEL_FAMILY = "xgboost_technical_binary"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "technical"
LATEST_MANIFEST_PATH = MODELS_DIR / "latest.json"
VERSION_PATTERN = re.compile(r"^xgboost_technical_binary_\d{8}T\d{6}\d{6}Z$")

LABELS = ["down_equal", "up"]
MINIMUM_TRAINING_ROWS = 200
MINIMUM_UNIQUE_DATES = 120
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MAX_FEATURES = 35
MAX_PREDICTION_HORIZON_DAYS = 20
DEFAULT_RETURN_THRESHOLD = 0.01
DEFAULT_PREDICTION_HORIZON_DAYS = 5

SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_available",
    "sentiment_raw",
    "sentiment_bullish_scaled",
    "sentiment_article_count_log",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
]

# A decision threshold is for the binary trade direction.  It is deliberately
# separate from DEFAULT_CONFIDENCE_THRESHOLD, which controls bullish/bearish/
# neutral labels returned to the frontend.
DECISION_THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.45, 0.651, 0.01)]
MINIMUM_PREDICTED_CLASS_RATE = 0.10

MINIMUM_ACTIVATION_TEST_ACCURACY = 0.50

PARAMETER_CANDIDATES = [
    {
        "n_estimators": 1100,
        "learning_rate": 0.005,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.5,
        "colsample_bytree": 0.50,
        "reg_alpha": 0.03,
        "reg_lambda": 8.0,
    },
    {
        "n_estimators": 1200,
        "learning_rate": 0.005,
        "max_depth": 4,
        "min_child_weight": 6,
        "subsample": 0.5,
        "colsample_bytree": 0.50,
        "reg_alpha": 0.04,
        "reg_lambda": 8.0,
    },
    {
        "n_estimators": 1000,
        "learning_rate": 0.005,
        "max_depth": 4,
        "min_child_weight": 6,
        "subsample": 0.60,
        "colsample_bytree": 0.50,
        "reg_alpha": 0.05,
        "reg_lambda": 8.0,
    },
]


def _load_ml_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import xgboost
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
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Binary technical ML dependencies are missing. Install "
            "backend/requirements-ml.txt in the active environment."
        ) from exc

    return {
        "joblib": joblib,
        "xgboost": xgboost,
        "XGBClassifier": XGBClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "matthews_corrcoef": matthews_corrcoef,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def is_binary_xgboost_model_version(model_version: str | None) -> bool:
    return bool(model_version and VERSION_PATTERN.fullmatch(model_version))


def _new_version_id(trained_at: datetime) -> str:
    return f"{MODEL_FAMILY}_{trained_at.strftime('%Y%m%dT%H%M%S%fZ')}"


def _validate_version(model_version: str) -> None:
    if not VERSION_PATTERN.fullmatch(model_version):
        raise ValueError(f"Invalid binary XGBoost technical model version: {model_version}")


def _version_paths(model_version: str) -> tuple[Path, Path]:
    _validate_version(model_version)
    version_dir = MODELS_DIR / model_version
    return version_dir / "model.joblib", version_dir / "metadata.json"


def _relative_backend_path(path: Path) -> str:
    backend_dir = Path(__file__).resolve().parents[2]
    return path.relative_to(backend_dir).as_posix()


def _candidate_feature_columns(use_sentiment_features: bool) -> list[str]:
    return [
        *FEATURE_COLUMNS,
        *(SENTIMENT_FEATURE_COLUMNS if use_sentiment_features else []),
    ]


def _add_sentiment_features(
    indicators: list | pd.DataFrame,
    sentiment_scores: list | pd.DataFrame | None,
) -> pd.DataFrame:
    """Join same-day, already-published sentiment without dropping no-news rows."""
    frame = pd.DataFrame(indicators).copy()
    for column in SENTIMENT_FEATURE_COLUMNS:
        frame[column] = 0.0
    if sentiment_scores is None or pd.DataFrame(sentiment_scores).empty:
        return frame

    sentiment = pd.DataFrame(sentiment_scores).copy()
    required = {"stock_id", "score_date"}
    if not required.issubset(sentiment.columns):
        return frame
    sentiment["score_date"] = pd.to_datetime(sentiment["score_date"], errors="coerce")
    sentiment = sentiment.dropna(subset=["stock_id", "score_date"]).copy()
    if sentiment.empty:
        return frame

    numeric = [
        "raw_sentiment", "bullish_score", "article_count", "positive_count", "negative_count",
    ]
    for column in numeric:
        sentiment[column] = pd.to_numeric(sentiment.get(column, 0), errors="coerce").fillna(0.0)
    article_count = sentiment["article_count"].clip(lower=0)
    sentiment = sentiment.assign(
        date=sentiment["score_date"].dt.normalize(),
        sentiment_available=1.0,
        sentiment_raw=sentiment["raw_sentiment"].clip(-1, 1),
        sentiment_bullish_scaled=(sentiment["bullish_score"] - 5.0) / 5.0,
        sentiment_article_count_log=np.log1p(article_count),
        sentiment_positive_ratio=sentiment["positive_count"].div(article_count.where(article_count > 0)).fillna(0.0),
        sentiment_negative_ratio=sentiment["negative_count"].div(article_count.where(article_count > 0)).fillna(0.0),
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.drop(columns=SENTIMENT_FEATURE_COLUMNS).merge(
        sentiment[["stock_id", "date", *SENTIMENT_FEATURE_COLUMNS]],
        on=["stock_id", "date"],
        how="left",
    ).fillna({column: 0.0 for column in SENTIMENT_FEATURE_COLUMNS})


def build_binary_dataset(
    indicators: list | pd.DataFrame,
    return_threshold: float = DEFAULT_RETURN_THRESHOLD,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
    use_sentiment_features: bool = False,
    sentiment_scores: list | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create normalized features and a binary multi-day return target."""
    if not 1 <= int(prediction_horizon_days) <= MAX_PREDICTION_HORIZON_DAYS:
        raise ValueError(
            f"prediction_horizon_days must be between 1 and {MAX_PREDICTION_HORIZON_DAYS}."
        )
    candidate_features = _candidate_feature_columns(use_sentiment_features)
    source = _add_sentiment_features(indicators, sentiment_scores) if use_sentiment_features else pd.DataFrame(indicators)
    df = engineer_model_features(source)
    grouped = df.groupby("stock_id", sort=False)
    future_close = grouped["close"].shift(-int(prediction_horizon_days))
    df["target_return"] = future_close.div(df["close"]) - 1
    df["target_end_date"] = grouped["date"].shift(-int(prediction_horizon_days))
    df["next_day_return"] = df["target_return"]
    df["target_direction"] = (df["target_return"] > return_threshold).astype(int)
    df.loc[df["target_return"].isna(), "target_direction"] = pd.NA
    return (
        df.dropna(subset=[*candidate_features, "target_return", "target_direction"])
        .assign(target_direction=lambda frame: frame["target_direction"].astype(int))
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )


def _filter_train_before_date(dataset: pd.DataFrame, train_before_date: str | None) -> pd.DataFrame:
    if not train_before_date:
        return dataset
    cutoff = pd.to_datetime(train_before_date, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError(f"Invalid train_before_date: {train_before_date}")
    return dataset[(dataset["date"] < cutoff) & (dataset["target_end_date"] < cutoff)].copy()


def split_dataset_by_date(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_dates: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split with a small date embargo."""
    unique_dates = pd.Index(sorted(dataset["date"].dropna().unique()))
    if len(unique_dates) < MINIMUM_UNIQUE_DATES:
        raise ValueError(
            f"At least {MINIMUM_UNIQUE_DATES} unique trading dates are required; "
            f"found {len(unique_dates)}."
        )

    train_end = int(len(unique_dates) * train_fraction)
    validation_end = int(len(unique_dates) * (train_fraction + validation_fraction))
    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end + embargo_dates:validation_end]
    test_dates = unique_dates[validation_end + embargo_dates:]

    if not len(train_dates) or not len(validation_dates) or not len(test_dates):
        raise ValueError("Chronological split produced an empty dataset partition.")

    train = dataset[dataset["date"].isin(train_dates)].copy()
    validation = dataset[dataset["date"].isin(validation_dates)].copy()
    test = dataset[dataset["date"].isin(test_dates)].copy()
    return train, validation, test


def _sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    class_count = max(1, len(counts))
    return y.map(
        lambda value: total / (class_count * max(1, counts.get(int(value), 1)))
    ).to_numpy(dtype=float)


def _new_classifier(XGBClassifier, parameters: dict[str, Any]):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
        tree_method="hist",
        **parameters,
    )


def _matrix(dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return dataset[feature_columns].apply(pd.to_numeric, errors="coerce")


def _labels(dataset: pd.DataFrame) -> pd.Series:
    return dataset["target_direction"].astype(int)


def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def _select_features(train: pd.DataFrame, parameters: dict[str, Any], dependencies: dict[str, Any], candidate_features: list[str]) -> list[str]:
    y_train = _labels(train)
    if y_train.nunique() < 2:
        raise ValueError("Training data needs both binary classes.")

    selector = _new_classifier(dependencies["XGBClassifier"], parameters)
    selector.fit(
        _matrix(train, candidate_features),
        y_train,
        sample_weight=_sample_weights(y_train),
    )
    importances = getattr(selector, "feature_importances_", None)
    if importances is None:
        return candidate_features[:DEFAULT_MAX_FEATURES]

    ranked = sorted(
        zip(candidate_features, importances),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    selected = [feature for feature, importance in ranked if float(importance) > 0]
    return selected[:DEFAULT_MAX_FEATURES] or candidate_features[:DEFAULT_MAX_FEATURES]


def _feature_importance(model, feature_columns: list[str]) -> list[dict[str, float]]:
    raw_importance = getattr(model, "feature_importances_", None)
    if raw_importance is None:
        return []
    return sorted(
        [
            {"feature": feature, "importance": round(float(importance), 6)}
            for feature, importance in zip(feature_columns, raw_importance)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


def _fit_model(
    train: pd.DataFrame,
    parameters: dict[str, Any],
    feature_columns: list[str],
    dependencies: dict[str, Any],
):
    y_train = _labels(train)
    model = _new_classifier(dependencies["XGBClassifier"], parameters)
    model.fit(
        _matrix(train, feature_columns),
        y_train,
        sample_weight=_sample_weights(y_train),
    )
    return model


def _best_decision_threshold(y_true: np.ndarray, probability_up: np.ndarray, dependencies: dict[str, Any]) -> float:
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_key = None
    for threshold in DECISION_THRESHOLD_GRID:
        predicted = (probability_up >= threshold).astype(int)
        predicted_up_rate = float(predicted.mean())
        # Reject cutoffs that effectively emit only one class. F1 can look
        # deceptively high for an imbalanced, almost-always-up classifier.
        if not MINIMUM_PREDICTED_CLASS_RATE <= predicted_up_rate <= (1 - MINIMUM_PREDICTED_CLASS_RATE):
            continue
        key = (
            float(dependencies["balanced_accuracy_score"](y_true, predicted)),
            float(dependencies["matthews_corrcoef"](y_true, predicted)),
            float(dependencies["accuracy_score"](y_true, predicted)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def _calculate_metrics(
    dataset: pd.DataFrame,
    model,
    feature_columns: list[str],
    decision_threshold: float,
    dependencies: dict[str, Any],
) -> dict[str, Any]:
    y_true = _labels(dataset).to_numpy(dtype=int)
    probability_up = _positive_probability(model, _matrix(dataset, feature_columns))
    predicted = (probability_up >= decision_threshold).astype(int)
    majority_class = int(pd.Series(y_true).mode().iloc[0])
    majority = np.full(len(y_true), majority_class, dtype=int)
    metrics = {
        "accuracy": round(float(dependencies["accuracy_score"](y_true, predicted)), 4),
        "balanced_accuracy": round(float(dependencies["balanced_accuracy_score"](y_true, predicted)), 4),
        "precision": round(float(dependencies["precision_score"](y_true, predicted, zero_division=0)), 4),
        "recall": round(float(dependencies["recall_score"](y_true, predicted, zero_division=0)), 4),
        "f1_score": round(float(dependencies["f1_score"](y_true, predicted, zero_division=0)), 4),
        "mcc": round(float(dependencies["matthews_corrcoef"](y_true, predicted)), 4),
        "majority_baseline_accuracy": round(float(dependencies["accuracy_score"](y_true, majority)), 4),
        "confusion_matrix": dependencies["confusion_matrix"](y_true, predicted, labels=[0, 1]).tolist(),
        "label_order": LABELS,
        "decision_threshold": round(float(decision_threshold), 4),
        "predicted_up_rate": round(float(predicted.mean()), 4),
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = round(float(dependencies["roc_auc_score"](y_true, probability_up)), 4)
    else:
        metrics["roc_auc"] = None
    return metrics


def _select_parameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dependencies: dict[str, Any],
    candidate_features: list[str],
) -> tuple[
    dict[str, Any],
    list[str],
    float,
    dict[str, Any],
    int,
    list[dict[str, Any]],
]:
    best_key = None
    best_payload = None
    candidate_results = []

    for candidate_index, parameters in enumerate(PARAMETER_CANDIDATES, start=1):
        selected_features = _select_features(
            train,
            parameters,
            dependencies,
            candidate_features,
        )
        model = _fit_model(train, parameters, selected_features, dependencies)
        validation_probability = _positive_probability(model, _matrix(validation, selected_features))
        validation_actual = _labels(validation).to_numpy(dtype=int)
        threshold = _best_decision_threshold(
            validation_actual,
            validation_probability,
            dependencies,
        )
        metrics = _calculate_metrics(
            validation,
            model,
            selected_features,
            threshold,
            dependencies,
        )
        key = (
            metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0,
            metrics["balanced_accuracy"],
            metrics["mcc"],
        )
        candidate_results.append({
            "candidate": candidate_index,
            "hyperparameters": dict(parameters),
            "selected_feature_count": len(selected_features),
            "decision_threshold": threshold,
            "validation_metrics": metrics,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_payload = (
                dict(parameters),
                selected_features,
                threshold,
                metrics,
                candidate_results[-1]["candidate"],
            )

    if best_payload is None:
        raise ValueError("Could not train a valid binary XGBoost candidate.")
    parameters, selected_features, threshold, metrics, selected_candidate = best_payload
    for result in candidate_results:
        result["selected"] = result["candidate"] == selected_candidate
    return (
        parameters,
        selected_features,
        threshold,
        metrics,
        selected_candidate,
        candidate_results,
    )


def _activation_eligibility(test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Allow activation when holdout accuracy is strictly above 50 percent."""
    accuracy = float(test_metrics["accuracy"])
    return {
        "passed": accuracy > MINIMUM_ACTIVATION_TEST_ACCURACY,
        "test_accuracy": round(accuracy, 4),
        "minimum_test_accuracy_exclusive": MINIMUM_ACTIVATION_TEST_ACCURACY,
    }


def train_model(
    indicators: list | pd.DataFrame,
    return_threshold: float = DEFAULT_RETURN_THRESHOLD,
    train_before_date: str | None = None,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
    use_sentiment_features: bool = False,
    sentiment_scores: list | pd.DataFrame | None = None,
) -> dict[str, Any]:
    dependencies = _load_ml_dependencies()
    candidate_features = _candidate_feature_columns(use_sentiment_features)
    dataset = build_binary_dataset(
        indicators,
        return_threshold,
        prediction_horizon_days,
        use_sentiment_features,
        sentiment_scores,
    )
    dataset = _filter_train_before_date(dataset, train_before_date)
    if len(dataset) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            f"At least {MINIMUM_TRAINING_ROWS} complete rows are required; "
            f"found {len(dataset)}."
        )

    train, validation, test = split_dataset_by_date(dataset)
    (
        parameters,
        selected_features,
        decision_threshold,
        validation_metrics,
        selected_candidate,
        candidate_validation_results,
    ) = _select_parameters(
        train,
        validation,
        dependencies,
        candidate_features,
    )

    evaluation_train = pd.concat([train, validation], ignore_index=True)
    evaluation_model = _fit_model(
        evaluation_train,
        parameters,
        selected_features,
        dependencies,
    )
    test_metrics = _calculate_metrics(
        test,
        evaluation_model,
        selected_features,
        decision_threshold,
        dependencies,
    )
    activation_eligibility = _activation_eligibility(test_metrics)

    final_features = _select_features(
        dataset,
        parameters,
        dependencies,
        candidate_features,
    )
    final_model = _fit_model(dataset, parameters, final_features, dependencies)
    feature_importance = _feature_importance(final_model, final_features)

    trained_at = datetime.now(timezone.utc)
    model_version = _new_version_id(trained_at)
    model_path, metadata_path = _version_paths(model_version)
    model_path.parent.mkdir(parents=True, exist_ok=False)
    dependencies["joblib"].dump(final_model, model_path)

    class_distribution = {
        LABELS[int(label)]: int(count)
        for label, count in dataset["target_direction"].value_counts().sort_index().items()
    }
    metadata = {
        "model_version": model_version,
        "model_family": MODEL_FAMILY,
        "trained_at": trained_at.isoformat(),
        "training_rows": int(len(dataset)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "dataset_start": dataset["date"].min().date().isoformat(),
        "dataset_end": dataset["date"].max().date().isoformat(),
        "train_before_date": train_before_date,
        "class_distribution": class_distribution,
        "hyperparameters": parameters,
        "selected_candidate": selected_candidate,
        "candidate_validation_results": candidate_validation_results,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "activation_eligibility": activation_eligibility,
        "feature_columns": final_features,
        "all_candidate_feature_columns": candidate_features,
        "top_features": feature_importance[:15],
        "labels": LABELS,
        "return_threshold": float(return_threshold),
        "prediction_horizon_trading_days": int(prediction_horizon_days),
        "use_sentiment_features": bool(use_sentiment_features),
        "decision_threshold": float(decision_threshold),
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "evaluation_mode": (
            "binary_xgboost_date_grouped_train_validation_test_with_one_date_embargo"
        ),
        "xgboost_version": dependencies["xgboost"].__version__,
        "model_path": _relative_backend_path(model_path),
        "metadata_path": _relative_backend_path(metadata_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def walk_forward_evaluate(
    indicators: list | pd.DataFrame,
    return_threshold: float = DEFAULT_RETURN_THRESHOLD,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
    use_sentiment_features: bool = False,
    sentiment_scores: list | pd.DataFrame | None = None,
    test_window_dates: int = 63,
    max_folds: int = 4,
) -> dict[str, Any]:
    """Run expanding-window evaluations without saving or activating a model."""
    if test_window_dates < 20 or max_folds < 1:
        raise ValueError("test_window_dates must be at least 20 and max_folds at least 1.")
    dependencies = _load_ml_dependencies()
    candidate_features = _candidate_feature_columns(use_sentiment_features)
    dataset = build_binary_dataset(
        indicators,
        return_threshold,
        prediction_horizon_days,
        use_sentiment_features,
        sentiment_scores,
    )
    unique_dates = pd.Index(sorted(dataset["date"].dropna().unique()))
    initial_train_dates = max(MINIMUM_UNIQUE_DATES, int(len(unique_dates) * 0.55))
    available_folds = (len(unique_dates) - initial_train_dates) // test_window_dates
    fold_count = min(max_folds, available_folds)
    if fold_count < 1:
        raise ValueError("Not enough dated rows for the requested walk-forward evaluation.")

    fold_starts = [
        initial_train_dates + (index * test_window_dates)
        for index in range(available_folds - fold_count, available_folds)
    ]
    folds = []
    for fold_number, start_index in enumerate(fold_starts, start=1):
        test_dates = unique_dates[start_index:start_index + test_window_dates]
        test_start = test_dates[0]
        history = dataset[
            (dataset["date"] < test_start)
            & (dataset["target_end_date"] < test_start)
        ].copy()
        train, validation, _ = split_dataset_by_date(history)
        (
            parameters,
            selected_features,
            decision_threshold,
            validation_metrics,
            selected_candidate,
            candidate_validation_results,
        ) = _select_parameters(
            train,
            validation,
            dependencies,
            candidate_features,
        )
        model = _fit_model(
            pd.concat([train, validation], ignore_index=True),
            parameters,
            selected_features,
            dependencies,
        )
        test = dataset[dataset["date"].isin(test_dates)].copy()
        metrics = _calculate_metrics(
            test,
            model,
            selected_features,
            decision_threshold,
            dependencies,
        )
        folds.append({
            "fold": fold_number,
            "train_end": history["date"].max().date().isoformat(),
            "test_start": test["date"].min().date().isoformat(),
            "test_end": test["date"].max().date().isoformat(),
            "training_rows": int(len(history)),
            "test_rows": int(len(test)),
            "decision_threshold": decision_threshold,
            "selected_candidate": selected_candidate,
            "selected_hyperparameters": parameters,
            "selected_feature_count": len(selected_features),
            "candidate_validation_results": candidate_validation_results,
            "validation_metrics": validation_metrics,
            "test_metrics": metrics,
        })

    summary_keys = ["accuracy", "balanced_accuracy", "mcc", "roc_auc"]
    mean_test_metrics = {
        key: round(float(np.mean([fold["test_metrics"][key] for fold in folds])), 4)
        for key in summary_keys
        if all(fold["test_metrics"].get(key) is not None for fold in folds)
    }
    candidate_selection_summary = [
        {
            "candidate": index,
            "hyperparameters": parameters,
            "selected_folds": sum(
                fold["selected_candidate"] == index for fold in folds
            ),
        }
        for index, parameters in enumerate(PARAMETER_CANDIDATES, start=1)
    ]
    return {
        "evaluation_mode": "binary_xgboost_expanding_window_walk_forward",
        "prediction_horizon_trading_days": int(prediction_horizon_days),
        "return_threshold": float(return_threshold),
        "use_sentiment_features": bool(use_sentiment_features),
        "test_window_dates": int(test_window_dates),
        "folds": folds,
        "mean_test_metrics": mean_test_metrics,
        "candidate_selection_summary": candidate_selection_summary,
    }


def load_model_metadata(model_version: str) -> dict[str, Any]:
    _, metadata_path = _version_paths(model_version)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Binary XGBoost technical model metadata not found for {model_version}."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_model(model_version: str):
    dependencies = _load_ml_dependencies()
    model_path, _ = _version_paths(model_version)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Binary XGBoost technical model file not found for {model_version}."
        )
    return dependencies["joblib"].load(model_path), load_model_metadata(model_version)


def activate_local_model(model_version: str) -> dict[str, str]:
    model_path, metadata_path = _version_paths(model_version)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Local binary XGBoost artifacts not found for {model_version}."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": model_version,
        "model_family": MODEL_FAMILY,
        "model_path": _relative_backend_path(model_path),
        "metadata_path": _relative_backend_path(metadata_path),
    }
    temporary = LATEST_MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(LATEST_MANIFEST_PATH)
    return manifest


def _prediction_label(probability_up: float, confidence_threshold: float) -> str:
    if probability_up >= confidence_threshold:
        return "bullish"
    if probability_up <= (1 - confidence_threshold):
        return "bearish"
    return "neutral"


def _score_from_probability(probability_up: float) -> tuple[float, float]:
    raw_outlook = max(-1.0, min(1.0, (probability_up * 2) - 1))
    if raw_outlook >= 0:
        score = 5 + (raw_outlook * 5)
    else:
        score = 5 + (raw_outlook * 4)
    return round(float(raw_outlook), 4), round(float(score), 2)


def predict_latest(
    indicators: list | pd.DataFrame,
    model_version: str,
    sentiment_scores: list | pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    model, metadata = load_model(model_version)
    use_sentiment_features = bool(metadata.get("use_sentiment_features", False))
    source = _add_sentiment_features(indicators, sentiment_scores) if use_sentiment_features else pd.DataFrame(indicators)
    features = engineer_model_features(source)
    selected_features = metadata["feature_columns"]
    complete = features.dropna(subset=selected_features).copy()
    if complete.empty:
        raise ValueError("No complete technical indicator rows are available.")

    latest = (
        complete.sort_values(["stock_id", "date"])
        .groupby("stock_id", as_index=False, sort=False)
        .tail(1)
    )
    probability_up = _positive_probability(model, _matrix(latest, selected_features))
    confidence_threshold = float(metadata.get("confidence_threshold") or DEFAULT_CONFIDENCE_THRESHOLD)
    created_at = datetime.now(timezone.utc).isoformat()
    predictions = []

    for position, (_, row) in enumerate(latest.iterrows()):
        up_probability = float(probability_up[position])
        raw_outlook, technical_score = _score_from_probability(up_probability)
        prediction = _prediction_label(up_probability, confidence_threshold)
        predictions.append({
            "stock_id": int(row["stock_id"]),
            "symbol": str(row["symbol"]).upper(),
            "latest_date": row["date"].date().isoformat(),
            "latest_close": float(row["close"]),
            "prediction": prediction,
            "probabilities": {
                "bearish": round(float(1 - up_probability), 4),
                "neutral": 0.0,
                "bullish": round(up_probability, 4),
            },
            "raw_outlook": raw_outlook,
            "technical_score": technical_score,
            "prediction_horizon": f"{int(metadata.get('prediction_horizon_trading_days', 1))}_trading_days",
            "model_version": metadata["model_version"],
            "created_at": created_at,
        })

    return predictions


def backtest_model(
    indicators: list | pd.DataFrame,
    model_version: str,
    start_date: str | None = None,
    end_date: str | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    transaction_cost_bps: float = 10.0,
    sentiment_scores: list | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate historical predictions and a simple long-only strategy.

    This is a diagnostic backtest. It is clean only when the model was trained
    before the requested backtest window.
    """
    dependencies = _load_ml_dependencies()
    model, metadata = load_model(model_version)
    dataset = build_binary_dataset(
        indicators,
        metadata.get("return_threshold", RETURN_THRESHOLD),
        metadata.get("prediction_horizon_trading_days", 1),
        bool(metadata.get("use_sentiment_features", False)),
        sentiment_scores,
    )
    if start_date:
        dataset = dataset[dataset["date"] >= pd.to_datetime(start_date)]
    if end_date:
        dataset = dataset[dataset["date"] <= pd.to_datetime(end_date)]
    if dataset.empty:
        raise ValueError("No complete technical rows were available for the backtest window.")

    selected_features = metadata["feature_columns"]
    dataset = dataset.dropna(subset=selected_features).copy()
    probability_up = _positive_probability(model, _matrix(dataset, selected_features))
    decision_threshold = float(metadata.get("decision_threshold") or DEFAULT_DECISION_THRESHOLD)
    predicted = (probability_up >= decision_threshold).astype(int)
    actual = dataset["target_direction"].to_numpy(dtype=int)
    confidence = np.where(predicted == 1, probability_up, 1 - probability_up)
    traded = confidence >= float(confidence_threshold)
    cost = float(transaction_cost_bps) / 10_000
    long_only_returns = np.where(
        (predicted == 1) & traded,
        dataset["target_return"].to_numpy(dtype=float) - cost,
        0.0,
    )
    cumulative_return = float(np.prod(1 + long_only_returns) - 1)

    all_metrics = {
        "accuracy": round(float(dependencies["accuracy_score"](actual, predicted)), 4),
        "balanced_accuracy": round(float(dependencies["balanced_accuracy_score"](actual, predicted)), 4),
        "precision": round(float(dependencies["precision_score"](actual, predicted, zero_division=0)), 4),
        "recall": round(float(dependencies["recall_score"](actual, predicted, zero_division=0)), 4),
        "f1_score": round(float(dependencies["f1_score"](actual, predicted, zero_division=0)), 4),
        "mcc": round(float(dependencies["matthews_corrcoef"](actual, predicted)), 4),
        "confusion_matrix": dependencies["confusion_matrix"](actual, predicted, labels=[0, 1]).tolist(),
    }
    if len(set(actual.tolist())) > 1:
        all_metrics["roc_auc"] = round(float(dependencies["roc_auc_score"](actual, probability_up)), 4)
    else:
        all_metrics["roc_auc"] = None

    training_end = metadata.get("dataset_end")
    leakage_warning = None
    if training_end and str(dataset["date"].min().date()) <= training_end:
        leakage_warning = (
            "Backtest window overlaps the model training period. Retrain with "
            "train_before_date before reporting this as out-of-sample performance."
        )

    return {
        "model_version": model_version,
        "symbol": (
            str(dataset["symbol"].dropna().iloc[0]).upper()
            if dataset["symbol"].nunique() == 1
            else None
        ),
        "window_start": dataset["date"].min().date().isoformat(),
        "window_end": dataset["date"].max().date().isoformat(),
        "rows": int(len(dataset)),
        "decision_threshold": decision_threshold,
        "confidence_threshold": float(confidence_threshold),
        "transaction_cost_bps": float(transaction_cost_bps),
        "prediction_horizon_trading_days": int(metadata.get("prediction_horizon_trading_days", 1)),
        "metrics": all_metrics,
        "trades_taken": int(traded.sum()),
        "trade_rate": round(float(traded.mean()), 4),
        "mean_strategy_return": round(float(np.mean(long_only_returns)), 6),
        "cumulative_strategy_return": round(cumulative_return, 6),
        "leakage_warning": leakage_warning,
    }
