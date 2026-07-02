"""Versioned XGBoost training, evaluation, storage, and inference."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.services.financial.feature_engineering import (
    FEATURE_COLUMNS,
    build_training_dataset,
    engineer_features,
)


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
BINARY_LABELS = ["bearish", "bullish"]
BINARY_LABEL_TO_ID = {label: index for index, label in enumerate(BINARY_LABELS)}
MODEL_FAMILY = "xgboost_financial"
BINARY_MODEL_FAMILY = "xgboost_financial_binary"
DEFAULT_BINARY_DISPLAY_THRESHOLDS = {
    "bearish_below": 0.45,
    "neutral_between": [0.45, 0.55],
    "bullish_above": 0.55,
}
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "financial"
LATEST_MANIFEST_PATH = MODELS_DIR / "latest.json"
MINIMUM_TRAINING_ROWS = 20
VERSION_PATTERN = re.compile(
    r"^(xgboost_financial|xgboost_financial_binary)_\d{8}T\d{6}\d{6}Z$"
)

BASE_HYPERPARAMETERS = {
    "objective": "multi:softprob",
    "num_class": len(LABELS),
    "n_estimators": 800,
    "learning_rate": 0.01,
    "max_depth": 2,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "gamma": 0.5,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": 0,
}
CONTINUATION_ESTIMATORS = 30
EARLY_STOPPING_ROUNDS = 25
MIN_ROLLING_TRAIN_ROWS = 40
MIN_ROLLING_EVAL_ROWS = 8
TUNING_WEIGHT_POWERS = [0.0, 0.25, 0.5, 0.75, 1.0]
TUNING_CANDIDATES = [
    {
        "name": "current",
        "parameters": {},
    },
    {
        "name": "looser_depth_2",
        "parameters": {
            "max_depth": 2,
            "min_child_weight": 5,
            "gamma": 0.5,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "learning_rate": 0.01,
            "n_estimators": 800,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    {
        "name": "more_flexible_depth_2",
        "parameters": {
            "max_depth": 2,
            "min_child_weight": 3,
            "gamma": 0.0,
            "reg_alpha": 0.2,
            "reg_lambda": 3.0,
            "learning_rate": 0.015,
            "n_estimators": 600,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    {
        "name": "fast_shallow",
        "parameters": {
            "max_depth": 1,
            "min_child_weight": 5,
            "gamma": 0.0,
            "reg_alpha": 0.2,
            "reg_lambda": 5.0,
            "learning_rate": 0.02,
            "n_estimators": 400,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
    },
    {
        "name": "regularized_depth_2",
        "parameters": {
            "max_depth": 2,
            "min_child_weight": 8,
            "gamma": 1.0,
            "reg_alpha": 0.8,
            "reg_lambda": 8.0,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
    },
]


def _load_ml_dependencies():
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            log_loss,
        )
        import xgboost
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Financial ML dependencies are missing. "
            "Install backend/requirements-ml.txt in the backend environment."
        ) from exc

    return {
        "xgboost": xgboost,
        "XGBClassifier": XGBClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "log_loss": log_loss,
    }


def _new_classifier(
    XGBClassifier,
    continuation: bool = False,
    parameters_override: dict = None,
    early_stopping: bool = False,
):
    parameters = dict(BASE_HYPERPARAMETERS)
    if continuation:
        parameters["n_estimators"] = CONTINUATION_ESTIMATORS
    if parameters_override:
        parameters.update(parameters_override)
    if parameters.get("num_class") is None:
        parameters.pop("num_class", None)
    if early_stopping:
        parameters["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBClassifier(**parameters)


def _sample_weights(labels: pd.Series, power: float = 1.0) -> np.ndarray:
    if power <= 0:
        return np.ones(len(labels))

    counts = labels.value_counts()
    total = len(labels)
    class_count = len(counts)
    weights = labels.map(lambda label: total / (class_count * counts[label])).to_numpy()
    if power == 1.0:
        return weights
    return np.power(weights, power)


def _feature_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def _chronological_holdout(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_indices = dataset.groupby("stock_id")["period"].idxmax()
    holdout = dataset.loc[holdout_indices].sort_values("period").copy()
    train = dataset.drop(index=holdout_indices).sort_values("period").copy()
    return train, holdout


def _calculate_metrics(actual, predicted, dependencies, probabilities=None) -> dict:
    metrics = {
        "accuracy": round(
            float(dependencies["accuracy_score"](actual, predicted)),
            4,
        ),
        "balanced_accuracy": round(
            float(dependencies["balanced_accuracy_score"](actual, predicted)),
            4,
        ),
        "macro_f1": round(
            float(
                dependencies["f1_score"](
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "confusion_matrix": dependencies["confusion_matrix"](
            actual,
            predicted,
            labels=range(len(LABELS)),
        ).tolist(),
        "label_order": LABELS,
    }
    if probabilities is not None:
        metrics["log_loss"] = round(
            float(
                dependencies["log_loss"](
                    actual,
                    probabilities,
                    labels=range(len(LABELS)),
                )
            ),
            4,
        )
    return metrics


def _label_ids(labels: pd.Series) -> np.ndarray:
    return labels.map(LABEL_TO_ID).to_numpy()


def _best_iteration_count(model) -> int | None:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return None
    return int(best_iteration) + 1


def _fit_classifier(
    training_set: pd.DataFrame,
    dependencies,
    validation_set: pd.DataFrame = None,
    parameters_override: dict = None,
    weight_power: float = 1.0,
):
    labels = training_set["target_label"]
    model = _new_classifier(
        dependencies["XGBClassifier"],
        parameters_override=parameters_override,
        early_stopping=validation_set is not None,
    )
    fit_arguments = {
        "sample_weight": _sample_weights(labels, weight_power),
    }
    if validation_set is not None:
        validation_labels = validation_set["target_label"]
        fit_arguments.update({
            "eval_set": [
                (
                    _feature_matrix(validation_set),
                    _label_ids(validation_labels),
                )
            ],
            "sample_weight_eval_set": [
                _sample_weights(validation_labels, weight_power),
            ],
            "verbose": False,
        })

    model.fit(
        _feature_matrix(training_set),
        _label_ids(labels),
        **fit_arguments,
    )
    return model


def _evaluate_model(model, evaluation_set: pd.DataFrame, dependencies) -> dict:
    actual = _label_ids(evaluation_set["target_label"])
    matrix = _feature_matrix(evaluation_set)
    probabilities = model.predict_proba(matrix)
    predicted = np.argmax(probabilities, axis=1)
    return _calculate_metrics(actual, predicted, dependencies, probabilities)


def _latest_validation_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    years = sorted(dataset["period"].dt.year.dropna().unique())
    for validation_year in reversed(years):
        training_set = dataset[dataset["period"].dt.year < validation_year].copy()
        validation_set = dataset[dataset["period"].dt.year == validation_year].copy()
        if (
            len(training_set) >= MIN_ROLLING_TRAIN_ROWS
            and len(validation_set) >= MIN_ROLLING_EVAL_ROWS
            and training_set["target_label"].nunique() == len(LABELS)
            and validation_set["target_label"].nunique() >= 2
        ):
            return training_set, validation_set
    return None, None


def _mean_metric(folds: list[dict], metric_name: str) -> float | None:
    values = [
        fold["metrics"][metric_name]
        for fold in folds
        if metric_name in fold["metrics"]
    ]
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def _rolling_validation(dataset: pd.DataFrame, dependencies) -> dict:
    folds = []
    years = sorted(dataset["period"].dt.year.dropna().unique())

    for test_year in years:
        previous_data = dataset[dataset["period"].dt.year < test_year].copy()
        test_set = dataset[dataset["period"].dt.year == test_year].copy()
        if (
            len(previous_data) < MIN_ROLLING_TRAIN_ROWS
            or len(test_set) < MIN_ROLLING_EVAL_ROWS
            or test_set["target_label"].nunique() < 2
        ):
            continue

        training_set, validation_set = _latest_validation_split(previous_data)
        if training_set is None:
            continue

        model = _fit_classifier(training_set, dependencies, validation_set)
        folds.append({
            "test_year": int(test_year),
            "train_rows": len(training_set),
            "validation_rows": len(validation_set),
            "test_rows": len(test_set),
            "best_n_estimators": _best_iteration_count(model),
            "test_class_distribution": {
                label: int(count)
                for label, count in test_set["target_label"].value_counts().items()
            },
            "metrics": _evaluate_model(model, test_set, dependencies),
        })

    return {
        "fold_count": len(folds),
        "folds": folds,
        "average_metrics": {
            "accuracy": _mean_metric(folds, "accuracy"),
            "balanced_accuracy": _mean_metric(folds, "balanced_accuracy"),
            "macro_f1": _mean_metric(folds, "macro_f1"),
            "log_loss": _mean_metric(folds, "log_loss"),
        },
        "method": (
            "For each test year, train on older years, use the latest prior "
            "year for early stopping, then score the unseen test year."
        ),
    }


def _parameters_for_candidate(candidate: dict) -> dict:
    parameters = dict(BASE_HYPERPARAMETERS)
    parameters.update(candidate["parameters"])
    return parameters


def _rank_tuning_result(result: dict) -> tuple:
    latest = result["latest_holdout_metrics"]
    rolling = result["rolling_validation"]["average_metrics"]
    return (
        rolling.get("macro_f1") or 0,
        rolling.get("balanced_accuracy") or 0,
        latest.get("accuracy") or 0,
        -(rolling.get("log_loss") or 999),
    )


def _evaluate_tuning_candidate(
    dataset: pd.DataFrame,
    candidate: dict,
    weight_power: float,
    dependencies,
) -> dict:
    train_set, holdout_set = _chronological_holdout(dataset)
    early_training_set, validation_set = _latest_validation_split(train_set)
    if early_training_set is None:
        early_training_set = train_set
        validation_set = None

    parameters = _parameters_for_candidate(candidate)
    model = _fit_classifier(
        early_training_set,
        dependencies,
        validation_set,
        parameters,
        weight_power,
    )
    latest_metrics = _evaluate_model(model, holdout_set, dependencies)
    return {
        "candidate_name": candidate["name"],
        "weight_power": weight_power,
        "parameters": parameters,
        "best_n_estimators": _best_iteration_count(model),
        "latest_holdout_rows": len(holdout_set),
        "latest_holdout_metrics": latest_metrics,
        "rolling_validation": _rolling_validation_for_parameters(
            dataset,
            parameters,
            weight_power,
            dependencies,
        ),
    }


def _rolling_validation_for_parameters(
    dataset: pd.DataFrame,
    parameters: dict,
    weight_power: float,
    dependencies,
) -> dict:
    folds = []
    years = sorted(dataset["period"].dt.year.dropna().unique())

    for test_year in years:
        previous_data = dataset[dataset["period"].dt.year < test_year].copy()
        test_set = dataset[dataset["period"].dt.year == test_year].copy()
        if (
            len(previous_data) < MIN_ROLLING_TRAIN_ROWS
            or len(test_set) < MIN_ROLLING_EVAL_ROWS
            or test_set["target_label"].nunique() < 2
        ):
            continue

        training_set, validation_set = _latest_validation_split(previous_data)
        if training_set is None:
            continue

        model = _fit_classifier(
            training_set,
            dependencies,
            validation_set,
            parameters,
            weight_power,
        )
        folds.append({
            "test_year": int(test_year),
            "train_rows": len(training_set),
            "validation_rows": len(validation_set),
            "test_rows": len(test_set),
            "best_n_estimators": _best_iteration_count(model),
            "test_class_distribution": {
                label: int(count)
                for label, count in test_set["target_label"].value_counts().items()
            },
            "metrics": _evaluate_model(model, test_set, dependencies),
        })

    return {
        "fold_count": len(folds),
        "folds": folds,
        "average_metrics": {
            "accuracy": _mean_metric(folds, "accuracy"),
            "balanced_accuracy": _mean_metric(folds, "balanced_accuracy"),
            "macro_f1": _mean_metric(folds, "macro_f1"),
            "log_loss": _mean_metric(folds, "log_loss"),
        },
    }


def _train_final_tuned_model(
    dataset: pd.DataFrame,
    tuning_result: dict,
    dependencies,
):
    parameters = dict(tuning_result["parameters"])
    best_n_estimators = tuning_result.get("best_n_estimators")
    if best_n_estimators:
        parameters["n_estimators"] = best_n_estimators

    labels = dataset["target_label"]
    model = _new_classifier(
        dependencies["XGBClassifier"],
        parameters_override=parameters,
    )
    model.fit(
        _feature_matrix(dataset),
        _label_ids(labels),
        sample_weight=_sample_weights(labels, tuning_result["weight_power"]),
    )
    return model, parameters


def _binary_parameters_for_candidate(candidate: dict) -> dict:
    parameters = dict(BASE_HYPERPARAMETERS)
    parameters.update(candidate["parameters"])
    parameters["objective"] = "binary:logistic"
    parameters["eval_metric"] = "logloss"
    parameters["num_class"] = None
    return parameters


def _prepare_binary_dataset(dataset: pd.DataFrame, strategy: str) -> pd.DataFrame:
    binary_dataset = dataset.copy()
    if strategy == "directional":
        binary_dataset["binary_label"] = np.where(
            binary_dataset["target_label"].eq("negative"),
            "bearish",
            "bullish",
        )
    elif strategy == "strong_only":
        binary_dataset = binary_dataset[
            binary_dataset["target_label"].isin(["negative", "positive"])
        ].copy()
        binary_dataset["binary_label"] = np.where(
            binary_dataset["target_label"].eq("negative"),
            "bearish",
            "bullish",
        )
    else:
        raise ValueError("Unknown binary strategy.")

    binary_dataset["binary_label_id"] = (
        binary_dataset["binary_label"].map(BINARY_LABEL_TO_ID).astype(int)
    )
    return binary_dataset.reset_index(drop=True)


def _binary_latest_validation_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    years = sorted(dataset["period"].dt.year.dropna().unique())
    for validation_year in reversed(years):
        training_set = dataset[dataset["period"].dt.year < validation_year].copy()
        validation_set = dataset[dataset["period"].dt.year == validation_year].copy()
        if (
            len(training_set) >= MIN_ROLLING_TRAIN_ROWS
            and len(validation_set) >= MIN_ROLLING_EVAL_ROWS
            and training_set["binary_label"].nunique() == len(BINARY_LABELS)
            and validation_set["binary_label"].nunique() == len(BINARY_LABELS)
        ):
            return training_set, validation_set
    return None, None


def _fit_binary_classifier(
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    parameters: dict,
    weight_power: float,
    dependencies,
):
    labels = training_set["binary_label"]
    model = _new_classifier(
        dependencies["XGBClassifier"],
        parameters_override=parameters,
        early_stopping=validation_set is not None,
    )
    fit_arguments = {
        "sample_weight": _sample_weights(labels, weight_power),
    }
    if validation_set is not None:
        validation_labels = validation_set["binary_label"]
        fit_arguments.update({
            "eval_set": [
                (
                    _feature_matrix(validation_set),
                    validation_set["binary_label_id"].to_numpy(),
                )
            ],
            "sample_weight_eval_set": [
                _sample_weights(validation_labels, weight_power),
            ],
            "verbose": False,
        })

    model.fit(
        _feature_matrix(training_set),
        training_set["binary_label_id"].to_numpy(),
        **fit_arguments,
    )
    return model


def _calculate_binary_metrics(actual, predicted, probabilities, dependencies) -> dict:
    return {
        "accuracy": round(
            float(dependencies["accuracy_score"](actual, predicted)),
            4,
        ),
        "balanced_accuracy": round(
            float(dependencies["balanced_accuracy_score"](actual, predicted)),
            4,
        ),
        "macro_f1": round(
            float(
                dependencies["f1_score"](
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "confusion_matrix": dependencies["confusion_matrix"](
            actual,
            predicted,
            labels=range(len(BINARY_LABELS)),
        ).tolist(),
        "label_order": BINARY_LABELS,
        "log_loss": round(
            float(
                dependencies["log_loss"](
                    actual,
                    probabilities,
                    labels=range(len(BINARY_LABELS)),
                )
            ),
            4,
        ),
    }


def _evaluate_binary_model(model, evaluation_set: pd.DataFrame, dependencies) -> dict:
    actual = evaluation_set["binary_label_id"].to_numpy()
    probabilities = model.predict_proba(_feature_matrix(evaluation_set))
    predicted = np.argmax(probabilities, axis=1)
    return _calculate_binary_metrics(actual, predicted, probabilities, dependencies)


def _binary_rolling_validation_for_parameters(
    dataset: pd.DataFrame,
    parameters: dict,
    weight_power: float,
    dependencies,
) -> dict:
    folds = []
    years = sorted(dataset["period"].dt.year.dropna().unique())

    for test_year in years:
        previous_data = dataset[dataset["period"].dt.year < test_year].copy()
        test_set = dataset[dataset["period"].dt.year == test_year].copy()
        if (
            len(previous_data) < MIN_ROLLING_TRAIN_ROWS
            or len(test_set) < MIN_ROLLING_EVAL_ROWS
            or test_set["binary_label"].nunique() != len(BINARY_LABELS)
        ):
            continue

        training_set, validation_set = _binary_latest_validation_split(previous_data)
        if training_set is None:
            continue

        model = _fit_binary_classifier(
            training_set,
            validation_set,
            parameters,
            weight_power,
            dependencies,
        )
        folds.append({
            "test_year": int(test_year),
            "train_rows": len(training_set),
            "validation_rows": len(validation_set),
            "test_rows": len(test_set),
            "best_n_estimators": _best_iteration_count(model),
            "test_class_distribution": {
                label: int(count)
                for label, count in test_set["binary_label"].value_counts().items()
            },
            "metrics": _evaluate_binary_model(model, test_set, dependencies),
        })

    return {
        "fold_count": len(folds),
        "folds": folds,
        "average_metrics": {
            "accuracy": _mean_metric(folds, "accuracy"),
            "balanced_accuracy": _mean_metric(folds, "balanced_accuracy"),
            "macro_f1": _mean_metric(folds, "macro_f1"),
            "log_loss": _mean_metric(folds, "log_loss"),
        },
    }


def _evaluate_binary_tuning_candidate(
    dataset: pd.DataFrame,
    strategy: str,
    candidate: dict,
    weight_power: float,
    dependencies,
) -> dict:
    binary_dataset = _prepare_binary_dataset(dataset, strategy)
    train_set, holdout_set = _chronological_holdout(binary_dataset)
    early_training_set, validation_set = _binary_latest_validation_split(train_set)
    if early_training_set is None:
        early_training_set = train_set
        validation_set = None

    parameters = _binary_parameters_for_candidate(candidate)
    model = _fit_binary_classifier(
        early_training_set,
        validation_set,
        parameters,
        weight_power,
        dependencies,
    )
    rolling_validation = _binary_rolling_validation_for_parameters(
        binary_dataset,
        parameters,
        weight_power,
        dependencies,
    )
    return {
        "strategy": strategy,
        "strategy_description": (
            "negative -> bearish; neutral and positive -> bullish"
            if strategy == "directional"
            else "neutral rows excluded; negative -> bearish; positive -> bullish"
        ),
        "candidate_name": candidate["name"],
        "weight_power": weight_power,
        "parameters": parameters,
        "best_n_estimators": _best_iteration_count(model),
        "dataset_rows": len(binary_dataset),
        "class_distribution": {
            label: int(count)
            for label, count in binary_dataset["binary_label"].value_counts().items()
        },
        "latest_holdout_rows": len(holdout_set),
        "latest_holdout_metrics": _evaluate_binary_model(
            model,
            holdout_set,
            dependencies,
        ),
        "rolling_validation": rolling_validation,
    }


def _rank_binary_tuning_result(result: dict) -> tuple:
    rolling = result["rolling_validation"]["average_metrics"]
    latest = result["latest_holdout_metrics"]
    return (
        rolling.get("macro_f1") or 0,
        rolling.get("balanced_accuracy") or 0,
        rolling.get("accuracy") or 0,
        latest.get("accuracy") or 0,
        -(rolling.get("log_loss") or 999),
    )


def calculate_fundamental_score(probabilities: dict) -> tuple[float, float]:
    """
    Convert class probabilities into a 1-10 fundamental outlook score.

    Positive contributes +1, neutral contributes 0, and negative contributes
    -1. The resulting -1..1 outlook uses the same mapping as sentiment scores.
    """
    positive = float(probabilities.get("positive", 0))
    negative = float(probabilities.get("negative", 0))
    raw_outlook = max(-1.0, min(1.0, positive - negative))

    if raw_outlook >= 0:
        fundamental_score = 5 + (raw_outlook * 5)
    else:
        fundamental_score = 5 + (raw_outlook * 4)

    return round(raw_outlook, 4), round(fundamental_score, 2)


def calculate_binary_fundamental_score(
    bullish_probability: float,
) -> tuple[float, float]:
    bullish_probability = max(0.0, min(1.0, float(bullish_probability)))
    bearish_probability = 1.0 - bullish_probability
    raw_outlook = max(-1.0, min(1.0, bullish_probability - bearish_probability))
    fundamental_score = max(1.0, min(10.0, bullish_probability * 10))
    return round(raw_outlook, 4), round(fundamental_score, 2)


def classify_binary_display_label(
    bullish_probability: float,
    thresholds: dict = None,
) -> str:
    thresholds = thresholds or DEFAULT_BINARY_DISPLAY_THRESHOLDS
    bearish_below = float(thresholds.get("bearish_below", 0.45))
    bullish_above = float(thresholds.get("bullish_above", 0.55))

    if bullish_probability < bearish_below:
        return "negative"
    if bullish_probability > bullish_above:
        return "positive"
    return "neutral"


def _new_version_id(trained_at: datetime, model_family: str = MODEL_FAMILY) -> str:
    timestamp = trained_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{model_family}_{timestamp}"


def _validate_version(model_version: str) -> None:
    if not VERSION_PATTERN.fullmatch(model_version):
        raise ValueError(f"Invalid financial model version: {model_version}")


def _version_paths(model_version: str) -> tuple[Path, Path]:
    _validate_version(model_version)
    version_dir = MODELS_DIR / model_version
    return version_dir / "model.ubj", version_dir / "metadata.json"


def _relative_backend_path(path: Path) -> str:
    backend_dir = Path(__file__).resolve().parents[2]
    return path.relative_to(backend_dir).as_posix()


def load_model_metadata(model_version: str) -> dict:
    _, metadata_path = _version_paths(model_version)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Financial model metadata not found for {model_version}."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_model(model_version: str = None):
    dependencies = _load_ml_dependencies()
    if model_version is None:
        if not LATEST_MANIFEST_PATH.exists():
            raise FileNotFoundError("No active local financial model. Train a model first.")
        manifest = json.loads(LATEST_MANIFEST_PATH.read_text(encoding="utf-8"))
        model_version = manifest["model_version"]

    model_path, _ = _version_paths(model_version)
    if not model_path.exists():
        raise FileNotFoundError(f"Financial model file not found for {model_version}.")

    model = dependencies["xgboost"].Booster()
    model.load_model(model_path)
    return model, load_model_metadata(model_version)


def activate_local_model(model_version: str) -> dict:
    model_path, metadata_path = _version_paths(model_version)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Local artifacts not found for {model_version}.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": model_version,
        "model_path": _relative_backend_path(model_path),
        "metadata_path": _relative_backend_path(metadata_path),
    }
    temporary_path = LATEST_MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(LATEST_MANIFEST_PATH)
    return manifest


def _save_version(model, metadata: dict) -> dict:
    model_path, metadata_path = _version_paths(metadata["model_version"])
    model_path.parent.mkdir(parents=True, exist_ok=False)
    model.save_model(model_path)

    metadata = {
        **metadata,
        "model_path": _relative_backend_path(model_path),
        "metadata_path": _relative_backend_path(metadata_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def _validate_dataset(statements: list | pd.DataFrame) -> pd.DataFrame:
    dataset = build_training_dataset(pd.DataFrame(statements))
    if len(dataset) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            f"At least {MINIMUM_TRAINING_ROWS} valid next-quarter pairs are required; "
            f"found {len(dataset)}."
        )
    return dataset


def _train_fresh(dataset: pd.DataFrame, dependencies) -> tuple[object, dict, pd.DataFrame, dict]:
    class_counts = dataset["target_label"].value_counts()
    missing_classes = [label for label in LABELS if label not in class_counts]
    if missing_classes:
        raise ValueError(
            "Training data does not contain every outlook class. Missing: "
            + ", ".join(missing_classes)
        )

    train_set, holdout_set = _chronological_holdout(dataset)
    if train_set["target_label"].nunique() != len(LABELS):
        raise ValueError("Chronological training split does not contain every outlook class.")

    early_training_set, validation_set = _latest_validation_split(train_set)
    early_stopping_used = early_training_set is not None
    if early_stopping_used:
        evaluation_model = _fit_classifier(
            early_training_set,
            dependencies,
            validation_set,
        )
        effective_estimators = _best_iteration_count(evaluation_model)
    else:
        evaluation_model = _fit_classifier(train_set, dependencies)
        effective_estimators = None

    metrics = _evaluate_model(evaluation_model, holdout_set, dependencies)
    final_labels = dataset["target_label"]
    final_parameters = {}
    if effective_estimators:
        final_parameters["n_estimators"] = effective_estimators
    final_model = _new_classifier(
        dependencies["XGBClassifier"],
        parameters_override=final_parameters,
    )
    final_model.fit(
        _feature_matrix(dataset),
        _label_ids(final_labels),
        sample_weight=_sample_weights(final_labels),
    )
    training_details = {
        "early_stopping": {
            "enabled": early_stopping_used,
            "rounds": EARLY_STOPPING_ROUNDS if early_stopping_used else None,
            "best_n_estimators": effective_estimators,
            "validation_rows": len(validation_set) if early_stopping_used else 0,
            "validation_year": (
                int(validation_set["period"].dt.year.max())
                if early_stopping_used
                else None
            ),
            "final_model_n_estimators": (
                effective_estimators
                if effective_estimators
                else BASE_HYPERPARAMETERS["n_estimators"]
            ),
        },
        "rolling_validation": _rolling_validation(dataset, dependencies),
    }
    return final_model, metrics, holdout_set, training_details


def _train_continued(
    dataset: pd.DataFrame,
    base_version: str,
    dependencies,
) -> tuple[object, dict, pd.DataFrame, dict]:
    base_model, base_metadata = load_model(base_version)
    if base_metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Base model feature columns do not match the current model.")
    if base_metadata["labels"] != LABELS:
        raise ValueError("Base model labels do not match the current model.")

    previous_dataset_end = pd.Timestamp(base_metadata["dataset_end"])
    new_dataset = dataset[dataset["period"] > previous_dataset_end].copy()
    if new_dataset.empty:
        raise ValueError(
            f"No new labeled quarters exist after {base_metadata['dataset_end']}; "
            "continuation training would only relearn old data."
        )

    actual = new_dataset["target_label"].map(LABEL_TO_ID).to_numpy()
    new_matrix = dependencies["xgboost"].DMatrix(
        _feature_matrix(new_dataset),
        label=actual,
        weight=_sample_weights(new_dataset["target_label"]),
        feature_names=FEATURE_COLUMNS,
    )
    predicted_probabilities = base_model.predict(new_matrix)
    predicted = np.argmax(predicted_probabilities, axis=1)
    metrics = _calculate_metrics(actual, predicted, dependencies)

    continuation_parameters = {
        key: value
        for key, value in BASE_HYPERPARAMETERS.items()
        if key not in {"n_estimators", "n_jobs", "random_state"}
    }
    continuation_parameters["seed"] = BASE_HYPERPARAMETERS["random_state"]
    continuation_parameters["nthread"] = BASE_HYPERPARAMETERS["n_jobs"]
    continued_model = dependencies["xgboost"].train(
        params=continuation_parameters,
        dtrain=new_matrix,
        num_boost_round=CONTINUATION_ESTIMATORS,
        xgb_model=base_model,
    )
    return continued_model, metrics, new_dataset, base_metadata


def train_model(
    statements: list | pd.DataFrame,
    training_mode: str = "fresh",
    base_version: str = None,
) -> dict:
    if training_mode not in {"fresh", "continue"}:
        raise ValueError("training_mode must be 'fresh' or 'continue'.")
    if training_mode == "continue" and not base_version:
        raise ValueError("base_version is required for continuation training.")

    dependencies = _load_ml_dependencies()
    dataset = _validate_dataset(statements)
    trained_at = datetime.now(timezone.utc)
    model_version = _new_version_id(trained_at)

    if training_mode == "fresh":
        model, metrics, evaluation_set, training_details = _train_fresh(
            dataset,
            dependencies,
        )
        training_set = dataset
        parent_version = None
        cumulative_training_rows = len(dataset)
        evaluation_mode = "rolling_year_validation_with_latest_quarter_holdout"
    else:
        model, metrics, training_set, base_metadata = _train_continued(
            dataset,
            base_version,
            dependencies,
        )
        evaluation_set = training_set
        parent_version = base_version
        cumulative_training_rows = (
            int(base_metadata["cumulative_training_rows"]) + len(training_set)
        )
        evaluation_mode = "parent_model_on_new_unseen_quarters"
        training_details = {
            "early_stopping": {"enabled": False},
            "rolling_validation": None,
        }

    class_distribution = {
        label: int(count)
        for label, count in training_set["target_label"].value_counts().items()
    }
    metadata = {
        "model_version": model_version,
        "model_family": MODEL_FAMILY,
        "parent_version": parent_version,
        "training_mode": training_mode,
        "trained_at": trained_at.isoformat(),
        "training_rows": len(training_set),
        "cumulative_training_rows": cumulative_training_rows,
        "holdout_rows": len(evaluation_set),
        "dataset_start": dataset["period"].min().date().isoformat(),
        "dataset_end": dataset["period"].max().date().isoformat(),
        "class_distribution": class_distribution,
        "hyperparameters": {
            **BASE_HYPERPARAMETERS,
            "n_estimators": (
                CONTINUATION_ESTIMATORS
                if training_mode == "continue"
                else training_details["early_stopping"].get(
                    "final_model_n_estimators",
                    BASE_HYPERPARAMETERS["n_estimators"],
                )
            ),
            "early_stopping_rounds": (
                EARLY_STOPPING_ROUNDS
                if training_details["early_stopping"].get("enabled")
                else None
            ),
        },
        "metrics": metrics,
        "training_details": training_details,
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "evaluation_mode": evaluation_mode,
        "xgboost_version": dependencies["xgboost"].__version__,
    }
    saved_metadata = _save_version(model, metadata)
    return {
        **saved_metadata,
        "warning": (
            "Prototype metrics are based on a small dataset. "
            "Continuation metrics measure the parent before it learns the new rows."
        ),
    }


def tune_model(
    statements: list | pd.DataFrame,
    save_best: bool = False,
    top_n: int = 5,
) -> dict:
    dependencies = _load_ml_dependencies()
    dataset = _validate_dataset(statements)
    results = []

    for candidate in TUNING_CANDIDATES:
        for weight_power in TUNING_WEIGHT_POWERS:
            results.append(
                _evaluate_tuning_candidate(
                    dataset,
                    candidate,
                    weight_power,
                    dependencies,
                )
            )

    results.sort(key=_rank_tuning_result, reverse=True)
    best_result = results[0]
    response = {
        "dataset_rows": len(dataset),
        "dataset_start": dataset["period"].min().date().isoformat(),
        "dataset_end": dataset["period"].max().date().isoformat(),
        "candidate_count": len(results),
        "ranking_metric": (
            "rolling macro_f1, rolling balanced_accuracy, latest holdout "
            "accuracy, then rolling log_loss"
        ),
        "best_result": best_result,
        "top_results": results[:top_n],
        "saved_model": None,
    }

    if not save_best:
        return response

    trained_at = datetime.now(timezone.utc)
    model_version = _new_version_id(trained_at)
    final_model, final_parameters = _train_final_tuned_model(
        dataset,
        best_result,
        dependencies,
    )
    class_distribution = {
        label: int(count)
        for label, count in dataset["target_label"].value_counts().items()
    }
    metadata = {
        "model_version": model_version,
        "model_family": MODEL_FAMILY,
        "parent_version": None,
        "training_mode": "fresh",
        "trained_at": trained_at.isoformat(),
        "training_rows": len(dataset),
        "cumulative_training_rows": len(dataset),
        "holdout_rows": best_result["latest_holdout_rows"],
        "dataset_start": response["dataset_start"],
        "dataset_end": response["dataset_end"],
        "class_distribution": class_distribution,
        "hyperparameters": {
            **final_parameters,
            "weight_power": best_result["weight_power"],
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        },
        "metrics": best_result["latest_holdout_metrics"],
        "training_details": {
            "tuning": {
                "candidate_count": len(results),
                "ranking_metric": response["ranking_metric"],
                "best_candidate_name": best_result["candidate_name"],
                "best_weight_power": best_result["weight_power"],
            },
            "rolling_validation": best_result["rolling_validation"],
        },
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "evaluation_mode": "parameter_tuning_with_rolling_year_validation",
        "xgboost_version": dependencies["xgboost"].__version__,
    }
    response["saved_model"] = _save_version(final_model, metadata)
    return response


def tune_binary_model(
    statements: list | pd.DataFrame,
    top_n: int = 5,
) -> dict:
    dependencies = _load_ml_dependencies()
    dataset = _validate_dataset(statements)
    results = []

    for strategy in ("directional", "strong_only"):
        for candidate in TUNING_CANDIDATES:
            for weight_power in TUNING_WEIGHT_POWERS:
                results.append(
                    _evaluate_binary_tuning_candidate(
                        dataset,
                        strategy,
                        candidate,
                        weight_power,
                        dependencies,
                    )
                )

    results.sort(key=_rank_binary_tuning_result, reverse=True)
    return {
        "source_dataset_rows": len(dataset),
        "source_dataset_start": dataset["period"].min().date().isoformat(),
        "source_dataset_end": dataset["period"].max().date().isoformat(),
        "candidate_count": len(results),
        "label_order": BINARY_LABELS,
        "ranking_metric": (
            "rolling binary macro_f1, rolling balanced_accuracy, rolling "
            "accuracy, latest holdout accuracy, then rolling log_loss"
        ),
        "note": (
            "Comparison-only experiment. This endpoint does not save, activate, "
            "or change the current 3-class financial model."
        ),
        "best_result": results[0],
        "top_results": results[:top_n],
    }


def train_binary_model(
    statements: list | pd.DataFrame,
    top_n: int = 5,
) -> dict:
    dependencies = _load_ml_dependencies()
    dataset = _validate_dataset(statements)
    tuned = tune_binary_model(statements, top_n=top_n)
    best_result = tuned["best_result"]
    binary_dataset = _prepare_binary_dataset(dataset, best_result["strategy"])

    parameters = dict(best_result["parameters"])
    best_n_estimators = best_result.get("best_n_estimators")
    if best_n_estimators:
        parameters["n_estimators"] = best_n_estimators

    labels = binary_dataset["binary_label"]
    model = _new_classifier(
        dependencies["XGBClassifier"],
        parameters_override=parameters,
    )
    model.fit(
        _feature_matrix(binary_dataset),
        binary_dataset["binary_label_id"].to_numpy(),
        sample_weight=_sample_weights(labels, best_result["weight_power"]),
    )

    trained_at = datetime.now(timezone.utc)
    model_version = _new_version_id(trained_at, BINARY_MODEL_FAMILY)
    metadata = {
        "model_version": model_version,
        "model_family": BINARY_MODEL_FAMILY,
        "parent_version": None,
        "training_mode": "fresh",
        "trained_at": trained_at.isoformat(),
        "training_rows": len(binary_dataset),
        "cumulative_training_rows": len(binary_dataset),
        "holdout_rows": best_result["latest_holdout_rows"],
        "dataset_start": binary_dataset["period"].min().date().isoformat(),
        "dataset_end": binary_dataset["period"].max().date().isoformat(),
        "class_distribution": {
            label: int(count)
            for label, count in binary_dataset["binary_label"].value_counts().items()
        },
        "hyperparameters": {
            **parameters,
            "weight_power": best_result["weight_power"],
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "display_thresholds": {
                "bearish_below": 0.45,
                "neutral_between": [0.45, 0.55],
                "bullish_above": 0.55,
            },
        },
        "metrics": best_result["latest_holdout_metrics"],
        "training_details": {
            "binary_strategy": best_result["strategy"],
            "binary_strategy_description": best_result["strategy_description"],
            "tuning": {
                "candidate_count": tuned["candidate_count"],
                "ranking_metric": tuned["ranking_metric"],
                "best_candidate_name": best_result["candidate_name"],
                "best_weight_power": best_result["weight_power"],
            },
            "rolling_validation": best_result["rolling_validation"],
            "note": (
                "Binary financial model. When activated, the normal financial "
                "prediction pipeline converts bullish probability into a "
                "negative, neutral, or positive display label."
            ),
        },
        "feature_columns": FEATURE_COLUMNS,
        "labels": BINARY_LABELS,
        "evaluation_mode": "binary_parameter_tuning_with_rolling_year_validation",
        "xgboost_version": dependencies["xgboost"].__version__,
    }
    saved_metadata = _save_version(model, metadata)
    return {
        "source_dataset_rows": len(dataset),
        "saved_model": saved_metadata,
        "best_result": best_result,
        "top_results": tuned["top_results"],
        "warning": (
            "Saved as a separate binary model family. Activate this model "
            "version to use it in the normal financial prediction pipeline."
        ),
    }


def predict_latest(
    statements: list | pd.DataFrame,
    model_version: str = None,
) -> dict:
    model, metadata = load_model(model_version)
    features = engineer_features(pd.DataFrame(statements))
    if features.empty:
        raise ValueError("No valid quarterly financial statements are available.")

    stock_count = features["stock_id"].nunique()
    if stock_count != 1:
        raise ValueError("Prediction input must contain statements for exactly one stock.")

    latest = features.sort_values("period").iloc[-1]
    X = latest[metadata["feature_columns"]].to_frame().T
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    dependencies = _load_ml_dependencies()
    prediction_matrix = dependencies["xgboost"].DMatrix(
        X,
        feature_names=metadata["feature_columns"],
    )
    probabilities = model.predict(prediction_matrix)[0]
    prediction_id = int(np.argmax(probabilities))
    prediction = metadata["labels"][prediction_id]
    probability_map = {
        label: round(float(probabilities[index]), 4)
        for index, label in enumerate(metadata["labels"])
    }
    confidence = float(probabilities[prediction_id])
    raw_outlook, fundamental_score = calculate_fundamental_score(
        probability_map
    )

    return {
        "stock_id": int(latest["stock_id"]),
        "ticker": str(latest["ticker"]).upper(),
        "prediction": prediction,
        "score": round(confidence, 4),
        "confidence": round(confidence * 100, 2),
        "model_version": metadata["model_version"],
        "period": latest["period"].date().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probabilities": probability_map,
        "raw_outlook": raw_outlook,
        "fundamental_score": fundamental_score,
    }


def predict_latest_binary(
    statements: list | pd.DataFrame,
    model_version: str,
) -> dict:
    model, metadata = load_model(model_version)
    if metadata.get("model_family") != BINARY_MODEL_FAMILY:
        raise ValueError(f"{model_version} is not a binary financial model.")
    if metadata.get("labels") != BINARY_LABELS:
        raise ValueError(f"{model_version} does not use the expected binary labels.")

    features = engineer_features(pd.DataFrame(statements))
    if features.empty:
        raise ValueError("No valid quarterly financial statements are available.")

    stock_count = features["stock_id"].nunique()
    if stock_count != 1:
        raise ValueError("Prediction input must contain statements for exactly one stock.")

    latest = features.sort_values("period").iloc[-1]
    X = latest[metadata["feature_columns"]].to_frame().T
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    dependencies = _load_ml_dependencies()
    prediction_matrix = dependencies["xgboost"].DMatrix(
        X,
        feature_names=metadata["feature_columns"],
    )
    raw_prediction = np.asarray(model.predict(prediction_matrix)[0])
    if raw_prediction.ndim == 0:
        bullish_probability = float(raw_prediction)
    else:
        bullish_probability = float(raw_prediction[BINARY_LABEL_TO_ID["bullish"]])

    bullish_probability = max(0.0, min(1.0, bullish_probability))
    bearish_probability = 1.0 - bullish_probability
    probability_map = {
        "bearish": round(bearish_probability, 4),
        "bullish": round(bullish_probability, 4),
    }
    binary_prediction = "bullish" if bullish_probability >= 0.5 else "bearish"
    thresholds = (
        metadata.get("hyperparameters", {}).get("display_thresholds")
        or DEFAULT_BINARY_DISPLAY_THRESHOLDS
    )
    display_prediction = classify_binary_display_label(
        bullish_probability,
        thresholds,
    )
    confidence = max(bullish_probability, bearish_probability)
    raw_outlook, fundamental_score = calculate_binary_fundamental_score(
        bullish_probability
    )

    return {
        "stock_id": int(latest["stock_id"]),
        "ticker": str(latest["ticker"]).upper(),
        "prediction": display_prediction,
        "binary_prediction": binary_prediction,
        "score": round(confidence, 4),
        "confidence": round(confidence * 100, 2),
        "model_version": metadata["model_version"],
        "period": latest["period"].date().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probabilities": probability_map,
        "bullish_probability": probability_map["bullish"],
        "bearish_probability": probability_map["bearish"],
        "raw_outlook": raw_outlook,
        "fundamental_score": fundamental_score,
        "display_thresholds": thresholds,
    }
