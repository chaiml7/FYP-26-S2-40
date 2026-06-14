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
MODEL_FAMILY = "xgboost_financial"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "financial"
LATEST_MANIFEST_PATH = MODELS_DIR / "latest.json"
MINIMUM_TRAINING_ROWS = 20
VERSION_PATTERN = re.compile(r"^xgboost_financial_\d{8}T\d{6}\d{6}Z$")

BASE_HYPERPARAMETERS = {
    "objective": "multi:softprob",
    "num_class": len(LABELS),
    "n_estimators": 120,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": 0,
}
CONTINUATION_ESTIMATORS = 30


def _load_ml_dependencies():
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
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
    }


def _new_classifier(XGBClassifier, continuation: bool = False):
    parameters = dict(BASE_HYPERPARAMETERS)
    if continuation:
        parameters["n_estimators"] = CONTINUATION_ESTIMATORS
    return XGBClassifier(**parameters)


def _sample_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    total = len(labels)
    class_count = len(counts)
    return labels.map(lambda label: total / (class_count * counts[label])).to_numpy()


def _feature_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def _chronological_holdout(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_indices = dataset.groupby("stock_id")["period"].idxmax()
    holdout = dataset.loc[holdout_indices].sort_values("period").copy()
    train = dataset.drop(index=holdout_indices).sort_values("period").copy()
    return train, holdout


def _calculate_metrics(actual, predicted, dependencies) -> dict:
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
            labels=range(len(LABELS)),
        ).tolist(),
        "label_order": LABELS,
    }


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


def _new_version_id(trained_at: datetime) -> str:
    timestamp = trained_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{MODEL_FAMILY}_{timestamp}"


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


def _train_fresh(dataset: pd.DataFrame, dependencies) -> tuple[object, dict, pd.DataFrame]:
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

    train_labels = train_set["target_label"]
    evaluation_model = _new_classifier(dependencies["XGBClassifier"])
    evaluation_model.fit(
        _feature_matrix(train_set),
        train_labels.map(LABEL_TO_ID).to_numpy(),
        sample_weight=_sample_weights(train_labels),
    )
    actual = holdout_set["target_label"].map(LABEL_TO_ID).to_numpy()
    predicted = evaluation_model.predict(_feature_matrix(holdout_set))

    final_labels = dataset["target_label"]
    final_model = _new_classifier(dependencies["XGBClassifier"])
    final_model.fit(
        _feature_matrix(dataset),
        final_labels.map(LABEL_TO_ID).to_numpy(),
        sample_weight=_sample_weights(final_labels),
    )
    metrics = _calculate_metrics(actual, predicted, dependencies)
    return final_model, metrics, holdout_set


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
        model, metrics, evaluation_set = _train_fresh(dataset, dependencies)
        training_set = dataset
        parent_version = None
        cumulative_training_rows = len(dataset)
        evaluation_mode = "chronological_latest_quarter_holdout"
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
                else BASE_HYPERPARAMETERS["n_estimators"]
            ),
        },
        "metrics": metrics,
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
