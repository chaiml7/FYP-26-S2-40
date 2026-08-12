"""Versioned three-class LightGBM training, evaluation, and inference."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.services.technical.feature_engineering import (
    FEATURE_COLUMNS,
    RETURN_THRESHOLD,
    build_training_dataset,
    engineer_model_features,
)


LABELS = ["bearish", "neutral", "bullish"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MODEL_FAMILY = "lightgbm_technical"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "technical"
LATEST_MANIFEST_PATH = MODELS_DIR / "latest.json"
VERSION_PATTERN = re.compile(r"^lightgbm_technical_\d{8}T\d{6}\d{6}Z$")
MINIMUM_TRAINING_ROWS = 500
MINIMUM_UNIQUE_DATES = 250

PARAMETER_CANDIDATES = [
    {
        "n_estimators": 250,
        "learning_rate": 0.04,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 40,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.025,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 35,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.02,
        "max_depth": 5,
        "num_leaves": 23,
        "min_child_samples": 30,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
]


def _load_ml_dependencies() -> dict:
    try:
        import lightgbm
        from lightgbm import Booster, LGBMClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            log_loss,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Technical ML dependencies are missing. Install "
            "backend/requirements-ml.txt in the active environment."
        ) from exc

    return {
        "lightgbm": lightgbm,
        "Booster": Booster,
        "LGBMClassifier": LGBMClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "log_loss": log_loss,
    }


def calculate_technical_score(probabilities: dict) -> tuple[float, float]:
    """Convert bullish and bearish probability balance to a 1-10 score."""
    bullish = float(probabilities.get("bullish", 0))
    bearish = float(probabilities.get("bearish", 0))
    raw_outlook = max(-1.0, min(1.0, bullish - bearish))
    if raw_outlook >= 0:
        score = 5 + (raw_outlook * 5)
    else:
        score = 5 + (raw_outlook * 4)
    return round(raw_outlook, 4), round(score, 2)


def split_dataset_by_date(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_dates: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole trading dates into train, validation, and untouched test."""
    unique_dates = pd.Index(sorted(dataset["date"].dropna().unique()))
    if len(unique_dates) < MINIMUM_UNIQUE_DATES:
        raise ValueError(
            f"At least {MINIMUM_UNIQUE_DATES} unique trading dates are required; "
            f"found {len(unique_dates)}."
        )

    train_end = int(len(unique_dates) * train_fraction)
    validation_end = int(
        len(unique_dates) * (train_fraction + validation_fraction)
    )
    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[
        train_end + embargo_dates:validation_end
    ]
    test_dates = unique_dates[validation_end + embargo_dates:]

    if not len(train_dates) or not len(validation_dates) or not len(test_dates):
        raise ValueError("Chronological split produced an empty dataset partition.")

    train = dataset[dataset["date"].isin(train_dates)].copy()
    validation = dataset[dataset["date"].isin(validation_dates)].copy()
    test = dataset[dataset["date"].isin(test_dates)].copy()
    return train, validation, test


def _new_classifier(LGBMClassifier, parameters: dict):
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        class_weight="balanced",
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
        **parameters,
    )


def _matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")


def _labels(dataset: pd.DataFrame) -> np.ndarray:
    return dataset["target_label"].map(LABEL_TO_ID).to_numpy()


def _calculate_metrics(actual, probabilities, dependencies) -> dict:
    predicted = np.argmax(probabilities, axis=1)
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
        "log_loss": round(
            float(
                dependencies["log_loss"](
                    actual,
                    probabilities,
                    labels=range(len(LABELS)),
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


def _ensure_all_classes(dataset: pd.DataFrame, partition: str) -> None:
    missing = [
        label
        for label in LABELS
        if label not in set(dataset["target_label"])
    ]
    if missing:
        raise ValueError(
            f"{partition} data does not contain every class. Missing: "
            + ", ".join(missing)
        )


def _select_parameters(train, validation, dependencies) -> tuple[dict, dict]:
    _ensure_all_classes(train, "Training")
    validation_actual = _labels(validation)
    best_parameters = None
    best_metrics = None
    best_key = (-1.0, -1.0, float("-inf"))

    for parameters in PARAMETER_CANDIDATES:
        model = _new_classifier(dependencies["LGBMClassifier"], parameters)
        model.fit(_matrix(train), _labels(train))
        metrics = _calculate_metrics(
            validation_actual,
            model.predict_proba(_matrix(validation)),
            dependencies,
        )
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -metrics["log_loss"],
        )
        if key > best_key:
            best_key = key
            best_parameters = parameters
            best_metrics = metrics

    return dict(best_parameters), dict(best_metrics)


def _new_version_id(trained_at: datetime) -> str:
    return (
        f"{MODEL_FAMILY}_"
        f"{trained_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )


def _validate_version(model_version: str) -> None:
    if not VERSION_PATTERN.fullmatch(model_version):
        raise ValueError(f"Invalid technical model version: {model_version}")


def _version_paths(model_version: str) -> tuple[Path, Path]:
    _validate_version(model_version)
    version_dir = MODELS_DIR / model_version
    return version_dir / "model.txt", version_dir / "metadata.json"


def _relative_backend_path(path: Path) -> str:
    backend_dir = Path(__file__).resolve().parents[2]
    return path.relative_to(backend_dir).as_posix()


def train_model(indicators: list | pd.DataFrame) -> dict:
    dependencies = _load_ml_dependencies()
    dataset = build_training_dataset(pd.DataFrame(indicators))
    if len(dataset) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            f"At least {MINIMUM_TRAINING_ROWS} complete rows are required; "
            f"found {len(dataset)}."
        )

    train, validation, test = split_dataset_by_date(dataset)
    parameters, validation_metrics = _select_parameters(
        train,
        validation,
        dependencies,
    )

    evaluation_train = pd.concat([train, validation], ignore_index=True)
    _ensure_all_classes(evaluation_train, "Train and validation")
    evaluation_model = _new_classifier(
        dependencies["LGBMClassifier"],
        parameters,
    )
    evaluation_model.fit(_matrix(evaluation_train), _labels(evaluation_train))
    test_metrics = _calculate_metrics(
        _labels(test),
        evaluation_model.predict_proba(_matrix(test)),
        dependencies,
    )

    _ensure_all_classes(dataset, "Full training")
    final_model = _new_classifier(
        dependencies["LGBMClassifier"],
        parameters,
    )
    final_model.fit(_matrix(dataset), _labels(dataset))

    trained_at = datetime.now(timezone.utc)
    model_version = _new_version_id(trained_at)
    model_path, metadata_path = _version_paths(model_version)
    model_path.parent.mkdir(parents=True, exist_ok=False)
    final_model.booster_.save_model(str(model_path))

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
        "class_distribution": {
            label: int(count)
            for label, count in dataset["target_label"].value_counts().items()
        },
        "hyperparameters": parameters,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "return_threshold": RETURN_THRESHOLD,
        "evaluation_mode": (
            "date_grouped_train_validation_test_with_one_date_embargo"
        ),
        "lightgbm_version": dependencies["lightgbm"].__version__,
        "model_path": _relative_backend_path(model_path),
        "metadata_path": _relative_backend_path(metadata_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def load_model_metadata(model_version: str) -> dict:
    _, metadata_path = _version_paths(model_version)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Technical model metadata not found for {model_version}."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_model(model_version: str):
    dependencies = _load_ml_dependencies()
    model_path, _ = _version_paths(model_version)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Technical model file not found for {model_version}."
        )
    return (
        dependencies["Booster"](model_file=str(model_path)),
        load_model_metadata(model_version),
    )


def activate_local_model(model_version: str) -> dict:
    model_path, metadata_path = _version_paths(model_version)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Local technical model artifacts not found for {model_version}."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": model_version,
        "model_path": _relative_backend_path(model_path),
        "metadata_path": _relative_backend_path(metadata_path),
    }
    temporary = LATEST_MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(LATEST_MANIFEST_PATH)
    return manifest


def predict_latest(
    indicators: list | pd.DataFrame,
    model_version: str,
) -> list[dict]:
    model, metadata = load_model(model_version)
    features = engineer_model_features(pd.DataFrame(indicators))
    complete = features.dropna(subset=metadata["feature_columns"]).copy()
    if complete.empty:
        raise ValueError("No complete technical indicator rows are available.")

    latest = (
        complete.sort_values(["stock_id", "date"])
        .groupby("stock_id", as_index=False, sort=False)
        .tail(1)
    )
    probabilities = model.predict(latest[metadata["feature_columns"]])
    created_at = datetime.now(timezone.utc).isoformat()
    predictions = []

    for position, (_, row) in enumerate(latest.iterrows()):
        probability_map = {
            label: round(float(probabilities[position][index]), 4)
            for index, label in enumerate(metadata["labels"])
        }
        prediction = max(probability_map, key=probability_map.get)
        raw_outlook, technical_score = calculate_technical_score(
            probability_map
        )
        predictions.append({
            "stock_id": int(row["stock_id"]),
            "symbol": str(row["symbol"]).upper(),
            "latest_date": row["date"].date().isoformat(),
            "latest_close": float(row["close"]),
            "prediction": prediction,
            "probabilities": probability_map,
            "raw_outlook": raw_outlook,
            "technical_score": technical_score,
            "prediction_horizon": "next_trading_day",
            "model_version": metadata["model_version"],
            "created_at": created_at,
        })

    return predictions
