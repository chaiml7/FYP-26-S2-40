from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
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
    roc_auc_score,
)

from services.technical.model_service import FEATURES, TARGET_RETURN_THRESHOLD
from services.technical.xgboost_model_service import (
    DEFAULT_LOOKAHEAD_DAYS,
    chronological_splits,
    date_range_summary,
    filter_training_history,
    normalize_optional_date,
    row_date,
    sort_for_chronological_training,
    sort_for_target_creation,
    target_group_columns,
)

CATBOOST_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "technical_catboost_model.joblib"
)

CATBOOST_CLASS_LABELS = {
    0: "down_equal",
    1: "up",
}

CATBOOST_OPTIONAL_CATEGORICAL_FEATURES = ["symbol"]
DEFAULT_CATBOOST_MAX_FEATURES = 45
CATBOOST_DECISION_THRESHOLD_GRID = [
    round(value, 2) for value in np.arange(0.40, 0.611, 0.02)
]

CATBOOST_PARAM_CANDIDATES = [
    {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.04,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "rsm": 0.85,
    },
    {
        "iterations": 450,
        "depth": 3,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "rsm": 0.90,
    },
    {
        "iterations": 250,
        "depth": 5,
        "learning_rate": 0.035,
        "l2_leaf_reg": 7.0,
        "random_strength": 0.75,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.80,
        "rsm": 0.80,
    },
]


def train_catboost_artifact(
    indicator_df: pd.DataFrame,
    model_scope: str,
    trained_symbol: str | None = None,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    train_before_date: str | None = None,
    n_splits: int = 5,
    max_features: int = DEFAULT_CATBOOST_MAX_FEATURES,
    artifact_path: Path | str = CATBOOST_ARTIFACT_PATH,
) -> dict[str, Any]:
    filtered_df = filter_training_history(indicator_df, train_before_date)
    X, y, clean_df, feature_names, categorical_features = prepare_catboost_training_data(
        filtered_df,
        lookahead_days=lookahead_days,
        threshold=threshold,
    )
    if clean_df.empty:
        return {
            "status": "no_data",
            "reason": "Not enough complete rows to train the CatBoost model",
            "indicator_rows": len(filtered_df),
        }
    if y.nunique() < 2:
        return {
            "status": "no_data",
            "reason": "CatBoost needs both up and down/equal target classes",
            "indicator_rows": len(filtered_df),
            "clean_training_rows": int(len(clean_df)),
            "target_distribution": target_distribution(y),
        }

    tuning_result = tune_catboost_classifier(
        X,
        y,
        categorical_features=categorical_features,
        n_splits=n_splits,
        max_features=max_features,
    )
    best_params = tuning_result.get("best_params", CATBOOST_PARAM_CANDIDATES[0])
    validation = walk_forward_catboost_validation(
        X,
        y,
        clean_df,
        categorical_features=categorical_features,
        model_params=best_params,
        n_splits=n_splits,
        max_features=max_features,
    )

    selected_features, feature_importance = select_catboost_features(
        X,
        y,
        model_params=best_params,
        categorical_features=categorical_features,
        max_features=max_features,
    )
    selected_categorical_features = [
        feature for feature in categorical_features if feature in selected_features
    ]
    classifier = get_catboost_classifier(best_params)
    classifier.fit(
        X[selected_features],
        y,
        cat_features=selected_categorical_features,
    )

    training_date_range = date_range_summary(clean_df)
    symbols = (
        sorted(str(value) for value in clean_df["symbol"].dropna().unique())
        if "symbol" in clean_df.columns
        else []
    )
    distribution = target_distribution(y)
    metadata = {
        "model_family": "catboost_binary",
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols": symbols,
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "decision_threshold": validation["decision_threshold"],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "feature_count": len(feature_names),
        "categorical_features": selected_categorical_features,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_features": feature_importance[:15],
        "tuned_params": best_params,
        "tuning": tuning_result,
        "metrics": validation["metrics"],
        "target_distribution": distribution,
        "class_labels": CATBOOST_CLASS_LABELS,
        "training_rows": int(len(clean_df)),
    }
    saved_path = save_catboost_artifact(
        classifier=classifier,
        metadata=metadata,
        path=artifact_path,
    )

    return {
        "status": "ok",
        "model_artifact_path": str(saved_path),
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols_trained": symbols,
        "symbols_trained_count": len(symbols),
        "indicator_rows": len(filtered_df),
        "clean_training_rows": int(len(clean_df)),
        "target_distribution": distribution,
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "decision_threshold": validation["decision_threshold"],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "selected_feature_count": len(selected_features),
        "categorical_features": selected_categorical_features,
        "metrics": validation["metrics"],
        "top_features": feature_importance[:15],
        "tuned_params": best_params,
    }


def prepare_catboost_training_data(
    df: pd.DataFrame,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[str]]:
    feature_names = resolve_catboost_features(df, features)
    if df is None or df.empty:
        return (
            pd.DataFrame(columns=feature_names),
            pd.Series(dtype=int),
            pd.DataFrame(),
            feature_names,
            [],
        )

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    lookahead_days = max(1, int(lookahead_days))
    clean_df = sort_for_target_creation(df)
    group_columns = target_group_columns(clean_df)
    if group_columns:
        future_close = clean_df.groupby(group_columns, sort=False)["close"].shift(
            -lookahead_days
        )
    else:
        future_close = clean_df["close"].shift(-lookahead_days)

    clean_df["future_close"] = future_close
    clean_df["future_close_return"] = future_close / clean_df["close"] - 1
    clean_df["catboost_target_direction"] = (
        clean_df["future_close_return"] > threshold
    ).astype(int)
    clean_df.loc[clean_df["future_close_return"].isna(), "catboost_target_direction"] = pd.NA
    clean_df = sort_for_chronological_training(clean_df)

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(
        subset=feature_names + ["catboost_target_direction"]
    ).copy()
    clean_df["catboost_target_direction"] = clean_df[
        "catboost_target_direction"
    ].astype(int)
    clean_df = clean_df.reset_index(drop=True)

    X = clean_df[feature_names].copy()
    categorical_features = [
        feature
        for feature in CATBOOST_OPTIONAL_CATEGORICAL_FEATURES
        if feature in X.columns
    ]
    X = normalize_catboost_feature_frame(X, categorical_features)
    y = clean_df["catboost_target_direction"].copy()
    return X, y, clean_df, feature_names, categorical_features


def resolve_catboost_features(
    df: pd.DataFrame | None,
    features: list[str] | None = None,
) -> list[str]:
    if features is not None:
        return features

    feature_names = list(FEATURES)
    if df is not None:
        for feature in CATBOOST_OPTIONAL_CATEGORICAL_FEATURES:
            if feature in df.columns and feature not in feature_names:
                feature_names.append(feature)
    return feature_names


def tune_catboost_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    n_splits: int = 5,
    max_features: int = DEFAULT_CATBOOST_MAX_FEATURES,
) -> dict[str, Any]:
    evaluations = []
    best_params = CATBOOST_PARAM_CANDIDATES[0]
    best_score = -np.inf

    for params in CATBOOST_PARAM_CANDIDATES:
        fold_scores = []
        for train_index, test_index in chronological_splits(X, n_splits):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if y_train.nunique() < 2:
                continue

            selected_features, _ = select_catboost_features(
                X_train,
                y_train,
                model_params=params,
                categorical_features=categorical_features,
                max_features=max_features,
            )
            selected_categorical_features = [
                feature for feature in categorical_features if feature in selected_features
            ]
            model = get_catboost_classifier(params)
            model.fit(
                X_train[selected_features],
                y_train,
                cat_features=selected_categorical_features,
            )
            y_pred = model.predict(X_test[selected_features]).astype(int)
            fold_scores.append(
                {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                }
            )

        accuracy = average_metric(fold_scores, "accuracy")
        f1 = average_metric(fold_scores, "f1_score")
        evaluation = {
            "params": params,
            "accuracy": accuracy,
            "f1_score": f1,
        }
        evaluations.append(evaluation)
        comparable_score = f1 if f1 is not None else -np.inf
        if comparable_score > best_score:
            best_score = comparable_score
            best_params = params

    return {
        "best_params": best_params,
        "best_f1_score": None if best_score == -np.inf else float(best_score),
        "evaluations": evaluations,
    }


def walk_forward_catboost_validation(
    X: pd.DataFrame,
    y: pd.Series,
    clean_df: pd.DataFrame,
    categorical_features: list[str],
    model_params: dict[str, Any],
    n_splits: int = 5,
    max_features: int = DEFAULT_CATBOOST_MAX_FEATURES,
) -> dict[str, Any]:
    fold_results = []
    all_true = []
    all_proba = []

    for fold_number, (train_index, test_index) in enumerate(
        chronological_splits(X, n_splits),
        start=1,
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if y_train.nunique() < 2:
            continue

        selected_features, _ = select_catboost_features(
            X_train,
            y_train,
            model_params=model_params,
            categorical_features=categorical_features,
            max_features=max_features,
        )
        selected_categorical_features = [
            feature for feature in categorical_features if feature in selected_features
        ]
        classifier = get_catboost_classifier(model_params)
        classifier.fit(
            X_train[selected_features],
            y_train,
            cat_features=selected_categorical_features,
        )
        probability_up = positive_class_probability(
            classifier,
            X_test[selected_features],
        )
        y_pred = (probability_up >= 0.5).astype(int)
        majority_class = int(y_train.mode().iloc[0])
        majority_pred = np.full(len(y_test), majority_class, dtype=int)

        fold_results.append(
            {
                "fold": fold_number,
                "test_start_date": row_date(clean_df, int(test_index[0])),
                "test_end_date": row_date(clean_df, int(test_index[-1])),
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                **binary_classification_metrics(y_test, y_pred, probability_up),
                "majority_baseline_accuracy": float(
                    accuracy_score(y_test, majority_pred)
                ),
            }
        )
        all_true.append(y_test.to_numpy(dtype=int))
        all_proba.append(probability_up)

    if not fold_results:
        return {
            "metrics": empty_binary_metrics("no_validation_folds"),
            "decision_threshold": 0.5,
            "folds": [],
        }

    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    decision_threshold = find_best_catboost_decision_threshold(y_true, y_proba)
    threshold_pred = (y_proba >= decision_threshold).astype(int)

    return {
        "metrics": {
            **binary_classification_metrics(y_true, threshold_pred, y_proba),
            "majority_baseline_accuracy": average_metric(
                fold_results,
                "majority_baseline_accuracy",
            ),
            "plain_0_5_accuracy": average_metric(fold_results, "accuracy"),
            "plain_0_5_f1_score": average_metric(fold_results, "f1_score"),
            "confusion_matrix": confusion_matrix(
                y_true,
                threshold_pred,
                labels=[0, 1],
            ).tolist(),
            "validation_folds": len(fold_results),
        },
        "decision_threshold": decision_threshold,
        "folds": fold_results,
    }


def get_catboost_classifier(model_params: dict[str, Any] | None = None) -> Any:
    from catboost import CatBoostClassifier

    params = {
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "thread_count": -1,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(model_params or {})
    return CatBoostClassifier(**params)


def select_catboost_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any],
    categorical_features: list[str],
    max_features: int = DEFAULT_CATBOOST_MAX_FEATURES,
) -> tuple[list[str], list[dict[str, float]]]:
    feature_names = list(X_train.columns)
    if len(feature_names) <= max_features or y_train.nunique() < 2:
        return feature_names, []

    selector = get_catboost_classifier(model_params)
    selector.fit(
        X_train,
        y_train,
        cat_features=[
            feature for feature in categorical_features if feature in X_train.columns
        ],
    )
    importances = feature_importance_from_catboost(selector, feature_names)
    selected = [
        item["feature"]
        for item in importances
        if item["importance"] > 0
    ][:max_features]
    if not selected:
        selected = feature_names[:max_features]

    selected_set = set(selected)
    return selected, [item for item in importances if item["feature"] in selected_set]


def feature_importance_from_catboost(
    model: Any,
    features: list[str],
) -> list[dict[str, float]]:
    raw_importance = model.get_feature_importance()
    importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(features, raw_importance)
    ]
    return sorted(importance, key=lambda item: item["importance"], reverse=True)


def predict_with_catboost_artifact(
    artifact: dict[str, Any],
    feature_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    classifier = artifact["classifier"]
    metadata = artifact.get("metadata", {})
    selected_features = metadata.get("selected_features", FEATURES)
    categorical_features = metadata.get("categorical_features", [])
    decision_threshold = float(metadata.get("decision_threshold") or 0.5)

    ready_rows = feature_rows.replace([np.inf, -np.inf], np.nan)
    ready_rows = ready_rows.dropna(subset=selected_features).copy()
    if ready_rows.empty:
        return []

    X = normalize_catboost_feature_frame(
        ready_rows[selected_features],
        [feature for feature in categorical_features if feature in selected_features],
    )
    probability_up = positive_class_probability(classifier, X)
    predictions = []
    for row_index, (_, row) in enumerate(ready_rows.iterrows()):
        predicted_class = int(probability_up[row_index] >= decision_threshold)
        confidence = (
            probability_up[row_index]
            if predicted_class == 1
            else 1 - probability_up[row_index]
        )
        predictions.append(
            {
                "stock_id": int(row["stock_id"]) if "stock_id" in row and not pd.isna(row["stock_id"]) else None,
                "symbol": str(row["symbol"]).upper() if "symbol" in row else None,
                "as_of_date": str(row["date"]),
                "as_of_close": float(row["close"]),
                "predicted_class": predicted_class,
                "predicted_direction": CATBOOST_CLASS_LABELS[predicted_class],
                "confidence": float(confidence),
                "probability_up": float(probability_up[row_index]),
                "decision_threshold": decision_threshold,
                "lookahead_days": metadata.get("lookahead_days"),
                "direction_threshold": metadata.get("direction_threshold"),
            }
        )

    return predictions


def save_catboost_artifact(
    classifier: Any,
    metadata: dict[str, Any],
    path: Path | str = CATBOOST_ARTIFACT_PATH,
) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": classifier,
        "metadata": {
            **metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        },
    }
    joblib.dump(payload, artifact_path)
    return artifact_path


def load_catboost_artifact(
    path: Path | str = CATBOOST_ARTIFACT_PATH,
) -> dict[str, Any]:
    return joblib.load(Path(path))


def normalize_catboost_feature_frame(
    X: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    result = X.copy()
    for feature in categorical_features:
        if feature in result.columns:
            result[feature] = result[feature].fillna("UNKNOWN").astype(str)
    return result


def positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def find_best_catboost_decision_threshold(
    y_true: np.ndarray,
    probability_up: np.ndarray,
) -> float:
    best_threshold = 0.5
    best_score = -np.inf
    best_accuracy = -np.inf

    for threshold in CATBOOST_DECISION_THRESHOLD_GRID:
        y_pred = (probability_up >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        if (f1, accuracy) > (best_score, best_accuracy):
            best_score = f1
            best_accuracy = accuracy
            best_threshold = threshold

    return float(best_threshold)


def binary_classification_metrics(
    y_true: Any,
    y_pred: Any,
    probability_up: Any | None = None,
) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if probability_up is not None and len(set(np.asarray(y_true).tolist())) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probability_up))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def empty_binary_metrics(reason: str) -> dict[str, Any]:
    return {
        "accuracy": None,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "mcc": None,
        "roc_auc": None,
        "reason": reason,
    }


def average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def target_distribution(y: pd.Series) -> dict[str, int]:
    return {
        CATBOOST_CLASS_LABELS[int(label)]: int(count)
        for label, count in y.value_counts().sort_index().items()
    }
