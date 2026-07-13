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
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor

from services.technical.model_service import FEATURES, TARGET_RETURN_THRESHOLD

XGBOOST_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "technical_xgboost_model.joblib"
)

XGBOOST_CLASS_LABELS = {
    0: "neutral",
    1: "down",
    2: "up",
}

OHLC_TARGET_COLUMNS = ["open", "high", "low", "close"]
REGRESSION_TARGET_COLUMNS = [
    f"target_next_{column}_return" for column in OHLC_TARGET_COLUMNS
]

DEFAULT_LOOKAHEAD_DAYS = 1
DEFAULT_XGBOOST_MAX_FEATURES = 45
DEFAULT_CLASS_PROBABILITY_THRESHOLD = 0.55
CLASS_THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]

XGBOOST_PARAM_CANDIDATES = [
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 450,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "min_child_weight": 2,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
    },
]


def train_xgboost_artifact(
    indicator_df: pd.DataFrame,
    model_scope: str,
    trained_symbol: str | None = None,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    train_before_date: str | None = None,
    n_splits: int = 5,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
    artifact_path: Path | str = XGBOOST_ARTIFACT_PATH,
) -> dict[str, Any]:
    filtered_df = filter_training_history(indicator_df, train_before_date)
    X, y_class, y_regression, clean_df = prepare_xgboost_training_data(
        filtered_df,
        lookahead_days=lookahead_days,
        threshold=threshold,
    )
    if clean_df.empty:
        return {
            "status": "no_data",
            "reason": "Not enough complete rows to train the XGBoost model",
            "indicator_rows": len(filtered_df),
        }

    tuning_result = tune_xgboost_classifier(
        X,
        y_class,
        n_splits=n_splits,
        max_features=max_features,
    )
    best_params = tuning_result.get("best_params", XGBOOST_PARAM_CANDIDATES[0])
    validation = walk_forward_xgboost_validation(
        X,
        y_class,
        y_regression,
        clean_df,
        model_params=best_params,
        n_splits=n_splits,
        max_features=max_features,
    )

    selected_features, feature_importance = select_xgboost_features(
        X,
        y_class,
        best_params,
        max_features=max_features,
    )
    classifier = get_xgboost_classifier(best_params)
    classifier.fit(X[selected_features], y_class)

    regressor = get_xgboost_regressor(best_params)
    regressor.fit(X[selected_features], y_regression)

    training_date_range = date_range_summary(clean_df)
    symbols = (
        sorted(str(value) for value in clean_df["symbol"].dropna().unique())
        if "symbol" in clean_df.columns
        else []
    )
    metadata = {
        "model_family": "xgboost",
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols": symbols,
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "feature_count": len(FEATURES),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_features": feature_importance[:15],
        "tuned_params": best_params,
        "tuning": tuning_result,
        "classification_metrics": validation["classification_metrics"],
        "regression_metrics": validation["regression_metrics"],
        "class_probability_thresholds": validation["class_probability_thresholds"],
        "class_labels": XGBOOST_CLASS_LABELS,
        "regression_targets": REGRESSION_TARGET_COLUMNS,
        "training_rows": int(len(clean_df)),
    }
    saved_path = save_xgboost_artifact(
        classifier=classifier,
        regressor=regressor,
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
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "selected_feature_count": len(selected_features),
        "classification_metrics": validation["classification_metrics"],
        "regression_metrics": validation["regression_metrics"],
        "class_probability_thresholds": validation["class_probability_thresholds"],
        "top_features": feature_importance[:15],
        "tuned_params": best_params,
    }


def prepare_xgboost_training_data(
    df: pd.DataFrame,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    feature_names = features or FEATURES
    if df is None or df.empty:
        return (
            pd.DataFrame(columns=feature_names),
            pd.Series(dtype=int),
            pd.DataFrame(columns=REGRESSION_TARGET_COLUMNS),
            pd.DataFrame(),
        )

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    lookahead_days = max(1, int(lookahead_days))
    clean_df = sort_for_target_creation(df)
    group_columns = target_group_columns(clean_df)
    for column in OHLC_TARGET_COLUMNS:
        if group_columns:
            future_value = clean_df.groupby(group_columns, sort=False)[column].shift(
                -lookahead_days
            )
        else:
            future_value = clean_df[column].shift(-lookahead_days)
        clean_df[f"future_{column}"] = future_value
        clean_df[f"target_next_{column}_return"] = future_value / clean_df["close"] - 1

    clean_df["future_close_return"] = clean_df["future_close"] / clean_df["close"] - 1
    clean_df["xgboost_direction"] = 0
    clean_df.loc[clean_df["future_close_return"] <= -threshold, "xgboost_direction"] = 1
    clean_df.loc[clean_df["future_close_return"] >= threshold, "xgboost_direction"] = 2
    clean_df.loc[clean_df["future_close_return"].isna(), "xgboost_direction"] = pd.NA
    clean_df = sort_for_chronological_training(clean_df)

    required_columns = feature_names + ["xgboost_direction", *REGRESSION_TARGET_COLUMNS]
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=required_columns).copy()
    clean_df["xgboost_direction"] = clean_df["xgboost_direction"].astype(int)
    clean_df = clean_df.reset_index(drop=True)

    X = clean_df[feature_names].copy()
    y_class = clean_df["xgboost_direction"].copy()
    y_regression = clean_df[REGRESSION_TARGET_COLUMNS].copy()
    return X, y_class, y_regression, clean_df


def tune_xgboost_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
) -> dict[str, Any]:
    evaluations = []
    best_params = XGBOOST_PARAM_CANDIDATES[0]
    best_score = -np.inf

    for params in XGBOOST_PARAM_CANDIDATES:
        fold_scores = []
        for train_index, test_index in chronological_splits(X, n_splits):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if y_train.nunique() < 2:
                continue

            selected_features, _ = select_xgboost_features(
                X_train,
                y_train,
                params,
                max_features=max_features,
            )
            model = get_xgboost_classifier(params)
            model.fit(X_train[selected_features], y_train)
            y_pred = model.predict(X_test[selected_features])
            fold_scores.append(
                {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                }
            )

        accuracy = average_metric(fold_scores, "accuracy")
        f1_macro = average_metric(fold_scores, "f1_macro")
        evaluation = {
            "params": params,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
        }
        evaluations.append(evaluation)
        comparable_score = f1_macro if f1_macro is not None else -np.inf
        if comparable_score > best_score:
            best_score = comparable_score
            best_params = params

    return {
        "best_params": best_params,
        "best_f1_macro": None if best_score == -np.inf else float(best_score),
        "evaluations": evaluations,
    }


def walk_forward_xgboost_validation(
    X: pd.DataFrame,
    y_class: pd.Series,
    y_regression: pd.DataFrame,
    clean_df: pd.DataFrame,
    model_params: dict[str, Any],
    n_splits: int = 5,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
) -> dict[str, Any]:
    fold_results = []
    all_true = []
    all_proba = []
    all_regression_true = []
    all_regression_pred = []

    for fold_number, (train_index, test_index) in enumerate(
        chronological_splits(X, n_splits),
        start=1,
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_class.iloc[train_index], y_class.iloc[test_index]
        y_reg_train = y_regression.iloc[train_index]
        y_reg_test = y_regression.iloc[test_index]

        if y_train.nunique() < 2:
            continue

        selected_features, _ = select_xgboost_features(
            X_train,
            y_train,
            model_params,
            max_features=max_features,
        )
        classifier = get_xgboost_classifier(model_params)
        classifier.fit(X_train[selected_features], y_train)
        proba = classifier.predict_proba(X_test[selected_features])
        y_pred = np.argmax(proba, axis=1)

        regressor = get_xgboost_regressor(model_params)
        regressor.fit(X_train[selected_features], y_reg_train)
        reg_pred = regressor.predict(X_test[selected_features])

        fold_results.append(
            {
                "fold": fold_number,
                "test_start_date": row_date(clean_df, int(test_index[0])),
                "test_end_date": row_date(clean_df, int(test_index[-1])),
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                **classification_metrics(y_test, y_pred),
                **regression_metric_summary(y_reg_test, reg_pred),
            }
        )
        all_true.append(y_test.to_numpy(dtype=int))
        all_proba.append(proba)
        all_regression_true.append(y_reg_test.to_numpy(dtype=float))
        all_regression_pred.append(np.asarray(reg_pred, dtype=float))

    if not fold_results:
        return {
            "classification_metrics": empty_classification_metrics("no_validation_folds"),
            "regression_metrics": empty_regression_metrics("no_validation_folds"),
            "class_probability_thresholds": {
                "down": DEFAULT_CLASS_PROBABILITY_THRESHOLD,
                "up": DEFAULT_CLASS_PROBABILITY_THRESHOLD,
            },
            "folds": [],
        }

    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    thresholds = find_best_class_probability_thresholds(y_true, y_proba)
    selective_pred = selective_class_predictions(
        y_proba,
        down_threshold=thresholds["down"],
        up_threshold=thresholds["up"],
    )
    regression_true = np.vstack(all_regression_true)
    regression_pred = np.vstack(all_regression_pred)

    return {
        "classification_metrics": {
            **classification_metrics(y_true, selective_pred),
            "plain_argmax_accuracy": average_metric(fold_results, "accuracy"),
            "plain_argmax_f1_macro": average_metric(fold_results, "f1_macro"),
            "confusion_matrix": confusion_matrix(y_true, selective_pred, labels=[0, 1, 2]).tolist(),
            "validation_folds": len(fold_results),
        },
        "regression_metrics": regression_metric_summary(regression_true, regression_pred),
        "class_probability_thresholds": thresholds,
        "folds": fold_results,
    }


def get_xgboost_classifier(model_params: dict[str, Any] | None = None) -> Any:
    from xgboost import XGBClassifier

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(model_params or {})
    return XGBClassifier(**params)


def get_xgboost_regressor(model_params: dict[str, Any] | None = None) -> MultiOutputRegressor:
    from xgboost import XGBRegressor

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(model_params or {})
    params.pop("num_class", None)
    params.pop("objective", None)
    regressor = XGBRegressor(objective="reg:squarederror", **params)
    return MultiOutputRegressor(regressor)


def select_xgboost_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any],
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
) -> tuple[list[str], list[dict[str, float]]]:
    feature_names = list(X_train.columns)
    if len(feature_names) <= max_features or y_train.nunique() < 2:
        return feature_names, []

    selector = get_xgboost_classifier(model_params)
    selector.fit(X_train, y_train)
    importances = feature_importance_from_model(selector, feature_names)
    selected = [
        item["feature"]
        for item in importances
        if item["importance"] > 0
    ][:max_features]
    if not selected:
        selected = feature_names[:max_features]

    selected_set = set(selected)
    return selected, [item for item in importances if item["feature"] in selected_set]


def feature_importance_from_model(
    model: Any,
    features: list[str],
) -> list[dict[str, float]]:
    raw_importance = getattr(model, "feature_importances_", None)
    if raw_importance is None:
        return []
    importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(features, raw_importance)
    ]
    return sorted(importance, key=lambda item: item["importance"], reverse=True)


def selective_class_predictions(
    probabilities: np.ndarray,
    down_threshold: float = DEFAULT_CLASS_PROBABILITY_THRESHOLD,
    up_threshold: float = DEFAULT_CLASS_PROBABILITY_THRESHOLD,
) -> np.ndarray:
    predictions = np.zeros(len(probabilities), dtype=int)
    down_mask = (
        (probabilities[:, 1] >= down_threshold)
        & (probabilities[:, 1] > probabilities[:, 2])
    )
    up_mask = (
        (probabilities[:, 2] >= up_threshold)
        & (probabilities[:, 2] > probabilities[:, 1])
    )
    predictions[down_mask] = 1
    predictions[up_mask] = 2
    return predictions


def find_best_class_probability_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    best_thresholds = {
        "down": DEFAULT_CLASS_PROBABILITY_THRESHOLD,
        "up": DEFAULT_CLASS_PROBABILITY_THRESHOLD,
    }
    best_score = -np.inf
    best_accuracy = -np.inf

    for down_threshold in CLASS_THRESHOLD_GRID:
        for up_threshold in CLASS_THRESHOLD_GRID:
            y_pred = selective_class_predictions(
                probabilities,
                down_threshold=down_threshold,
                up_threshold=up_threshold,
            )
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            if (f1_macro, accuracy) > (best_score, best_accuracy):
                best_score = f1_macro
                best_accuracy = accuracy
                best_thresholds = {
                    "down": float(down_threshold),
                    "up": float(up_threshold),
                }

    return best_thresholds


def predict_with_xgboost_artifact(
    artifact: dict[str, Any],
    feature_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    classifier = artifact["classifier"]
    regressor = artifact["regressor"]
    metadata = artifact.get("metadata", {})
    selected_features = metadata.get("selected_features", FEATURES)
    thresholds = metadata.get("class_probability_thresholds", {})
    down_threshold = float(
        thresholds.get("down", DEFAULT_CLASS_PROBABILITY_THRESHOLD)
    )
    up_threshold = float(thresholds.get("up", DEFAULT_CLASS_PROBABILITY_THRESHOLD))

    ready_rows = feature_rows.replace([np.inf, -np.inf], np.nan)
    ready_rows = ready_rows.dropna(subset=selected_features).copy()
    if ready_rows.empty:
        return []

    probabilities = classifier.predict_proba(ready_rows[selected_features])
    class_predictions = selective_class_predictions(
        probabilities,
        down_threshold=down_threshold,
        up_threshold=up_threshold,
    )
    regression_returns = regressor.predict(ready_rows[selected_features])

    predictions = []
    for row_index, (_, row) in enumerate(ready_rows.iterrows()):
        class_id = int(class_predictions[row_index])
        close_price = float(row["close"])
        predicted_returns = regression_returns[row_index]
        predicted_prices = {
            OHLC_TARGET_COLUMNS[index]: float(close_price * (1 + predicted_returns[index]))
            for index in range(len(OHLC_TARGET_COLUMNS))
        }
        class_probabilities = {
            XGBOOST_CLASS_LABELS[index]: float(probabilities[row_index][index])
            for index in range(min(probabilities.shape[1], len(XGBOOST_CLASS_LABELS)))
        }
        predictions.append(
            {
                "stock_id": int(row["stock_id"]) if "stock_id" in row and not pd.isna(row["stock_id"]) else None,
                "symbol": str(row["symbol"]).upper() if "symbol" in row else None,
                "as_of_date": str(row["date"]),
                "as_of_close": close_price,
                "predicted_class": class_id,
                "predicted_direction": XGBOOST_CLASS_LABELS[class_id],
                "confidence": float(class_probabilities[XGBOOST_CLASS_LABELS[class_id]]),
                "class_probabilities": class_probabilities,
                "predicted_next_prices": predicted_prices,
                "lookahead_days": metadata.get("lookahead_days"),
                "direction_threshold": metadata.get("direction_threshold"),
            }
        )

    return predictions


def save_xgboost_artifact(
    classifier: Any,
    regressor: Any,
    metadata: dict[str, Any],
    path: Path | str = XGBOOST_ARTIFACT_PATH,
) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": classifier,
        "regressor": regressor,
        "metadata": {
            **metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        },
    }
    joblib.dump(payload, artifact_path)
    return artifact_path


def load_xgboost_artifact(path: Path | str = XGBOOST_ARTIFACT_PATH) -> dict[str, Any]:
    return joblib.load(Path(path))


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def regression_metric_summary(y_true: Any, y_pred: Any) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    metrics = {
        "mse": float(mean_squared_error(y_true_array, y_pred_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "r2": float(r2_score(y_true_array, y_pred_array, multioutput="uniform_average")),
        "median_absolute_error": float(
            median_absolute_error(y_true_array.ravel(), y_pred_array.ravel())
        ),
        "explained_variance": float(
            explained_variance_score(
                y_true_array,
                y_pred_array,
                multioutput="uniform_average",
            )
        ),
    }

    for index, target in enumerate(REGRESSION_TARGET_COLUMNS):
        metrics[target] = {
            "mse": float(mean_squared_error(y_true_array[:, index], y_pred_array[:, index])),
            "mae": float(mean_absolute_error(y_true_array[:, index], y_pred_array[:, index])),
            "r2": float(r2_score(y_true_array[:, index], y_pred_array[:, index])),
        }

    return metrics


def empty_classification_metrics(reason: str) -> dict[str, Any]:
    return {
        "accuracy": None,
        "balanced_accuracy": None,
        "f1_macro": None,
        "precision_macro": None,
        "recall_macro": None,
        "mcc": None,
        "reason": reason,
    }


def empty_regression_metrics(reason: str) -> dict[str, Any]:
    return {
        "mse": None,
        "mae": None,
        "r2": None,
        "median_absolute_error": None,
        "explained_variance": None,
        "reason": reason,
    }


def chronological_splits(
    X: pd.DataFrame,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_samples = len(X)
    if n_samples < 3:
        return []
    effective_splits = max(2, min(int(n_splits), n_samples - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    return list(splitter.split(X))


def filter_training_history(
    df: pd.DataFrame,
    train_before_date: str | None = None,
) -> pd.DataFrame:
    normalized_date = normalize_optional_date(train_before_date)
    if not normalized_date:
        return df
    cutoff = pd.to_datetime(normalized_date, errors="raise").date()
    result = df.copy()
    result["_date_filter"] = pd.to_datetime(
        result["date"],
        errors="coerce",
        utc=True,
    ).dt.date
    result = result[result["_date_filter"] < cutoff]
    return result.drop(columns=["_date_filter"], errors="ignore").reset_index(drop=True)


def normalize_optional_date(value: str | None) -> str | None:
    if not value:
        return None
    return pd.to_datetime(value, errors="raise").date().isoformat()


def date_range_summary(df: pd.DataFrame) -> dict[str, str | None]:
    if df is None or df.empty or "date" not in df.columns:
        return {"start_date": None, "end_date": None}

    dates = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date.dropna()
    if dates.empty:
        return {"start_date": None, "end_date": None}

    return {
        "start_date": dates.min().isoformat(),
        "end_date": dates.max().isoformat(),
    }


def sort_for_target_creation(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["_date_sort"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
    sort_columns = [*target_group_columns(result), "_date_sort"]
    return result.sort_values(sort_columns, ascending=True).reset_index(drop=True)


def sort_for_chronological_training(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "_date_sort" not in result.columns:
        result["_date_sort"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
    sort_columns = ["_date_sort"]
    for column in ["symbol", "stock_id"]:
        if column in result.columns:
            sort_columns.append(column)
    result = result.sort_values(sort_columns, ascending=True).reset_index(drop=True)
    return result.drop(columns=["_date_sort"], errors="ignore")


def target_group_columns(df: pd.DataFrame) -> list[str]:
    if "stock_id" in df.columns:
        return ["stock_id"]
    if "symbol" in df.columns:
        return ["symbol"]
    return []


def row_date(df: pd.DataFrame, index: int) -> str | None:
    if "date" not in df.columns or index >= len(df):
        return None
    return str(df.iloc[index]["date"])


def average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))
