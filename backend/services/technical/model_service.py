from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_RETURN_THRESHOLD = 0.002
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_MAX_FEATURES = 40
MODEL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "technical_direction_model.joblib"
)

MARKET_CONTEXT_FEATURES = [
    "market_spy_return_1d",
    "market_spy_return_5d",
    "market_spy_above_sma_200",
    "market_qqq_return_1d",
    "market_qqq_return_5d",
    "market_qqq_above_sma_200",
    "market_vix_level",
    "market_vix_return_1d",
    "market_vix_return_5d",
    "market_sector_return_1d",
    "market_sector_return_5d",
    "market_sector_above_sma_200",
]

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "log_return",
    "return_5d",
    "high_low_range",
    "open_close_gap",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_10",
    "ema_20",
    "ema_50",
    "trend_filter_50_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "volume_sma_20",
    "volume_change",
    "relative_volume",
    "vwap_20",
    "support_20",
    "resistance_20",
    "distance_to_support",
    "distance_to_resistance",
    "breakout_indicator",
    "breakdown_indicator",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "close_lag_1",
    "close_lag_2",
    "close_lag_5",
    *MARKET_CONTEXT_FEATURES,
]

LIGHTGBM_PARAM_CANDIDATES = [
    {},
    {
        "n_estimators": 250,
        "learning_rate": 0.04,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 30,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.025,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 25,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.02,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": 0.5,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": -1,
        "num_leaves": 15,
        "min_child_samples": 40,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
]

DECISION_THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.40, 0.611, 0.02)]


class ConstantDirectionClassifier:
    """Tiny fallback for rare histories where training has only one class."""

    classes_ = np.array([0, 1])
    selected_features_ = FEATURES
    feature_importances_: list[dict[str, float]] = []

    def __init__(self, constant: int):
        self.constant = int(constant)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstantDirectionClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.constant, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(X), 2), dtype=float)
        proba[:, self.constant] = 1.0
        return proba


class FeatureSelectedClassifier:
    """Wrap a fitted classifier so callers can pass the full feature frame."""

    def __init__(
        self,
        model: Any,
        selected_features: list[str],
        feature_importances: list[dict[str, float]],
    ):
        self.model = model
        self.selected_features_ = selected_features
        self.feature_importances_ = feature_importances
        self.classes_ = getattr(model, "classes_", np.array([0, 1]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.selected_features_])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.selected_features_])


def prepare_training_data(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target_return_threshold: float = TARGET_RETURN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create the next-day direction target and return cleaned X/y data."""
    feature_names = features or FEATURES
    if df is None or df.empty:
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=int), pd.DataFrame()

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {', '.join(missing_features)}")

    clean_df = _sort_for_target_creation(df)
    group_columns = _target_group_columns(clean_df)
    if group_columns:
        next_close = clean_df.groupby(group_columns, sort=False)["close"].shift(-1)
    else:
        next_close = clean_df["close"].shift(-1)

    clean_df["next_day_return"] = next_close / clean_df["close"] - 1
    clean_df["target_direction"] = (
        clean_df["next_day_return"] > target_return_threshold
    ).astype(int)
    clean_df.loc[clean_df["next_day_return"].isna(), "target_direction"] = pd.NA
    clean_df = _sort_for_chronological_training(clean_df)

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=feature_names + ["target_direction"]).copy()
    clean_df["target_direction"] = clean_df["target_direction"].astype(int)
    if group_columns:
        clean_df["previous_direction_baseline"] = (
            clean_df.groupby(group_columns, sort=False)["target_direction"].shift(1)
        )
    else:
        clean_df["previous_direction_baseline"] = clean_df["target_direction"].shift(1)
    clean_df = clean_df.reset_index(drop=True)

    X = clean_df[feature_names].copy()
    y = clean_df["target_direction"].copy()
    return X, y, clean_df


def get_model(model_params: dict[str, Any] | None = None) -> Any:
    """Prefer tree-based LightGBM, then fall back to XGBoost or RandomForest."""
    try:
        from lightgbm import LGBMClassifier

        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 25,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        params.update(model_params or {})
        return LGBMClassifier(**params)
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )


def tune_lightgbm_params(
    df: pd.DataFrame,
    candidate_params: list[dict[str, Any]] | None = None,
    initial_train_size: int | None = None,
    test_size: int = 60,
    max_windows: int = 6,
    target_return_threshold: float = TARGET_RETURN_THRESHOLD,
) -> dict[str, Any]:
    """Choose LightGBM parameters using chronological walk-forward validation."""
    candidates = candidate_params or LIGHTGBM_PARAM_CANDIDATES
    evaluations = []
    best_params: dict[str, Any] = {}
    best_score = -np.inf
    best_f1 = -np.inf

    for params in candidates:
        metrics = walk_forward_validation(
            df,
            initial_train_size=initial_train_size,
            test_size=test_size,
            model_params=params,
            tune_threshold=True,
            use_feature_selection=False,
            max_windows=max_windows,
            target_return_threshold=target_return_threshold,
        )
        score = metrics.get("accuracy")
        f1 = metrics.get("f1_score")
        evaluation = {
            "params": params,
            "accuracy": score,
            "f1_score": f1,
            "decision_threshold": metrics.get("decision_threshold"),
        }
        evaluations.append(evaluation)

        comparable_score = score if score is not None else -np.inf
        comparable_f1 = f1 if f1 is not None else -np.inf
        if (comparable_score, comparable_f1) > (best_score, best_f1):
            best_score = comparable_score
            best_f1 = comparable_f1
            best_params = params

    return {
        "best_params": best_params,
        "evaluations": evaluations,
        "best_accuracy": None if best_score == -np.inf else float(best_score),
        "best_f1_score": None if best_f1 == -np.inf else float(best_f1),
    }


def walk_forward_validation(
    df: pd.DataFrame,
    initial_train_size: int | None = None,
    test_size: int = 30,
    model_params: dict[str, Any] | None = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    tune_threshold: bool = True,
    threshold_metric: str = "accuracy",
    use_feature_selection: bool = True,
    max_features: int = DEFAULT_MAX_FEATURES,
    max_windows: int | None = None,
    target_return_threshold: float = TARGET_RETURN_THRESHOLD,
) -> dict[str, Any]:
    """Run expanding-window walk-forward validation without shuffling."""
    X, y, clean_df = prepare_training_data(
        df,
        target_return_threshold=target_return_threshold,
    )
    sample_count = len(clean_df)

    if sample_count < 2:
        return _empty_metrics("not_enough_data", target_return_threshold)

    if initial_train_size is None:
        initial_train_size = int(sample_count * 0.7)

    initial_train_size = max(1, min(initial_train_size, sample_count - 1))
    test_size = max(1, test_size)

    window_payloads = []
    test_start = initial_train_size
    while test_start < sample_count:
        test_end = min(test_start + test_size, sample_count)

        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if X_test.empty:
            break

        model = _fit_window_model(
            X_train,
            y_train,
            model_params=model_params,
            use_feature_selection=use_feature_selection,
            max_features=max_features,
        )
        y_proba = _positive_class_probability(model, X_test)
        fallback_pred = model.predict(X_test)

        majority_class = int(y_train.mode().iloc[0])
        majority_pred = np.full(len(y_test), majority_class, dtype=int)

        previous_direction_pred = clean_df["previous_direction_baseline"].iloc[
            test_start:test_end
        ]
        previous_direction_pred = previous_direction_pred.fillna(majority_class).astype(int)

        window_payloads.append(
            {
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_start_date": _row_date(clean_df, test_start),
                "test_end_date": _row_date(clean_df, test_end - 1),
                "y_test": y_test.to_numpy(dtype=int),
                "y_proba": y_proba,
                "fallback_pred": np.asarray(fallback_pred, dtype=int),
                "majority_pred": majority_pred,
                "previous_direction_pred": previous_direction_pred.to_numpy(dtype=int),
                "selected_features": getattr(model, "selected_features_", list(X_train.columns)),
            }
        )

        if max_windows is not None and len(window_payloads) >= max_windows:
            break

        test_start = test_end

    if not window_payloads:
        return _empty_metrics("no_validation_windows", target_return_threshold)

    selected_threshold = decision_threshold
    if tune_threshold and all(payload["y_proba"] is not None for payload in window_payloads):
        all_true = np.concatenate([payload["y_test"] for payload in window_payloads])
        all_proba = np.concatenate([payload["y_proba"] for payload in window_payloads])
        selected_threshold = _find_best_decision_threshold(
            all_true,
            all_proba,
            metric=threshold_metric,
        )

    window_results = []
    for payload in window_payloads:
        y_test = payload["y_test"]
        y_proba = payload["y_proba"]
        y_pred = (
            (y_proba >= selected_threshold).astype(int)
            if y_proba is not None
            else payload["fallback_pred"]
        )

        roc_auc = None
        if y_proba is not None and len(set(y_test.tolist())) > 1:
            try:
                roc_auc = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                roc_auc = None

        window_results.append(
            {
                "train_rows": payload["train_rows"],
                "test_rows": payload["test_rows"],
                "test_start_date": payload["test_start_date"],
                "test_end_date": payload["test_end_date"],
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": roc_auc,
                "baseline_accuracy": float(accuracy_score(y_test, payload["previous_direction_pred"])),
                "majority_baseline_accuracy": float(accuracy_score(y_test, payload["majority_pred"])),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
                "selected_feature_count": int(len(payload["selected_features"])),
            }
        )

    return {
        "accuracy": _average_metric(window_results, "accuracy"),
        "precision": _average_metric(window_results, "precision"),
        "recall": _average_metric(window_results, "recall"),
        "f1_score": _average_metric(window_results, "f1_score"),
        "roc_auc": _average_metric(window_results, "roc_auc"),
        "baseline_accuracy": _average_metric(window_results, "baseline_accuracy"),
        "majority_baseline_accuracy": _average_metric(window_results, "majority_baseline_accuracy"),
        "decision_threshold": float(selected_threshold),
        "target_return_threshold": float(target_return_threshold),
        "selected_feature_count": _average_metric(window_results, "selected_feature_count"),
        "windows": window_results,
    }


def train_final_model(
    df: pd.DataFrame,
    model_params: dict[str, Any] | None = None,
    target_return_threshold: float = TARGET_RETURN_THRESHOLD,
    use_feature_selection: bool = True,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> tuple[Any, pd.DataFrame, str]:
    """Train on all cleaned historical rows with known next-day labels."""
    X, y, clean_df = prepare_training_data(
        df,
        target_return_threshold=target_return_threshold,
    )
    if X.empty or y.empty:
        raise ValueError("Not enough cleaned technical data to train a model")

    model = _fit_window_model(
        X,
        y,
        model_params=model_params,
        use_feature_selection=use_feature_selection,
        max_features=max_features,
    )
    base_name = _base_model_name(model)
    selected_count = len(getattr(model, "selected_features_", FEATURES))
    model_used = f"{base_name}(selected_features={selected_count})"
    return model, clean_df, model_used


def get_feature_importance(model: Any, limit: int = 15) -> list[dict[str, float]]:
    importance = getattr(model, "feature_importances_", [])
    return importance[:limit]


def save_model_artifact(
    model: Any,
    metadata: dict[str, Any],
    path: Path | str = MODEL_ARTIFACT_PATH,
) -> Path:
    """Persist the trained model and training metadata locally."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "metadata": {
            **metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        },
    }
    joblib.dump(payload, artifact_path)
    return artifact_path


def load_model_artifact(path: Path | str = MODEL_ARTIFACT_PATH) -> dict[str, Any]:
    """Load a locally saved technical model artifact."""
    return joblib.load(Path(path))


def _fit_window_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any] | None = None,
    use_feature_selection: bool = True,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> Any:
    if y_train.nunique() < 2:
        constant = int(y_train.mode().iloc[0])
        return ConstantDirectionClassifier(constant).fit(X_train, y_train)

    selected_features = list(X_train.columns)
    feature_importances: list[dict[str, float]] = []
    if use_feature_selection and len(selected_features) > max_features:
        selected_features, feature_importances = _select_features_from_training(
            X_train,
            y_train,
            model_params=model_params,
            max_features=max_features,
        )

    model = get_model(model_params)
    model.fit(X_train[selected_features], y_train)
    if not feature_importances:
        feature_importances = _feature_importance_from_model(model, selected_features)

    return FeatureSelectedClassifier(model, selected_features, feature_importances)


def _select_features_from_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any] | None,
    max_features: int,
) -> tuple[list[str], list[dict[str, float]]]:
    selector = get_model(model_params)
    selector.fit(X_train, y_train)
    importances = _feature_importance_from_model(selector, list(X_train.columns))

    selected = [
        item["feature"]
        for item in importances
        if item["importance"] > 0
    ][:max_features]

    if not selected:
        selected = list(X_train.columns)[:max_features]

    selected_set = set(selected)
    return selected, [item for item in importances if item["feature"] in selected_set]


def _feature_importance_from_model(model: Any, features: list[str]) -> list[dict[str, float]]:
    raw_importance = getattr(model, "feature_importances_", None)
    if raw_importance is None:
        return []

    importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(features, raw_importance)
    ]
    return sorted(importance, key=lambda item: item["importance"], reverse=True)


def _positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return None

    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def _find_best_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "accuracy",
) -> float:
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_score = -np.inf

    for threshold in DECISION_THRESHOLD_GRID:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == "f1_score":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def _average_metric(windows: list[dict[str, Any]], key: str) -> float | None:
    values = [window[key] for window in windows if window.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _empty_metrics(reason: str, target_return_threshold: float) -> dict[str, Any]:
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "roc_auc": None,
        "baseline_accuracy": None,
        "majority_baseline_accuracy": None,
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "target_return_threshold": float(target_return_threshold),
        "selected_feature_count": None,
        "windows": [],
        "reason": reason,
    }


def _row_date(df: pd.DataFrame, index: int) -> str | None:
    if "date" not in df.columns or index >= len(df):
        return None
    return str(df.iloc[index]["date"])


def _base_model_name(model: Any) -> str:
    if isinstance(model, FeatureSelectedClassifier):
        return model.model.__class__.__name__
    return model.__class__.__name__


def _sort_for_target_creation(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["_date_sort"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
    sort_columns = [*_target_group_columns(result), "_date_sort"]
    return result.sort_values(sort_columns, ascending=True).reset_index(drop=True)


def _sort_for_chronological_training(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "_date_sort" not in result.columns:
        result["_date_sort"] = pd.to_datetime(result["date"], errors="coerce", utc=True)

    sort_columns = ["_date_sort"]
    for column in ["symbol", "stock_id"]:
        if column in result.columns:
            sort_columns.append(column)

    result = result.sort_values(sort_columns, ascending=True).reset_index(drop=True)
    return result.drop(columns=["_date_sort"], errors="ignore")


def _target_group_columns(df: pd.DataFrame) -> list[str]:
    if "stock_id" in df.columns:
        return ["stock_id"]
    if "symbol" in df.columns:
        return ["symbol"]
    return []
