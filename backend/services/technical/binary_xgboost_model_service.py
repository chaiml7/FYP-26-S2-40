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
from sklearn.model_selection import TimeSeriesSplit

from services.technical.model_service import FEATURES, TARGET_RETURN_THRESHOLD
from services.technical.xgboost_model_service import (
    DEFAULT_LOOKAHEAD_DAYS,
    DEFAULT_XGBOOST_MAX_FEATURES,
    chronological_splits,
    date_range_summary,
    feature_importance_from_model,
    filter_training_history,
    normalize_optional_date,
    row_date,
    sort_for_chronological_training,
    sort_for_target_creation,
    target_group_columns,
)

BINARY_XGBOOST_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "technical_xgboost_binary_model.joblib"
)

BINARY_XGBOOST_CLASS_LABELS = {
    0: "down_equal",
    1: "up",
}

BINARY_DECISION_THRESHOLD_GRID = [
    round(value, 2) for value in np.arange(0.40, 0.611, 0.02)
]

BINARY_XGBOOST_PARAM_CANDIDATES = [
    {
        "n_estimators": 900,
        "max_depth": 3,
        "learning_rate": 0.015,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "min_child_weight": 10,
        "gamma": 0.25,
        "reg_alpha": 0.25,
        "reg_lambda": 8.0,
    },
    {
        "n_estimators": 1100,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.60,
        "colsample_bytree": 0.65,
        "min_child_weight": 15,
        "gamma": 0.50,
        "reg_alpha": 0.75,
        "reg_lambda": 10.0,
    },
    {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.02,
        "subsample": 0.75,
        "colsample_bytree": 0.60,
        "min_child_weight": 12,
        "gamma": 0.35,
        "reg_alpha": 0.50,
        "reg_lambda": 6.0,
    },
    {
        "n_estimators": 1200,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.55,
        "colsample_bytree": 0.55,
        "min_child_weight": 20,
        "gamma": 0.75,
        "reg_alpha": 1.25,
        "reg_lambda": 12.0,
    },
]

DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE = 3
DEFAULT_BINARY_XGBOOST_PURGE_DAYS = 1
DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS = 0
DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS = 50
DEFAULT_BINARY_XGBOOST_SAMPLE_WEIGHTING = True


class AveragedBinaryXGBoostClassifier:
    """Average class probabilities across several fitted XGBoost models."""

    classes_ = np.array([0, 1])

    def __init__(
        self,
        models: list[Any],
        selected_features: list[str],
        feature_importances: list[dict[str, float]],
    ):
        self.models = models
        self.selected_features_ = selected_features
        self.feature_importances_ = feature_importances

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        feature_frame = X[self.selected_features_]
        probabilities = [
            positive_class_probability(model, feature_frame)
            for model in self.models
        ]
        probability_up = np.mean(np.vstack(probabilities), axis=0)
        return np.column_stack([1 - probability_up, probability_up])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_binary_xgboost_artifact(
    indicator_df: pd.DataFrame,
    model_scope: str,
    trained_symbol: str | None = None,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    train_before_date: str | None = None,
    n_splits: int = 5,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
    ensemble_size: int = DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE,
    use_sample_weighting: bool = DEFAULT_BINARY_XGBOOST_SAMPLE_WEIGHTING,
    purge_days: int = DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    embargo_days: int = DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
    early_stopping_rounds: int = DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
    artifact_path: Path | str = BINARY_XGBOOST_ARTIFACT_PATH,
) -> dict[str, Any]:
    filtered_df = filter_training_history(indicator_df, train_before_date)
    X, y, clean_df = prepare_binary_xgboost_training_data(
        filtered_df,
        lookahead_days=lookahead_days,
        threshold=threshold,
    )
    if clean_df.empty:
        return {
            "status": "no_data",
            "reason": "Not enough complete rows to train the binary XGBoost model",
            "indicator_rows": len(filtered_df),
        }

    sample_weight = build_binary_xgboost_sample_weights(
        clean_df,
        enabled=use_sample_weighting,
    )
    tuning_result = tune_binary_xgboost_classifier(
        X,
        y,
        clean_df,
        sample_weight=sample_weight,
        n_splits=n_splits,
        max_features=max_features,
        purge_days=purge_days,
        embargo_days=embargo_days,
        early_stopping_rounds=early_stopping_rounds,
    )
    best_params = tuning_result.get("best_params", BINARY_XGBOOST_PARAM_CANDIDATES[0])
    validation = walk_forward_binary_xgboost_validation(
        X,
        y,
        clean_df,
        model_params=best_params,
        sample_weight=sample_weight,
        n_splits=n_splits,
        max_features=max_features,
        purge_days=purge_days,
        embargo_days=embargo_days,
        early_stopping_rounds=early_stopping_rounds,
    )

    selected_features, feature_importance = select_binary_xgboost_features(
        X,
        y,
        best_params,
        sample_weight=sample_weight,
        max_features=max_features,
        early_stopping_rounds=early_stopping_rounds,
    )
    classifier = fit_binary_xgboost_ensemble(
        X[selected_features],
        y,
        model_params=best_params,
        sample_weight=sample_weight,
        selected_features=selected_features,
        feature_importance=feature_importance,
        ensemble_size=ensemble_size,
        early_stopping_rounds=early_stopping_rounds,
    )

    training_date_range = date_range_summary(clean_df)
    symbols = (
        sorted(str(value) for value in clean_df["symbol"].dropna().unique())
        if "symbol" in clean_df.columns
        else []
    )
    target_distribution = {
        BINARY_XGBOOST_CLASS_LABELS[int(label)]: int(count)
        for label, count in y.value_counts().sort_index().items()
    }
    metadata = {
        "model_family": "xgboost_binary",
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols": symbols,
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "decision_threshold": validation["decision_threshold"],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "feature_count": len(FEATURES),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "top_features": feature_importance[:15],
        "tuned_params": best_params,
        "ensemble_size": int(max(1, ensemble_size)),
        "sample_weighting": bool(use_sample_weighting),
        "purge_days": int(max(0, purge_days)),
        "embargo_days": int(max(0, embargo_days)),
        "early_stopping_rounds": int(max(0, early_stopping_rounds)),
        "tuning": tuning_result,
        "metrics": validation["metrics"],
        "target_distribution": target_distribution,
        "class_labels": BINARY_XGBOOST_CLASS_LABELS,
        "training_rows": int(len(clean_df)),
    }
    saved_path = save_binary_xgboost_artifact(
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
        "target_distribution": target_distribution,
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "decision_threshold": validation["decision_threshold"],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "selected_feature_count": len(selected_features),
        "ensemble_size": int(max(1, ensemble_size)),
        "sample_weighting": bool(use_sample_weighting),
        "purge_days": int(max(0, purge_days)),
        "embargo_days": int(max(0, embargo_days)),
        "early_stopping_rounds": int(max(0, early_stopping_rounds)),
        "metrics": validation["metrics"],
        "top_features": feature_importance[:15],
        "tuned_params": best_params,
    }


def prepare_binary_xgboost_training_data(
    df: pd.DataFrame,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_names = features or FEATURES
    if df is None or df.empty:
        return pd.DataFrame(columns=feature_names), pd.Series(dtype=int), pd.DataFrame()

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
    clean_df["binary_target_direction"] = (
        clean_df["future_close_return"] > threshold
    ).astype(int)
    clean_df.loc[clean_df["future_close_return"].isna(), "binary_target_direction"] = pd.NA
    clean_df = sort_for_chronological_training(clean_df)

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=feature_names + ["binary_target_direction"]).copy()
    clean_df["binary_target_direction"] = clean_df["binary_target_direction"].astype(int)
    clean_df = clean_df.reset_index(drop=True)

    X = clean_df[feature_names].copy()
    y = clean_df["binary_target_direction"].copy()
    return X, y, clean_df


def tune_binary_xgboost_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    clean_df: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
    purge_days: int = DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    embargo_days: int = DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
    early_stopping_rounds: int = DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
) -> dict[str, Any]:
    evaluations = []
    best_params = BINARY_XGBOOST_PARAM_CANDIDATES[0]
    best_score = -np.inf

    for params in BINARY_XGBOOST_PARAM_CANDIDATES:
        fold_scores = []
        for train_index, test_index in purged_chronological_splits(
            clean_df,
            n_splits=n_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        ):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if y_train.nunique() < 2:
                continue
            train_weight = sample_weight.iloc[train_index] if sample_weight is not None else None

            selected_features, _ = select_binary_xgboost_features(
                X_train,
                y_train,
                params,
                sample_weight=train_weight,
                max_features=max_features,
                early_stopping_rounds=early_stopping_rounds,
            )
            model = get_binary_xgboost_classifier(params, y_train)
            fit_binary_xgboost_model(
                model,
                X_train[selected_features],
                y_train,
                sample_weight=train_weight,
                early_stopping_rounds=early_stopping_rounds,
            )
            y_pred = model.predict(X_test[selected_features])
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


def walk_forward_binary_xgboost_validation(
    X: pd.DataFrame,
    y: pd.Series,
    clean_df: pd.DataFrame,
    model_params: dict[str, Any],
    sample_weight: pd.Series | None = None,
    n_splits: int = 5,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
    purge_days: int = DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    embargo_days: int = DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
    early_stopping_rounds: int = DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
) -> dict[str, Any]:
    fold_results = []
    all_true = []
    all_proba = []

    for fold_number, (train_index, test_index) in enumerate(
        purged_chronological_splits(
            clean_df,
            n_splits=n_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        ),
        start=1,
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if y_train.nunique() < 2:
            continue
        train_weight = sample_weight.iloc[train_index] if sample_weight is not None else None

        selected_features, _ = select_binary_xgboost_features(
            X_train,
            y_train,
            model_params,
            sample_weight=train_weight,
            max_features=max_features,
            early_stopping_rounds=early_stopping_rounds,
        )
        classifier = get_binary_xgboost_classifier(model_params, y_train)
        fit_binary_xgboost_model(
            classifier,
            X_train[selected_features],
            y_train,
            sample_weight=train_weight,
            early_stopping_rounds=early_stopping_rounds,
        )
        probability_up = positive_class_probability(classifier, X_test[selected_features])
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
    decision_threshold = find_best_binary_decision_threshold(y_true, y_proba)
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


def get_binary_xgboost_classifier(
    model_params: dict[str, Any] | None = None,
    y_train: pd.Series | None = None,
    random_state: int = 42,
) -> Any:
    from xgboost import XGBClassifier

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": int(random_state),
    }
    params.update(model_params or {})
    if y_train is not None:
        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        if positive_count > 0 and negative_count > 0:
            params["scale_pos_weight"] = negative_count / positive_count
    return XGBClassifier(**params)


def select_binary_xgboost_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any],
    sample_weight: pd.Series | None = None,
    max_features: int = DEFAULT_XGBOOST_MAX_FEATURES,
    early_stopping_rounds: int = DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
) -> tuple[list[str], list[dict[str, float]]]:
    feature_names = list(X_train.columns)
    if len(feature_names) <= max_features or y_train.nunique() < 2:
        return feature_names, []

    selector = get_binary_xgboost_classifier(model_params, y_train)
    fit_binary_xgboost_model(
        selector,
        X_train,
        y_train,
        sample_weight=sample_weight,
        early_stopping_rounds=early_stopping_rounds,
    )
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


def fit_binary_xgboost_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict[str, Any],
    sample_weight: pd.Series | None,
    selected_features: list[str],
    feature_importance: list[dict[str, float]],
    ensemble_size: int,
    early_stopping_rounds: int,
) -> Any:
    models = []
    ensemble_size = max(1, int(ensemble_size))
    for seed_offset in range(ensemble_size):
        seed = 42 + seed_offset * 17
        params = {
            **model_params,
            "subsample": min(0.95, float(model_params.get("subsample", 0.8))),
            "colsample_bytree": min(
                0.95,
                float(model_params.get("colsample_bytree", 0.8)),
            ),
        }
        model = get_binary_xgboost_classifier(params, y, random_state=seed)
        fit_binary_xgboost_model(
            model,
            X,
            y,
            sample_weight=sample_weight,
            early_stopping_rounds=early_stopping_rounds,
        )
        models.append(model)

    if len(models) == 1:
        return models[0]
    return AveragedBinaryXGBoostClassifier(
        models=models,
        selected_features=selected_features,
        feature_importances=feature_importance,
    )


def fit_binary_xgboost_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | None = None,
    early_stopping_rounds: int = DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
) -> Any:
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.to_numpy(dtype=float)

    early_stopping_rounds = max(0, int(early_stopping_rounds))
    split = early_stopping_split(y_train)
    if early_stopping_rounds and split is not None:
        train_index, eval_index = split
        eval_set = [(X_train.iloc[eval_index], y_train.iloc[eval_index])]
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["verbose"] = False
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_index].to_numpy(dtype=float)
            fit_kwargs["sample_weight_eval_set"] = [
                sample_weight.iloc[eval_index].to_numpy(dtype=float)
            ]

        try:
            return model.fit(
                X_train.iloc[train_index],
                y_train.iloc[train_index],
                early_stopping_rounds=early_stopping_rounds,
                **fit_kwargs,
            )
        except TypeError:
            fit_kwargs.pop("sample_weight_eval_set", None)
            fit_kwargs.pop("eval_set", None)
            fit_kwargs.pop("verbose", None)
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight.to_numpy(dtype=float)

    return model.fit(X_train, y_train, **fit_kwargs)


def early_stopping_split(y_train: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    if len(y_train) < 80:
        return None

    eval_size = max(25, int(len(y_train) * 0.15))
    if eval_size >= len(y_train):
        return None

    train_index = np.arange(0, len(y_train) - eval_size)
    eval_index = np.arange(len(y_train) - eval_size, len(y_train))
    if y_train.iloc[train_index].nunique() < 2 or y_train.iloc[eval_index].nunique() < 2:
        return None
    return train_index, eval_index


def build_binary_xgboost_sample_weights(
    clean_df: pd.DataFrame,
    enabled: bool = DEFAULT_BINARY_XGBOOST_SAMPLE_WEIGHTING,
) -> pd.Series | None:
    if not enabled or clean_df.empty:
        return None

    weights = pd.Series(np.ones(len(clean_df), dtype=float), index=clean_df.index)

    if "relative_volume" in clean_df.columns:
        relative_volume = pd.to_numeric(
            clean_df["relative_volume"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        volume_factor = np.log1p(relative_volume.clip(lower=0.0)) / np.log(2.0)
        volume_factor = volume_factor.clip(lower=0.75, upper=1.25).fillna(1.0)
        weights *= volume_factor

    if "future_close_return" in clean_df.columns:
        abs_return = clean_df["future_close_return"].abs()
        cutoff = abs_return.quantile(0.98)
        if pd.notna(cutoff) and cutoff > 0:
            weights.loc[abs_return > cutoff] *= 0.35

    if "rolling_volatility_20" in clean_df.columns:
        volatility = pd.to_numeric(
            clean_df["rolling_volatility_20"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        high_volatility_cutoff = volatility.quantile(0.98)
        if pd.notna(high_volatility_cutoff) and high_volatility_cutoff > 0:
            weights.loc[volatility > high_volatility_cutoff] *= 0.50

    mean_weight = weights.mean()
    if pd.notna(mean_weight) and mean_weight > 0:
        weights = weights / mean_weight
    return weights.clip(lower=0.10, upper=3.00)


def purged_chronological_splits(
    clean_df: pd.DataFrame,
    n_splits: int,
    purge_days: int = DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    embargo_days: int = DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    sample_count = len(clean_df)
    if sample_count < 3:
        return []

    effective_splits = max(2, min(int(n_splits), sample_count - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    dates = pd.to_datetime(clean_df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    splits = []
    for train_index, test_index in splitter.split(np.arange(sample_count)):
        if len(train_index) == 0 or len(test_index) == 0:
            continue

        test_dates = dates.iloc[test_index].dropna()
        if test_dates.empty:
            splits.append((train_index, test_index))
            continue

        test_start = test_dates.min()
        test_end = test_dates.max()
        purge_start = test_start - pd.Timedelta(days=max(0, int(purge_days)))
        embargo_end = test_end + pd.Timedelta(days=max(0, int(embargo_days)))

        train_dates = dates.iloc[train_index]
        keep_mask = (train_dates < purge_start) | (train_dates > embargo_end)
        purged_train_index = train_index[np.asarray(keep_mask, dtype=bool)]
        if len(purged_train_index) == 0:
            continue
        splits.append((purged_train_index, test_index))
    return splits


def predict_with_binary_xgboost_artifact(
    artifact: dict[str, Any],
    feature_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    classifier = artifact["classifier"]
    metadata = artifact.get("metadata", {})
    selected_features = metadata.get("selected_features", FEATURES)
    decision_threshold = float(metadata.get("decision_threshold") or 0.5)

    ready_rows = feature_rows.replace([np.inf, -np.inf], np.nan)
    ready_rows = ready_rows.dropna(subset=selected_features).copy()
    if ready_rows.empty:
        return []

    probability_up = positive_class_probability(
        classifier,
        ready_rows[selected_features],
    )
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
                "predicted_direction": BINARY_XGBOOST_CLASS_LABELS[predicted_class],
                "confidence": float(confidence),
                "probability_up": float(probability_up[row_index]),
                "decision_threshold": decision_threshold,
                "lookahead_days": metadata.get("lookahead_days"),
                "direction_threshold": metadata.get("direction_threshold"),
            }
        )

    return predictions


def save_binary_xgboost_artifact(
    classifier: Any,
    metadata: dict[str, Any],
    path: Path | str = BINARY_XGBOOST_ARTIFACT_PATH,
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


def load_binary_xgboost_artifact(
    path: Path | str = BINARY_XGBOOST_ARTIFACT_PATH,
) -> dict[str, Any]:
    return joblib.load(Path(path))


def positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def find_best_binary_decision_threshold(
    y_true: np.ndarray,
    probability_up: np.ndarray,
) -> float:
    best_threshold = 0.5
    best_score = -np.inf
    best_accuracy = -np.inf

    for threshold in BINARY_DECISION_THRESHOLD_GRID:
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
