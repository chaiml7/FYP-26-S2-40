from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from services.technical.model_service import FEATURES, TARGET_RETURN_THRESHOLD
from services.technical.xgboost_model_service import (
    date_range_summary,
    filter_training_history,
    normalize_optional_date,
    sort_for_chronological_training,
    sort_for_target_creation,
    target_group_columns,
)

LSTM_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "technical_lstm_model.pt"
)

LSTM_CLASS_LABELS = {
    0: "down_equal",
    1: "up",
}

DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_LOOKAHEAD_DAYS = 1
DEFAULT_LSTM_EPOCHS = 20
DEFAULT_LSTM_BATCH_SIZE = 64
DEFAULT_LSTM_HIDDEN_SIZE = 64
DEFAULT_LSTM_NUM_LAYERS = 1
DEFAULT_LSTM_DROPOUT = 0.10
DEFAULT_LSTM_LEARNING_RATE = 0.001
LSTM_DECISION_THRESHOLD_GRID = [
    round(value, 2) for value in np.arange(0.40, 0.611, 0.02)
]


class TechnicalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE,
        num_layers: int = DEFAULT_LSTM_NUM_LAYERS,
        dropout: float = DEFAULT_LSTM_DROPOUT,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(features)
        return self.output(hidden_state[-1]).squeeze(-1)


def train_lstm_artifact(
    indicator_df: pd.DataFrame,
    model_scope: str,
    trained_symbol: str | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    train_before_date: str | None = None,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE,
    num_layers: int = DEFAULT_LSTM_NUM_LAYERS,
    dropout: float = DEFAULT_LSTM_DROPOUT,
    learning_rate: float = DEFAULT_LSTM_LEARNING_RATE,
    n_splits: int = 5,
    artifact_path: Path | str = LSTM_ARTIFACT_PATH,
) -> dict[str, Any]:
    torch.manual_seed(42)
    np.random.seed(42)

    filtered_df = filter_training_history(indicator_df, train_before_date)
    clean_df = prepare_lstm_training_frame(
        filtered_df,
        lookahead_days=lookahead_days,
        threshold=threshold,
    )
    sequences, targets, sequence_meta = build_lstm_sequences(
        clean_df,
        sequence_length=sequence_length,
    )
    if len(sequences) < 2 or len(np.unique(targets)) < 2:
        return {
            "status": "no_data",
            "reason": "Not enough complete LSTM sequences with both classes",
            "indicator_rows": len(filtered_df),
            "clean_training_rows": len(clean_df),
            "sequence_count": int(len(sequences)),
        }

    validation = walk_forward_lstm_validation(
        sequences,
        targets,
        sequence_meta,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        n_splits=n_splits,
    )
    scaler = fit_sequence_scaler(sequences)
    scaled_sequences = transform_sequences(sequences, scaler)
    model = train_lstm_model(
        scaled_sequences,
        targets,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    training_date_range = date_range_summary(clean_df)
    symbols = (
        sorted(str(value) for value in clean_df["symbol"].dropna().unique())
        if "symbol" in clean_df.columns
        else []
    )
    target_distribution = {
        LSTM_CLASS_LABELS[int(label)]: int(count)
        for label, count in pd.Series(targets).value_counts().sort_index().items()
    }
    metadata = {
        "model_family": "lstm_binary",
        "model_scope": model_scope,
        "trained_symbol": trained_symbol,
        "symbols": symbols,
        "sequence_length": int(sequence_length),
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "decision_threshold": validation["decision_threshold"],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "metrics": validation["metrics"],
        "target_distribution": target_distribution,
        "class_labels": LSTM_CLASS_LABELS,
        "clean_training_rows": int(len(clean_df)),
        "sequence_count": int(len(sequences)),
    }
    saved_path = save_lstm_artifact(
        model=model,
        scaler=scaler,
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
        "sequence_count": int(len(sequences)),
        "target_distribution": target_distribution,
        "sequence_length": int(sequence_length),
        "lookahead_days": int(lookahead_days),
        "direction_threshold": float(threshold),
        "decision_threshold": validation["decision_threshold"],
        "train_before_date": normalize_optional_date(train_before_date),
        "training_start_date": training_date_range["start_date"],
        "training_end_date": training_date_range["end_date"],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "metrics": validation["metrics"],
    }


def prepare_lstm_training_frame(
    df: pd.DataFrame,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    threshold: float = TARGET_RETURN_THRESHOLD,
    features: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = features or FEATURES
    if df is None or df.empty:
        return pd.DataFrame()

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
    clean_df["lstm_target_direction"] = (
        clean_df["future_close_return"] > threshold
    ).astype(int)
    clean_df.loc[clean_df["future_close_return"].isna(), "lstm_target_direction"] = pd.NA
    clean_df = sort_for_chronological_training(clean_df)
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=feature_names + ["lstm_target_direction"]).copy()
    clean_df["lstm_target_direction"] = clean_df["lstm_target_direction"].astype(int)
    return clean_df.reset_index(drop=True)


def build_lstm_sequences(
    clean_df: pd.DataFrame,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_names = features or FEATURES
    sequence_length = max(2, int(sequence_length))
    if clean_df is None or clean_df.empty:
        return empty_sequence_arrays(sequence_length, len(feature_names))

    sequences = []
    targets = []
    meta_rows = []
    group_columns = target_group_columns(clean_df)
    grouped = (
        clean_df.groupby(group_columns, sort=False)
        if group_columns
        else [(None, clean_df)]
    )
    for _, group in grouped:
        group = group.sort_values("date", ascending=True).reset_index(drop=True)
        if len(group) < sequence_length:
            continue
        values = group[feature_names].to_numpy(dtype=np.float32)
        for end_index in range(sequence_length - 1, len(group)):
            window = values[end_index - sequence_length + 1 : end_index + 1]
            if np.isnan(window).any():
                continue
            row = group.iloc[end_index]
            sequences.append(window)
            targets.append(int(row["lstm_target_direction"]))
            meta_rows.append(meta_payload(row))

    if not sequences:
        return empty_sequence_arrays(sequence_length, len(feature_names))

    meta_df = pd.DataFrame(meta_rows)
    meta_df["_date_sort"] = pd.to_datetime(meta_df["date"], errors="coerce", utc=True)
    order = meta_df.sort_values(["_date_sort", "symbol"], ascending=True).index.to_numpy()
    meta_df = meta_df.iloc[order].drop(columns=["_date_sort"], errors="ignore").reset_index(drop=True)
    return (
        np.asarray(sequences, dtype=np.float32)[order],
        np.asarray(targets, dtype=np.int64)[order],
        meta_df,
    )


def walk_forward_lstm_validation(
    sequences: np.ndarray,
    targets: np.ndarray,
    sequence_meta: pd.DataFrame,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    n_splits: int,
) -> dict[str, Any]:
    fold_results = []
    all_true = []
    all_proba = []

    for fold_number, (train_index, test_index) in enumerate(
        chronological_splits(len(sequences), n_splits),
        start=1,
    ):
        X_train, X_test = sequences[train_index], sequences[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        if len(np.unique(y_train)) < 2:
            continue

        scaler = fit_sequence_scaler(X_train)
        X_train_scaled = transform_sequences(X_train, scaler)
        X_test_scaled = transform_sequences(X_test, scaler)
        model = train_lstm_model(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
        )
        probability_up = predict_lstm_probabilities(model, X_test_scaled)
        y_pred = (probability_up >= 0.5).astype(int)
        majority_class = int(pd.Series(y_train).mode().iloc[0])
        majority_pred = np.full(len(y_test), majority_class, dtype=int)

        fold_results.append(
            {
                "fold": fold_number,
                "test_start_date": str(sequence_meta.iloc[int(test_index[0])]["date"]),
                "test_end_date": str(sequence_meta.iloc[int(test_index[-1])]["date"]),
                "train_sequences": int(len(train_index)),
                "test_sequences": int(len(test_index)),
                **lstm_metrics(y_test, y_pred, probability_up),
                "majority_baseline_accuracy": float(
                    accuracy_score(y_test, majority_pred)
                ),
            }
        )
        all_true.append(y_test.astype(int))
        all_proba.append(probability_up)

    if not fold_results:
        return {
            "metrics": empty_lstm_metrics("no_validation_folds"),
            "decision_threshold": 0.5,
            "folds": [],
        }

    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    decision_threshold = find_best_lstm_decision_threshold(y_true, y_proba)
    threshold_pred = (y_proba >= decision_threshold).astype(int)

    return {
        "metrics": {
            **lstm_metrics(y_true, threshold_pred, y_proba),
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


def train_lstm_model(
    sequences: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
) -> TechnicalLSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TechnicalLSTM(
        input_size=sequences.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )

    positive_count = float((targets == 1).sum())
    negative_count = float((targets == 0).sum())
    pos_weight_value = negative_count / positive_count if positive_count else 1.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))

    model.train()
    for _ in range(max(1, int(epochs))):
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model.cpu()


def predict_lstm_probabilities(
    model: TechnicalLSTM,
    sequences: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(sequences, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def build_lstm_prediction_sequences(
    indicator_df: pd.DataFrame,
    sequence_length: int,
    features: list[str] | None = None,
    as_of_date: str | None = None,
    requested_symbol: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    feature_names = features or FEATURES
    rows = indicator_df.replace([np.inf, -np.inf], np.nan)
    rows = rows.dropna(subset=feature_names).copy()
    if rows.empty:
        return empty_prediction_arrays(sequence_length, len(feature_names))

    rows["date"] = pd.to_datetime(rows["date"], errors="coerce", utc=True).dt.date.astype(str)
    if requested_symbol and "symbol" in rows.columns:
        rows = rows[rows["symbol"].astype(str).str.upper() == requested_symbol.upper()]

    target_date = normalize_optional_date(as_of_date)
    sequences = []
    meta_rows = []
    group_columns = target_group_columns(rows)
    grouped = rows.groupby(group_columns, sort=False) if group_columns else [(None, rows)]
    for _, group in grouped:
        group = group.sort_values("date", ascending=True).reset_index(drop=True)
        if target_date:
            date_matches = group.index[group["date"] == target_date].tolist()
            if not date_matches:
                continue
            end_index = date_matches[-1]
        else:
            end_index = len(group) - 1

        start_index = end_index - sequence_length + 1
        if start_index < 0:
            continue

        window = group.iloc[start_index : end_index + 1][feature_names].to_numpy(
            dtype=np.float32
        )
        if len(window) != sequence_length or np.isnan(window).any():
            continue
        sequences.append(window)
        meta_rows.append(meta_payload(group.iloc[end_index]))

    if not sequences:
        return empty_prediction_arrays(sequence_length, len(feature_names))

    return np.asarray(sequences, dtype=np.float32), pd.DataFrame(meta_rows)


def predict_with_lstm_artifact(
    artifact: dict[str, Any],
    indicator_df: pd.DataFrame,
    as_of_date: str | None = None,
    requested_symbol: str | None = None,
) -> list[dict[str, Any]]:
    metadata = artifact.get("metadata", {})
    features = metadata.get("features", FEATURES)
    sequence_length = int(metadata.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    sequences, meta_df = build_lstm_prediction_sequences(
        indicator_df,
        sequence_length=sequence_length,
        features=features,
        as_of_date=as_of_date,
        requested_symbol=requested_symbol,
    )
    if len(sequences) == 0:
        return []

    scaler = artifact["scaler"]
    model = model_from_artifact(artifact)
    scaled_sequences = transform_sequences(sequences, scaler)
    probability_up = predict_lstm_probabilities(model, scaled_sequences)
    decision_threshold = float(metadata.get("decision_threshold") or 0.5)

    predictions = []
    for index, (_, row) in enumerate(meta_df.iterrows()):
        predicted_class = int(probability_up[index] >= decision_threshold)
        confidence = (
            probability_up[index]
            if predicted_class == 1
            else 1 - probability_up[index]
        )
        predictions.append(
            {
                "stock_id": row.get("stock_id"),
                "symbol": row.get("symbol"),
                "as_of_date": str(row["date"]),
                "as_of_close": float(row["close"]),
                "predicted_class": predicted_class,
                "predicted_direction": LSTM_CLASS_LABELS[predicted_class],
                "confidence": float(confidence),
                "probability_up": float(probability_up[index]),
                "decision_threshold": decision_threshold,
                "sequence_length": sequence_length,
                "lookahead_days": metadata.get("lookahead_days"),
                "direction_threshold": metadata.get("direction_threshold"),
            }
        )

    return predictions


def save_lstm_artifact(
    model: TechnicalLSTM,
    scaler: dict[str, Any],
    metadata: dict[str, Any],
    path: Path | str = LSTM_ARTIFACT_PATH,
) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "scaler": {
            "mean": np.asarray(scaler["mean"], dtype=float).tolist(),
            "scale": np.asarray(scaler["scale"], dtype=float).tolist(),
        },
        "metadata": {
            **metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        },
    }
    torch.save(payload, artifact_path)
    return artifact_path


def load_lstm_artifact(path: Path | str = LSTM_ARTIFACT_PATH) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def model_from_artifact(artifact: dict[str, Any]) -> TechnicalLSTM:
    metadata = artifact.get("metadata", {})
    model = TechnicalLSTM(
        input_size=int(metadata.get("feature_count", len(FEATURES))),
        hidden_size=int(metadata.get("hidden_size", DEFAULT_LSTM_HIDDEN_SIZE)),
        num_layers=int(metadata.get("num_layers", DEFAULT_LSTM_NUM_LAYERS)),
        dropout=float(metadata.get("dropout", DEFAULT_LSTM_DROPOUT)),
    )
    model.load_state_dict(artifact["model_state"])
    return model.cpu()


def fit_sequence_scaler(sequences: np.ndarray) -> dict[str, np.ndarray]:
    flat = sequences.reshape(-1, sequences.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale == 0] = 1.0
    return {"mean": mean, "scale": scale}


def transform_sequences(
    sequences: np.ndarray,
    scaler: dict[str, Any],
) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return ((sequences - mean) / scale).astype(np.float32)


def find_best_lstm_decision_threshold(
    y_true: np.ndarray,
    probability_up: np.ndarray,
) -> float:
    best_threshold = 0.5
    best_score = -np.inf
    best_accuracy = -np.inf
    for threshold in LSTM_DECISION_THRESHOLD_GRID:
        y_pred = (probability_up >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        if (f1, accuracy) > (best_score, best_accuracy):
            best_score = f1
            best_accuracy = accuracy
            best_threshold = threshold
    return float(best_threshold)


def lstm_metrics(
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


def empty_lstm_metrics(reason: str) -> dict[str, Any]:
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


def chronological_splits(
    sample_count: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if sample_count < 3:
        return []
    effective_splits = max(2, min(int(n_splits), sample_count - 1))
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    placeholder = np.arange(sample_count)
    return list(splitter.split(placeholder))


def meta_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "stock_id": (
            int(row["stock_id"])
            if "stock_id" in row and not pd.isna(row["stock_id"])
            else None
        ),
        "symbol": str(row["symbol"]).upper() if "symbol" in row else None,
        "date": str(row["date"]),
        "close": float(row["close"]),
    }


def empty_sequence_arrays(
    sequence_length: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    return (
        np.empty((0, sequence_length, feature_count), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        pd.DataFrame(),
    )


def empty_prediction_arrays(
    sequence_length: int,
    feature_count: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    return (
        np.empty((0, sequence_length, feature_count), dtype=np.float32),
        pd.DataFrame(),
    )


def average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))
