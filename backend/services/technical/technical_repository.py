"""Supabase persistence for technical model versions and predictions."""

from backend.database.supabase_client import supabase


PREDICTION_COLUMNS = {
    "stock_id",
    "symbol",
    "latest_date",
    "latest_close",
    "prediction",
    "probabilities",
    "raw_outlook",
    "technical_score",
    "prediction_horizon",
    "model_version",
    "created_at",
}

MODEL_REGISTRY_COLUMNS = {
    "model_version",
    "model_path",
    "metadata_path",
    "trained_at",
    "training_rows",
    "train_rows",
    "validation_rows",
    "test_rows",
    "dataset_start",
    "dataset_end",
    "class_distribution",
    "hyperparameters",
    "validation_metrics",
    "test_metrics",
    "feature_columns",
    "labels",
    "return_threshold",
    "evaluation_mode",
}


def save_technical_prediction(prediction: dict) -> dict:
    payload = {
        key: value
        for key, value in prediction.items()
        if key in PREDICTION_COLUMNS
    }
    response = (
        supabase.table("direction_predictions")
        .upsert(
            payload,
            on_conflict="stock_id,latest_date,model_version",
        )
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise RuntimeError("Supabase did not return the technical prediction.")

    stored = rows[0]
    required = (
        "prediction",
        "probabilities",
        "raw_outlook",
        "technical_score",
        "model_version",
    )
    missing = [field for field in required if stored.get(field) is None]
    if missing:
        raise RuntimeError(
            "Technical prediction was returned without fields: "
            + ", ".join(missing)
        )
    return stored


def save_model_version(metadata: dict) -> dict:
    payload = {
        key: value
        for key, value in metadata.items()
        if key in MODEL_REGISTRY_COLUMNS
    }
    response = (
        supabase.table("technical_model_versions")
        .insert(payload)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else payload


def get_model_version(model_version: str) -> dict:
    response = (
        supabase.table("technical_model_versions")
        .select("*")
        .eq("model_version", model_version)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def get_active_model_version() -> dict:
    response = (
        supabase.table("technical_model_versions")
        .select("*")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def list_model_versions() -> list:
    response = (
        supabase.table("technical_model_versions")
        .select("*")
        .order("trained_at", desc=True)
        .execute()
    )
    return response.data or []


def activate_model_version(model_version: str) -> dict:
    (
        supabase.table("technical_model_versions")
        .update({"is_active": False})
        .eq("is_active", True)
        .execute()
    )
    response = (
        supabase.table("technical_model_versions")
        .update({"is_active": True})
        .eq("model_version", model_version)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def get_latest_prediction(
    symbol: str,
    model_version: str = None,
) -> dict:
    query = (
        supabase.table("direction_predictions")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("latest_date", desc=True)
        .order("created_at", desc=True)
        .limit(1)
    )
    if model_version:
        query = query.eq("model_version", model_version)
    response = query.execute()
    rows = response.data or []
    return rows[0] if rows else None


def get_prediction_history(symbol: str) -> list:
    response = (
        supabase.table("direction_predictions")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("latest_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []
