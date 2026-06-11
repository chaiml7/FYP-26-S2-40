"""Supabase persistence for financial statements and outlook predictions."""

from database.supabase_client import supabase


PREDICTION_COLUMNS = {
    "stock_id",
    "ticker",
    "prediction",
    "score",
    "confidence",
    "model_version",
    "period",
    "created_at",
    "probabilities",
    "raw_outlook",
    "fundamental_score",
}


def load_financial_statements(symbol: str = None) -> list:
    query = (
        supabase.table("financial_statements")
        .select("*")
        .eq("period_type", "quarterly")
        .order("stock_id")
        .order("period")
    )
    if symbol:
        query = query.eq("ticker", symbol.upper())

    response = query.execute()
    return response.data or []


def save_financial_statements(statements: list) -> list:
    if not statements:
        return []
    response = (
        supabase.table("financial_statements")
        .upsert(
            statements,
            on_conflict="stock_id,period,period_type",
        )
        .execute()
    )
    return response.data or []


def save_financial_prediction(prediction: dict) -> dict:
    payload = {
        key: value
        for key, value in prediction.items()
        if key in PREDICTION_COLUMNS
    }
    response = (
        supabase.table("financial_predictions")
        .upsert(payload, on_conflict="stock_id,period,model_version")
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise RuntimeError("Supabase did not return the saved financial prediction.")

    stored = rows[0]
    required_score_fields = (
        "probabilities",
        "raw_outlook",
        "fundamental_score",
    )
    missing = [
        field
        for field in required_score_fields
        if stored.get(field) is None
    ]
    if missing:
        raise RuntimeError(
            "Financial prediction was returned without saved score fields: "
            + ", ".join(missing)
        )
    return stored


def get_financial_prediction_history(symbol: str) -> list:
    response = (
        supabase.table("financial_predictions")
        .select("*")
        .eq("ticker", symbol.upper())
        .order("period", desc=True)
        .execute()
    )
    return response.data or []


def get_latest_financial_prediction(
    symbol: str,
    model_version: str = None,
) -> dict:
    query = (
        supabase.table("financial_predictions")
        .select("*")
        .eq("ticker", symbol.upper())
        .order("period", desc=True)
        .order("created_at", desc=True)
        .limit(1)
    )
    if model_version:
        query = query.eq("model_version", model_version)

    response = query.execute()
    rows = response.data or []
    return rows[0] if rows else None


def save_model_version(model_metadata: dict) -> dict:
    response = (
        supabase.table("financial_model_versions")
        .insert(model_metadata)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else model_metadata


def activate_model_version(model_version: str) -> dict:
    (
        supabase.table("financial_model_versions")
        .update({"is_active": False})
        .eq("is_active", True)
        .execute()
    )
    response = (
        supabase.table("financial_model_versions")
        .update({"is_active": True})
        .eq("model_version", model_version)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def get_model_version(model_version: str) -> dict:
    response = (
        supabase.table("financial_model_versions")
        .select("*")
        .eq("model_version", model_version)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def get_active_model_version() -> dict:
    response = (
        supabase.table("financial_model_versions")
        .select("*")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def list_model_versions() -> list:
    response = (
        supabase.table("financial_model_versions")
        .select("*")
        .order("trained_at", desc=True)
        .execute()
    )
    return response.data or []
