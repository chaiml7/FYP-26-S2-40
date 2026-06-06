from database.supabase_client import supabase


def save_prediction(prediction_data: dict):
    response = (
        supabase
        .table("predictions")
        .insert(prediction_data)
        .execute()
    )

    return response.data


def get_predictions_by_symbol(symbol: str):
    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def get_latest_prediction_by_symbol(symbol: str):
    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    return response.data
