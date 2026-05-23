from database.supabase_client import supabase


def get_all_stocks():
    response = supabase.table("stocks").select("*").execute()
    return response.data


def get_stock_by_symbol(symbol: str):
    response = (
        supabase
        .table("stocks")
        .select("*")
        .eq("symbol", symbol.upper())
        .execute()
    )

    return response.data


def add_stock(stock_data: dict):
    response = (
        supabase
        .table("stocks")
        .insert(stock_data)
        .execute()
    )

    return response.data


def update_stock(symbol: str, stock_data: dict):
    response = (
        supabase
        .table("stocks")
        .update(stock_data)
        .eq("symbol", symbol.upper())
        .execute()
    )

    return response.data


def get_stock_history(symbol: str):
    response = (
        supabase
        .table("daily_ohlcv")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("trade_date", desc=False)
        .execute()
    )

    return response.data


def save_stock_history(rows: list):
    response = (
        supabase
        .table("daily_ohlcv")
        .upsert(rows, on_conflict="symbol,trade_date")
        .execute()
    )

    return response.data


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