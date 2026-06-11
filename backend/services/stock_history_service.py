from backend.database.supabase_client import supabase


def save_stock_history(rows: list):
    if len(rows) == 0:
        return {
            "success": False,
            "message": "No rows to save",
            "rows_saved": 0
        }

    response = (
        supabase
        .table("daily_ohlcv")
        .upsert(rows, on_conflict="stock_id,trade_date")
        .execute()
    )

    return {
        "success": True,
        "message": "Stock history saved successfully",
        "rows_saved": len(rows),
        "data": response.data
    }


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


def get_latest_stock_price(symbol: str):
    response = (
        supabase
        .table("daily_ohlcv")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("trade_date", desc=True)
        .limit(1)
        .execute()
    )

    return response.data


def get_stock_history_by_date_range(symbol: str, start_date: str, end_date: str):
    response = (
        supabase
        .table("daily_ohlcv")
        .select("*")
        .eq("symbol", symbol.upper())
        .gte("trade_date", start_date)
        .lte("trade_date", end_date)
        .order("trade_date", desc=False)
        .execute()
    )

    return response.data


def delete_stock_history(symbol: str):
    response = (
        supabase
        .table("daily_ohlcv")
        .delete()
        .eq("symbol", symbol.upper())
        .execute()
    )

    return response.data
