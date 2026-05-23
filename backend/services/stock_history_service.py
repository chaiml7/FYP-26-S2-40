from database.supabase_client import supabase


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
        .upsert(rows, on_conflict="symbol,trade_date")
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