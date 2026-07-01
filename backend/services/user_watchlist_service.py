from backend.database.supabase_client import supabase
from backend.services.stock_list_service import get_stock_by_symbol


def get_user_watchlist(user_id: str):
    response = (
        supabase
        .table("user_watchlists")
        .select("id, user_id, stock_id, created_at, stocks(*)")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def add_user_watchlist_stock(user_id: str, stock_id: int):
    response = (
        supabase
        .table("user_watchlists")
        .upsert(
            {
                "user_id": user_id,
                "stock_id": stock_id,
            },
            on_conflict="user_id,stock_id",
        )
        .execute()
    )

    return response.data


def remove_user_watchlist_stock(user_id: str, stock_id: int):
    response = (
        supabase
        .table("user_watchlists")
        .delete()
        .eq("user_id", user_id)
        .eq("stock_id", stock_id)
        .execute()
    )

    return response.data


def add_watchlist_by_symbol(user_id: str, symbol: str) -> dict:
    stocks = get_stock_by_symbol(symbol)
    if not stocks:
        raise ValueError(f"{symbol.upper()} is not in the stocks table")

    stock_id = stocks[0]["id"]
    result = add_user_watchlist_stock(user_id, stock_id) or []
    return result[0] if result else {"user_id": user_id, "stock_id": stock_id}


def remove_watchlist_by_symbol(user_id: str, symbol: str) -> dict:
    stocks = get_stock_by_symbol(symbol)
    if not stocks:
        raise ValueError(f"{symbol.upper()} is not in the stocks table")

    stock_id = stocks[0]["id"]
    removed = remove_user_watchlist_stock(user_id, stock_id)
    return {"stock_id": stock_id, "rows_deleted": len(removed)}


def get_user_watchlist_symbols(user_id: str) -> list:
    watchlist = get_user_watchlist(user_id)
    return [
        item["stocks"]["symbol"]
        for item in watchlist
        if item.get("stocks") and item["stocks"].get("symbol")
    ]
