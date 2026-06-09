from database.supabase_client import supabase


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
