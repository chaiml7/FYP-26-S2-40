from backend.database.supabase_client import supabase
from backend.services.stock_list_service import get_stock_by_symbol
from backend.services.dashboard_service import _price_summary
from backend.services.prediction_service import get_latest_prediction_by_symbol
from backend.services.sentiment.sentiment_aggregator import get_sentiment_summary


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


def get_user_watchlist_summary(user_id: str) -> list:
    watchlist = get_user_watchlist(user_id)
    summary = []

    for item in watchlist:
        stock = item.get("stocks") or {}
        symbol = stock.get("symbol")

        price = _price_summary(symbol) if symbol else {
            "price": None, "change": None, "change_percent": None, "trade_date": None,
        }
        predictions = get_latest_prediction_by_symbol(symbol) if symbol else []
        prediction = predictions[0] if predictions else None

        try:
            sentiment = get_sentiment_summary(symbol) if symbol else {}
        except Exception:
            sentiment = {}

        weighted_scores = sentiment.get("weighted_scores") or []
        legacy_daily_scores = sentiment.get("daily_scores") or []

        if weighted_scores:
            sentiment_label = weighted_scores[0].get("sentiment_label")
            sentiment_score = weighted_scores[0].get("bullish_score")
        elif legacy_daily_scores:
            sentiment_label = legacy_daily_scores[0].get("label")
            sentiment_score = legacy_daily_scores[0].get("avg_score")
        else:
            sentiment_label = None
            sentiment_score = None

        summary.append({
            "watchlist_id": item["id"],
            "stock_id": item["stock_id"],
            "symbol": symbol,
            "company_name": stock.get("company_name"),
            "sector": stock.get("sector"),
            "price": price["price"],
            "change": price["change"],
            "change_percent": price["change_percent"],
            "trade_date": price["trade_date"],
            "prediction_signal": prediction.get("signal") if prediction else None,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "added_at": item["created_at"],
        })

    return summary
