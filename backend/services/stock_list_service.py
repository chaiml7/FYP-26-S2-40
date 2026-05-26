from database.supabase_client import supabase
from datetime import datetime, timezone

def get_active_stocks():
    response = (
        supabase
        .table("stocks")
        .select("*")
        .eq("is_active", True)
        .execute()
    )

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
    stock_data["symbol"] = stock_data["symbol"].upper()

    response = (
        supabase
        .table("stocks")
        .insert(stock_data)
        .execute()
    )

    return response.data


def deactivate_stock(symbol: str):
    response = (
        supabase
        .table("stocks")
        .update({"is_active": False})
        .eq("symbol", symbol.upper())
        .execute()
    )

    return response.data

def update_last_imported_at(symbol: str):
    response = (
        supabase
        .table("stocks")
        .update({"last_imported_at": datetime.now(timezone.utc).isoformat()})
        .eq("symbol", symbol.upper())
        .execute()
    )

    return response.data