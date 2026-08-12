from backend.database.supabase_client import supabase
from datetime import datetime, timezone


def get_all_stocks():
    try:
        response = (
            supabase
            .table("stocks")
            .select("*")
            .order("symbol")
            .execute()
        )
        return response.data if response.data else []
        
    except Exception as e:
        print(f"Database Exception in get_all_stocks service: {e}")
        return []


def get_active_stocks():
    response = (
        supabase
        .table("stocks")
        .select("*")
        .eq("is_active", True)
        .execute()
    )

    return response.data


def get_inactive_stocks():
    response = (
        supabase
        .table("stocks")
        .select("*")
        .eq("is_active", False)
        .order("symbol")
        .execute()
    )

    return response.data


def get_stocks_by_sector(sector: str):
    response = (
        supabase
        .table("stocks")
        .select("*")
        .eq("sector", sector)
        .order("symbol")
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


def update_stock(symbol: str, stock_data: dict):
    response = (
        supabase
        .table("stocks")
        .update(stock_data)
        .eq("symbol", symbol.upper())
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


def search_active_stocks(q: str, limit: int = 8) -> list[dict]:
    """Case-insensitive search over active stocks by symbol-prefix or
    company-name substring. Symbol-prefix matches rank before company-name
    matches; each group preserves get_active_stocks()'s original order."""
    if not q or not q.strip():
        return []

    needle = q.strip().lower()
    stocks = get_active_stocks() or []

    symbol_matches = [s for s in stocks if s.get("symbol", "").lower().startswith(needle)]
    name_matches = [
        s for s in stocks
        if not s.get("symbol", "").lower().startswith(needle)
        and needle in (s.get("company_name") or "").lower()
    ]

    return (symbol_matches + name_matches)[:limit]
