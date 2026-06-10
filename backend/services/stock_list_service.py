from backend.database.supabase_client import get_client

TICKERS = ["AAPL", "MSFT", "TSLA", "AMD", "AMZN", "GOOGL", "META", "NVDA", "PLTTR", "AVGO", "ORCL"]


def get_tracked_tickers() -> list[str]:
    return TICKERS


def get_stock_by_symbol(symbol: str) -> dict | None:
    supabase = get_client()
    resp = (
        supabase.table("stocks")
        .select("*")
        .eq("symbol", symbol.upper())
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None