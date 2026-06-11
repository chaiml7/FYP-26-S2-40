import pandas as pd
from backend.database import get_client
 
# Maps stock symbol in Supabase stocks table → Yahoo Finance ticker
SYMBOL_TO_YFINANCE = {
    
}
 
 
def get_stock_id_map() -> dict:
    """Load {symbol: stock_id} from the stocks table."""
    supabase = get_client()
    resp = supabase.table("stocks").select("id, symbol").execute()
    mapping = {row["symbol"].upper(): row["id"] for row in resp.data}
    print(f"  Loaded {len(mapping)} stocks from stocks table.")
    return mapping
 
 
def get_yfinance_ticker(symbol: str) -> str:
    """Return the correct Yahoo Finance ticker for a given DB symbol."""
    return SYMBOL_TO_YFINANCE.get(symbol.upper(), symbol.upper())
 
 
def store_statements(records: list[dict]) -> None:
    """Upsert financial statement records into financial_statements."""
    if not records:
        print("  No records to store.")
        return
    supabase = get_client()
    supabase.table("financial_statements").upsert(
        records,
        on_conflict="stock_id,period,period_type"
    ).execute()
    print(f"  Stored {len(records)} rows → financial_statements")
 
 
def load_statements() -> pd.DataFrame:
    """Load all rows from financial_statements for model training."""
    supabase = get_client()
    resp = supabase.table("financial_statements").select("*").execute()
    df = pd.DataFrame(resp.data)
    print(f"  Loaded {len(df)} rows from financial_statements")
    return df
 
 
def load_statements_by_ticker(ticker: str) -> pd.DataFrame:
    """Load all financial_statements rows for a specific ticker."""
    supabase = get_client()
    resp = (
        supabase.table("financial_statements")
        .select("*")
        .eq("ticker", ticker.upper())
        .order("period")
        .execute()
    )
    return pd.DataFrame(resp.data)