import pandas as pd
from backend.database import get_client
 
 
def store_predictions(predictions: list[dict]) -> None:
    """Upsert all quarter predictions into financial_predictions."""
    if not predictions:
        print("  No predictions to store.")
        return
    supabase = get_client()
    supabase.table("financial_predictions").upsert(
        predictions,
        on_conflict="stock_id,period,model_version"
    ).execute()
    print(f"  Stored {len(predictions)} rows → financial_predictions")
 
 
def get_latest_predictions() -> pd.DataFrame:
    """Return the most recent prediction per ticker."""
    supabase = get_client()
    resp = (
        supabase.table("financial_predictions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    df = pd.DataFrame(resp.data)
    if df.empty:
        return df
    return df.sort_values("created_at", ascending=False).drop_duplicates("ticker")
 
 
def get_all_predictions() -> pd.DataFrame:
    """Return all predictions for all quarters and tickers."""
    supabase = get_client()
    resp = (
        supabase.table("financial_predictions")
        .select("*")
        .order("ticker")
        .order("period")
        .execute()
    )
    return pd.DataFrame(resp.data)
 
 
def get_predictions_by_ticker(ticker: str) -> pd.DataFrame:
    """Return full prediction history for a specific ticker."""
    supabase = get_client()
    resp = (
        supabase.table("financial_predictions")
        .select("*")
        .eq("ticker", ticker.upper())
        .order("period")
        .execute()
    )
    return pd.DataFrame(resp.data)
 
 
def get_predictions_by_outlook(prediction: str) -> pd.DataFrame:
    """Return all predictions matching a given outlook."""
    if prediction not in ("positive", "neutral", "negative"):
        raise ValueError("prediction must be 'positive', 'neutral', or 'negative'")
    supabase = get_client()
    resp = (
        supabase.table("financial_predictions")
        .select("*")
        .eq("prediction", prediction)
        .order("confidence", desc=True)
        .execute()
    )
    return pd.DataFrame(resp.data)