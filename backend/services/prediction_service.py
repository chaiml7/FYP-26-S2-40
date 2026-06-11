from backend.database.supabase_client import supabase
from datetime import date


def save_prediction(prediction_data: dict):
    response = (
        supabase
        .table("predictions")
        .insert(prediction_data)
        .execute()
    )

    return response.data


def get_predictions_by_symbol(symbol: str):
    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def get_latest_prediction_by_symbol(symbol: str):
    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    return response.data

def get_technical_score(symbol: str, score_date: date = None) -> int:
    score_date = score_date or date.today()
    day = score_date.isoformat()

    try:
        response = (
            supabase.table("direction_predictions")
            .select("predicted_probability")
            .eq("symbol", symbol.upper())
            .eq("latest_date", day)
            .limit(1)
            .execute()
        )
        
        rows = response.data or []
        if not rows:
            return 0 
            
        raw_score = rows[0].get("predicted_probability")
        if raw_score is None:
            return 0
            
        return int(round(float(raw_score) * 10))
        
    except Exception as e:
        print(f"Database error in get_technical_score for {symbol}: {e}")
        return 0 


def get_financial_score(symbol: str, score_date: date = None) -> int:
    score_date = score_date or date.today()
    day = score_date.isoformat()

    try:
        response = (
            supabase.table("financial_predictions")
            .select("score")
            .eq("ticker", symbol.upper())
            .eq("period", day)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        rows = response.data or []
        if not rows:
            return 0 
            
        raw_score = rows[0].get("score")
        if raw_score is None:
            return 0
            
        return int(round(float(raw_score) * 10))
        
    except Exception as e:
        print(f"Database error in get_financial_score for {symbol}: {e}")
        return 0