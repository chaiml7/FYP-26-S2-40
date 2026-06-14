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

def get_technical_score(symbol: str, score_date: date = None) -> float:
    try:
        query = (
            supabase.table("direction_predictions")
            .select("technical_score")
            .eq("symbol", symbol.upper())
            .order("latest_date", desc=True)
            .order("created_at", desc=True)
            .limit(1)
        )
        if score_date is not None:
            query = query.eq("latest_date", score_date.isoformat())
        response = query.execute()
        
        rows = response.data or []
        if not rows:
            return 0 
            
        technical_score = rows[0].get("technical_score")
        if technical_score is None:
            return 0
            
        return round(float(technical_score), 2)
        
    except Exception as e:
        print(f"Database error in get_technical_score for {symbol}: {e}")
        return 0 


def get_financial_score(symbol: str, score_date: date = None) -> float:
    try:
        query = (
            supabase.table("financial_predictions")
            .select("fundamental_score")
            .eq("ticker", symbol.upper())
            .order("created_at", desc=True)
            .limit(1)
        )
        if score_date is not None:
            query = query.eq("period", score_date.isoformat())
        response = query.execute()
        
        rows = response.data or []
        if not rows:
            return 0 
            
        fundamental_score = rows[0].get("fundamental_score")
        if fundamental_score is None:
            return 0
            
        return round(float(fundamental_score), 2)
        
    except Exception as e:
        print(f"Database error in get_financial_score for {symbol}: {e}")
        return 0
