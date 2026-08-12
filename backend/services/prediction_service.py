from backend.database.supabase_client import supabase
from datetime import date

from backend.services.sentiment.sentiment_aggregator import (
    get_latest_weighted_sentiment_score,
    get_weighted_sentiment_score,
)


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


def get_effective_weights(user_id: str) -> dict:
    """User's saved technical/sentiment/financial weights, falling back to
    the admin default (id=1), falling back to a hardcoded default."""
    try:
        user_w_res = (
            supabase.table("weightages")
            .select("technical, sentiment, financial")
            .eq("user_id", user_id)
            .execute()
        )
        if user_w_res.data:
            return user_w_res.data[0]

        admin_w_res = (
            supabase.table("weightages")
            .select("technical, sentiment, financial")
            .eq("id", "1")
            .execute()
        )
        if admin_w_res.data:
            return admin_w_res.data[0]
    except Exception as e:
        print(f"Error matching weight records: {e}")

    return {"technical": 40, "sentiment": 30, "financial": 30}


def score_action_label(composite: float) -> str:
    if composite >= 6.5:
        return "BUY"
    if composite <= 3.5:
        return "SELL"
    return "HOLD"


def get_latest_technical_signals(symbol: str) -> dict:
    """Latest stored RSI/MACD/volume/momentum/volatility snapshot for a symbol."""
    try:
        response = (
            supabase.table("technical_indicators")
            .select("date, rsi_14, macd, macd_signal, relative_volume, return_5d, rolling_volatility_20")
            .eq("symbol", symbol.upper())
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as e:
        print(f"Database error in get_latest_technical_signals for {symbol}: {e}")
        return None


def get_latest_direction_outlook(symbol: str) -> dict:
    """Latest directional model output (bullish/bearish/neutral + confidence)."""
    try:
        response = (
            supabase.table("direction_predictions")
            .select("prediction, probabilities, latest_date")
            .eq("symbol", symbol.upper())
            .order("latest_date", desc=True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as e:
        print(f"Database error in get_latest_direction_outlook for {symbol}: {e}")
        return None


# Heuristic thresholds on 20-day rolling stdev of daily returns. Large-cap
# equities typically run ~1-2% daily stdev; growth/small caps run higher.
RISK_TIER_THRESHOLDS = (0.015, 0.03)


def risk_tier_from_volatility(rolling_volatility_20) -> tuple[str, str]:
    """Maps 20D return volatility to a (slug, label) risk tier.

    Returns ("moderate", "Medium") when volatility data isn't available yet,
    since that's the neutral/unknown case rather than a real assessment.
    """
    if rolling_volatility_20 is None:
        return "moderate", "Medium"

    volatility = float(rolling_volatility_20)
    low, high = RISK_TIER_THRESHOLDS
    if volatility < low:
        return "conservative", "Low"
    if volatility > high:
        return "aggressive", "High"
    return "moderate", "Medium"


def build_composite_scorecard(symbol: str, weights: dict, sentiment_date: date = None) -> dict:
    """Real technical + sentiment + financial composite score for a symbol.

    This is the single source of truth for the weighted score shown on both
    the Recommendations list and the Prediction Breakdown page, so the two
    can never drift out of sync with each other.
    """
    target_symbol = symbol.upper()

    tech_w = weights.get("technical", 40)
    sent_w = weights.get("sentiment", 30)
    fin_w = weights.get("financial", 30)

    if sentiment_date is not None:
        sentiment_data = get_weighted_sentiment_score(target_symbol, sentiment_date)
    else:
        sentiment_data = get_latest_weighted_sentiment_score(target_symbol)

    raw_sent = int((sentiment_data.get("bullish_score") or 0)) if sentiment_data else 0

    try:
        tech_data = get_technical_score(target_symbol)
        raw_tech = tech_data if isinstance(tech_data, (int, float)) else tech_data.get("score", 0)
    except Exception:
        raw_tech = 0

    try:
        fin_data = get_financial_score(target_symbol)
        raw_fin = fin_data if isinstance(fin_data, (int, float)) else fin_data.get("score", 0)
    except Exception:
        raw_fin = 0

    composite_score = round(
        (raw_tech * (tech_w / 100.0)) +
        (raw_sent * (sent_w / 100.0)) +
        (raw_fin * (fin_w / 100.0)),
        2,
    )

    return {
        "symbol": target_symbol,
        "action": score_action_label(composite_score),
        "composite": composite_score,
        "technical_score": round(float(raw_tech), 2),
        "sentiment_score": round(float(raw_sent), 2),
        "financial_score": round(float(raw_fin), 2),
        "tech_weight": tech_w,
        "sent_weight": sent_w,
        "fin_weight": fin_w,
    }
