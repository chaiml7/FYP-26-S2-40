"""Read models for the authenticated stock dashboard."""

from backend.database.supabase_client import supabase
from backend.services.stock_list_service import get_active_stocks, get_stock_by_symbol


def _first(rows: list) -> dict | None:
    return rows[0] if rows else None


def _score_tone(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score >= 6:
        return "bullish"
    if score < 4:
        return "bearish"
    return "neutral"


def _recent_prices(symbol: str, limit: int = 2) -> list:
    response = (
        supabase.table("daily_ohlcv")
        .select("trade_date,open,high,low,close,volume")
        .eq("symbol", symbol.upper())
        .order("trade_date", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def _price_summary(symbol: str) -> dict:
    rows = _recent_prices(symbol)
    latest = _first(rows)
    previous = rows[1] if len(rows) > 1 else None

    if not latest or latest.get("close") is None:
        return {
            "price": None,
            "change": None,
            "change_percent": None,
            "trade_date": None,
        }

    price = float(latest["close"])
    previous_close = (
        float(previous["close"])
        if previous and previous.get("close") is not None
        else None
    )
    change = price - previous_close if previous_close else None
    change_percent = (
        (change / previous_close) * 100
        if change is not None and previous_close
        else None
    )
    return {
        "price": round(price, 2),
        "change": round(change, 2) if change is not None else None,
        "change_percent": (
            round(change_percent, 2) if change_percent is not None else None
        ),
        "trade_date": latest.get("trade_date"),
    }


def _technical_prediction(symbol: str) -> dict | None:
    response = (
        supabase.table("direction_predictions")
        .select(
            "prediction,probabilities,raw_outlook,technical_score,"
            "prediction_horizon,model_version,latest_date"
        )
        .eq("symbol", symbol.upper())
        .order("latest_date", desc=True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return _first(response.data or [])


def _sentiment_prediction(symbol: str) -> dict | None:
    response = (
        supabase.table("sentiment_daily_scores")
        .select(
            "bullish_score,sentiment_label,score_date,article_count,"
            "positive_count,neutral_count,negative_count,model_version"
        )
        .eq("symbol", symbol.upper())
        .order("score_date", desc=True)
        .limit(1)
        .execute()
    )
    return _first(response.data or [])


def _financial_prediction(symbol: str) -> dict | None:
    response = (
        supabase.table("financial_predictions")
        .select(
            "prediction,probabilities,raw_outlook,fundamental_score,"
            "confidence,period,model_version,created_at"
        )
        .eq("ticker", symbol.upper())
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return _first(response.data or [])


def _score_card(name: str, score: float | None, detail: dict | None) -> dict:
    normalized = round(float(score), 2) if score is not None else None
    return {
        "name": name,
        "score": normalized,
        "tone": _score_tone(normalized),
        "detail": detail,
    }


def get_dashboard_stocks() -> list:
    stocks = get_active_stocks() or []
    dashboard_stocks = []

    for stock in sorted(stocks, key=lambda item: item.get("symbol", "")):
        symbol = stock.get("symbol", "").upper()
        dashboard_stocks.append({
            **stock,
            "symbol": symbol,
            "company_name": stock.get("company_name") or symbol,
            **_price_summary(symbol),
        })

    return dashboard_stocks


def get_stock_dashboard(symbol: str) -> dict | None:
    stocks = get_stock_by_symbol(symbol)
    stock = _first(stocks or [])
    if stock is None or stock.get("is_active") is False:
        return None

    symbol = stock["symbol"].upper()
    technical = _technical_prediction(symbol)
    sentiment = _sentiment_prediction(symbol)
    financial = _financial_prediction(symbol)
    history = list(reversed(_recent_prices(symbol, limit=10)))

    return {
        **stock,
        "symbol": symbol,
        "company_name": stock.get("company_name") or symbol,
        **_price_summary(symbol),
        "scores": [
            _score_card(
                "Technical",
                technical.get("technical_score") if technical else None,
                technical,
            ),
            _score_card(
                "Sentiment",
                sentiment.get("bullish_score") if sentiment else None,
                sentiment,
            ),
            _score_card(
                "Financial",
                financial.get("fundamental_score") if financial else None,
                financial,
            ),
        ],
        "price_history": history,
    }
