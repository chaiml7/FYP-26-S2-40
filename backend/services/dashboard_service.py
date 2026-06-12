"""Read models for the authenticated stock dashboard."""

from datetime import date

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


def _recent_prices(
    symbol: str,
    limit: int = 2,
    selected_date: date = None,
) -> list:
    query = (
        supabase.table("daily_ohlcv")
        .select("trade_date,open,high,low,close,volume")
        .eq("symbol", symbol.upper())
        .order("trade_date", desc=True)
        .limit(limit)
    )
    if selected_date is not None:
        query = query.lte("trade_date", selected_date.isoformat())
    response = query.execute()
    return response.data or []


def _empty_price_summary() -> dict:
    return {
        "price": None,
        "change": None,
        "change_percent": None,
        "trade_date": None,
    }


def _price_summary_from_rows(
    rows: list,
    selected_date: date = None,
) -> dict:
    latest = _first(rows)
    previous = rows[1] if len(rows) > 1 else None

    if (
        not latest
        or latest.get("close") is None
        or (
            selected_date is not None
            and latest.get("trade_date") != selected_date.isoformat()
        )
    ):
        return _empty_price_summary()

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


def _price_summary(symbol: str, selected_date: date = None) -> dict:
    rows = _recent_prices(symbol, selected_date=selected_date)
    return _price_summary_from_rows(rows, selected_date)


def _dashboard_price_summaries(
    symbols: list[str],
    selected_date: date = None,
) -> dict[str, dict]:
    if not symbols:
        return {}

    query = (
        supabase.table("daily_ohlcv")
        .select("symbol,trade_date,open,high,low,close,volume")
        .in_("symbol", symbols)
        .order("trade_date", desc=True)
        .limit(max(200, len(symbols) * 6))
    )
    if selected_date is not None:
        query = query.lte("trade_date", selected_date.isoformat())

    response = query.execute()
    rows_by_symbol = {symbol: [] for symbol in symbols}
    for row in response.data or []:
        symbol = str(row.get("symbol", "")).upper()
        if symbol in rows_by_symbol and len(rows_by_symbol[symbol]) < 2:
            rows_by_symbol[symbol].append(row)

    return {
        symbol: _price_summary_from_rows(rows, selected_date)
        for symbol, rows in rows_by_symbol.items()
    }


def _technical_prediction(
    symbol: str,
    selected_date: date = None,
) -> dict | None:
    query = (
        supabase.table("direction_predictions")
        .select(
            "prediction,probabilities,raw_outlook,technical_score,"
            "prediction_horizon,model_version,latest_date"
        )
        .eq("symbol", symbol.upper())
        .order("latest_date", desc=True)
        .order("created_at", desc=True)
        .limit(1)
    )
    if selected_date is not None:
        query = query.eq("latest_date", selected_date.isoformat())
    response = query.execute()
    return _first(response.data or [])


def _sentiment_prediction(
    symbol: str,
    selected_date: date = None,
) -> dict | None:
    query = (
        supabase.table("sentiment_daily_scores")
        .select(
            "bullish_score,sentiment_label,score_date,article_count,"
            "positive_count,neutral_count,negative_count,model_version"
        )
        .eq("symbol", symbol.upper())
        .order("score_date", desc=True)
        .limit(1)
    )
    if selected_date is not None:
        query = query.eq("score_date", selected_date.isoformat())
    response = query.execute()
    return _first(response.data or [])


def _financial_prediction(
    symbol: str,
    selected_date: date = None,
) -> dict | None:
    query = (
        supabase.table("financial_predictions")
        .select(
            "prediction,probabilities,raw_outlook,fundamental_score,"
            "confidence,period,model_version,created_at"
        )
        .eq("ticker", symbol.upper())
        .order("period", desc=True)
        .order("created_at", desc=True)
        .limit(1)
    )
    if selected_date is not None:
        query = query.lte("period", selected_date.isoformat())
    response = query.execute()
    return _first(response.data or [])


def _score_card(name: str, score: float | None, detail: dict | None) -> dict:
    normalized = round(float(score), 2) if score is not None else None
    return {
        "name": name,
        "score": normalized,
        "tone": _score_tone(normalized),
        "detail": detail,
    }


def get_dashboard_stocks(selected_date: date = None) -> list:
    stocks = get_active_stocks() or []
    dashboard_stocks = []
    sorted_stocks = sorted(stocks, key=lambda item: item.get("symbol", ""))
    symbols = [
        stock.get("symbol", "").upper()
        for stock in sorted_stocks
        if stock.get("symbol")
    ]

    try:
        price_summaries = _dashboard_price_summaries(
            symbols,
            selected_date,
        )
    except Exception as exc:
        print(f"Dashboard price batch lookup failed: {exc}")
        price_summaries = {}

    for stock in sorted_stocks:
        symbol = stock.get("symbol", "").upper()
        dashboard_stocks.append({
            **stock,
            "symbol": symbol,
            "company_name": stock.get("company_name") or symbol,
            **price_summaries.get(symbol, _empty_price_summary()),
        })

    return dashboard_stocks


def get_stock_dashboard(
    symbol: str,
    selected_date: date = None,
) -> dict | None:
    stocks = get_stock_by_symbol(symbol)
    stock = _first(stocks or [])
    if stock is None or stock.get("is_active") is False:
        return None

    symbol = stock["symbol"].upper()
    technical = _technical_prediction(symbol, selected_date)
    sentiment = _sentiment_prediction(symbol, selected_date)
    financial = _financial_prediction(symbol, selected_date)
    history = list(reversed(
        _recent_prices(symbol, limit=10, selected_date=selected_date)
    ))
    chart_history = list(reversed(
        _recent_prices(symbol, limit=60, selected_date=selected_date)
    ))

    return {
        **stock,
        "symbol": symbol,
        "company_name": stock.get("company_name") or symbol,
        **_price_summary(symbol, selected_date),
        "selected_date": (
            selected_date.isoformat() if selected_date is not None else None
        ),
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
        "chart_history": chart_history,
        "price_history": history,
    }
