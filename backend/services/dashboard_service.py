"""Read models for the authenticated stock dashboard."""

from datetime import date

from backend.database.supabase_client import supabase
from backend.services.prediction_service import score_action_label
from backend.services.stock_list_service import get_active_stocks, get_stock_by_symbol


DEFAULT_MODEL_WEIGHTS = {
    "technical": 40,
    "sentiment": 30,
    "financial": 30,
}


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


def _validated_model_weights(row: dict | None) -> dict[str, int] | None:
    if not isinstance(row, dict):
        return None
    try:
        weights = {
            name: int(row.get(name))
            for name in DEFAULT_MODEL_WEIGHTS
        }
    except (TypeError, ValueError):
        return None
    if any(value < 0 or value > 100 for value in weights.values()):
        return None
    return weights if sum(weights.values()) == 100 else None


def get_model_weights(user_id: str | None = None) -> dict[str, int]:
    """Return premium-user weights, then platform defaults, then safe defaults."""
    try:
        if user_id:
            user_response = (
                supabase.table("weightages")
                .select("technical,sentiment,financial")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            user_rows = user_response.data or []
            user_weights = _validated_model_weights(
                user_rows[0] if isinstance(user_rows, list) and user_rows else None
            )
            if user_weights:
                return user_weights

        default_response = (
            supabase.table("weightages")
            .select("technical,sentiment,financial")
            .eq("id", "1")
            .limit(1)
            .execute()
        )
        default_rows = default_response.data or []
        default_weights = _validated_model_weights(
            default_rows[0]
            if isinstance(default_rows, list) and default_rows
            else None
        )
        if default_weights:
            return default_weights
    except Exception as exc:
        print(f"Model weightage lookup failed: {exc}")

    return DEFAULT_MODEL_WEIGHTS.copy()


def _weighted_overall_score(
    technical_score: float | None,
    sentiment_score: float | None,
    financial_score: float | None,
    weights: dict[str, int],
) -> float | None:
    scores = {
        "technical": technical_score,
        "sentiment": sentiment_score,
        "financial": financial_score,
    }
    active_weight = sum(weights.values())
    if active_weight <= 0:
        return None
    if any(scores[name] is None for name, weight in weights.items() if weight > 0):
        return None
    weighted_total = sum(
        float(scores[name]) * weight
        for name, weight in weights.items()
        if weight > 0
    )
    return round(weighted_total / active_weight, 1)


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


def get_top_movers(limit: int = 5) -> dict[str, list]:
    """Top gainers/losers by day-over-day % change, restricted to the active
    stocks we list — real daily_ohlcv data, no external symbols."""
    stocks = get_active_stocks() or []
    stocks_by_symbol = {
        str(stock.get("symbol", "")).upper(): stock
        for stock in stocks
        if stock.get("symbol")
    }
    symbols = list(stocks_by_symbol.keys())
    summaries = _dashboard_price_summaries(symbols)

    movers = []
    for symbol, summary in summaries.items():
        if summary.get("change_percent") is None:
            continue
        stock = stocks_by_symbol.get(symbol, {})
        movers.append({
            "symbol": symbol,
            "company_name": stock.get("company_name"),
            "price": summary["price"],
            "change_percent": summary["change_percent"],
            "trade_date": summary["trade_date"],
        })

    ranked = sorted(movers, key=lambda row: row["change_percent"], reverse=True)
    return {
        "gainers": [row for row in ranked if row["change_percent"] > 0][:limit],
        "losers": [row for row in ranked if row["change_percent"] < 0][-limit:][::-1],
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


def _technical_indicator_snapshot(
    symbol: str,
    indicator_date: date | str = None,
) -> dict | None:
    query = (
        supabase.table("technical_indicators")
        .select(
            "date,close,return_1d,return_5d,rsi_14,macd,macd_signal,"
            "macd_histogram,bb_middle,bb_upper,bb_lower,bb_width,atr_14,"
            "sma_20,sma_50,sma_200,vwap_20,relative_volume,support_20,"
            "resistance_20"
        )
        .eq("symbol", symbol.upper())
        .order("date", desc=True)
        .limit(1)
    )
    if indicator_date is not None:
        date_value = (
            indicator_date.isoformat()
            if isinstance(indicator_date, date)
            else str(indicator_date)
        )
        query = query.eq("date", date_value)
    response = query.execute()
    return _first(response.data or [])


def _format_indicator_value(value, style: str = "number") -> str:
    if value is None:
        return "--"
    number = float(value)
    if style == "price":
        return f"${number:,.2f}"
    if style == "percent":
        return f"{number * 100:+.2f}%"
    if style == "unsigned_percent":
        return f"{number * 100:.2f}%"
    if style == "multiple":
        return f"{number:.2f}x"
    return f"{number:.2f}"


def _technical_indicator_groups(snapshot: dict | None) -> list:
    if not snapshot:
        return []

    groups = [
        (
            "Momentum",
            [
                ("RSI (14)", "rsi_14", "number"),
                ("MACD", "macd", "number"),
                ("MACD signal", "macd_signal", "number"),
                ("MACD histogram", "macd_histogram", "number"),
                ("1-day return", "return_1d", "percent"),
                ("5-day return", "return_5d", "percent"),
            ],
        ),
        (
            "Bollinger & volatility",
            [
                ("Upper band", "bb_upper", "price"),
                ("Middle band", "bb_middle", "price"),
                ("Lower band", "bb_lower", "price"),
                ("Band width", "bb_width", "unsigned_percent"),
                ("ATR (14)", "atr_14", "price"),
                ("Relative volume", "relative_volume", "multiple"),
            ],
        ),
        (
            "Trend & price levels",
            [
                ("Latest close", "close", "price"),
                ("SMA (20)", "sma_20", "price"),
                ("SMA (50)", "sma_50", "price"),
                ("SMA (200)", "sma_200", "price"),
                ("VWAP (20)", "vwap_20", "price"),
                ("Support (20)", "support_20", "price"),
                ("Resistance (20)", "resistance_20", "price"),
            ],
        ),
    ]
    return [
        {
            "title": title,
            "items": [
                {
                    "label": label,
                    "value": _format_indicator_value(snapshot.get(key), style),
                }
                for label, key, style in items
            ],
        }
        for title, items in groups
    ]


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


def _format_volume(volume: int | float | None) -> str:
    if volume is None:
        return "--"
    value = float(volume)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value):,}"


def _combined_score(
    technical_score: float | None,
    sentiment_score: float | None,
    financial_score: float | None,
) -> float | None:
    if any(
        score is None
        for score in (technical_score, sentiment_score, financial_score)
    ):
        return None
    return round(
        (float(technical_score) * 0.50)
        + (float(sentiment_score) * 0.30)
        + (float(financial_score) * 0.20),
        1,
    )


def get_public_market_leaders(limit: int = 5) -> list:
    stocks = get_active_stocks() or []
    stocks_by_symbol = {
        str(stock.get("symbol", "")).upper(): stock
        for stock in stocks
        if stock.get("symbol")
    }
    if not stocks_by_symbol:
        return []

    latest_response = (
        supabase.table("daily_ohlcv")
        .select("trade_date")
        .in_("symbol", list(stocks_by_symbol))
        .order("trade_date", desc=True)
        .limit(1)
        .execute()
    )
    latest_row = _first(latest_response.data or [])
    if latest_row is None:
        return []

    market_date = latest_row["trade_date"]
    volume_response = (
        supabase.table("daily_ohlcv")
        .select("symbol,trade_date,close,volume")
        .in_("symbol", list(stocks_by_symbol))
        .eq("trade_date", market_date)
        .order("volume", desc=True)
        .limit(limit)
        .execute()
    )

    leaders = []
    for rank, row in enumerate(volume_response.data or [], start=1):
        symbol = str(row.get("symbol", "")).upper()
        stock = stocks_by_symbol.get(symbol)
        if stock is None:
            continue

        technical = _technical_prediction(symbol)
        sentiment = _sentiment_prediction(symbol)
        financial = _financial_prediction(symbol)
        technical_score = technical.get("technical_score") if technical else None
        sentiment_score = sentiment.get("bullish_score") if sentiment else None
        financial_score = financial.get("fundamental_score") if financial else None
        overall_score = _combined_score(
            technical_score,
            sentiment_score,
            financial_score,
        )
        leaders.append({
            "rank": rank,
            "symbol": symbol,
            "company_name": stock.get("company_name") or symbol,
            "market_date": market_date,
            "volume": row.get("volume"),
            "volume_display": _format_volume(row.get("volume")),
            "overall_score": overall_score,
            "analysis": _score_tone(overall_score),
            "technical_score": (
                round(float(technical_score), 1)
                if technical_score is not None else None
            ),
            "sentiment_score": (
                round(float(sentiment_score), 1)
                if sentiment_score is not None else None
            ),
            "financial_score": (
                round(float(financial_score), 1)
                if financial_score is not None else None
            ),
        })

    return leaders


def _model_performance_row(
    name: str,
    version: str | None,
    metrics: dict | None,
    evaluation_mode: str | None,
) -> dict:
    metrics = metrics or {}
    f1_value = (
        metrics.get("macro_f1")
        if metrics.get("macro_f1") is not None
        else metrics.get("f1_score")
    )
    return {
        "name": name,
        "model_version": version or "Not available",
        "evaluation_mode": evaluation_mode or "Not evaluated",
        "evaluation_status": (
            "Held-out test"
            if metrics
            else "Not recorded"
        ),
        "accuracy": (
            round(float(metrics["accuracy"]) * 100, 1)
            if metrics.get("accuracy") is not None else None
        ),
        "balanced_accuracy": (
            round(float(metrics["balanced_accuracy"]) * 100, 1)
            if metrics.get("balanced_accuracy") is not None else None
        ),
        "macro_f1": (
            round(float(metrics["macro_f1"]) * 100, 1)
            if metrics.get("macro_f1") is not None else None
        ),
        "f1": (
            round(float(f1_value) * 100, 1)
            if f1_value is not None else None
        ),
        "log_loss": (
            round(float(metrics["log_loss"]), 4)
            if metrics.get("log_loss") is not None else None
        ),
    }


def get_public_model_metrics() -> list:
    financial_response = (
        supabase.table("financial_model_versions")
        .select("model_version,metrics,evaluation_mode,trained_at")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    technical_response = (
        supabase.table("technical_model_versions")
        .select("model_version,test_metrics,evaluation_mode,trained_at")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    sentiment_response = (
        supabase.table("sentiment_model_versions")
        .select("model_version,metrics,evaluation_mode,trained_at")
        .eq("is_active", True)
        .limit(1)
        .execute()
    )

    financial = _first(financial_response.data or []) or {}
    technical = _first(technical_response.data or []) or {}
    sentiment = _first(sentiment_response.data or []) or {}
    return [
        _model_performance_row(
            "Technical",
            technical.get("model_version"),
            technical.get("test_metrics"),
            technical.get("evaluation_mode"),
        ),
        _model_performance_row(
            "Sentiment",
            sentiment.get("model_version"),
            sentiment.get("metrics"),
            sentiment.get("evaluation_mode"),
        ),
        _model_performance_row(
            "Financial",
            financial.get("model_version"),
            financial.get("metrics"),
            financial.get("evaluation_mode"),
        ),
    ]


def get_dashboard_stocks(selected_date: date = None) -> list:
    stocks = get_active_stocks() or []
    dashboard_stocks = []
    sorted_stocks = sorted(
        stocks,
        key=lambda item: (
            str(item.get("company_name") or item.get("symbol") or "").casefold(),
            str(item.get("symbol") or "").casefold(),
        ),
    )
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
    include_technical_indicators: bool = False,
    weight_user_id: str | None = None,
) -> dict | None:
    stocks = get_stock_by_symbol(symbol)
    stock = _first(stocks or [])
    if stock is None or stock.get("is_active") is False:
        return None

    symbol = stock["symbol"].upper()
    technical = _technical_prediction(symbol, selected_date)
    indicator_date = (
        technical.get("latest_date")
        if technical and technical.get("latest_date")
        else selected_date
    )
    technical_indicators = None
    if include_technical_indicators:
        try:
            technical_indicators = _technical_indicator_snapshot(
                symbol,
                indicator_date,
            )
        except Exception as exc:
            print(f"Technical indicator detail lookup failed for {symbol}: {exc}")
    sentiment = _sentiment_prediction(symbol, selected_date)
    financial = _financial_prediction(symbol, selected_date)
    technical_score = technical.get("technical_score") if technical else None
    sentiment_score = sentiment.get("bullish_score") if sentiment else None
    financial_score = financial.get("fundamental_score") if financial else None
    model_weights = get_model_weights(weight_user_id)
    overall_score = _weighted_overall_score(
        technical_score,
        sentiment_score,
        financial_score,
        model_weights,
    )
    history = _recent_prices(
        symbol,
        limit=10,
        selected_date=selected_date,
    )
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
                technical_score,
                technical,
            ),
            _score_card(
                "Sentiment",
                sentiment_score,
                sentiment,
            ),
            _score_card(
                "Financial",
                financial_score,
                financial,
            ),
        ],
        "overall_score": overall_score,
        "overall_tone": _score_tone(overall_score),
        "action": score_action_label(overall_score) if overall_score is not None else None,
        "model_weights": model_weights,
        "component_scores": {
            "technical": technical_score,
            "sentiment": sentiment_score,
            "financial": financial_score,
        },
        "technical_indicator_date": (
            technical_indicators.get("date") if technical_indicators else None
        ),
        "technical_indicator_groups": _technical_indicator_groups(
            technical_indicators
        ),
        "chart_history": chart_history,
        "price_history": history,
    }
