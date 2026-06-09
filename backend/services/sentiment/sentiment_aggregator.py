import time
from datetime import date, timedelta
from database.supabase_client import supabase

MAX_RETRIES = 3
BACKOFF_BASE = 2
MODEL_VERSION = "ProsusAI/finbert"

LABEL_WEIGHTS = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}


def save_scores(symbol: str, scored_headlines: list) -> dict:
    if not scored_headlines:
        return {"rows_saved": 0}

    stock_id = _get_stock_id(symbol)

    if stock_id is None:
        return {"rows_saved": 0, "reason": "stock_not_found"}

    rows = [
        {
            "symbol": symbol.upper(),
            "stock_id": stock_id,
            "headline": h["headline"],
            "source": h["source"],
            "published_at": h["published_at"],
            "label": h["label"],
            "score": h["score"],
            "model_version": MODEL_VERSION,
        }
        for h in scored_headlines
    ]
    for attempt in range(MAX_RETRIES):
        try:
            response = (
                supabase.table("sentiment_scores")
                .upsert(rows, on_conflict="symbol,headline,published_at")
                .execute()
            )
            return {"rows_saved": len(response.data)}
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            raise


def save_daily_sentiment_score(symbol: str, score_date: date = None) -> dict:
    score_date = score_date or date.today()
    symbol = symbol.upper()
    stock_id = _get_stock_id(symbol)

    if stock_id is None:
        return {"rows_saved": 0, "reason": "stock_not_found"}

    rows = _get_sentiment_rows_for_date(symbol, score_date)
    daily_score = calculate_daily_sentiment_score(symbol, stock_id, score_date, rows)

    if daily_score is None:
        return {"rows_saved": 0, "reason": "no_sentiment_rows"}

    response = (
        supabase.table("sentiment_daily_scores")
        .upsert(daily_score, on_conflict="stock_id,score_date")
        .execute()
    )

    return {
        "rows_saved": len(response.data or []),
        "daily_score": daily_score,
    }


def get_sentiment_summary(symbol: str, days: int = 7) -> dict:
    from_date = (date.today() - timedelta(days=days)).isoformat()
    response = (
        supabase.table("sentiment_scores")
        .select("*")
        .eq("symbol", symbol.upper())
        .gte("published_at", f"{from_date}T00:00:00Z")
        .order("published_at", desc=True)
        .execute()
    )
    rows = response.data or []
    by_date = {}
    for row in rows:
        d = row["published_at"][:10]
        by_date.setdefault(d, []).append(row["score"])
    daily_scores = [
        {
            "date": d,
            "avg_score": round(sum(scores) / len(scores), 4),
            "label": _score_to_label(sum(scores) / len(scores)),
            "headline_count": len(scores),
        }
        for d, scores in sorted(by_date.items(), reverse=True)
    ]
    headlines = [
        {
            "headline": r["headline"],
            "source": r["source"],
            "published_at": r["published_at"],
            "label": r["label"],
            "score": r["score"],
        }
        for r in rows
    ]

    score_response = (
        supabase.table("sentiment_daily_scores")
        .select("*")
        .eq("symbol", symbol.upper())
        .gte("score_date", from_date)
        .order("score_date", desc=True)
        .execute()
    )

    return {
        "daily_scores": daily_scores,
        "weighted_scores": score_response.data or [],
        "headlines": headlines,
    }


def get_weighted_sentiment_score(symbol: str, score_date: date = None) -> dict:
    score_date = score_date or date.today()

    response = (
        supabase.table("sentiment_daily_scores")
        .select("*")
        .eq("symbol", symbol.upper())
        .eq("score_date", score_date.isoformat())
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        return None

    return rows[0]


def has_data_for_today(symbol: str) -> bool:
    today = date.today().isoformat()
    response = (
        supabase.table("sentiment_scores")
        .select("id")
        .eq("symbol", symbol.upper())
        .gte("created_at", f"{today}T00:00:00Z")
        .limit(1)
        .execute()
    )
    return len(response.data) > 0


def _score_to_label(avg_score: float) -> str:
    if avg_score > 0.6:
        return "positive"
    if avg_score < 0.4:
        return "negative"
    return "neutral"


def _get_stock_id(symbol: str):
    response = (
        supabase.table("stocks")
        .select("id")
        .eq("symbol", symbol.upper())
        .limit(1)
        .execute()
    )

    rows = response.data if isinstance(response.data, list) else []
    if not rows:
        return None

    return rows[0]["id"]


def _get_sentiment_rows_for_date(symbol: str, score_date: date) -> list:
    day = score_date.isoformat()
    response = (
        supabase.table("sentiment_scores")
        .select("stock_id, symbol, label, score, published_at, model_version")
        .eq("symbol", symbol.upper())
        .gte("published_at", f"{day}T00:00:00Z")
        .lt("published_at", f"{day}T23:59:59.999999Z")
        .execute()
    )

    return response.data or []


def calculate_daily_sentiment_score(
    symbol: str,
    stock_id: int,
    score_date: date,
    rows: list,
) -> dict:
    if not rows:
        return None

    weighted_values = [_weighted_sentiment_value(row) for row in rows]
    raw_sentiment = sum(weighted_values) / len(weighted_values)
    bullish_score = _raw_sentiment_to_bullish_score(raw_sentiment)
    labels = [row["label"] for row in rows]

    return {
        "stock_id": stock_id,
        "symbol": symbol.upper(),
        "score_date": score_date.isoformat(),
        "article_count": len(rows),
        "positive_count": labels.count("positive"),
        "neutral_count": labels.count("neutral"),
        "negative_count": labels.count("negative"),
        "raw_sentiment": round(raw_sentiment, 4),
        "bullish_score": round(bullish_score, 2),
        "sentiment_label": _bullish_score_to_label(bullish_score),
        "model_version": rows[0].get("model_version") or MODEL_VERSION,
    }


def _weighted_sentiment_value(row: dict) -> float:
    label = row["label"]
    score = float(row["score"])
    return LABEL_WEIGHTS[label] * score


def _raw_sentiment_to_bullish_score(raw_sentiment: float) -> float:
    raw_sentiment = max(-1, min(1, raw_sentiment))

    if raw_sentiment >= 0:
        return 5 + (raw_sentiment * 5)

    return 5 + (raw_sentiment * 4)


def _bullish_score_to_label(bullish_score: float) -> str:
    if bullish_score >= 6:
        return "bullish"
    if bullish_score < 4:
        return "bearish"
    return "neutral"
