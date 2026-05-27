import time
from datetime import date, timedelta
from database.supabase_client import supabase

MAX_RETRIES = 3
BACKOFF_BASE = 2


def save_scores(symbol: str, scored_headlines: list) -> dict:
    if not scored_headlines:
        return {"rows_saved": 0}
    rows = [
        {
            "symbol": symbol.upper(),
            "headline": h["headline"],
            "source": h["source"],
            "published_at": h["published_at"],
            "label": h["label"],
            "score": h["score"],
            "model_version": "ProsusAI/finbert",
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
    return {"daily_scores": daily_scores, "headlines": headlines}


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
