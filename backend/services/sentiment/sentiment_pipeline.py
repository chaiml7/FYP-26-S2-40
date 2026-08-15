import time
import logging
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.services.sentiment.finnhub_service import fetch_news as fetch_finnhub
from backend.services.sentiment.gnews_service import fetch_news as fetch_gnews
from backend.services.sentiment.finbert_service import score_headlines
from backend.services.sentiment.sentiment_aggregator import (
    has_data_for_today,
    save_daily_sentiment_score,
    save_neutral_daily_sentiment_score,
    save_scores,
)
from backend.services.stock_list_service import get_active_stocks

logger = logging.getLogger(__name__)


def _current_score_date() -> date:
    timezone_name = os.getenv(
        "SENTIMENT_TIMEZONE",
        os.getenv("ANALYSIS_SCHEDULER_TIMEZONE", "Asia/Singapore"),
    )
    try:
        return datetime.now(ZoneInfo(timezone_name)).date()
    except ZoneInfoNotFoundError:
        logger.warning(
            "Unknown SENTIMENT_TIMEZONE %s; falling back to UTC.",
            timezone_name,
        )
        return datetime.now(ZoneInfo("UTC")).date()


def run_pipeline(
    *,
    refresh_existing: bool = False,
    score_date: date | None = None,
) -> dict:
    results = []
    from_date = score_date or _current_score_date()
    active_stocks = get_active_stocks() or []

    for stock in active_stocks:
        symbol = str(stock.get("symbol", "")).upper()
        if not symbol:
            logger.warning("Skipping active stock with no symbol: %s", stock)
            continue
        company_name = stock.get("company_name") or symbol
        if not refresh_existing and has_data_for_today(symbol, from_date):
            results.append({"symbol": symbol, "headlines_scored": 0, "status": "skipped"})
            continue
        try:
            headlines = []
            if not symbol.endswith(".SI"):
                headlines = list(fetch_finnhub(symbol, from_date=from_date))
                time.sleep(0.5)
            headlines += list(fetch_gnews(symbol, company_name, from_date=from_date))
            if not headlines:
                if refresh_existing:
                    existing_result = save_daily_sentiment_score(symbol, from_date)
                    if existing_result.get("rows_saved", 0) > 0:
                        results.append({
                            "symbol": symbol,
                            "headlines_scored": 0,
                            "daily_score_saved": existing_result["rows_saved"],
                            "sentiment_fallback": "existing_headlines",
                            "status": "no_new_data",
                        })
                        continue
                daily_score_result = save_neutral_daily_sentiment_score(symbol, from_date)
                results.append({
                    "symbol": symbol,
                    "headlines_scored": 0,
                    "daily_score_saved": daily_score_result["rows_saved"],
                    "sentiment_fallback": "neutral",
                    "status": "no_data",
                })
                continue
            scores = score_headlines([h["headline"] for h in headlines])
            scored = [{**headlines[i], **scores[i]} for i in range(len(headlines))]
            save_scores(symbol, scored)
            daily_score_result = save_daily_sentiment_score(symbol, from_date)
            results.append({
                "symbol": symbol,
                "headlines_scored": len(scored),
                "daily_score_saved": daily_score_result["rows_saved"],
                "status": "ok",
            })
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", symbol, e)
            results.append({"symbol": symbol, "headlines_scored": 0, "status": "error", "reason": str(e)})
    return {
        "message": "Pipeline complete",
        "score_date": from_date.isoformat(),
        "refresh_existing": refresh_existing,
        "active_stocks_found": len(active_stocks),
        "symbols_processed": len([r for r in results if r["status"] not in ("skipped",)]),
        "results": results,
    }
