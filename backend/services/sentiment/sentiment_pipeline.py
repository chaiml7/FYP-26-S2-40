import time
import logging
from datetime import date

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

def run_pipeline() -> dict:
    results = []
    from_date = date.today()
    active_stocks = get_active_stocks() or []

    for stock in active_stocks:
        symbol = str(stock.get("symbol", "")).upper()
        if not symbol:
            logger.warning("Skipping active stock with no symbol: %s", stock)
            continue
        company_name = stock.get("company_name") or symbol
        if has_data_for_today(symbol):
            results.append({"symbol": symbol, "headlines_scored": 0, "status": "skipped"})
            continue
        try:
            headlines = list(fetch_finnhub(symbol, from_date=from_date))
            time.sleep(0.5)
            headlines += list(fetch_gnews(symbol, company_name, from_date=from_date))
            if not headlines:
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
        "active_stocks_found": len(active_stocks),
        "symbols_processed": len([r for r in results if r["status"] not in ("skipped",)]),
        "results": results,
    }
