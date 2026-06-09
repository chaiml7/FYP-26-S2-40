import time
import logging
from datetime import date

from services.sentiment.finnhub_service import fetch_news as fetch_finnhub
from services.sentiment.news_scraper_service import fetch_news as fetch_newsapi
from services.sentiment.finbert_service import score_headlines
from services.sentiment.sentiment_aggregator import (
    save_daily_sentiment_score,
    save_scores,
    has_data_for_today,
)

logger = logging.getLogger(__name__)

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "BABA"]

COMPANY_NAMES = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "NVIDIA", "MSFT": "Microsoft",
    "AMZN": "Amazon", "META": "Meta", "GOOGL": "Google", "NFLX": "Netflix",
    "AMD": "AMD", "BABA": "Alibaba",
}


def run_pipeline() -> dict:
    results = []
    from_date = date.today()
    for symbol in WATCHLIST:
        if has_data_for_today(symbol):
            results.append({"symbol": symbol, "headlines_scored": 0, "status": "skipped"})
            continue
        try:
            headlines = list(fetch_finnhub(symbol, from_date=from_date))
            time.sleep(0.5)
            headlines += list(fetch_newsapi(symbol, COMPANY_NAMES[symbol], from_date=from_date))
            if not headlines:
                results.append({"symbol": symbol, "headlines_scored": 0, "status": "no_data"})
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
        "symbols_processed": len([r for r in results if r["status"] not in ("skipped",)]),
        "results": results,
    }
