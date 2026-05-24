import os
import time
import requests
from datetime import date, datetime, timedelta, timezone

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"
MAX_RETRIES = 3
BACKOFF_BASE = 2


def fetch_news(symbol: str, from_date: date = None) -> list:
    if from_date is None:
        from_date = date.today() - timedelta(days=1)
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": date.today().isoformat(),
        "token": FINNHUB_API_KEY,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/company-news", params=params, timeout=10)
            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE ** (attempt + 1))
                    continue
                raise Exception(f"FinnHub rate limit exceeded after {MAX_RETRIES} attempts")
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                return []
            return [
                {"headline": item["headline"], "source": "finnhub", "published_at": _unix_to_iso(item.get("datetime", 0))}
                for item in data
                if item.get("headline")
            ]
        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            return []
    return []


def _unix_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
