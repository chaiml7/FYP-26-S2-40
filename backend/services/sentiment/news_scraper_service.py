import os
import time
import requests
from datetime import date, timedelta

BASE_URL = "https://newsapi.org/v2"
MAX_RETRIES = 3
BACKOFF_BASE = 2


def fetch_news(symbol: str, company_name: str, from_date: date = None) -> list:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY is not set")
    if from_date is None:
        from_date = date.today() - timedelta(days=1)
    params = {
        "q": company_name,
        "from": from_date.isoformat(),
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/everything", params=params, timeout=10)
            if response.status_code in (429, 426):
                return []
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            if not isinstance(articles, list):
                return []
            return [
                {"headline": a["title"], "source": "newsapi", "published_at": a.get("publishedAt", "")}
                for a in articles
                if a.get("title")
            ]
        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            return []
        except Exception:
            return []
    return []
