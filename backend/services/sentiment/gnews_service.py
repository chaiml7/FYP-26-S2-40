from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime

from gnews import GNews


def fetch_news(symbol: str, company_name: str, from_date: date = None) -> list:
    if from_date is None:
        from_date = date.today() - timedelta(days=1)

    days_back = (date.today() - from_date).days or 1
    period = f"{days_back}d"

    is_sgx = symbol.upper().endswith(".SI")
    country = "SG" if is_sgx else "US"
    query = f"{company_name} Singapore" if is_sgx else f"{company_name} {symbol} stock"

    g = GNews(language="en", country=country, period=period, max_results=10)
    try:
        articles = g.get_news(query)
    except Exception:
        return []

    results = []
    for a in articles:
        try:
            dt = parsedate_to_datetime(a.get("published date", ""))
            published_at = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue

        url = a.get("url") or ""
        results.append({
            "headline": a.get("title", "").strip(),
            "source": "gnews",
            "published_at": published_at,
            "url": url if url.startswith(("https://", "http://")) else None,
        })

    return results
