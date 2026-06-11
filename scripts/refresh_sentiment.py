"""
Bulk refresh sentiment data for watchlist stocks.
Deletes stale base-model data and replaces with fine-tuned FinBERT scores.
Run from project root: python scripts/refresh_sentiment.py
"""
import sys
import os
import time
import requests
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

# Bridge env var name mismatch: .env has SUPABASE_SECRET_KEY, client reads SUPABASE_KEY
if not os.getenv("SUPABASE_KEY") and os.getenv("SUPABASE_SECRET_KEY"):
    os.environ["SUPABASE_KEY"] = os.getenv("SUPABASE_SECRET_KEY")

from database.supabase_client import supabase

MODEL_VERSION = "balibpt/finbert-stocklens"
DAYS_BACK = 7
BATCH_SIZE = 16

WATCHLIST = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "BABA"]

COMPANY_NAMES = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Google",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "BABA": "Alibaba",
}


def get_stock_ids():
    response = supabase.table("stocks").select("id, symbol").in_("symbol", WATCHLIST).execute()
    return {row["symbol"]: row["id"] for row in (response.data or [])}


def fetch_finnhub_news(symbol, from_date):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print(f"  [WARN] FINNHUB_API_KEY not set, skipping Finnhub for {symbol}")
        return []
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": date.today().isoformat(),
        "token": api_key,
    }
    for attempt in range(3):
        try:
            resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)
            if resp.status_code == 429:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"  [WARN] Finnhub rate limit for {symbol}")
                return []
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return []
            return [
                {
                    "headline": item["headline"],
                    "source": "finnhub",
                    "published_at": datetime.fromtimestamp(item["datetime"], tz=timezone.utc).isoformat(),
                }
                for item in data
                if item.get("headline") and item.get("datetime")
            ]
        except requests.Timeout:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return []
        except Exception as e:
            print(f"  [WARN] Finnhub error for {symbol}: {e}")
            return []
    return []


def fetch_newsapi_news(symbol, company_name, from_date):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print(f"  [WARN] NEWSAPI_KEY not set, skipping NewsAPI for {symbol}")
        return []
    params = {
        "q": company_name,
        "from": from_date.isoformat(),
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
    }
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        if resp.status_code in (429, 426):
            print(f"  [WARN] NewsAPI quota/upgrade for {symbol}")
            return []
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "headline": a["title"],
                "source": "newsapi",
                "published_at": a.get("publishedAt", ""),
            }
            for a in articles
            if a.get("title") and a["title"] != "[Removed]"
        ]
    except Exception as e:
        print(f"  [WARN] NewsAPI error for {symbol}: {e}")
        return []


def deduplicate_headlines(headlines):
    seen = set()
    unique = []
    for h in headlines:
        key = h["headline"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def fetch_all_news():
    from_date = date.today() - timedelta(days=DAYS_BACK)
    all_news = {}
    for symbol in WATCHLIST:
        try:
            finnhub = fetch_finnhub_news(symbol, from_date)
            time.sleep(0.5)
            newsapi = fetch_newsapi_news(symbol, COMPANY_NAMES[symbol], from_date)
            combined = deduplicate_headlines(finnhub + newsapi)
            all_news[symbol] = combined
            print(f"  {symbol}: {len(finnhub)} finnhub + {len(newsapi)} newsapi = {len(combined)} unique")
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            all_news[symbol] = []
    return all_news


def score_all_headlines(all_news):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F

    print(f"  Loading model {MODEL_VERSION}...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_VERSION)
    model.eval()
    label_map = {int(k): v for k, v in model.config.id2label.items()}
    print(f"  Label map: {label_map}")

    scored = {}
    total = sum(len(v) for v in all_news.values())
    processed = 0

    for symbol, headlines in all_news.items():
        if not headlines:
            scored[symbol] = []
            continue
        texts = [h["headline"] for h in headlines]
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            for j, prob in enumerate(probs):
                idx = prob.argmax().item()
                h = headlines[i + j]
                results.append({
                    "headline": h["headline"],
                    "source": h["source"],
                    "published_at": h["published_at"],
                    "label": label_map[idx],
                    "score": round(prob[idx].item(), 4),
                })
            processed += len(batch)
        print(f"  {symbol}: scored {len(results)} headlines ({processed}/{total} total)")
        scored[symbol] = results
    return scored


def delete_old_data():
    print("  Deleting sentiment_daily_scores...")
    resp1 = supabase.table("sentiment_daily_scores").delete().in_("symbol", WATCHLIST).execute()
    count1 = len(resp1.data or [])
    print(f"    Deleted {count1} daily score rows")

    print("  Deleting sentiment_scores...")
    resp2 = supabase.table("sentiment_scores").delete().in_("symbol", WATCHLIST).execute()
    count2 = len(resp2.data or [])
    print(f"    Deleted {count2} headline score rows")


def insert_new_data(scored, stock_ids):
    total_inserted = 0
    for symbol, headlines in scored.items():
        if not headlines:
            continue
        stock_id = stock_ids.get(symbol)
        if not stock_id:
            print(f"  [WARN] No stock_id for {symbol}, skipping insert")
            continue
        rows = [
            {
                "symbol": symbol,
                "stock_id": stock_id,
                "headline": h["headline"][:500],
                "source": h["source"],
                "published_at": h["published_at"],
                "label": h["label"],
                "score": h["score"],
                "model_version": MODEL_VERSION,
            }
            for h in headlines
        ]
        for i in range(0, len(rows), 100):
            chunk = rows[i:i + 100]
            supabase.table("sentiment_scores").insert(chunk).execute()
        total_inserted += len(rows)
        print(f"  {symbol}: upserted {len(rows)} rows")
    print(f"  Total: {total_inserted} rows upserted into sentiment_scores")


def compute_daily_scores(scored, stock_ids):
    total_days = 0
    for symbol, headlines in scored.items():
        if not headlines:
            continue
        stock_id = stock_ids.get(symbol)
        if not stock_id:
            continue

        by_date = defaultdict(list)
        for h in headlines:
            pub = h["published_at"][:10] if h["published_at"] else None
            if pub:
                by_date[pub].append(h)

        rows = []
        for score_date, day_headlines in by_date.items():
            pos = [h for h in day_headlines if h["label"] == "positive"]
            neg = [h for h in day_headlines if h["label"] == "negative"]
            neu = [h for h in day_headlines if h["label"] == "neutral"]

            weighted = sum(
                (1.0 * h["score"] if h["label"] == "positive"
                 else -1.0 * h["score"] if h["label"] == "negative"
                 else 0.0)
                for h in day_headlines
            ) / len(day_headlines)

            raw_sentiment = round(weighted, 4)

            if raw_sentiment >= 0:
                bullish_score = round(5 + raw_sentiment * 5, 2)
            else:
                bullish_score = round(5 + raw_sentiment * 4, 2)

            if bullish_score >= 6:
                label = "bullish"
            elif bullish_score < 4:
                label = "bearish"
            else:
                label = "neutral"

            rows.append({
                "stock_id": stock_id,
                "symbol": symbol,
                "score_date": score_date,
                "article_count": len(day_headlines),
                "positive_count": len(pos),
                "neutral_count": len(neu),
                "negative_count": len(neg),
                "raw_sentiment": raw_sentiment,
                "bullish_score": bullish_score,
                "sentiment_label": label,
                "model_version": MODEL_VERSION,
            })

        if rows:
            supabase.table("sentiment_daily_scores").insert(rows).execute()
            total_days += len(rows)
            print(f"  {symbol}: {len(rows)} daily scores")

    print(f"  Total: {total_days} daily score rows upserted")


def verify():
    print("\n--- Verification ---")

    resp = supabase.table("sentiment_scores").select("*").in_("symbol", WATCHLIST).execute()
    rows = resp.data or []
    print(f"\nsentiment_scores: {len(rows)} total rows")

    by_symbol = defaultdict(int)
    scores = []
    models = set()
    for r in rows:
        by_symbol[r["symbol"]] += 1
        scores.append(r["score"])
        models.add(r.get("model_version", "unknown"))

    print(f"  Per stock: {dict(by_symbol)}")
    if scores:
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}, mean: {sum(scores)/len(scores):.4f}")
        buckets = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
        for s in scores:
            if s < 0.5:
                buckets["0.0-0.5"] += 1
            elif s < 0.7:
                buckets["0.5-0.7"] += 1
            elif s < 0.9:
                buckets["0.7-0.9"] += 1
            else:
                buckets["0.9-1.0"] += 1
        print(f"  Score distribution: {buckets}")
    print(f"  Model versions: {models}")

    print(f"\n  Sample records:")
    for r in rows[:5]:
        print(f"    {r['symbol']} | {r['label']} ({r['score']}) | {r['headline'][:60]}")

    resp2 = supabase.table("sentiment_daily_scores").select("*").in_("symbol", WATCHLIST).execute()
    daily = resp2.data or []
    print(f"\nsentiment_daily_scores: {len(daily)} total rows")
    for r in daily[:5]:
        print(f"    {r['symbol']} | {r['score_date']} | {r['sentiment_label']} | bullish={r['bullish_score']} | articles={r['article_count']}")


def main():
    print("=" * 60)
    print("StockLens Sentiment Refresh")
    print(f"Model: {MODEL_VERSION}")
    print(f"Watchlist: {WATCHLIST}")
    print(f"Days back: {DAYS_BACK}")
    print("=" * 60)

    print("\n[Phase 1] Fetching stock IDs...")
    stock_ids = get_stock_ids()
    print(f"  Found {len(stock_ids)} stocks: {list(stock_ids.keys())}")
    missing = [s for s in WATCHLIST if s not in stock_ids]
    if missing:
        print(f"  [WARN] Missing from DB: {missing}")

    print("\n[Phase 2] Fetching news headlines...")
    all_news = fetch_all_news()
    total_headlines = sum(len(v) for v in all_news.values())
    print(f"  Total headlines fetched: {total_headlines}")
    if total_headlines == 0:
        print("[ABORT] No headlines fetched. Check API keys and network.")
        sys.exit(1)

    print("\n[Phase 3] Scoring with FinBERT...")
    scored = score_all_headlines(all_news)
    total_scored = sum(len(v) for v in scored.values())
    print(f"  Total scored: {total_scored}")

    print("\n[Phase 4] Replacing database data...")
    print("  Step 4a: Delete old data")
    delete_old_data()
    print("  Step 4b: Insert new headline scores")
    insert_new_data(scored, stock_ids)
    print("  Step 4c: Compute and insert daily scores")
    compute_daily_scores(scored, stock_ids)

    verify()

    print("\n" + "=" * 60)
    print("Sentiment refresh complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
