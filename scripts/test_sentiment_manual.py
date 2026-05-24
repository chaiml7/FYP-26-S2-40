"""
Manual end-to-end test for the sentiment pipeline.
Run from repo root: python scripts/test_sentiment_manual.py
Requires: backend server running at localhost:8000, valid API keys in backend/.env
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

import httpx
from datetime import date, timedelta

BASE_URL = "http://localhost:8000/api"
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


def step(n, label):
    print(f"\n--- Step {n}: {label} ---")


# Step 1: FinBERT smoke test
step(1, "FinBERT scoring")
try:
    from services.sentiment.finbert_service import score_headlines
    results = score_headlines([
        "Apple reports record quarterly profits",
        "Tesla faces massive safety recall affecting 500000 vehicles",
        "Market trading volume remains unchanged today",
    ])
    check("FinBERT returns 3 results", len(results) == 3)
    check("FinBERT labels are valid", all(r["label"] in ("positive", "negative", "neutral") for r in results))
    check("FinBERT scores in range", all(0.0 <= r["score"] <= 1.0 for r in results))
    check("Apple headline is positive", results[0]["label"] == "positive", f"got {results[0]['label']}")
    check("Tesla recall is negative", results[1]["label"] == "negative", f"got {results[1]['label']}")
    print(f"  Results: {results}")
except Exception as e:
    check("FinBERT smoke test", False, str(e))


# Step 2: FinnHub fetch
step(2, "FinnHub news fetch")
try:
    from services.sentiment.finnhub_service import fetch_news as fetch_finnhub
    yesterday = date.today() - timedelta(days=1)
    headlines = fetch_finnhub("AAPL", from_date=yesterday)
    check("FinnHub returns a list", isinstance(headlines, list))
    check("FinnHub headlines have correct keys",
          all("headline" in h and "source" in h and "published_at" in h for h in headlines) if headlines else True)
    check("FinnHub source is 'finnhub'", all(h["source"] == "finnhub" for h in headlines) if headlines else True)
    print(f"  Got {len(headlines)} headlines. First 2: {[h['headline'] for h in headlines[:2]]}")
except Exception as e:
    check("FinnHub fetch", False, str(e))


# Step 3: NewsAPI fetch
step(3, "NewsAPI news fetch")
try:
    from services.sentiment.news_scraper_service import fetch_news as fetch_newsapi
    yesterday = date.today() - timedelta(days=1)
    headlines = fetch_newsapi("AAPL", "Apple", from_date=yesterday)
    check("NewsAPI returns a list", isinstance(headlines, list))
    check("NewsAPI source is 'newsapi'", all(h["source"] == "newsapi" for h in headlines) if headlines else True)
    print(f"  Got {len(headlines)} headlines. First 2: {[h['headline'] for h in headlines[:2]]}")
    if not headlines:
        print("  Warning: no headlines returned — may be quota exceeded or no news today")
except Exception as e:
    check("NewsAPI fetch", False, str(e))


# Step 4: Pipeline trigger via API
step(4, "Pipeline trigger (POST /api/sentiment/run-pipeline)")
try:
    response = httpx.post(f"{BASE_URL}/sentiment/run-pipeline", timeout=300)
    check("POST /run-pipeline returns 200", response.status_code == 200, f"got {response.status_code}")
    data = response.json()
    check("Response has symbols_processed", "symbols_processed" in data)
    check("Response has results list", "results" in data and isinstance(data["results"], list))
    print(f"  Pipeline summary: {data['message']}, processed: {data['symbols_processed']}")
    for r in data["results"]:
        status_icon = "✓" if r["status"] == "ok" else ("~" if r["status"] in ("skipped", "no_data") else "✗")
        print(f"  {status_icon} {r['symbol']}: {r['status']} ({r.get('headlines_scored', 0)} headlines)")
except Exception as e:
    check("Pipeline trigger", False, str(e))


# Step 5: Supabase row check
step(5, "Supabase row count after pipeline")
try:
    from database.supabase_client import supabase
    today = date.today().isoformat()
    response = supabase.table("sentiment_scores").select("symbol").gte("created_at", f"{today}T00:00:00Z").execute()
    rows = response.data or []
    check("Supabase has rows for today", len(rows) > 0, f"got {len(rows)} rows")
    by_symbol = {}
    for row in rows:
        by_symbol[row["symbol"]] = by_symbol.get(row["symbol"], 0) + 1
    print(f"  Rows by symbol: {by_symbol}")
except Exception as e:
    check("Supabase row check", False, str(e))


# Step 6: Idempotency check
step(6, "Idempotency (re-run pipeline, row count unchanged)")
try:
    from database.supabase_client import supabase
    today = date.today().isoformat()
    before = supabase.table("sentiment_scores").select("id").gte("created_at", f"{today}T00:00:00Z").execute()
    count_before = len(before.data or [])

    response = httpx.post(f"{BASE_URL}/sentiment/run-pipeline", timeout=300)
    data = response.json()
    skipped = [r for r in data.get("results", []) if r["status"] == "skipped"]

    after = supabase.table("sentiment_scores").select("id").gte("created_at", f"{today}T00:00:00Z").execute()
    count_after = len(after.data or [])

    check("Row count unchanged after re-run", count_before == count_after, f"before={count_before}, after={count_after}")
    check("All symbols skipped on re-run", len(skipped) == 10, f"only {len(skipped)} skipped")
except Exception as e:
    check("Idempotency check", False, str(e))


# Step 7: Sentiment read endpoint
step(7, "GET /api/stocks/AAPL/sentiment")
try:
    response = httpx.get(f"{BASE_URL}/stocks/AAPL/sentiment", timeout=30)
    check("GET /sentiment returns 200", response.status_code == 200, f"got {response.status_code}")
    data = response.json()
    check("Response has symbol", data.get("symbol") == "AAPL")
    check("Response has daily_scores", "daily_scores" in data)
    check("Response has headlines", "headlines" in data)
    if data.get("daily_scores"):
        ds = data["daily_scores"][0]
        check("daily_scores entry has required keys",
              all(k in ds for k in ("date", "avg_score", "label", "headline_count")))
    print(f"  daily_scores: {data.get('daily_scores', [])[:2]}")
    print(f"  headline count: {len(data.get('headlines', []))}")
except Exception as e:
    check("Sentiment read endpoint", False, str(e))


# Step 8: Error simulation (bad API key)
step(8, "Error simulation (invalid FinnHub key)")
try:
    import services.sentiment.finnhub_service as fh_module
    original_key = os.environ.get("FINNHUB_API_KEY")
    os.environ["FINNHUB_API_KEY"] = "INVALID_KEY_TEST"
    from services.sentiment.finnhub_service import fetch_news
    try:
        result = fetch_news("AAPL", from_date=date.today() - timedelta(days=1))
        check("Invalid key returns empty list or raises cleanly",
              isinstance(result, list),
              f"got unexpected type: {type(result)}")
    except Exception as inner_e:
        check("Error simulation completed without crash", True)
        print(f"  Exception raised (expected): {inner_e}")
    finally:
        if original_key is not None:
            os.environ["FINNHUB_API_KEY"] = original_key
        else:
            del os.environ["FINNHUB_API_KEY"]
except Exception as e:
    check("Error simulation setup", False, str(e))


# --- Summary ---
print(f"\n{'='*50}")
if failures:
    print(f"\033[91m{len(failures)} FAILED: {', '.join(failures)}\033[0m")
    sys.exit(1)
else:
    print(f"\033[92mAll steps PASSED\033[0m")
