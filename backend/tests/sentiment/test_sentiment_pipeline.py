import pytest
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from services.sentiment.sentiment_pipeline import run_pipeline, WATCHLIST

MODULE = "services.sentiment.sentiment_pipeline"

MOCK_HEADLINES = [
    {"headline": "Apple profits up", "source": "finnhub", "published_at": "2026-05-24T09:00:00+00:00"},
    {"headline": "Apple new product launch", "source": "newsapi", "published_at": "2026-05-24T10:00:00+00:00"},
]
MOCK_SCORES = [
    {"label": "positive", "score": 0.91},
    {"label": "positive", "score": 0.85},
]


@contextmanager
def patched_pipeline(has_data=False, finnhub_data=None, newsapi_data=None, scores=None,
                     finnhub_side_effect=None, score_side_effect=None):
    _finnhub = finnhub_data if finnhub_data is not None else MOCK_HEADLINES[:1]
    _newsapi = newsapi_data if newsapi_data is not None else MOCK_HEADLINES[1:]
    _scores = scores if scores is not None else MOCK_SCORES

    with patch(f"{MODULE}.has_data_for_today", return_value=has_data) as m_has, \
         patch(f"{MODULE}.fetch_finnhub", return_value=_finnhub, side_effect=finnhub_side_effect) as m_fh, \
         patch(f"{MODULE}.fetch_newsapi", return_value=_newsapi) as m_fa, \
         patch(f"{MODULE}.score_headlines", return_value=_scores, side_effect=score_side_effect) as m_sc, \
         patch(f"{MODULE}.save_scores", return_value={"rows_saved": 2}) as m_sv, \
         patch(f"{MODULE}.time.sleep") as m_sl:
        yield {"has_data": m_has, "fetch_finnhub": m_fh, "fetch_newsapi": m_fa,
               "score_headlines": m_sc, "save_scores": m_sv, "sleep": m_sl}


def test_run_pipeline_processes_all_watchlist():
    with patched_pipeline() as m:
        result = run_pipeline()
    symbols = [r["symbol"] for r in result["results"]]
    for symbol in WATCHLIST:
        assert symbol in symbols


def test_idempotency_skips_existing():
    with patched_pipeline(has_data=True) as m:
        result = run_pipeline()
    skipped = [r for r in result["results"] if r["status"] == "skipped"]
    assert len(skipped) == len(WATCHLIST)
    m["fetch_finnhub"].assert_not_called()


def test_idempotency_processes_missing():
    with patched_pipeline(has_data=False) as m:
        result = run_pipeline()
    ok = [r for r in result["results"] if r["status"] == "ok"]
    assert len(ok) == len(WATCHLIST)


def test_one_symbol_failure_continues():
    side_effect = [Exception("FinnHub down")] + [MOCK_HEADLINES[:1]] * (len(WATCHLIST) - 1)
    with patched_pipeline(finnhub_side_effect=side_effect):
        result = run_pipeline()
    statuses = [r["status"] for r in result["results"]]
    assert "error" in statuses
    assert statuses.count("ok") >= len(WATCHLIST) - 1


def test_no_headlines_returns_no_data_status():
    with patched_pipeline(finnhub_data=[], newsapi_data=[]):
        result = run_pipeline()
    no_data = [r for r in result["results"] if r["status"] == "no_data"]
    assert len(no_data) == len(WATCHLIST)


def test_finbert_failure_caught_per_symbol():
    with patched_pipeline(score_side_effect=OSError("model not found")):
        result = run_pipeline()
    error = [r for r in result["results"] if r["status"] == "error"]
    assert len(error) == len(WATCHLIST)


def test_both_fetchers_called_per_symbol():
    with patched_pipeline() as m:
        run_pipeline()
    assert m["fetch_finnhub"].call_count == len(WATCHLIST)
    assert m["fetch_newsapi"].call_count == len(WATCHLIST)


def test_newsapi_empty_still_scores_finnhub_results():
    with patched_pipeline(newsapi_data=[]):
        result = run_pipeline()
    ok = [r for r in result["results"] if r["status"] == "ok"]
    assert len(ok) == len(WATCHLIST)


def test_result_has_required_keys():
    with patched_pipeline():
        result = run_pipeline()
    assert "message" in result
    assert "symbols_processed" in result
    assert "results" in result
    assert all("symbol" in r and "status" in r for r in result["results"])


def test_rate_limit_sleep_called_between_symbols():
    with patched_pipeline() as m:
        run_pipeline()
    assert m["sleep"].call_count == len(WATCHLIST)
    m["sleep"].assert_any_call(0.5)
