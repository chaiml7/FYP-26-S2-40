from contextlib import contextmanager
from unittest.mock import patch

from backend.services.sentiment.sentiment_pipeline import run_pipeline


MODULE = "backend.services.sentiment.sentiment_pipeline"

MOCK_HEADLINES = [
    {"headline": "Apple profits up", "source": "finnhub", "published_at": "2026-05-24T09:00:00+00:00"},
    {"headline": "Apple new product launch", "source": "gnews", "published_at": "2026-05-24T10:00:00+00:00"},
]
MOCK_SCORES = [
    {"label": "positive", "score": 0.91},
    {"label": "positive", "score": 0.85},
]
ACTIVE_STOCKS = [
    {"symbol": "AAPL", "company_name": "Apple"},
    {"symbol": "NVDA", "company_name": "NVIDIA"},
]


@contextmanager
def patched_pipeline(
    has_data=False,
    finnhub_data=None,
    gnews_data=None,
    scores=None,
    finnhub_side_effect=None,
    score_side_effect=None,
):
    finnhub_results = finnhub_data if finnhub_data is not None else MOCK_HEADLINES[:1]
    gnews_results = gnews_data if gnews_data is not None else MOCK_HEADLINES[1:]
    scoring_results = scores if scores is not None else MOCK_SCORES

    with (
        patch(f"{MODULE}.get_active_stocks", return_value=ACTIVE_STOCKS) as active_stocks,
        patch(f"{MODULE}.has_data_for_today", return_value=has_data) as has_data_for_today,
        patch(
            f"{MODULE}.fetch_finnhub",
            return_value=finnhub_results,
            side_effect=finnhub_side_effect,
        ) as fetch_finnhub,
        patch(f"{MODULE}.fetch_gnews", return_value=gnews_results) as fetch_gnews,
        patch(
            f"{MODULE}.score_headlines",
            return_value=scoring_results,
            side_effect=score_side_effect,
        ) as score_headlines,
        patch(f"{MODULE}.save_scores", return_value={"rows_saved": 2}) as save_scores,
        patch(
            f"{MODULE}.save_daily_sentiment_score",
            return_value={"rows_saved": 1},
        ) as save_daily_score,
        patch(
            f"{MODULE}.save_neutral_daily_sentiment_score",
            return_value={"rows_saved": 1},
        ) as save_neutral_daily_score,
        patch(f"{MODULE}.time.sleep") as sleep,
    ):
        yield {
            "active_stocks": active_stocks,
            "has_data": has_data_for_today,
            "fetch_finnhub": fetch_finnhub,
            "fetch_gnews": fetch_gnews,
            "score_headlines": score_headlines,
            "save_scores": save_scores,
            "save_daily_score": save_daily_score,
            "save_neutral_daily_score": save_neutral_daily_score,
            "sleep": sleep,
        }


def test_run_pipeline_processes_all_active_stocks():
    with patched_pipeline():
        result = run_pipeline()

    assert [row["symbol"] for row in result["results"]] == ["AAPL", "NVDA"]
    assert result["active_stocks_found"] == len(ACTIVE_STOCKS)


def test_idempotency_skips_existing_scores():
    with patched_pipeline(has_data=True) as mocks:
        result = run_pipeline()

    assert [row["status"] for row in result["results"]] == ["skipped", "skipped"]
    mocks["fetch_finnhub"].assert_not_called()


def test_one_symbol_failure_does_not_stop_other_active_stocks():
    side_effect = [Exception("Finnhub down"), MOCK_HEADLINES[:1]]
    with patched_pipeline(finnhub_side_effect=side_effect):
        result = run_pipeline()

    assert [row["status"] for row in result["results"]] == ["error", "ok"]


def test_no_headlines_returns_no_data_status():
    with patched_pipeline(finnhub_data=[], gnews_data=[]) as mocks:
        result = run_pipeline()

    assert [row["status"] for row in result["results"]] == ["no_data", "no_data"]
    assert all(row["sentiment_fallback"] == "neutral" for row in result["results"])
    assert mocks["save_neutral_daily_score"].call_count == len(ACTIVE_STOCKS)


def test_gnews_query_uses_database_company_name():
    with patched_pipeline() as mocks:
        run_pipeline()

    mocks["fetch_gnews"].assert_any_call("AAPL", "Apple", from_date=mocks["fetch_gnews"].call_args.kwargs["from_date"])
    mocks["fetch_gnews"].assert_any_call("NVDA", "NVIDIA", from_date=mocks["fetch_gnews"].call_args.kwargs["from_date"])


def test_both_fetchers_called_for_each_active_stock():
    with patched_pipeline() as mocks:
        run_pipeline()

    assert mocks["fetch_finnhub"].call_count == len(ACTIVE_STOCKS)
    assert mocks["fetch_gnews"].call_count == len(ACTIVE_STOCKS)


def test_result_has_required_keys():
    with patched_pipeline():
        result = run_pipeline()

    assert {"message", "active_stocks_found", "symbols_processed", "results"} <= result.keys()
    assert all("symbol" in row and "status" in row for row in result["results"])


def test_finnhub_skipped_for_sgx_symbol():
    sgx_stocks = [{"symbol": "D05.SI", "company_name": "DBS Group Holdings Ltd"}]
    with patched_pipeline() as mocks:
        mocks["active_stocks"].return_value = sgx_stocks
        result = run_pipeline()

    mocks["fetch_finnhub"].assert_not_called()
    mocks["sleep"].assert_not_called()
    mocks["fetch_gnews"].assert_called_once_with(
        "D05.SI", "DBS Group Holdings Ltd", from_date=mocks["fetch_gnews"].call_args.kwargs["from_date"]
    )
    assert result["results"][0]["status"] == "ok"


def test_finnhub_called_only_for_non_sgx_symbols_in_mixed_batch():
    mixed_stocks = [
        {"symbol": "AAPL", "company_name": "Apple"},
        {"symbol": "D05.SI", "company_name": "DBS Group Holdings Ltd"},
    ]
    with patched_pipeline() as mocks:
        mocks["active_stocks"].return_value = mixed_stocks
        run_pipeline()

    assert mocks["fetch_finnhub"].call_count == 1
    called_symbols = [c.args[0] for c in mocks["fetch_finnhub"].call_args_list]
    assert called_symbols == ["AAPL"]
    assert mocks["fetch_gnews"].call_count == 2
