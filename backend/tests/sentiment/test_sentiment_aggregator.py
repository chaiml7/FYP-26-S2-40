import pytest
from unittest.mock import patch, MagicMock, call
from datetime import date

from backend.services.sentiment.sentiment_aggregator import (
    calculate_daily_sentiment_score,
    get_recent_news,
    save_neutral_daily_sentiment_score,
    save_scores,
    get_sentiment_summary,
    has_data_for_today,
    _score_to_label,
)
from tests.sentiment.conftest import SAMPLE_SCORED_HEADLINES

MODULE = "backend.services.sentiment.sentiment_aggregator"


@patch(f"{MODULE}._get_stock_id", return_value=1)
@patch(f"{MODULE}.supabase")
def test_save_scores_upserts_rows(mock_supa, mock_get_stock_id):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    mock_supa.table.assert_called_with("sentiment_scores")
    rows = mock_supa.table.return_value.upsert.call_args[0][0]
    assert len(rows) == 3
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["stock_id"] == 1
    assert rows[0]["model_version"] == "balibpt/finbert-stocklens"


@patch(f"{MODULE}.supabase")
def test_save_scores_empty_list_does_not_upsert(mock_supa):
    result = save_scores("AAPL", [])
    mock_supa.table.return_value.upsert.assert_not_called()
    assert result["rows_saved"] == 0


@patch(f"{MODULE}._get_stock_id", return_value=1)
@patch(f"{MODULE}.supabase")
def test_save_scores_includes_required_fields(mock_supa, mock_get_stock_id):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    rows = mock_supa.table.return_value.upsert.call_args[0][0]
    required = {"symbol", "stock_id", "headline", "source", "published_at", "label", "score", "model_version", "url"}
    assert required.issubset(rows[0].keys())


@patch(f"{MODULE}._get_stock_id", return_value=1)
@patch(f"{MODULE}.supabase")
def test_save_scores_url_passed_through(mock_supa, mock_get_stock_id):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    rows = mock_supa.table.return_value.upsert.call_args[0][0]
    assert rows[0]["url"] == "https://example.com/1"
    assert rows[2]["url"] is None


@patch(f"{MODULE}.supabase")
def test_get_sentiment_summary_headlines_include_url(mock_supa):
    mock_rows = [
        {
            "published_at": "2026-05-24T09:00:00+00:00",
            "headline": "Apple profits up",
            "source": "finnhub",
            "label": "positive",
            "score": 0.92,
            "url": "https://example.com/apple",
        }
    ]
    score_rows = []

    # Two supabase calls: sentiment_scores then sentiment_daily_scores
    execute_mock_scores = MagicMock()
    execute_mock_scores.data = mock_rows
    execute_mock_daily = MagicMock()
    execute_mock_daily.data = score_rows

    chain_scores = MagicMock()
    chain_scores.execute.return_value = execute_mock_scores
    chain_daily = MagicMock()
    chain_daily.execute.return_value = execute_mock_daily

    def table_side_effect(name):
        mock = MagicMock()
        if name == "sentiment_scores":
            mock.select.return_value.eq.return_value.gte.return_value.order.return_value = chain_scores
        else:
            mock.select.return_value.eq.return_value.gte.return_value.order.return_value = chain_daily
        return mock

    mock_supa.table.side_effect = table_side_effect

    result = get_sentiment_summary("AAPL", days=7)
    assert result["headlines"][0]["url"] == "https://example.com/apple"


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}._get_stock_id", return_value=1)
@patch(f"{MODULE}.supabase")
def test_save_scores_retries_on_failure(mock_supa, mock_get_stock_id, mock_sleep):
    mock_supa.table.return_value.upsert.return_value.execute.side_effect = [
        Exception("db error"),
        MagicMock(data=SAMPLE_SCORED_HEADLINES),
    ]
    result = save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    assert mock_supa.table.return_value.upsert.call_count == 2
    mock_sleep.assert_called_once_with(2)


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}._get_stock_id", return_value=1)
@patch(f"{MODULE}.supabase")
def test_save_scores_all_retries_fail_raises(mock_supa, mock_get_stock_id, mock_sleep):
    mock_supa.table.return_value.upsert.return_value.execute.side_effect = Exception("db error")
    with pytest.raises(Exception, match="db error"):
        save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    assert mock_supa.table.return_value.upsert.call_count == 3


def test_label_positive_threshold():
    assert _score_to_label(0.61) == "positive"


def test_label_negative_threshold():
    assert _score_to_label(0.39) == "negative"


def test_label_neutral_threshold():
    assert _score_to_label(0.50) == "neutral"


def test_label_boundary_exactly_06():
    assert _score_to_label(0.60) == "neutral"


def test_label_boundary_exactly_04():
    assert _score_to_label(0.40) == "neutral"


@patch(f"{MODULE}.supabase")
def test_get_summary_returns_correct_shape(mock_supa):
    rows = [
        {"headline": "h1", "source": "finnhub", "published_at": "2026-05-24T09:00:00+00:00", "label": "positive", "score": 0.9},
        {"headline": "h2", "source": "newsapi", "published_at": "2026-05-24T10:00:00+00:00", "label": "negative", "score": 0.8},
    ]
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = rows
    result = get_sentiment_summary("AAPL")
    assert "daily_scores" in result
    assert "weighted_scores" in result
    assert "headlines" in result
    assert isinstance(result["daily_scores"], list)
    assert isinstance(result["weighted_scores"], list)
    assert isinstance(result["headlines"], list)


@patch(f"{MODULE}.supabase")
def test_get_summary_empty_db_returns_empty_lists(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = []
    result = get_sentiment_summary("AAPL")
    assert result == {"daily_scores": [], "weighted_scores": [], "headlines": []}


@patch(f"{MODULE}.supabase")
def test_daily_aggregation_groups_by_date(mock_supa):
    rows = [
        {"headline": f"h{i}", "source": "finnhub", "published_at": f"2026-05-24T0{i}:00:00+00:00", "label": "positive", "score": 0.8}
        for i in range(5)
    ]
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = rows
    result = get_sentiment_summary("AAPL")
    assert len(result["daily_scores"]) == 1
    assert result["daily_scores"][0]["headline_count"] == 5


@patch(f"{MODULE}.supabase")
def test_has_data_for_today_true(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = [{"id": "abc"}]
    assert has_data_for_today("AAPL") is True


@patch(f"{MODULE}.supabase")
def test_has_data_for_today_false(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
    assert has_data_for_today("AAPL") is False


@patch(f"{MODULE}._get_stock_id", return_value=1)
@patch(f"{MODULE}.supabase")
def test_save_neutral_daily_score_uses_zero_article_neutral_payload(mock_supa, _mock_get_stock_id):
    mock_supa.table.return_value.upsert.return_value.execute.return_value.data = [{}]

    result = save_neutral_daily_sentiment_score("AAPL", date(2026, 7, 14))

    payload = mock_supa.table.return_value.upsert.call_args.args[0]
    assert payload["score_date"] == "2026-07-14"
    assert payload["article_count"] == 0
    assert payload["bullish_score"] == 5.0
    assert payload["sentiment_label"] == "neutral"
    assert result["rows_saved"] == 1


def test_calculate_daily_sentiment_score_weights_labels():
    rows = [
        {"label": "positive", "score": 0.8, "model_version": "ProsusAI/finbert"},
        {"label": "negative", "score": 0.6, "model_version": "ProsusAI/finbert"},
        {"label": "neutral", "score": 0.9, "model_version": "ProsusAI/finbert"},
    ]

    result = calculate_daily_sentiment_score("AAPL", 1, date(2026, 6, 9), rows)

    assert result["raw_sentiment"] == 0.0667
    assert result["bullish_score"] == 5.33
    assert result["sentiment_label"] == "neutral"
    assert result["article_count"] == 3


@patch(f"{MODULE}.get_active_stocks")
@patch(f"{MODULE}.supabase")
def test_get_recent_news_paginates(mock_supa, mock_get_stocks):
    mock_get_stocks.return_value = [{"symbol": "AAPL", "company_name": "Apple Inc."}]
    rows = [
        {
            "symbol": "AAPL",
            "headline": f"Headline {i}",
            "source": "gnews",
            "published_at": f"2026-07-{i + 1:02d}T09:00:00+00:00",
            "label": "positive",
            "score": 0.9,
            "url": f"https://example.com/{i}",
        }
        for i in range(25)
    ]
    mock_supa.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = rows

    result = get_recent_news(page=1, page_size=20)

    assert result["total_count"] == 25
    assert result["total_pages"] == 2
    assert result["page"] == 1
    assert len(result["articles"]) == 20


@patch(f"{MODULE}.get_active_stocks")
@patch(f"{MODULE}.supabase")
def test_get_recent_news_second_page(mock_supa, mock_get_stocks):
    mock_get_stocks.return_value = [{"symbol": "AAPL", "company_name": "Apple Inc."}]
    rows = [
        {
            "symbol": "AAPL",
            "headline": f"Headline {i}",
            "source": "gnews",
            "published_at": f"2026-07-{i + 1:02d}T09:00:00+00:00",
            "label": "positive",
            "score": 0.9,
            "url": f"https://example.com/{i}",
        }
        for i in range(25)
    ]
    mock_supa.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = rows

    result = get_recent_news(page=2, page_size=20)

    assert result["page"] == 2
    assert len(result["articles"]) == 5


@patch(f"{MODULE}.get_active_stocks")
@patch(f"{MODULE}.supabase")
def test_get_recent_news_filters_by_query_text(mock_supa, mock_get_stocks):
    mock_get_stocks.return_value = [
        {"symbol": "AAPL", "company_name": "Apple Inc."},
        {"symbol": "TSLA", "company_name": "Tesla Inc."},
    ]
    rows = [
        {"symbol": "AAPL", "headline": "Apple posts strong earnings", "source": "gnews",
         "published_at": "2026-07-20T09:00:00+00:00", "label": "positive", "score": 0.9,
         "url": "https://example.com/a"},
        {"symbol": "TSLA", "headline": "Tesla recalls vehicles", "source": "gnews",
         "published_at": "2026-07-21T09:00:00+00:00", "label": "negative", "score": 0.8,
         "url": "https://example.com/b"},
    ]
    mock_supa.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = rows

    result = get_recent_news(q="tesla")

    assert result["total_count"] == 1
    assert result["articles"][0]["symbol"] == "TSLA"


@patch(f"{MODULE}.get_active_stocks")
@patch(f"{MODULE}.supabase")
def test_get_recent_news_excludes_non_gnews_and_dead_urls(mock_supa, mock_get_stocks):
    mock_get_stocks.return_value = [{"symbol": "AAPL", "company_name": "Apple Inc."}]
    rows = [
        {"symbol": "AAPL", "headline": "From finnhub", "source": "finnhub",
         "published_at": "2026-07-20T09:00:00+00:00", "label": "positive", "score": 0.9,
         "url": "https://finnhub.io/api/news?id=1"},
        {"symbol": "AAPL", "headline": "Good gnews article", "source": "gnews",
         "published_at": "2026-07-21T09:00:00+00:00", "label": "positive", "score": 0.9,
         "url": "https://news.google.com/rss/articles/xyz"},
    ]
    mock_supa.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = rows

    result = get_recent_news()

    assert result["total_count"] == 1
    assert result["articles"][0]["headline"] == "Good gnews article"


@patch(f"{MODULE}.get_active_stocks")
@patch(f"{MODULE}.supabase")
def test_get_recent_news_empty_result(mock_supa, mock_get_stocks):
    mock_get_stocks.return_value = []
    mock_supa.table.return_value.select.return_value.gte.return_value.order.return_value.limit.return_value.execute.return_value.data = []

    result = get_recent_news()

    assert result == {"articles": [], "page": 1, "total_pages": 0, "total_count": 0}
