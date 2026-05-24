import pytest
from unittest.mock import patch, MagicMock, call
from services.sentiment.sentiment_aggregator import save_scores, get_sentiment_summary, has_data_for_today, _score_to_label
from tests.sentiment.conftest import SAMPLE_SCORED_HEADLINES

MODULE = "services.sentiment.sentiment_aggregator"


def make_supabase_chain(upsert_data=None, select_data=None):
    mock = MagicMock()
    # upsert chain: .table().upsert().execute()
    mock.table.return_value.upsert.return_value.execute.return_value.data = upsert_data or SAMPLE_SCORED_HEADLINES
    # select chain for get_sentiment_summary: .table().select().eq().gte().order().execute()
    mock.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = select_data or []
    # select chain for has_data_for_today: .table().select().eq().gte().limit().execute()
    mock.table.return_value.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
    return mock


@patch(f"{MODULE}.supabase")
def test_save_scores_upserts_rows(mock_supa):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    mock_supa.table.assert_called_with("sentiment_scores")
    rows = mock_supa.table.return_value.upsert.call_args[0][0]
    assert len(rows) == 3
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["model_version"] == "ProsusAI/finbert"


@patch(f"{MODULE}.supabase")
def test_save_scores_empty_list_does_not_upsert(mock_supa):
    result = save_scores("AAPL", [])
    mock_supa.table.return_value.upsert.assert_not_called()
    assert result["rows_saved"] == 0


@patch(f"{MODULE}.supabase")
def test_save_scores_includes_required_fields(mock_supa):
    save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    rows = mock_supa.table.return_value.upsert.call_args[0][0]
    required = {"symbol", "headline", "source", "published_at", "label", "score", "model_version"}
    assert required.issubset(rows[0].keys())


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.supabase")
def test_save_scores_retries_on_failure(mock_supa, mock_sleep):
    mock_supa.table.return_value.upsert.return_value.execute.side_effect = [
        Exception("db error"),
        MagicMock(data=SAMPLE_SCORED_HEADLINES),
    ]
    result = save_scores("AAPL", SAMPLE_SCORED_HEADLINES)
    assert mock_supa.table.return_value.upsert.call_count == 2


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.supabase")
def test_save_scores_all_retries_fail_raises(mock_supa, mock_sleep):
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
    assert "headlines" in result
    assert isinstance(result["daily_scores"], list)
    assert isinstance(result["headlines"], list)


@patch(f"{MODULE}.supabase")
def test_get_summary_empty_db_returns_empty_lists(mock_supa):
    mock_supa.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = []
    result = get_sentiment_summary("AAPL")
    assert result == {"daily_scores": [], "headlines": []}


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
