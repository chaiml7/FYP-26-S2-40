from datetime import date
from unittest.mock import patch, MagicMock

from backend.services.sentiment.gnews_service import fetch_news

MODULE = "backend.services.sentiment.gnews_service"

SAMPLE_ARTICLE = {
    "title": "DBS posts record quarterly profit",
    "published date": "Mon, 24 May 2026 09:00:00 GMT",
    "url": "https://example.com/dbs-article",
}


def mock_gnews(articles=None):
    instance = MagicMock()
    instance.get_news.return_value = articles if articles is not None else [SAMPLE_ARTICLE]
    return instance


@patch(f"{MODULE}.GNews")
def test_sgx_symbol_uses_singapore_query(mock_gnews_cls):
    mock_gnews_cls.return_value = mock_gnews()
    fetch_news("D05.SI", "DBS Group Holdings Ltd", from_date=date(2026, 5, 23))
    query_arg = mock_gnews_cls.return_value.get_news.call_args.args[0]
    assert query_arg == "DBS Group Holdings Ltd Singapore"
    assert "D05.SI" not in query_arg
    assert "stock" not in query_arg


@patch(f"{MODULE}.GNews")
def test_sgx_symbol_uses_sg_country(mock_gnews_cls):
    mock_gnews_cls.return_value = mock_gnews()
    fetch_news("D05.SI", "DBS Group Holdings Ltd", from_date=date(2026, 5, 23))
    assert mock_gnews_cls.call_args.kwargs["country"] == "SG"


@patch(f"{MODULE}.GNews")
def test_sgx_detection_is_case_insensitive(mock_gnews_cls):
    mock_gnews_cls.return_value = mock_gnews()
    fetch_news("d05.si", "DBS Group Holdings Ltd", from_date=date(2026, 5, 23))
    assert mock_gnews_cls.call_args.kwargs["country"] == "SG"


@patch(f"{MODULE}.GNews")
def test_non_sgx_symbol_uses_original_query(mock_gnews_cls):
    mock_gnews_cls.return_value = mock_gnews()
    fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    query_arg = mock_gnews_cls.return_value.get_news.call_args.args[0]
    assert query_arg == "Apple AAPL stock"


@patch(f"{MODULE}.GNews")
def test_non_sgx_symbol_uses_us_country(mock_gnews_cls):
    mock_gnews_cls.return_value = mock_gnews()
    fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert mock_gnews_cls.call_args.kwargs["country"] == "US"


@patch(f"{MODULE}.GNews")
def test_sgx_result_shape_unchanged(mock_gnews_cls):
    mock_gnews_cls.return_value = mock_gnews()
    result = fetch_news("D05.SI", "DBS Group Holdings Ltd", from_date=date(2026, 5, 23))
    assert len(result) == 1
    assert result[0]["headline"] == "DBS posts record quarterly profit"
    assert result[0]["source"] == "gnews"
    assert result[0]["url"] == "https://example.com/dbs-article"


@patch(f"{MODULE}.GNews")
def test_gnews_exception_returns_empty_for_sgx(mock_gnews_cls):
    instance = mock_gnews()
    instance.get_news.side_effect = Exception("network error")
    mock_gnews_cls.return_value = instance
    result = fetch_news("D05.SI", "DBS Group Holdings Ltd", from_date=date(2026, 5, 23))
    assert result == []
