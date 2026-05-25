import pytest
import requests
from unittest.mock import patch, MagicMock
from datetime import date
from services.sentiment.news_scraper_service import fetch_news

MODULE = "services.sentiment.news_scraper_service"

SAMPLE_NEWSAPI_RESPONSE = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {"title": "Apple sales surge globally", "publishedAt": "2026-05-24T09:00:00Z"},
        {"title": "Apple faces antitrust probe", "publishedAt": "2026-05-24T10:00:00Z"},
    ],
}


def mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data if json_data is not None else SAMPLE_NEWSAPI_RESPONSE
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


@pytest.fixture(autouse=True)
def patch_api_key(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "test_key")


@patch(f"{MODULE}.requests.get")
def test_fetch_returns_headlines(mock_get):
    mock_get.return_value = mock_response()
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert result[0]["source"] == "newsapi"
    assert "headline" in result[0]
    assert "published_at" in result[0]


@patch(f"{MODULE}.requests.get")
def test_fetch_empty_articles(mock_get):
    mock_get.return_value = mock_response(json_data={"status": "ok", "articles": []})
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_missing_articles_key(mock_get):
    mock_get.return_value = mock_response(json_data={"status": "ok"})
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_quota_exceeded_429_returns_empty(mock_get):
    mock_get.return_value = mock_response(status_code=429)
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_quota_exceeded_does_not_retry(mock_get):
    mock_get.return_value = mock_response(status_code=429)
    fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert mock_get.call_count == 1


@patch(f"{MODULE}.requests.get")
def test_quota_exceeded_426_returns_empty(mock_get):
    mock_get.return_value = mock_response(status_code=426)
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_timeout_retries(mock_get, mock_sleep):
    mock_get.side_effect = [requests.Timeout(), mock_response()]
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert mock_get.call_count == 2


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_timeout_exhausted_returns_empty(mock_get, mock_sleep):
    mock_get.side_effect = requests.Timeout()
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert result == []
    assert mock_get.call_count == 3


@patch(f"{MODULE}.requests.get")
def test_article_missing_title_filtered_out(mock_get):
    mock_get.return_value = mock_response(json_data={
        "articles": [
            {"title": "Valid title", "publishedAt": "2026-05-24T09:00:00Z"},
            {"title": "", "publishedAt": "2026-05-24T10:00:00Z"},
            {"publishedAt": "2026-05-24T11:00:00Z"},
        ]
    })
    result = fetch_news("AAPL", "Apple", from_date=date(2026, 5, 23))
    assert len(result) == 1
    assert result[0]["headline"] == "Valid title"
