import base64
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from frontend.main import app

client = TestClient(app)
MODULE = "backend.routes.user_routes"


def _session_cookie(role="basic_user", email="test@example.com", user_id="test-id"):
    signer = TimestampSigner("my-super-secret-key")
    data = {"user_role": role, "user_email": email, "user_id": user_id}
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


MOCK_NEWS_RESULT = {
    "articles": [{
        "symbol": "AAPL", "company_name": "Apple Inc.", "headline": "Apple news",
        "source": "gnews", "published_at": "2026-07-20T09:00:00+00:00",
        "label": "positive", "score": 0.9, "url": "https://example.com/a",
    }],
    "page": 1,
    "total_pages": 1,
    "total_count": 1,
}


@patch(f"{MODULE}.get_recent_news", return_value=MOCK_NEWS_RESULT)
def test_api_news_requires_login(mock_news):
    client.cookies.clear()
    response = client.get("/api/news")
    assert response.status_code == 401


@patch(f"{MODULE}.get_recent_news", return_value=MOCK_NEWS_RESULT)
def test_api_news_returns_json_for_logged_in_user(mock_news):
    client.cookies.set("session", _session_cookie())
    response = client.get("/api/news")
    client.cookies.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert data["articles"][0]["symbol"] == "AAPL"


@patch(f"{MODULE}.get_recent_news", return_value=MOCK_NEWS_RESULT)
def test_api_news_passes_filters_through(mock_news):
    client.cookies.set("session", _session_cookie())
    client.get("/api/news?q=apple&label=positive&symbol=aapl&page=2")
    client.cookies.clear()
    mock_news.assert_called_once_with(symbol="AAPL", label="positive", q="apple", page=2)


@patch(f"{MODULE}.get_recent_news", return_value=MOCK_NEWS_RESULT)
def test_api_news_ignores_invalid_label(mock_news):
    client.cookies.set("session", _session_cookie())
    client.get("/api/news?label=bogus")
    client.cookies.clear()
    mock_news.assert_called_once_with(symbol=None, label=None, q=None, page=1)


@patch(f"{MODULE}.get_active_stocks", return_value=[
    {"symbol": "AAPL", "company_name": "Apple Inc.", "is_active": True},
])
@patch(f"{MODULE}.get_recent_news", return_value=MOCK_NEWS_RESULT)
def test_news_social_page_preselects_symbol_from_stock_link(mock_news, _mock_stocks):
    client.cookies.set("session", _session_cookie(role="premium_user"))
    response = client.get("/user/news_social?symbol=aapl")
    client.cookies.clear()

    assert response.status_code == 200
    assert '<option value="AAPL" selected>' in response.text
    mock_news.assert_called_once_with(
        symbol="AAPL", label=None, q="", page=1
    )


@patch(f"{MODULE}.search_active_stocks", return_value=[{"symbol": "AAPL", "company_name": "Apple Inc."}])
def test_api_stocks_search_requires_login(mock_search):
    client.cookies.clear()
    response = client.get("/api/stocks/search?q=app")
    assert response.status_code == 401


@patch(f"{MODULE}.search_active_stocks", return_value=[{"symbol": "AAPL", "company_name": "Apple Inc.", "is_active": True}])
def test_api_stocks_search_returns_matches(mock_search):
    client.cookies.set("session", _session_cookie())
    response = client.get("/api/stocks/search?q=app")
    client.cookies.clear()
    assert response.status_code == 200
    assert response.json() == [{"symbol": "AAPL", "company_name": "Apple Inc."}]
    mock_search.assert_called_once_with("app")
