import base64
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from frontend.main import app

client = TestClient(app)


def _session_cookie(role="frontend_admin", email="admin@example.com", user_id="admin-id"):
    signer = TimestampSigner("my-super-secret-key")
    data = {"user_role": role, "user_email": email, "user_id": user_id}
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def test_admin_sentiment_requires_login():
    client.cookies.clear()
    response = client.get("/admin/sentiment", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@patch(
    "backend.routes.admin_routes.get_all_sentiment_sources",
    return_value=[
        {"id": "1", "source_type": "RSS", "account": "Reuters", "relevance": "Market-wide", "is_active": True},
        {"id": "2", "source_type": "API", "account": "FinnHub", "relevance": None, "is_active": False},
    ],
)
def test_admin_sentiment_lists_sources(mock_get_sources):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/sentiment")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Reuters" in response.text
    assert "FinnHub" in response.text
    assert "Suspended" in response.text


@patch("backend.routes.admin_routes.add_sentiment_source")
def test_admin_add_sentiment_source_redirects(mock_add):
    client.cookies.set("session", _session_cookie())
    response = client.post(
        "/admin/sentiment/add",
        data={"source_type": "RSS", "account": "MarketWatch", "relevance": "Market-wide"},
        follow_redirects=False,
    )
    client.cookies.clear()

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/sentiment"
    mock_add.assert_called_once_with("RSS", "MarketWatch", "Market-wide")


@patch("backend.routes.admin_routes.set_sentiment_source_active")
def test_admin_suspend_sentiment_source(mock_set_active):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/sentiment/1/suspend", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    mock_set_active.assert_called_once_with("1", False)


@patch("backend.routes.admin_routes.set_sentiment_source_active")
def test_admin_reactivate_sentiment_source(mock_set_active):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/sentiment/1/reactivate", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    mock_set_active.assert_called_once_with("1", True)


@patch("backend.routes.admin_routes.delete_sentiment_source")
def test_admin_delete_sentiment_source(mock_delete):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/sentiment/1/delete", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/sentiment"
    mock_delete.assert_called_once_with("1")
