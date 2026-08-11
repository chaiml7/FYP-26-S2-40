import base64
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from frontend.main import app


client = TestClient(app)
MODULE = "backend.routes.user_routes"


def _session_cookie(role="premium_user", user_id="user-1"):
    signer = TimestampSigner("my-super-secret-key")
    data = {
        "user_role": role,
        "user_email": "premium@example.com",
        "user_id": user_id,
    }
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


@patch(f"{MODULE}.get_user_watchlist_summary", return_value=[])
@patch(
    f"{MODULE}.get_notification_preference",
    return_value={"user_id": "user-1", "analysis_ready_email": True},
)
def test_watchlist_renders_saved_email_opt_in(mock_preference, _mock_watchlist):
    client.cookies.set("session", _session_cookie())

    response = client.get("/user/watchlist")

    client.cookies.clear()
    assert response.status_code == 200
    assert "Email notifications" in response.text
    assert 'name="analysis_ready_email"' in response.text
    assert "checked" in response.text
    mock_preference.assert_called_once_with("user-1")


@patch(f"{MODULE}.set_notification_preference")
def test_premium_user_can_enable_email_notifications(mock_set):
    client.cookies.set("session", _session_cookie())

    response = client.post(
        "/user/notifications/email",
        data={"analysis_ready_email": "true"},
        follow_redirects=False,
    )

    client.cookies.clear()
    assert response.status_code == 303
    assert response.headers["location"] == "/user/watchlist?notifications=updated"
    mock_set.assert_called_once_with("user-1", True)


@patch(f"{MODULE}.set_notification_preference")
def test_free_user_cannot_enable_watchlist_emails(mock_set):
    client.cookies.set("session", _session_cookie(role="basic_user"))

    response = client.post(
        "/user/notifications/email",
        data={"analysis_ready_email": "true"},
    )

    client.cookies.clear()
    assert response.status_code == 403
    mock_set.assert_not_called()
