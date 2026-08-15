import base64
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from frontend.main import app

client = TestClient(app)


def _session_cookie(role="admin", email="admin@example.com", user_id="admin-id"):
    signer = TimestampSigner("my-super-secret-key")
    data = {"user_role": role, "user_email": email, "user_id": user_id}
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def test_admin_activity_log_requires_login():
    client.cookies.clear()
    response = client.get("/admin/activity_log", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@patch(
    "backend.routes.admin_routes.get_activity_log",
    return_value=[
        {"email": "user@example.com", "action": "account_suspended", "detail": "Suspended by admin@example.com", "created_at": "2026-08-15T10:00:00"},
    ],
)
def test_admin_activity_log_renders_entries(mock_get_log):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/activity_log")
    client.cookies.clear()

    assert response.status_code == 200
    assert "user@example.com" in response.text
    assert "account suspended" in response.text
