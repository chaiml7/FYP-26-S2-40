import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from frontend.main import app

client = TestClient(app)


def _session_cookie(role="basic_user", email="user@example.com", user_id="user-1"):
    signer = TimestampSigner("my-super-secret-key")
    data = {"user_role": role, "user_email": email, "user_id": user_id}
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def _auth_response(user_id="user-1"):
    return SimpleNamespace(user=SimpleNamespace(id=user_id))


@patch("frontend.main.log_activity")
@patch("frontend.main.get_profile")
@patch("frontend.main._supabase_auth")
def test_login_blocked_for_suspended_user(mock_auth, mock_get_profile, mock_log):
    mock_auth.auth.sign_in_with_password.return_value = _auth_response()
    mock_get_profile.return_value = [{"role_id": "basic_user", "is_active": False}]

    response = client.post(
        "/login",
        data={"email": "suspended@example.com", "password": "password123"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "suspended" in response.text.lower()
    mock_auth.auth.sign_out.assert_called_once()
    mock_log.assert_not_called()


@patch("frontend.main.reconcile_user_subscription_role", side_effect=lambda user_id, role: role)
@patch("frontend.main.log_activity")
@patch("frontend.main.get_profile")
@patch("frontend.main._supabase_auth")
def test_login_allowed_for_active_user(mock_auth, mock_get_profile, mock_log, mock_reconcile):
    mock_auth.auth.sign_in_with_password.return_value = _auth_response()
    mock_get_profile.return_value = [{"role_id": "basic_user", "is_active": True}]

    response = client.post(
        "/login",
        data={"email": "active@example.com", "password": "password123"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"
    mock_log.assert_called_once_with("active@example.com", "login")


@patch("frontend.main.get_profile")
def test_suspended_session_is_kicked_out_mid_session(mock_get_profile):
    mock_get_profile.return_value = [{"role_id": "basic_user", "is_active": False}]

    client.cookies.set("session", _session_cookie())
    response = client.get("/user/account", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
