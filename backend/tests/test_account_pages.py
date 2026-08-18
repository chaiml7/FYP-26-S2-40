import base64
import json
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from backend.services.auth_service import AuthServiceError
from frontend.main import app


client = TestClient(app)
MODULE = "backend.routes.user_routes"


def _session_cookie(role="basic_user", email="user@example.com", user_id="user-id"):
    signer = TimestampSigner("my-super-secret-key")
    data = {
        "user_role": role,
        "user_email": email,
        "user_id": user_id,
        "role_checked_at": time.time(),
    }
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def _get_as(path, role="basic_user", email="user@example.com"):
    client.cookies.set("session", _session_cookie(role=role, email=email))
    response = client.get(path)
    client.cookies.clear()
    return response


def test_account_overview_requires_login():
    client.cookies.clear()
    response = client.get("/user/account", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@patch(f"{MODULE}.get_profile", return_value=[{
    "id": "user-id",
    "full_name": "Jamie Tan",
    "email": "jamie@example.com",
    "role_id": "basic_user",
    "created_at": "2026-02-03T10:20:30+00:00",
    "is_active": True,
}])
def test_account_overview_shows_free_user_details_and_separate_actions(_mock_profile):
    response = _get_as("/user/account", email="jamie@example.com")

    assert response.status_code == 200
    assert "Jamie Tan" in response.text
    assert "jamie@example.com" in response.text
    assert "2026-02-03" in response.text
    assert "Free" in response.text
    assert 'href="/user/account/password"' in response.text
    assert 'href="/user/account/delete"' in response.text
    assert 'action="/user/account/password"' not in response.text
    assert 'action="/user/account/delete"' not in response.text


@patch(f"{MODULE}.get_profile", return_value=[{
    "full_name": "Priya Lim",
    "email": "priya@example.com",
    "created_at": "2025-11-12T08:00:00+00:00",
    "is_active": True,
}])
def test_account_overview_uses_current_premium_membership(_mock_profile):
    response = _get_as(
        "/user/account", role="premium_user", email="priya@example.com"
    )

    assert response.status_code == 200
    assert "Premium" in response.text
    assert "account-plan-premium" in response.text


@patch(f"{MODULE}.get_profile", return_value=[{
    "full_name": "StockLens Admin",
    "email": "admin@example.com",
    "created_at": "2025-01-01T00:00:00+00:00",
    "is_active": True,
}])
def test_account_overview_uses_administrator_membership(_mock_profile):
    response = _get_as(
        "/user/account", role="admin", email="admin@example.com"
    )

    assert response.status_code == 200
    assert "Administrator" in response.text
    assert "account-plan-admin" in response.text


def test_change_password_has_its_own_page():
    response = _get_as("/user/account/password")

    assert response.status_code == 200
    assert "Change password" in response.text
    assert 'action="/user/account/password"' in response.text
    assert 'href="/user/account"' in response.text
    assert 'action="/user/account/delete"' not in response.text


def test_invalid_new_password_stays_on_change_password_page():
    client.cookies.set("session", _session_cookie())
    response = client.post(
        "/user/account/password",
        data={
            "current_password": "current-password",
            "new_password": "short",
            "confirm_password": "short",
        },
    )
    client.cookies.clear()

    assert response.status_code == 200
    assert (
        "Password must be at least 8 characters and contain an uppercase "
        "letter, a number and a special character." in response.text
    )
    assert 'action="/user/account/password"' in response.text
    assert 'action="/user/account/delete"' not in response.text


@patch(f"{MODULE}.get_profile", return_value=[{
    "full_name": "Jamie Tan",
    "email": "jamie@example.com",
    "is_active": True,
}])
def test_update_profile_rejects_empty_full_name(_mock_profile):
    client.cookies.set("session", _session_cookie(email="jamie@example.com"))
    response = client.post("/user/account/profile", data={"full_name": "   "})
    client.cookies.clear()

    assert response.status_code == 200
    assert "Full name is required." in response.text


@patch(f"{MODULE}.update_profile")
@patch(f"{MODULE}.get_profile", return_value=[{
    "full_name": "Jamie Tan",
    "email": "jamie@example.com",
    "is_active": True,
}])
def test_update_profile_saves_new_full_name(_mock_profile, mock_update_profile):
    client.cookies.set("session", _session_cookie(email="jamie@example.com"))
    response = client.post("/user/account/profile", data={"full_name": "Jamie Lee"})
    client.cookies.clear()

    assert response.status_code == 200
    assert "Jamie Lee" in response.text
    mock_update_profile.assert_called_once_with("user-id", {"full_name": "Jamie Lee"})


def test_delete_account_has_its_own_page_and_premium_billing_warning():
    response = _get_as("/user/account/delete", role="premium_user")

    assert response.status_code == 200
    assert "Permanently delete account" in response.text
    assert 'action="/user/account/delete"' in response.text
    assert 'action="/billing/portal"' in response.text
    assert "does not cancel Stripe billing automatically" in response.text


@patch(f"{MODULE}.login", side_effect=AuthServiceError(401, "Invalid password"))
def test_wrong_delete_password_stays_on_delete_page(_mock_login):
    client.cookies.set("session", _session_cookie())
    response = client.post(
        "/user/account/delete",
        data={"current_password": "incorrect-password"},
    )
    client.cookies.clear()

    assert response.status_code == 200
    assert "Current password is incorrect." in response.text
    assert 'action="/user/account/delete"' in response.text
    assert 'action="/user/account/password"' not in response.text
