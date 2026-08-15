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


@patch("backend.routes.admin_routes.log_activity")
@patch(
    "backend.routes.admin_routes.update_user_status",
    return_value=[{"id": "target-id", "email": "target@example.com"}],
)
def test_admin_suspend_user_redirects(mock_update, mock_log):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/users/target-id/suspend", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/user_management"
    mock_update.assert_called_once_with("target-id", False)
    mock_log.assert_called_once_with(
        "target@example.com", "account_suspended", "Suspended by admin@example.com"
    )


@patch("backend.routes.admin_routes.update_user_status")
def test_admin_cannot_suspend_self(mock_update):
    client.cookies.set("session", _session_cookie(user_id="admin-id"))
    response = client.post("/admin/users/admin-id/suspend", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    mock_update.assert_not_called()


@patch("backend.routes.admin_routes.log_activity")
@patch(
    "backend.routes.admin_routes.update_user_status",
    return_value=[{"id": "target-id", "email": "target@example.com"}],
)
def test_admin_unsuspend_user_redirects(mock_update, mock_log):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/users/target-id/unsuspend", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 303
    mock_update.assert_called_once_with("target-id", True)
    mock_log.assert_called_once_with(
        "target@example.com", "account_reactivated", "Reactivated by admin@example.com"
    )


@patch("backend.routes.admin_routes.update_user_status", return_value=[])
def test_admin_suspend_user_404_when_missing(mock_update):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/users/missing-id/suspend", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 404


@patch("backend.routes.admin_routes.update_user_status", return_value=[])
def test_admin_unsuspend_user_404_when_missing(mock_update):
    client.cookies.set("session", _session_cookie())
    response = client.post("/admin/users/missing-id/unsuspend", follow_redirects=False)
    client.cookies.clear()

    assert response.status_code == 404


@patch(
    "backend.routes.admin_routes.get_profile",
    return_value=[{
        "id": "target-id",
        "username": "jdoe",
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "role_id": "basic_user",
        "is_active": True,
        "created_at": "2026-01-05T00:00:00Z",
    }],
)
def test_admin_user_detail_renders(mock_get_profile):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/users/target-id")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Jane Doe" in response.text
    assert "jane@example.com" in response.text


@patch("backend.routes.admin_routes.get_profile", return_value=[])
def test_admin_user_detail_404_when_missing(mock_get_profile):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/users/missing-id")
    client.cookies.clear()

    assert response.status_code == 404


_ALL_USERS = [
    {"id": "1", "username": "jdoe", "full_name": "Jane Doe", "email": "jane@example.com", "role_id": "basic_user", "is_active": True},
    {"id": "2", "username": "bsmith", "full_name": "Bob Smith", "email": "bob@example.com", "role_id": "premium_user", "is_active": False},
]


@patch("backend.routes.admin_routes.supabase")
def test_admin_user_management_search_by_name(mock_supabase):
    mock_supabase.table.return_value.select.return_value.execute.return_value.data = _ALL_USERS
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/user_management?q=jane")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Jane Doe" in response.text
    assert "Bob Smith" not in response.text


@patch("backend.routes.admin_routes.supabase")
def test_admin_user_management_search_by_email(mock_supabase):
    mock_supabase.table.return_value.select.return_value.execute.return_value.data = _ALL_USERS
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/user_management?q=bob@example.com")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Bob Smith" in response.text
    assert "Jane Doe" not in response.text


@patch("backend.routes.admin_routes.supabase")
def test_admin_user_management_search_by_id(mock_supabase):
    mock_supabase.table.return_value.select.return_value.execute.return_value.data = _ALL_USERS
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/user_management?q=2")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Bob Smith" in response.text
    assert "Jane Doe" not in response.text


@patch("backend.routes.admin_routes.supabase")
def test_admin_user_management_search_no_match(mock_supabase):
    mock_supabase.table.return_value.select.return_value.execute.return_value.data = _ALL_USERS
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/user_management?q=nonexistent")
    client.cookies.clear()

    assert response.status_code == 200
    assert "No users found." in response.text
