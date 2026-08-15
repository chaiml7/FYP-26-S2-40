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


@patch("backend.routes.admin_routes.get_all_stocks", return_value=[])
def test_stock_database_links_to_admin_prefix(mock_stocks):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/stocks")
    client.cookies.clear()

    assert response.status_code == 200
    assert "/admin/stocks/new" in response.text
    assert "/backend_admin/" not in response.text


def test_add_stock_page_links_to_admin_prefix():
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/stocks/new")
    client.cookies.clear()

    assert response.status_code == 200
    assert "/admin/stocks" in response.text
    assert "/backend_admin/" not in response.text


@patch("backend.routes.admin_routes.supabase")
def test_weightages_form_posts_to_admin_prefix(mock_supabase):
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/weightages")
    client.cookies.clear()

    assert response.status_code == 200
    assert 'action="/admin/weightages"' in response.text
