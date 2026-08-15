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


def test_admin_prediction_logs_requires_login():
    client.cookies.clear()
    response = client.get("/admin/prediction_logs", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@patch(
    "backend.routes.admin_routes.build_model_accuracy_log",
    return_value={
        "technical": [
            {"version": "tech_v3", "trained_at": "2026-08-01", "is_active": True, "accuracy": "72.0%", "balanced_accuracy": "70.0%", "macro_f1": "69.0%", "log_loss": "0.5000"},
        ],
        "financial": [],
        "errors": [],
    },
)
def test_admin_prediction_logs_renders(mock_build_log):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/prediction_logs")
    client.cookies.clear()

    assert response.status_code == 200
    assert "tech_v3" in response.text
    assert "72.0%" in response.text


def test_admin_system_health_requires_login():
    client.cookies.clear()
    response = client.get("/admin/system_health", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@patch(
    "backend.routes.admin_routes.build_system_health",
    return_value={
        "checked_at": "2026-08-15 10:00 +08",
        "db_status": "Operational",
        "db_latency_ms": 42.0,
        "freshness_summary": {"stale_after_days": 7, "rows": []},
        "integrations": [{"name": "FinnHub (sentiment news)", "configured": True}],
        "errors": [],
    },
)
def test_admin_system_health_renders(mock_build_health):
    client.cookies.set("session", _session_cookie())
    response = client.get("/admin/system_health")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Operational" in response.text
    assert "FinnHub" in response.text
