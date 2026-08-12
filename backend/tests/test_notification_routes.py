from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_dispatch_requires_configured_admin_key(monkeypatch):
    monkeypatch.delenv("NOTIFICATION_ADMIN_KEY", raising=False)

    response = client.post("/api/notifications/analysis-ready/dispatch")

    assert response.status_code == 503


@patch("backend.routes.notification_routes.dispatch_analysis_ready_emails")
def test_dispatch_defaults_to_safe_dry_run(mock_dispatch, monkeypatch):
    monkeypatch.setenv("NOTIFICATION_ADMIN_KEY", "test-admin-key")
    mock_dispatch.return_value = {"dry_run": True, "emails_sent": 0}

    response = client.post(
        "/api/notifications/analysis-ready/dispatch",
        headers={"X-Notification-Admin-Key": "test-admin-key"},
    )

    assert response.status_code == 200
    assert response.json()["dry_run"] is True
    mock_dispatch.assert_called_once_with(notification_date=None, dry_run=True)


def test_dispatch_rejects_wrong_admin_key(monkeypatch):
    monkeypatch.setenv("NOTIFICATION_ADMIN_KEY", "correct-key")

    response = client.post(
        "/api/notifications/analysis-ready/dispatch",
        headers={"X-Notification-Admin-Key": "wrong-key"},
    )

    assert response.status_code == 403
