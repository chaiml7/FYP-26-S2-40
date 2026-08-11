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


def test_admin_reports_requires_login():
    client.cookies.clear()
    response = client.get("/admin/reports", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_admin_reports_page_renders_for_frontend_admin():
    client.cookies.set("session", _session_cookie(role="frontend_admin"))
    response = client.get("/admin/reports")
    client.cookies.clear()

    assert response.status_code == 200
    assert "Performance Reports" in response.text
    assert "Generate Admin Report" in response.text


MOCK_REPORT = {
    "generated_at": "2026-08-04 20:30 +08",
    "generated_by": "admin@example.com",
    "row_limit": 1000,
    "summary_cards": [
        {"label": "Total Users", "value": "7", "detail": "1 premium users"},
    ],
    "user_summary": {
        "total_users": 7,
        "active_users": 6,
        "suspended_users": 1,
        "admin_users": 1,
        "role_distribution": [{"label": "Frontend admin", "count": 1, "percent": "14.3%"}],
    },
    "stock_summary": {
        "total_stocks": 14,
        "active_stocks": 14,
        "inactive_stocks": 0,
        "recent_stocks": [],
    },
    "watchlist_summary": {
        "total_entries": 1,
        "users_with_watchlists": 1,
        "average_per_user": "1.00",
        "top_stocks": [],
        "top_users": [],
    },
    "model_summary": {
        "rows": [
            {
                "name": "Technical",
                "details": [
                    {"label": "Version", "value": "technical_v1"},
                    {"label": "Accuracy", "value": "70.0%"},
                ],
            },
            {
                "name": "Sentiment",
                "details": [
                    {"label": "Version", "value": "finbert-stocklens"},
                ],
            },
            {"name": "Financial", "details": []},
        ],
    },
    "prediction_summary": {
        "rows": [{
            "name": "Technical",
            "total": 1,
            "bullish": 1,
            "neutral": 0,
            "bearish": 0,
            "unknown": 0,
            "average_score": "7.00",
            "latest_date": "2026-08-04",
        }],
    },
    "sentiment_summary": {
        "average_bullish_score": "5.00",
        "article_label_rows": [],
        "daily_label_rows": [],
    },
    "freshness_summary": {
        "stale_after_days": 7,
        "rows": [],
    },
    "weightage_summary": {
        "technical": "40%",
        "sentiment": "30%",
        "financial": "30%",
        "custom_record_count": 0,
    },
    "errors": [],
}


@patch("backend.routes.admin_routes.build_admin_report", return_value=MOCK_REPORT)
def test_generate_admin_report_renders_model_cards_without_na(mock_report):
    client.cookies.set("session", _session_cookie(role="frontend_admin"))
    response = client.post("/admin/reports/generate")
    client.cookies.clear()

    assert response.status_code == 200
    assert "technical_v1" in response.text
    assert "finbert-stocklens" in response.text
    assert "No stored model data." in response.text
    assert "N/A" not in response.text
    assert "Technical Model" not in response.text
    assert "Financial Model" not in response.text
    assert "Prediction Summary" not in response.text
    assert "Data Health" in response.text
    mock_report.assert_called_once_with("admin@example.com")
