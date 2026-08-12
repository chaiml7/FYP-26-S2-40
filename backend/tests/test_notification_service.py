import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.services.notification_service import (
    _processing_delivery_is_active,
    _send_sendgrid_email,
    build_analysis_ready_email,
    dispatch_analysis_ready_emails,
    get_in_app_notifications,
)


MODULE = "backend.services.notification_service"
DAY = date(2026, 8, 6)


def _ready_analysis(symbol="AAPL"):
    return {
        "symbol": symbol,
        "latest_market_date": "2026-08-05",
        "price": 210.5,
        "technical": {
            "latest_date": "2026-08-05",
            "prediction": "bullish",
            "technical_score": 6.7,
        },
        "sentiment": {
            "score_date": DAY.isoformat(),
            "sentiment_label": "neutral",
            "bullish_score": 5.0,
        },
        "financial": {
            "period": "2026-Q2",
            "prediction": "bullish",
        },
        "ready": True,
        "missing": [],
    }


@patch(f"{MODULE}._send_sendgrid_email")
@patch(f"{MODULE}._enabled_users")
@patch(f"{MODULE}._active_watchlist")
@patch(f"{MODULE}.load_symbol_analysis")
def test_dry_run_reports_ready_without_sending(
    mock_load, mock_watchlist, mock_users, mock_send
):
    mock_users.return_value = [{"id": "u1", "email": "user@example.com", "is_active": True}]
    mock_watchlist.return_value = [
        {"stock_id": 1, "symbol": "AAPL", "company_name": "Apple Inc."}
    ]
    mock_load.return_value = _ready_analysis()

    with patch(f"{MODULE}._existing_delivery", return_value=None):
        result = dispatch_analysis_ready_emails(DAY, dry_run=True)

    assert result["emails_sent"] == 0
    assert result["results"][0]["status"] == "ready"
    mock_send.assert_not_called()


@patch(f"{MODULE}._enabled_users")
@patch(f"{MODULE}._active_watchlist")
@patch(f"{MODULE}.load_symbol_analysis")
def test_dispatch_waits_until_every_watchlist_stock_is_ready(
    mock_load, mock_watchlist, mock_users
):
    mock_users.return_value = [{"id": "u1", "email": "user@example.com"}]
    mock_watchlist.return_value = [
        {"stock_id": 1, "symbol": "AAPL", "company_name": "Apple Inc."},
        {"stock_id": 2, "symbol": "MSFT", "company_name": "Microsoft"},
    ]
    mock_load.side_effect = [
        _ready_analysis("AAPL"),
        {
            **_ready_analysis("MSFT"),
            "sentiment": None,
            "ready": False,
            "missing": ["sentiment"],
        },
    ]

    result = dispatch_analysis_ready_emails(DAY, dry_run=True)

    assert result["results"][0] == {
        "user_id": "u1",
        "status": "not_ready",
        "missing": [{"symbol": "MSFT", "missing": ["sentiment"]}],
    }


@patch(f"{MODULE}._update_delivery")
@patch(f"{MODULE}._send_sendgrid_email", return_value="sendgrid-message-123")
@patch(f"{MODULE}._claim_delivery")
@patch(f"{MODULE}._existing_delivery", return_value=None)
@patch(f"{MODULE}._enabled_users")
@patch(f"{MODULE}._active_watchlist")
@patch(f"{MODULE}.load_symbol_analysis")
def test_dispatch_claims_sends_and_marks_delivery(
    mock_load,
    mock_watchlist,
    mock_users,
    _mock_existing,
    mock_claim,
    mock_send,
    mock_update,
):
    mock_users.return_value = [
        {"id": "u1", "email": "user@example.com", "full_name": "Test User"}
    ]
    mock_watchlist.return_value = [
        {"stock_id": 1, "symbol": "AAPL", "company_name": "Apple Inc."}
    ]
    mock_load.return_value = _ready_analysis()
    mock_claim.return_value = {
        "id": 10,
        "user_id": "u1",
        "notification_date": DAY.isoformat(),
    }

    result = dispatch_analysis_ready_emails(DAY)

    assert result["emails_sent"] == 1
    assert result["results"][0]["status"] == "sent"
    mock_send.assert_called_once()
    assert mock_update.call_args.kwargs["status"] == "sent"
    assert mock_update.call_args.kwargs["provider_message_id"] == "sendgrid-message-123"


@patch(f"{MODULE}._enabled_users")
@patch(f"{MODULE}._active_watchlist")
@patch(f"{MODULE}.load_symbol_analysis")
@patch(f"{MODULE}._existing_delivery", return_value={"status": "sent"})
def test_dispatch_is_idempotent_for_a_user_and_date(
    _mock_existing, mock_load, mock_watchlist, mock_users
):
    mock_users.return_value = [{"id": "u1", "email": "user@example.com"}]
    mock_watchlist.return_value = [
        {"stock_id": 1, "symbol": "AAPL", "company_name": "Apple Inc."}
    ]
    mock_load.return_value = _ready_analysis()

    result = dispatch_analysis_ready_emails(DAY)

    assert result["emails_sent"] == 0
    assert result["results"][0]["status"] == "already_sent"


def test_email_escapes_database_and_profile_text():
    user = {"email": "user@example.com", "full_name": "<script>alert(1)</script>"}
    stock = {
        "company_name": "A&B <Holdings>",
        **_ready_analysis(),
    }

    subject, body = build_analysis_ready_email(user, [stock], DAY)

    assert DAY.isoformat() in subject
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert "A&amp;B &lt;Holdings&gt;" in body


def test_stale_processing_delivery_can_be_retried():
    stale = {
        "status": "processing",
        "updated_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }
    current = {
        "status": "processing",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    assert _processing_delivery_is_active(stale) is False
    assert _processing_delivery_is_active(current) is True


@patch(f"{MODULE}.requests.post")
def test_sendgrid_sender_uses_mail_send_api_and_verified_sender(mock_post):
    mock_post.return_value.status_code = 202
    mock_post.return_value.headers = {"X-Message-Id": "sendgrid-message-123"}
    settings = {
        "SENDGRID_API_KEY": "SG.test-key",
        "SENDGRID_FROM_EMAIL": "verified@example.com",
        "SENDGRID_FROM_NAME": "StockLens Alerts",
        "APP_PUBLIC_URL": "https://stocklens.example",
    }

    with patch.dict(os.environ, settings, clear=False):
        message_id = _send_sendgrid_email(
            "user@example.com", "Analysis ready", "<p>Ready</p>"
        )

    request = mock_post.call_args
    assert request.args[0] == "https://api.sendgrid.com/v3/mail/send"
    assert request.kwargs["headers"]["Authorization"] == "Bearer SG.test-key"
    assert request.kwargs["timeout"] == 30.0
    assert request.kwargs["json"]["from"] == {
        "email": "verified@example.com",
        "name": "StockLens Alerts",
    }
    assert request.kwargs["json"]["personalizations"][0]["to"] == [
        {"email": "user@example.com"}
    ]
    assert request.kwargs["json"]["content"][1] == {
        "type": "text/html",
        "value": "<p>Ready</p>",
    }
    assert message_id == "sendgrid-message-123"


@patch(f"{MODULE}.requests.post")
def test_sendgrid_sender_fails_closed_when_credentials_are_missing(mock_post):
    with patch.dict(
        os.environ,
        {"SENDGRID_API_KEY": "", "SENDGRID_FROM_EMAIL": ""},
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="SENDGRID_API_KEY is not configured"):
            _send_sendgrid_email("user@example.com", "Analysis ready", "<p>Ready</p>")

    mock_post.assert_not_called()


@patch(f"{MODULE}.requests.post")
def test_sendgrid_sender_reports_api_rejection(mock_post):
    mock_post.return_value.status_code = 401
    mock_post.return_value.text = '{"errors":[{"message":"authorization required"}]}'
    mock_post.return_value.headers = {}
    settings = {
        "SENDGRID_API_KEY": "SG.invalid-key",
        "SENDGRID_FROM_EMAIL": "verified@example.com",
    }

    with patch.dict(os.environ, settings, clear=False):
        with pytest.raises(RuntimeError, match="SendGrid rejected.*status 401"):
            _send_sendgrid_email("user@example.com", "Analysis ready", "<p>Ready</p>")


@patch(f"{MODULE}._load_in_app_watchlist_data")
@patch(f"{MODULE}._active_watchlist")
def test_premium_in_app_notifications_include_ready_analysis_and_large_move(
    mock_watchlist, mock_data
):
    mock_watchlist.return_value = [
        {"stock_id": 1, "symbol": "AAPL", "company_name": "Apple Inc."}
    ]
    ready = _ready_analysis()
    mock_data.return_value = {
        "AAPL": {
            "prices": [
                {"trade_date": "2026-08-05", "close": 210.5},
                {"trade_date": "2026-08-04", "close": 203.58},
            ],
            "technical": ready["technical"],
            "sentiment": ready["sentiment"],
        }
    }

    with patch(f"{MODULE}.notification_today", return_value=DAY):
        result = get_in_app_notifications("user-1", "premium_user")

    assert [item["kind"] for item in result["notifications"]] == [
        "analysis",
        "price_up",
    ]
    assert result["notifications"][0]["href"] == "/stocks/AAPL/view"
    assert result["notifications"][0]["title"] == "AAPL analysis is ready"


@patch(f"{MODULE}._load_in_app_watchlist_data")
@patch(f"{MODULE}._active_watchlist")
def test_premium_in_app_notifications_report_pending_analysis(
    mock_watchlist, mock_data
):
    mock_watchlist.return_value = [
        {"stock_id": 1, "symbol": "MSFT", "company_name": "Microsoft"}
    ]
    ready = _ready_analysis("MSFT")
    mock_data.return_value = {
        "MSFT": {
            "prices": [{"trade_date": "2026-08-05", "close": 510.0}],
            "technical": ready["technical"],
            "sentiment": None,
        }
    }

    with patch(f"{MODULE}.notification_today", return_value=DAY):
        result = get_in_app_notifications("user-1", "premium_user")

    assert len(result["notifications"]) == 1
    assert result["notifications"][0]["kind"] == "status"
    assert "1 stock(s)" in result["notifications"][0]["message"]


def test_free_in_app_notifications_do_not_query_watchlist():
    with patch(f"{MODULE}._active_watchlist") as mock_watchlist:
        result = get_in_app_notifications("user-1", "basic_user")

    assert result["notifications"][0]["id"] == "free:premium-watchlist-alerts:v1"
    mock_watchlist.assert_not_called()
