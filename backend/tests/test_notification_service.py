import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.services.notification_service import (
    _processing_delivery_is_active,
    _send_gmail_email,
    build_analysis_ready_email,
    dispatch_analysis_ready_emails,
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


@patch(f"{MODULE}._send_gmail_email")
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
@patch(f"{MODULE}._send_gmail_email", return_value="gmail-message-123")
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
    assert mock_update.call_args.kwargs["provider_message_id"] == "gmail-message-123"


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


@patch(f"{MODULE}.smtplib.SMTP")
def test_gmail_sender_uses_tls_app_password_and_multipart_email(mock_smtp):
    smtp = mock_smtp.return_value.__enter__.return_value
    smtp.send_message.return_value = {}
    settings = {
        "GMAIL_SMTP_USER": "stocklens.notifications@gmail.com",
        "GMAIL_SMTP_APP_PASSWORD": "abcd efgh ijkl mnop",
        "GMAIL_FROM_NAME": "StockLens Alerts",
        "APP_PUBLIC_URL": "https://stocklens.example",
    }

    with patch.dict(os.environ, settings, clear=False):
        message_id = _send_gmail_email(
            "user@example.com", "Analysis ready", "<p>Ready</p>"
        )

    mock_smtp.assert_called_once_with("smtp.gmail.com", 587, timeout=30.0)
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with(
        "stocklens.notifications@gmail.com", "abcdefghijklmnop"
    )
    sent_message = smtp.send_message.call_args.args[0]
    assert sent_message["To"] == "user@example.com"
    assert sent_message["From"] == "StockLens Alerts <stocklens.notifications@gmail.com>"
    assert sent_message.is_multipart()
    assert message_id


@patch(f"{MODULE}.smtplib.SMTP")
def test_gmail_sender_fails_closed_when_credentials_are_missing(mock_smtp):
    with patch.dict(
        os.environ,
        {"GMAIL_SMTP_USER": "", "GMAIL_SMTP_APP_PASSWORD": ""},
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="GMAIL_SMTP_USER is not configured"):
            _send_gmail_email("user@example.com", "Analysis ready", "<p>Ready</p>")

    mock_smtp.assert_not_called()


@patch(f"{MODULE}.smtplib.SMTP")
def test_gmail_sender_reports_refused_recipient(mock_smtp):
    smtp = mock_smtp.return_value.__enter__.return_value
    smtp.send_message.return_value = {"user@example.com": (550, b"Rejected")}
    settings = {
        "GMAIL_SMTP_USER": "stocklens.notifications@gmail.com",
        "GMAIL_SMTP_APP_PASSWORD": "abcdefghijklmnop",
    }

    with patch.dict(os.environ, settings, clear=False):
        with pytest.raises(RuntimeError, match="Gmail refused recipient"):
            _send_gmail_email("user@example.com", "Analysis ready", "<p>Ready</p>")
