"""Opt-in, idempotent watchlist analysis-ready email notifications."""

from __future__ import annotations

import html
import logging
import os
import smtplib
import ssl
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from hmac import compare_digest
from typing import Any
from zoneinfo import ZoneInfo

from backend.database.supabase_client import supabase


logger = logging.getLogger(__name__)

NOTIFICATION_TYPE = "analysis_ready"
DEFAULT_TIMEZONE = "Asia/Singapore"
PROCESSING_TIMEOUT = timedelta(minutes=30)


def notification_today() -> date:
    timezone_name = os.getenv("NOTIFICATION_TIMEZONE", DEFAULT_TIMEZONE)
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception:
        logger.warning("Unknown NOTIFICATION_TIMEZONE=%s; using %s", timezone_name, DEFAULT_TIMEZONE)
        timezone = ZoneInfo(DEFAULT_TIMEZONE)
    return datetime.now(timezone).date()


def auto_dispatch_enabled() -> bool:
    return os.getenv("AUTO_SEND_ANALYSIS_READY_EMAILS", "false").lower() == "true"


def admin_key_is_valid(value: str | None) -> bool:
    expected = os.getenv("NOTIFICATION_ADMIN_KEY", "")
    return bool(expected and value and compare_digest(value, expected))


def get_notification_preference(user_id: str) -> dict[str, Any]:
    response = (
        supabase.table("user_notification_preferences")
        .select("user_id,analysis_ready_email,created_at,updated_at")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if rows:
        return rows[0]
    return {"user_id": user_id, "analysis_ready_email": False}


def set_notification_preference(user_id: str, enabled: bool) -> dict[str, Any]:
    response = (
        supabase.table("user_notification_preferences")
        .upsert(
            {
                "user_id": user_id,
                "analysis_ready_email": bool(enabled),
                "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            },
            on_conflict="user_id",
        )
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else {"user_id": user_id, "analysis_ready_email": bool(enabled)}


def _enabled_users() -> list[dict[str, Any]]:
    preference_response = (
        supabase.table("user_notification_preferences")
        .select("user_id")
        .eq("analysis_ready_email", True)
        .execute()
    )
    user_ids = [row.get("user_id") for row in (preference_response.data or []) if row.get("user_id")]
    if not user_ids:
        return []

    profile_response = (
        supabase.table("user_profiles")
        .select("id,email,full_name,is_active")
        .in_("id", user_ids)
        .execute()
    )
    return [
        row
        for row in (profile_response.data or [])
        if row.get("email") and row.get("is_active") is not False
    ]


def _active_watchlist(user_id: str) -> list[dict[str, Any]]:
    response = (
        supabase.table("user_watchlists")
        .select("stock_id,stocks(id,symbol,company_name,is_active)")
        .eq("user_id", user_id)
        .execute()
    )
    stocks = []
    for row in response.data or []:
        stock = row.get("stocks") or {}
        symbol = str(stock.get("symbol") or "").strip().upper()
        if symbol and stock.get("is_active") is True:
            stocks.append(
                {
                    "stock_id": stock.get("id") or row.get("stock_id"),
                    "symbol": symbol,
                    "company_name": stock.get("company_name") or symbol,
                }
            )
    return stocks


def _latest_row(table: str, symbol_column: str, symbol: str, order_columns: list[str], select: str) -> dict | None:
    query = supabase.table(table).select(select).eq(symbol_column, symbol)
    for column in order_columns:
        query = query.order(column, desc=True)
    response = query.limit(1).execute()
    rows = response.data or []
    return rows[0] if rows else None


def load_symbol_analysis(symbol: str, notification_date: date) -> dict[str, Any]:
    symbol = symbol.upper()
    price = _latest_row(
        "daily_ohlcv",
        "symbol",
        symbol,
        ["trade_date"],
        "trade_date,close",
    )
    technical = _latest_row(
        "direction_predictions",
        "symbol",
        symbol,
        ["latest_date", "created_at"],
        "latest_date,latest_close,prediction,technical_score,model_version,created_at",
    )
    sentiment_response = (
        supabase.table("sentiment_daily_scores")
        .select("score_date,sentiment_label,bullish_score,article_count,model_version")
        .eq("symbol", symbol)
        .eq("score_date", notification_date.isoformat())
        .limit(1)
        .execute()
    )
    sentiment_rows = sentiment_response.data or []
    sentiment = sentiment_rows[0] if sentiment_rows else None
    financial = _latest_row(
        "financial_predictions",
        "ticker",
        symbol,
        ["created_at", "period"],
        "period,prediction,fundamental_score,confidence,model_version,created_at",
    )

    latest_market_date = price.get("trade_date") if price else None
    technical_date = technical.get("latest_date") if technical else None
    missing = []
    if not latest_market_date:
        missing.append("latest_price")
    if not technical or not latest_market_date or technical_date != latest_market_date:
        missing.append("technical")
    if not sentiment:
        missing.append("sentiment")

    return {
        "symbol": symbol,
        "latest_market_date": latest_market_date,
        "price": (technical or {}).get("latest_close") or (price or {}).get("close"),
        "technical": technical,
        "sentiment": sentiment,
        "financial": financial,
        "ready": not missing,
        "missing": missing,
    }


def _existing_delivery(user_id: str, notification_date: date) -> dict | None:
    response = (
        supabase.table("notification_deliveries")
        .select("*")
        .eq("user_id", user_id)
        .eq("notification_type", NOTIFICATION_TYPE)
        .eq("notification_date", notification_date.isoformat())
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def _processing_delivery_is_active(delivery: dict[str, Any]) -> bool:
    if delivery.get("status") != "processing":
        return False
    updated_at = delivery.get("updated_at") or delivery.get("created_at")
    if not updated_at:
        return True
    try:
        updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=ZoneInfo("UTC"))
    except (TypeError, ValueError):
        return True
    return updated > datetime.now(ZoneInfo("UTC")) - PROCESSING_TIMEOUT


def _claim_delivery(user_id: str, notification_date: date, symbols: list[str]) -> dict | None:
    existing = _existing_delivery(user_id, notification_date)
    if existing and (
        existing.get("status") == "sent" or _processing_delivery_is_active(existing)
    ):
        return None
    payload = {
        "user_id": user_id,
        "notification_type": NOTIFICATION_TYPE,
        "notification_date": notification_date.isoformat(),
        "watchlist_symbols": symbols,
        "status": "processing",
        "error_message": None,
        "provider_message_id": None,
        "sent_at": None,
        "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
    }
    if existing:
        previous_status = existing.get("status")
        response = (
            supabase.table("notification_deliveries")
            .update(payload)
            .eq("id", existing["id"])
            .eq("status", previous_status)
            .execute()
        )
    else:
        try:
            response = supabase.table("notification_deliveries").insert(payload).execute()
        except Exception:
            concurrent = _existing_delivery(user_id, notification_date)
            if concurrent and concurrent.get("status") in {"processing", "sent"}:
                return None
            raise
    rows = response.data or []
    return rows[0] if rows else None


def _update_delivery(delivery: dict[str, Any], **changes: Any) -> None:
    payload = {**changes, "updated_at": datetime.now(ZoneInfo("UTC")).isoformat()}
    query = supabase.table("notification_deliveries").update(payload)
    if delivery.get("id") is not None:
        query = query.eq("id", delivery["id"])
    else:
        query = (
            query.eq("user_id", delivery["user_id"])
            .eq("notification_type", NOTIFICATION_TYPE)
            .eq("notification_date", delivery["notification_date"])
        )
    query.execute()


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.1f}/10"
    except (TypeError, ValueError):
        return "N/A"


def _format_price(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def build_analysis_ready_email(
    user: dict[str, Any],
    stocks: list[dict[str, Any]],
    notification_date: date,
) -> tuple[str, str]:
    display_name = str(user.get("full_name") or user.get("email", "StockLens user").split("@")[0])
    app_url = os.getenv("APP_PUBLIC_URL", "http://localhost:8001").rstrip("/")
    rows = []
    for stock in stocks:
        technical = stock.get("technical") or {}
        sentiment = stock.get("sentiment") or {}
        financial = stock.get("financial") or {}
        technical_label = str(technical.get("prediction") or "N/A").replace("_", " ").title()
        sentiment_label = str(sentiment.get("sentiment_label") or "Neutral").title()
        if financial:
            financial_label = str(financial.get("prediction") or "Available").replace("_", " ").title()
            financial_period = str(financial.get("period") or "latest report")
            financial_text = f"{financial_label} ({financial_period})"
        else:
            financial_text = "Not available"
        rows.append(
            "<tr>"
            f"<td>{html.escape(stock['symbol'])}<br><small>{html.escape(stock.get('company_name') or '')}</small></td>"
            f"<td>{html.escape(technical_label)}<br><small>{html.escape(_format_score(technical.get('technical_score')))}</small></td>"
            f"<td>{html.escape(sentiment_label)}<br><small>{html.escape(_format_score(sentiment.get('bullish_score')))}</small></td>"
            f"<td>{html.escape(financial_text)}</td>"
            f"<td>{html.escape(_format_price(stock.get('price')))}</td>"
            "</tr>"
        )

    subject = f"StockLens watchlist analysis is ready - {notification_date.isoformat()}"
    body = f"""<!doctype html>
<html><body style="margin:0;background:#f3f6fb;font-family:Arial,sans-serif;color:#172033">
<div style="max-width:720px;margin:24px auto;background:#fff;border:1px solid #dce5f2;border-radius:12px;overflow:hidden">
<div style="background:#2563eb;color:#fff;padding:22px 28px"><h1 style="margin:0;font-size:22px">StockLens</h1><p style="margin:6px 0 0">Your watchlist analysis is ready</p></div>
<div style="padding:24px 28px"><p>Hi {html.escape(display_name)},</p>
<p>Technical and sentiment analysis for your active watchlist stocks is ready for {notification_date.isoformat()}.</p>
<table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr style="background:#eaf2ff;text-align:left">
<th style="padding:10px">Stock</th><th style="padding:10px">Technical</th><th style="padding:10px">Sentiment</th><th style="padding:10px">Latest financial</th><th style="padding:10px">Price</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p style="font-size:12px;color:#68758a">Financial analysis uses the latest available company report and may not change daily.</p>
<p style="text-align:center;margin:24px 0"><a href="{html.escape(app_url)}/dashboard" style="background:#2563eb;color:#fff;text-decoration:none;padding:11px 22px;border-radius:7px">View dashboard</a></p>
<p style="font-size:11px;color:#7a8799">StockLens analysis is informational and is not financial advice. You can turn off these emails from <a href="{html.escape(app_url)}/user/watchlist?notifications=manage">your watchlist settings</a>.</p>
</div></div></body></html>"""
    return subject, body


def _gmail_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def _send_gmail_email(to_email: str, subject: str, html_content: str) -> str:
    """Send one notification through Gmail SMTP using an App Password."""
    username = _gmail_setting("GMAIL_SMTP_USER")
    app_password = _gmail_setting("GMAIL_SMTP_APP_PASSWORD").replace(" ", "")
    from_name = os.getenv("GMAIL_FROM_NAME", "StockLens").strip() or "StockLens"
    host = os.getenv("GMAIL_SMTP_HOST", "smtp.gmail.com").strip() or "smtp.gmail.com"
    try:
        port = int(os.getenv("GMAIL_SMTP_PORT", "587"))
        timeout = float(os.getenv("GMAIL_SMTP_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        raise RuntimeError("Gmail SMTP port and timeout must be numeric") from exc

    app_url = os.getenv("APP_PUBLIC_URL", "http://localhost:8001").rstrip("/")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((from_name, username))
    message["To"] = to_email
    message["Message-ID"] = make_msgid(
        domain=username.split("@", 1)[1] if "@" in username else None
    )
    message.set_content(
        "Your StockLens watchlist analysis is ready. "
        f"Sign in to view it: {app_url}/dashboard"
    )
    message.add_alternative(html_content, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=timeout) as smtp:
        smtp.ehlo()
        smtp.starttls(context=context)
        smtp.ehlo()
        smtp.login(username, app_password)
        refused = smtp.send_message(message)
    if refused:
        refused_addresses = ", ".join(sorted(str(address) for address in refused))
        raise RuntimeError(f"Gmail refused recipient(s): {refused_addresses}")
    return str(message["Message-ID"]).strip("<>")


def dispatch_analysis_ready_emails(
    notification_date: date | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_date = notification_date or notification_today()
    users = _enabled_users()
    results = []
    sent = 0

    for user in users:
        user_id = str(user["id"])
        watchlist = _active_watchlist(user_id)
        if not watchlist:
            results.append({"user_id": user_id, "status": "skipped", "reason": "no_active_watchlist"})
            continue

        analyses = []
        for stock in watchlist:
            analysis = load_symbol_analysis(stock["symbol"], selected_date)
            analyses.append({**stock, **analysis})
        missing = [
            {"symbol": row["symbol"], "missing": row["missing"]}
            for row in analyses
            if not row["ready"]
        ]
        if missing:
            results.append(
                {"user_id": user_id, "status": "not_ready", "missing": missing}
            )
            continue

        existing = _existing_delivery(user_id, selected_date)
        if existing and existing.get("status") == "sent":
            results.append({"user_id": user_id, "status": "already_sent"})
            continue
        if existing and _processing_delivery_is_active(existing):
            results.append({"user_id": user_id, "status": "already_processing"})
            continue

        if dry_run:
            results.append(
                {
                    "user_id": user_id,
                    "status": "ready",
                    "symbols": [row["symbol"] for row in analyses],
                }
            )
            continue

        symbols = [row["symbol"] for row in analyses]
        delivery = _claim_delivery(user_id, selected_date, symbols)
        if delivery is None:
            results.append({"user_id": user_id, "status": "already_claimed"})
            continue
        try:
            subject, email_html = build_analysis_ready_email(user, analyses, selected_date)
            provider_id = _send_gmail_email(user["email"], subject, email_html)
            _update_delivery(
                delivery,
                status="sent",
                provider_message_id=provider_id,
                sent_at=datetime.now(ZoneInfo("UTC")).isoformat(),
            )
            sent += 1
            results.append({"user_id": user_id, "status": "sent", "symbols": symbols})
        except Exception as exc:
            logger.exception("Analysis-ready email failed for user_id=%s", user_id)
            _update_delivery(delivery, status="failed", error_message=str(exc)[:1000])
            results.append({"user_id": user_id, "status": "failed", "reason": str(exc)})

    return {
        "notification_type": NOTIFICATION_TYPE,
        "notification_date": selected_date.isoformat(),
        "dry_run": dry_run,
        "opted_in_users": len(users),
        "emails_sent": sent,
        "results": results,
    }
