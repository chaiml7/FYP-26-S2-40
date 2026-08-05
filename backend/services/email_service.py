"""Daily watchlist email notification service using SendGrid."""
import os
from datetime import date
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from backend.database.supabase_client import supabase
from backend.services.user_watchlist_service import get_user_watchlist_summary


def get_all_users_with_watchlists():
    """Get all users who have at least one stock in their watchlist."""
    try:
        response = (
            supabase
            .table("user_watchlists")
            .select("user_id, user_profiles(email)")
            .execute()
        )
        seen = set()
        users = []
        for item in response.data:
            uid = item["user_id"]
            if uid not in seen:
                seen.add(uid)
                profile = item.get("user_profiles") or {}
                email = profile.get("email")
                if email:
                    users.append({"user_id": uid, "email": email})
        return users
    except Exception as e:
        print(f"Error fetching users with watchlists: {e}")
        return []


def build_email_html(email: str, stocks: list) -> str:
    """Build the HTML email body."""
    today = date.today().strftime("%A, %d %B %Y")
    username = email.split("@")[0]

    signal_colors = {
        "BUY": ("#e8f5e9", "#2e7d32"),
        "STRONG BUY": ("#e8f5e9", "#2e7d32"),
        "SELL": ("#fce4ec", "#c62828"),
        "STRONG SELL": ("#fce4ec", "#c62828"),
        "NEUTRAL": ("#f5f5f5", "#616161"),
    }

    rows = ""
    for stock in stocks:
        symbol = stock.get("symbol", "")
        signal = (stock.get("prediction_signal") or "NEUTRAL").upper()
        sentiment = (stock.get("sentiment_label") or "Neutral").capitalize()
        company = stock.get("company_name", "")
        price = stock.get("price")
        price_str = f"${price:.2f}" if price else "N/A"

        bg, color = signal_colors.get(signal, ("#f5f5f5", "#616161"))

        if sentiment.lower() in ["positive", "bullish"]:
            sentiment_color = "#2e7d32"
        elif sentiment.lower() in ["negative", "bearish"]:
            sentiment_color = "#c62828"
        else:
            sentiment_color = "#616161"

        rows += f"""
        <tr style="border-bottom: 1px solid #f9f9f9;">
            <td style="padding: 10px 0;">
                <strong style="color: #0f1117;">{symbol}</strong><br>
                <span style="font-size: 11px; color: #aaa;">{company}</span>
            </td>
            <td style="padding: 10px 0; text-align: center;">
                <span style="background: {bg}; color: {color}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600;">{signal}</span>
            </td>
            <td style="padding: 10px 0; text-align: center; color: #0f1117;">{price_str}</td>
            <td style="padding: 10px 0; text-align: right; color: {sentiment_color}; font-weight: 500;">{sentiment}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="margin: 0; padding: 0; background: #f4f4f4; font-family: Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background: #f4f4f4; padding: 30px 0;">
            <tr>
                <td align="center">
                    <table width="560" cellpadding="0" cellspacing="0" style="background: #ffffff; border-radius: 12px; overflow: hidden;">

                        <!-- Header -->
                        <tr>
                            <td style="background: #0f1117; padding: 28px; text-align: center;">
                                <p style="color: #ffffff; font-size: 20px; font-weight: 500; margin: 0 0 4px;">📈 StockLens</p>
                                <p style="color: #aaaaaa; font-size: 13px; margin: 0;">Daily stock predictions</p>
                            </td>
                        </tr>

                        <!-- Body -->
                        <tr>
                            <td style="padding: 24px 28px;">
                                <p style="color: #888; font-size: 12px; margin: 0 0 12px;">{today}</p>
                                <p style="color: #0f1117; font-size: 16px; font-weight: 500; margin: 0 0 8px;">Hi {username},</p>
                                <p style="color: #444; font-size: 14px; margin: 0 0 20px;">Your stock predictions for today are ready! Here's a summary of your watchlist:</p>

                                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; font-size: 13px;">
                                    <thead>
                                        <tr style="border-bottom: 1px solid #f0f0f0;">
                                            <th style="padding: 8px 0; text-align: left; color: #888; font-weight: 500;">Stock</th>
                                            <th style="padding: 8px 0; text-align: center; color: #888; font-weight: 500;">Signal</th>
                                            <th style="padding: 8px 0; text-align: center; color: #888; font-weight: 500;">Price</th>
                                            <th style="padding: 8px 0; text-align: right; color: #888; font-weight: 500;">Sentiment</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {rows}
                                    </tbody>
                                </table>

                                <div style="text-align: center; margin: 24px 0 16px;">
                                    <a href="http://localhost:8001/dashboard" style="background: #0f1117; color: #ffffff; text-decoration: none; padding: 10px 28px; border-radius: 6px; font-size: 13px; font-weight: 500;">View full breakdown</a>
                                </div>
                            </td>
                        </tr>

                        <!-- Footer -->
                        <tr>
                            <td style="border-top: 1px solid #f0f0f0; padding: 16px 28px; text-align: center;">
                                <p style="color: #bbb; font-size: 11px; margin: 0;">
                                    You're receiving this because you have stocks in your StockLens watchlist.<br>
                                    StockLens &middot; Singapore
                                </p>
                            </td>
                        </tr>

                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    return html


def send_watchlist_email(to_email: str, html_content: str):
    """Send email via SendGrid."""
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL", "stocklensdaily@outlook.com")

    if not api_key:
        print("SENDGRID_API_KEY not set in .env")
        return False

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject="📈 Your StockLens stock predictions are ready!",
        html_content=html_content,
    )

    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(f"Email sent to {to_email} — Status: {response.status_code}")
        return True
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")
        return False


def send_daily_watchlist_emails():
    """Main function — gets all users and sends them their daily summary."""
    print("Starting daily watchlist email job...")
    users = get_all_users_with_watchlists()

    if not users:
        print("No users with watchlists found.")
        return

    for user in users:
        user_id = user["user_id"]
        email = user["email"]
        try:
            stocks = get_user_watchlist_summary(user_id)
            if not stocks:
                print(f"No watchlist stocks for {email}, skipping.")
                continue
            html = build_email_html(email, stocks)
            send_watchlist_email(email, html)
        except Exception as e:
            print(f"Error processing {email}: {e}")

    print("Daily watchlist email job complete.")
