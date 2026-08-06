# Gmail analysis-ready email setup

StockLens sends one daily digest only to users who opt in on their watchlist page. A digest is ready when every active stock in that user's watchlist has a technical prediction for its newest imported market date and a sentiment score for the notification date. The latest financial analysis is included when available, but quarterly financial data does not block a daily notification.

## Gmail configuration

Use a dedicated Gmail account for StockLens notifications, enable Google 2-Step Verification, then create an App Password for the account. Put the following values in `backend/.env`:

```dotenv
GMAIL_SMTP_USER=stocklens.notifications@gmail.com
GMAIL_SMTP_APP_PASSWORD=your_16_character_google_app_password
GMAIL_FROM_NAME=StockLens
GMAIL_SMTP_HOST=smtp.gmail.com
GMAIL_SMTP_PORT=587
GMAIL_SMTP_TIMEOUT_SECONDS=30
```

Do not use the Gmail account's normal password and do not commit `backend/.env`. Spaces in an App Password are accepted and removed before authentication.

Keep these application settings alongside the Gmail settings:

```dotenv
APP_PUBLIC_URL=http://localhost:8001
NOTIFICATION_TIMEZONE=Asia/Singapore
NOTIFICATION_ADMIN_KEY=replace_with_a_long_random_value
AUTO_SEND_ANALYSIS_READY_EMAILS=false
ENABLE_EMAIL_NOTIFICATION_SCHEDULER=false
```

## Verify before sending

Restart the backend after changing `.env`. Preview readiness without sending an email:

```text
POST /api/notifications/analysis-ready/dispatch?dry_run=true
X-Notification-Admin-Key: <NOTIFICATION_ADMIN_KEY>
```

When the result shows `ready`, send by changing `dry_run=false`. Set `ENABLE_EMAIL_NOTIFICATION_SCHEDULER=true` to check every 15 minutes, or set `AUTO_SEND_ANALYSIS_READY_EMAILS=true` to queue a readiness check after technical or sentiment prediction endpoints finish.

The database delivery record prevents a successful notification from being sent twice to the same user on the same notification date. Failed deliveries remain eligible for a later retry.
