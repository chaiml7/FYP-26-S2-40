# SendGrid analysis-ready email setup

StockLens sends one daily digest only to users who opt in on their watchlist page. A digest is ready when every active stock in that user's watchlist has a technical prediction for its newest imported market date and a sentiment score for the notification date. The latest financial analysis is included when available, but quarterly financial data does not block a daily notification.

## SendGrid configuration

Create a restricted SendGrid API key with only Mail Send access, then verify the sender address in SendGrid. Put the following values in `backend/.env`:

```dotenv
SENDGRID_API_KEY=SG.your_sendgrid_api_key
SENDGRID_FROM_EMAIL=your_verified_sender@example.com
SENDGRID_FROM_NAME=StockLens
SENDGRID_TIMEOUT_SECONDS=30
```

`SENDGRID_FROM_EMAIL` must match a verified SendGrid Sender Identity. Do not commit `backend/.env` or expose the API key in frontend code.

Keep these application settings alongside the SendGrid settings:

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
