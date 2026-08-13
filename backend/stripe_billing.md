# Stripe Premium billing setup

StockLens uses hosted Stripe Checkout for a recurring Premium subscription,
Stripe webhooks for entitlement changes, and the Stripe Customer Portal for
billing management.

## Stripe Dashboard

1. Switch Stripe to a sandbox/test environment.
2. Create a `StockLens Premium` product and a recurring monthly Price.
3. Put the test secret key and recurring `price_...` ID in `backend/.env`.
4. Enable the Customer Portal in Stripe and allow payment method updates,
   invoice history, and cancellation at the end of the billing period.
5. In production, register this endpoint in Stripe Workbench:
   `https://your-api-host/api/billing/stripe/webhook`.

Subscribe the endpoint to:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `customer.subscription.paused`
- `customer.subscription.resumed`
- `invoice.paid`
- `invoice.payment_failed`

## Local webhook forwarding

Start the backend API on port 8000 and frontend on port 8001. Then run:

```powershell
stripe login
stripe listen --forward-to http://localhost:8000/api/billing/stripe/webhook
```

Copy the `whsec_...` secret printed by `stripe listen` into
`STRIPE_WEBHOOK_SECRET`. This local secret is different from a deployed
webhook endpoint secret.

## Required environment variables

```env
APP_PUBLIC_URL=http://localhost:8001
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PREMIUM_PRICE_ID=price_...
SESSION_SECRET_KEY=replace_with_a_long_random_value
SESSION_HTTPS_ONLY=false
```

Set `SESSION_HTTPS_ONLY=true` when the deployed frontend is served over HTTPS.

Run the Supabase migration before testing Checkout:

`supabase/migrations/20260812022726_add_stripe_subscriptions.sql`

Use Stripe test card `4242 4242 4242 4242`, any future expiry date, and any
three-digit CVC. Do not enter real card information in a test environment.
