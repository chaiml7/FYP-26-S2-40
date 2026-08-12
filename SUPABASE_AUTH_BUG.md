# Bug: Shared Supabase Client Auth Contamination

**Status:** Open  
**Severity:** High — blocks all server-side DB queries after any user login  
**Discovered:** 2026-06-28 during premium news feed verification  
**Affects:** All server-rendered routes that query Supabase after a login event

---

## Summary

`frontend/main.py` calls `supabase.auth.sign_in_with_password()` on the **shared service-role Supabase singleton**. This mutates the singleton's `Authorization` header from the service-role JWT to the logged-in user's anon JWT. From that point forward, every DB query in the same server process runs as the user — subject to RLS policies — instead of as the service role.

The result: any route that reads tables with restrictive RLS (e.g. `sentiment_scores`) returns 0 rows after a user logs in, even though the data exists in the DB.

---

## Root Cause

### The singleton

`frontend/database/supabase_client.py` creates one client at import time:

```python
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
```

### The contaminating call

`frontend/main.py` (login route) calls auth on that same singleton:

```python
response = supabase.auth.sign_in_with_password({"email": email, "password": password})
```

`sign_in_with_password` stores the returned session on the client and updates its internal `Authorization` header to the user's access token. All subsequent `.table(...).select(...)` calls on `supabase` now use that user token — not the service key.

### Why RLS blocks it

Supabase RLS policies on `sentiment_scores` (and likely other tables) are set up to allow reads only via the service role. When queries arrive with a user JWT, RLS denies them and Supabase returns an empty result set rather than an error.

---

## Evidence

- `sentiment_scores` table contains 343+ rows for GOOGL with `published_at` within the last 30 days.
- `GET /api/stocks/GOOGL/sentiment?days=30` returns `{"articles": [], "summary": {...}}` (0 articles) after logging in.
- Direct `curl` to the same endpoint (no session cookie, service role still intact) returns data before any login has occurred.
- Widening to `?days=30` made no difference — confirmed it's not a data freshness issue.

---

## Fix Options

### Option A — Separate anon client for auth (Recommended)

Create a second Supabase client using the **anon key** specifically for auth operations. The service-role client remains untouched.

```python
# frontend/database/supabase_client.py
from supabase import create_client, Client
import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SECRET_KEY"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)      # for DB queries
supabase_auth: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)    # for auth only
```

Then in `frontend/main.py` login route:

```python
from database.supabase_client import supabase_auth

response = supabase_auth.auth.sign_in_with_password({"email": email, "password": password})
```

**Why this is correct:** The service-role singleton is never touched by auth flows, so its JWT is never overwritten.

### Option B — Sign out after extracting the session token

After `sign_in_with_password`, immediately call `supabase.auth.sign_out()` to reset the client's session, then store the token manually in the session cookie without relying on the client's internal state.

```python
response = supabase.auth.sign_in_with_password({"email": email, "password": password})
token = response.session.access_token
supabase.auth.sign_out()   # reset client back to service role
# store token in session cookie as before
```

**Downside:** Fragile — a crash between sign_in and sign_out leaves the client contaminated. Option A is cleaner.

### Option C — Per-request client instantiation

Instantiate a fresh Supabase client for each request (or at least for auth routes). Expensive and changes the import pattern everywhere.

---

## Files to Change

| File | Change |
|---|---|
| `frontend/database/supabase_client.py` | Add `supabase_auth` client using anon key |
| `frontend/main.py` | Import `supabase_auth`, use it in login route |
| `frontend/.env` / `frontend/.env.example` | Ensure `SUPABASE_ANON_KEY` is present |

---

## Testing the Fix

1. Start the frontend server fresh.
2. Log in as a premium user.
3. Hit `GET /api/stocks/GOOGL/sentiment?days=30` — should return articles.
4. Visit `/premium/news/GOOGL` — news cards should populate.
5. Log out and log back in — repeat step 3, confirm still works (session not re-contaminated).

---

## Related

- Premium news feed feature: `frontend/routes/premium_routes.py`, `frontend/templates/premium/news_feed.html`
- Sentiment service: `backend/services/sentiment/sentiment_aggregator.py`
- The news feed page renders correctly (template, access control, routing all verified) — this bug is the only thing preventing live data from showing.
