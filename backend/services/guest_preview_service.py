"""Daily preview allowance for unregistered (guest) visitors.

Tracked in the signed Starlette session cookie StockLens already sets for
every visitor, logged in or not — no separate table or cookie needed. The
counter resets automatically whenever the stored date no longer matches
today, so there is no cron job or cleanup involved.
"""

from datetime import date

from fastapi import Request

GUEST_PREVIEW_LIMIT = 3
SESSION_KEY = "guest_preview"


def _today() -> str:
    return date.today().isoformat()


def _current_state(request: Request) -> dict:
    state = request.session.get(SESSION_KEY)
    if not isinstance(state, dict) or state.get("date") != _today():
        return {"date": _today(), "count": 0}
    return state


def remaining_guest_previews(request: Request) -> int:
    state = _current_state(request)
    return max(GUEST_PREVIEW_LIMIT - state["count"], 0)


def guest_previews_exhausted(request: Request) -> bool:
    return remaining_guest_previews(request) <= 0


def record_guest_preview(request: Request) -> int:
    """Consume one preview and return the remaining count."""
    state = _current_state(request)
    state["count"] += 1
    request.session[SESSION_KEY] = state
    return max(GUEST_PREVIEW_LIMIT - state["count"], 0)
