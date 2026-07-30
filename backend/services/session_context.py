"""Single source of truth for the template context every authenticated page needs.

top_navbar.html (included by every base layout) reads user_role/user_email/
user_initial to render the role badge and user chip. Every route that renders
a page must pass these — previously each route file duplicated this logic ad
hoc and several copies were incomplete, causing the badge/user chip to fall
back to FREE / "StockLens user" on pages that forgot a field.
"""

from fastapi import Request

BASE_LAYOUT_BY_ROLE = {
    "premium_user": "premium_users/base.html",
    "basic_user": "free_users/base.html",
    "frontend_admin": "user_admin/base.html",
}


def get_session_context(request: Request) -> dict | None:
    """Return the shared template context for the logged-in user, or None if
    there is no recognized session (caller should redirect to /login)."""
    role = request.session.get("user_role")
    base_layout = BASE_LAYOUT_BY_ROLE.get(role)
    if not role or not base_layout:
        return None

    email = request.session.get("user_email", "")

    return {
        "user_role": role,
        "user_id": request.session.get("user_id"),
        "user_email": email,
        "user_initial": email[:1].upper() if email else "U",
        "base_layout": base_layout,
    }
