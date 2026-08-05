from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from backend.schemas import (
    AccountCreate,
    EmailUpdate,
    LoginRequest,
    PasswordUpdate,
    ProfileUpdate,
    UserRoleUpdate,
    UserStatusUpdate,
    WatchlistAdd,
)
from backend.services.auth_service import (
    AuthServiceError,
    admin_get_user,
    admin_list_users,
    create_account,
    get_auth_user,
    login,
    logout,
    update_email,
    update_password,
)
from backend.services.sentiment.sentiment_aggregator import get_recent_news
from backend.services.session_context import get_session_context
from backend.services.stock_list_service import get_active_stocks, get_stock_by_symbol, search_active_stocks
from backend.services.user_profile_service import (
    get_profile,
    get_profiles,
    update_profile,
    update_user_role,
    update_user_status,
)
from backend.services.user_watchlist_service import (
    add_user_watchlist_stock,
    get_user_watchlist,
    get_user_watchlist_summary,
    get_user_watchlist_symbols,
    remove_user_watchlist_stock,
)

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


def _payload(model, exclude_none: bool = False):
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=exclude_none, mode="json")
    return model.dict(exclude_none=exclude_none)


def _auth_error(error: AuthServiceError):
    raise HTTPException(status_code=error.status_code, detail=error.detail)


def _access_token(authorization: str = None):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    return authorization.split(" ", 1)[1].strip()


def _current_user(authorization: str = None):
    token = _access_token(authorization)

    try:
        user = get_auth_user(token)
    except AuthServiceError as error:
        _auth_error(error)

    return user, token


def _require_backend_admin(authorization: str = None):
    user, token = _current_user(authorization)
    profile = get_profile(user["id"])

    if len(profile) == 0 or profile[0].get("role_id") != "backend_admin":
        raise HTTPException(status_code=403, detail="Backend admin access required")

    return user, token, profile[0]


def _combine_auth_user_with_profile(auth_user: dict, profile: dict = None):
    if auth_user and "user" in auth_user:
        auth_user = auth_user["user"]

    return {
        "id": auth_user.get("id"),
        "email": auth_user.get("email"),
        "created_at": auth_user.get("created_at"),
        "last_sign_in_at": auth_user.get("last_sign_in_at"),
        "profile": profile,
    }


@router.post("/auth/signup")
def signup(account_data: AccountCreate):
    try:
        return create_account(
            account_data.email,
            account_data.password,
            account_data.full_name,
        )
    except AuthServiceError as error:
        _auth_error(error)


@router.post("/auth/login")
def login_user(login_data: LoginRequest):
    try:
        return login(login_data.email, login_data.password)
    except AuthServiceError as error:
        _auth_error(error)


@router.post("/auth/logout")
def logout_user(authorization: str = Header(default=None)):
    _, token = _current_user(authorization)

    try:
        logout(token)
    except AuthServiceError as error:
        _auth_error(error)

    return {"message": "Logged out successfully"}


@router.get("/users/me")
def view_current_user(authorization: str = Header(default=None)):
    user, _ = _current_user(authorization)
    profile = get_profile(user["id"])

    return {
        "user": user,
        "profile": profile[0] if len(profile) > 0 else None,
    }


@router.get("/users/me/role")
def view_current_user_role(authorization: str = Header(default=None)):
    user, _ = _current_user(authorization)
    profile = get_profile(user["id"])

    if len(profile) == 0:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {
        "user_id": user["id"],
        "role_id": profile[0]["role_id"],
        "role": profile[0].get("roles"),
    }


@router.patch("/users/me/profile")
def edit_current_user_profile(
    profile_data: ProfileUpdate,
    authorization: str = Header(default=None),
):
    user, _ = _current_user(authorization)
    payload = _payload(profile_data, exclude_none=True)

    if len(payload) == 0:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    updated = update_profile(user["id"], payload)

    if len(updated) == 0:
        raise HTTPException(status_code=404, detail="Profile not found")

    return updated[0]


@router.patch("/users/me/password")
def edit_current_user_password(
    password_data: PasswordUpdate,
    authorization: str = Header(default=None),
):
    _, token = _current_user(authorization)

    try:
        update_password(token, password_data.new_password)
    except AuthServiceError as error:
        _auth_error(error)

    return {"message": "Password updated successfully"}


@router.patch("/users/me/email")
def edit_current_user_email(
    email_data: EmailUpdate,
    authorization: str = Header(default=None),
):
    _, token = _current_user(authorization)

    try:
        return update_email(token, email_data.email)
    except AuthServiceError as error:
        _auth_error(error)


@router.get("/users/me/watchlist")
def view_current_user_watchlist(authorization: str = Header(default=None)):
    user, _ = _current_user(authorization)
    return get_user_watchlist(user["id"])


@router.get("/users/me/watchlist/symbols")
def view_current_user_watchlist_symbols(authorization: str = Header(default=None)):
    user, _ = _current_user(authorization)
    return get_user_watchlist_symbols(user["id"])


@router.get("/users/me/watchlist/summary")
def view_current_user_watchlist_summary(authorization: str = Header(default=None)):
    user, _ = _current_user(authorization)
    return get_user_watchlist_summary(user["id"])


@router.post("/users/me/watchlist")
def add_current_user_watchlist_stock(
    watchlist_data: WatchlistAdd,
    authorization: str = Header(default=None),
):
    user, _ = _current_user(authorization)

    if watchlist_data.stock_id is None and not watchlist_data.symbol:
        raise HTTPException(
            status_code=400,
            detail="Provide either stock_id or symbol"
        )

    stock_id = watchlist_data.stock_id

    if stock_id is None:
        stock = get_stock_by_symbol(watchlist_data.symbol)

        if len(stock) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"{watchlist_data.symbol.upper()} is not in the stocks table"
            )

        stock_id = stock[0]["id"]

    result = add_user_watchlist_stock(user["id"], stock_id) or []
    return result[0] if len(result) > 0 else {"user_id": user["id"], "stock_id": stock_id}


@router.delete("/users/me/watchlist/{stock_id}")
def remove_current_user_watchlist_stock(
    stock_id: int,
    authorization: str = Header(default=None),
):
    user, _ = _current_user(authorization)
    removed = remove_user_watchlist_stock(user["id"], stock_id)

    return {
        "stock_id": stock_id,
        "rows_deleted": len(removed),
        "message": "Watchlist stock removed",
    }


@router.get("/admin/users")
def view_admin_users(authorization: str = Header(default=None)):
    _require_backend_admin(authorization)

    try:
        auth_result = admin_list_users()
    except AuthServiceError as error:
        _auth_error(error)

    profiles = {profile["id"]: profile for profile in get_profiles()}
    auth_users = auth_result.get("users", [])

    return [
        _combine_auth_user_with_profile(user, profiles.get(user.get("id")))
        for user in auth_users
    ]


@router.get("/admin/users/{user_id}")
def view_admin_user(user_id: str, authorization: str = Header(default=None)):
    _require_backend_admin(authorization)

    try:
        auth_user = admin_get_user(user_id)
    except AuthServiceError as error:
        _auth_error(error)

    profile = get_profile(user_id)
    return _combine_auth_user_with_profile(
        auth_user,
        profile[0] if len(profile) > 0 else None,
    )


@router.patch("/admin/users/{user_id}/role")
def edit_admin_user_role(
    user_id: str,
    role_data: UserRoleUpdate,
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    updated = update_user_role(user_id, role_data.role_id)

    if len(updated) == 0:
        raise HTTPException(status_code=404, detail="Profile not found")

    return updated[0]


@router.patch("/admin/users/{user_id}/status")
def edit_admin_user_status(
    user_id: str,
    status_data: UserStatusUpdate,
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    updated = update_user_status(user_id, status_data.is_active)

    if len(updated) == 0:
        raise HTTPException(status_code=404, detail="Profile not found")

    return updated[0]


# ==========================================
# Shared User Routes (Free & Premium)
# ==========================================

@router.get("/user/watchlist")
async def watchlist(request: Request):
    session = get_session_context(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    is_premium = session["user_role"] == "premium_user"

    watchlist_rows = []
    if is_premium:
        try:
            watchlist_rows = get_user_watchlist_summary(session["user_id"])
        except Exception:
            watchlist_rows = []

    return templates.TemplateResponse(
        request=request,
        name="free_users/watchlist.html",
        context={
            **session,
            "request": request,
            "is_premium": is_premium,
            "watchlist_rows": watchlist_rows,
        }
    )

@router.get("/user/market_overview")
async def market_overview(request: Request):
    session = get_session_context(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="free_users/user_market_overview.html",
        context={**session, "request": request}
    )

@router.get("/user/news_social")
async def news_social(
    request: Request,
    q: str = None,
    label: str = None,
    symbol: str = None,
):
    session = get_session_context(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    try:
        active_stocks = get_active_stocks() or []
    except Exception:
        active_stocks = []

    label_filter = label if label in ("positive", "neutral", "negative") else None
    symbol_filter = symbol.upper() if symbol else None
    search_query = (q or "").strip()

    try:
        result = get_recent_news(symbol=symbol_filter, label=label_filter, q=search_query, page=1)
    except Exception:
        result = {"articles": [], "page": 1, "total_pages": 0, "total_count": 0}

    return templates.TemplateResponse(
        request=request,
        name="free_users/news_social.html",
        context={
            **session,
            "request": request,
            "articles": result["articles"],
            "page": result["page"],
            "total_pages": result["total_pages"],
            "total_count": result["total_count"],
            "active_stocks": active_stocks,
            "q": search_query,
            "selected_label": label_filter or "all",
            "selected_symbol": symbol_filter or "",
        }
    )


@router.get("/api/news")
async def api_news(
    request: Request,
    q: str = None,
    label: str = None,
    symbol: str = None,
    page: int = 1,
):
    session = get_session_context(request)
    if not session:
        raise HTTPException(status_code=401, detail="Login required")

    label_filter = label if label in ("positive", "neutral", "negative") else None
    symbol_filter = symbol.upper() if symbol else None

    return get_recent_news(symbol=symbol_filter, label=label_filter, q=q, page=page)


@router.get("/api/stocks/search")
async def api_stocks_search(request: Request, q: str = ""):
    session = get_session_context(request)
    if not session:
        raise HTTPException(status_code=401, detail="Login required")

    matches = search_active_stocks(q)
    return [{"symbol": s["symbol"], "company_name": s.get("company_name")} for s in matches]