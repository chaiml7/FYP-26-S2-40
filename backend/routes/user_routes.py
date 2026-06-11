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
from backend.services.stock_list_service import get_stock_by_symbol
from backend.services.stock_history_service import get_latest_stock_price
from backend.services.prediction_service import get_latest_prediction_by_symbol
from backend.services.sentiment.sentiment_aggregator import get_sentiment_summary
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
    watchlist = get_user_watchlist(user["id"])

    return [
        item["stocks"]["symbol"]
        for item in watchlist
        if item.get("stocks") and item["stocks"].get("symbol")
    ]


@router.get("/users/me/watchlist/summary")
def view_current_user_watchlist_summary(authorization: str = Header(default=None)):
    user, _ = _current_user(authorization)
    watchlist = get_user_watchlist(user["id"])
    summary = []

    for item in watchlist:
        stock = item.get("stocks") or {}
        symbol = stock.get("symbol")

        latest_price = get_latest_stock_price(symbol) if symbol else []
        latest_prediction = get_latest_prediction_by_symbol(symbol) if symbol else []

        try:
            sentiment = get_sentiment_summary(symbol) if symbol else {}
        except Exception:
            sentiment = {}

        weighted_scores = sentiment.get("weighted_scores") or []
        legacy_daily_scores = sentiment.get("daily_scores") or []

        summary.append({
            "watchlist_id": item["id"],
            "stock_id": item["stock_id"],
            "symbol": symbol,
            "company_name": stock.get("company_name"),
            "sector": stock.get("sector"),
            "latest_price": latest_price[0] if len(latest_price) > 0 else None,
            "latest_prediction": latest_prediction[0] if len(latest_prediction) > 0 else None,
            "sentiment": {
                "latest_daily_score": weighted_scores[0]
                if weighted_scores
                else (legacy_daily_scores[0] if legacy_daily_scores else None)
            },
            "added_at": item["created_at"],
        })

    return summary


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
    role = request.session.get("user_role")
    if not role:
        return RedirectResponse(url="/login", status_code=303)

    # DYNAMIC LAYOUT SELECTOR: Chooses the shell based on the user's role
    layout = "premium_users/base.html" if role == "premium_user" else "free_users/base.html"

    return templates.TemplateResponse(
        request=request, 
        name="free_users/watchlist.html",
        context={
            "request": request,
            "base_layout": layout  # Passes the chosen layout to Jinja2
        }
    )

@router.get("/user/market_overview")
async def market_overview(request: Request):
    role = request.session.get("user_role")
    if not role:
        return RedirectResponse(url="/login", status_code=303)

    layout = "premium_users/base.html" if role == "premium_user" else "free_users/base.html"

    return templates.TemplateResponse(
        request=request, 
        name="free_users/user_market_overview.html",
        context={
            "request": request,
            "base_layout": layout
        }
    )

@router.get("/user/news_social")
async def news_social(request: Request):
    role = request.session.get("user_role")
    if not role:
        return RedirectResponse(url="/login", status_code=303)

    layout = "premium_users/base.html" if role == "premium_user" else "free_users/base.html"

    return templates.TemplateResponse(
        request=request, 
        name="free_users/news_social.html",
        context={
            "request": request,
            "base_layout": layout
        }
    )