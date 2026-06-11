"""Authenticated dashboard pages shared by all user roles."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.services.dashboard_service import (
    get_dashboard_stocks,
    get_stock_dashboard,
)


router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


def _session_context(request: Request) -> dict | None:
    role = request.session.get("user_role")
    if not role:
        return None

    email = request.session.get("user_email", "")
    base_layout = (
        "premium_users/base.html"
        if role == "premium_user"
        else "free_users/base.html"
    )
    return {
        "user_role": role,
        "user_email": email,
        "user_initial": email[:1].upper() if email else "U",
        "base_layout": base_layout,
    }


@router.get("/dashboard")
async def dashboard(request: Request):
    session = _session_context(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=303)

    stocks = get_dashboard_stocks()
    sectors = sorted({
        stock["sector"]
        for stock in stocks
        if stock.get("sector")
    })
    return templates.TemplateResponse(
        request=request,
        name="dashboard/index.html",
        context={
            **session,
            "stocks": stocks,
            "sectors": sectors,
        },
    )


@router.get("/stocks/{symbol}/view")
async def stock_detail(request: Request, symbol: str):
    session = _session_context(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=303)

    stock = get_stock_dashboard(symbol)
    if stock is None:
        return templates.TemplateResponse(
            request=request,
            name="dashboard/not_found.html",
            context={**session, "symbol": symbol.upper()},
            status_code=404,
        )

    return templates.TemplateResponse(
        request=request,
        name="dashboard/stock_detail.html",
        context={**session, "stock": stock},
    )
