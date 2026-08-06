"""Authenticated dashboard pages shared by all user roles."""

from datetime import date

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.services.dashboard_service import (
    get_dashboard_stocks,
    get_stock_dashboard,
)
from backend.services.session_context import get_session_context
from backend.services.user_watchlist_service import get_user_watchlist_symbols


router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


def _session_context(request: Request) -> dict | None:
    return get_session_context(request)


@router.get("/dashboard")
async def dashboard(request: Request, selected_date: date = None):
    session = _session_context(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=303)

    stocks = get_dashboard_stocks(selected_date)
    sectors = sorted({
        stock["sector"]
        for stock in stocks
        if stock.get("sector")
    })

    watchlisted_symbols = []
    if session["user_role"] == "premium_user":
        try:
            watchlisted_symbols = get_user_watchlist_symbols(session["user_id"])
        except Exception:
            watchlisted_symbols = []

    return templates.TemplateResponse(
        request=request,
        name="dashboard/index.html",
        context={
            **session,
            "stocks": stocks,
            "sectors": sectors,
            "selected_date": (
                selected_date.isoformat() if selected_date else ""
            ),
            "watchlisted_symbols": watchlisted_symbols,
        },
    )


@router.get("/stocks/{symbol}/view")
async def stock_detail(
    request: Request,
    symbol: str,
    selected_date: date = None,
):
    session = _session_context(request)
    if session is None:
        return RedirectResponse(url="/login", status_code=303)

    stock = get_stock_dashboard(
        symbol,
        selected_date,
        include_technical_indicators=(session["user_role"] == "premium_user"),
        weight_user_id=(
            session["user_id"]
            if session["user_role"] == "premium_user"
            else None
        ),
    )
    if stock is None:
        return templates.TemplateResponse(
            request=request,
            name="dashboard/not_found.html",
            context={
                **session,
                "symbol": symbol.upper(),
                "selected_date": (
                    selected_date.isoformat() if selected_date else ""
                ),
            },
            status_code=404,
        )

    is_watchlisted = False
    if session["user_role"] == "premium_user":
        try:
            is_watchlisted = stock["symbol"] in get_user_watchlist_symbols(session["user_id"])
        except Exception:
            is_watchlisted = False

    return templates.TemplateResponse(
        request=request,
        name="dashboard/stock_detail.html",
        context={**session, "stock": stock, "is_watchlisted": is_watchlisted},
    )
