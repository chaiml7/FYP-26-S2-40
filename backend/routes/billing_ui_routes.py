"""Authenticated browser billing routes served by the frontend app."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.services.billing_service import (
    AlreadyPremiumError,
    BillingConfigurationError,
    BillingError,
    PREMIUM_SUBSCRIPTION_STATUSES,
    create_checkout_session,
    create_customer_portal_session,
    get_user_subscription,
)
from backend.services.session_context import get_session_context
from backend.services.user_profile_service import get_profile


router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")
CUSTOMER_ROLES = {"basic_user", "premium_user"}


def _authenticated_customer(request: Request) -> dict | None:
    context = get_session_context(request)
    if not context or context["user_role"] not in CUSTOMER_ROLES:
        return None
    if not context.get("user_id"):
        return None
    return context


def _public_url(request: Request) -> str:
    configured = os.getenv("APP_PUBLIC_URL", "").strip().rstrip("/")
    return configured or str(request.base_url).rstrip("/")


def _refresh_session_role(request: Request) -> str | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    profiles = get_profile(str(user_id))
    if not profiles:
        return None
    role = str(profiles[0].get("role_id") or "basic_user").lower()
    request.session["user_role"] = role
    return role


def _billing_error_page(
    request: Request, context: dict, message: str, *, status_code: int
):
    return templates.TemplateResponse(
        request=request,
        name="billing/error.html",
        context={**context, "billing_error": message},
        status_code=status_code,
    )


@router.post("/billing/checkout")
async def start_checkout(request: Request):
    context = _authenticated_customer(request)
    if context is None:
        return RedirectResponse(url="/login", status_code=303)

    base_url = _public_url(request)
    try:
        checkout_url = create_checkout_session(
            user_id=str(context["user_id"]),
            email=str(context.get("user_email") or ""),
            success_url=f"{base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base_url}/billing/cancel",
        )
    except AlreadyPremiumError:
        return RedirectResponse(url="/dashboard", status_code=303)
    except BillingConfigurationError as exc:
        return _billing_error_page(request, context, str(exc), status_code=503)
    except BillingError as exc:
        return _billing_error_page(request, context, str(exc), status_code=400)

    return RedirectResponse(url=checkout_url, status_code=303)


@router.post("/billing/portal")
async def open_customer_portal(request: Request):
    context = _authenticated_customer(request)
    if context is None:
        return RedirectResponse(url="/login", status_code=303)

    try:
        portal_url = create_customer_portal_session(
            user_id=str(context["user_id"]),
            return_url=f"{_public_url(request)}/dashboard",
        )
    except BillingConfigurationError as exc:
        return _billing_error_page(request, context, str(exc), status_code=503)
    except BillingError as exc:
        return _billing_error_page(request, context, str(exc), status_code=400)

    return RedirectResponse(url=portal_url, status_code=303)


@router.get("/billing/success")
async def billing_success(request: Request):
    if _authenticated_customer(request) is None:
        return RedirectResponse(url="/login", status_code=303)
    _refresh_session_role(request)
    context = get_session_context(request)
    if context is None:
        return RedirectResponse(url="/login", status_code=303)
    subscription = get_user_subscription(str(context["user_id"])) or {}
    status = str(subscription.get("status") or "processing").lower()
    return templates.TemplateResponse(
        request=request,
        name="billing/success.html",
        context={
            **context,
            "subscription_status": status,
            "is_premium": (
                context["user_role"] == "premium_user"
                and status in PREMIUM_SUBSCRIPTION_STATUSES
            ),
        },
    )


@router.get("/billing/cancel")
async def billing_cancel(request: Request):
    context = _authenticated_customer(request)
    if context is None:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="billing/cancel.html",
        context=context,
    )


@router.get("/billing/status")
async def billing_status(request: Request):
    context = _authenticated_customer(request)
    if context is None:
        raise HTTPException(status_code=401, detail="Login required")
    role = _refresh_session_role(request) or context["user_role"]
    subscription = get_user_subscription(str(context["user_id"])) or {}
    status = str(subscription.get("status") or "inactive").lower()
    return JSONResponse(
        {
            "status": status,
            "is_premium": (
                role == "premium_user"
                and status in PREMIUM_SUBSCRIPTION_STATUSES
            ),
            "cancel_at_period_end": bool(
                subscription.get("cancel_at_period_end", False)
            ),
            "current_period_end": subscription.get("current_period_end"),
        }
    )
