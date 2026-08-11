"""Authenticated user feedback and frontend-admin review pages."""

import logging
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from backend.services.feedback_service import (
    FEEDBACK_TOPICS,
    FeedbackValidationError,
    create_feedback,
    get_feedback,
    list_feedback,
)


logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

USER_ROLES = {"basic_user", "premium_user"}
FRONTEND_ADMIN_ROLE = "frontend_admin"


def _user_context(request: Request) -> dict | None:
    role = request.session.get("user_role")
    if role not in USER_ROLES:
        return None

    email = request.session.get("user_email", "")
    return {
        "base_layout": (
            "premium_users/base.html"
            if role == "premium_user"
            else "free_users/base.html"
        ),
        "user_id": request.session.get("user_id"),
        "user_email": email,
        "user_initial": email[:1].upper() if email else "U",
        "user_role": role,
    }


def _admin_is_authenticated(request: Request) -> bool:
    return request.session.get("user_role") == FRONTEND_ADMIN_ROLE


def _admin_context(request: Request) -> dict:
    email = request.session.get("user_email") or ""
    return {
        "user_role": request.session.get("user_role"),
        "user_email": email,
        "user_initial": email[:1].upper() if email else "A",
    }


def _feedback_form_context(
    request: Request,
    *,
    submitted: bool = False,
    error: str | None = None,
    form_data: dict | None = None,
) -> dict:
    context = _user_context(request)
    if context is None:
        return {}
    return {
        **context,
        "topics": FEEDBACK_TOPICS,
        "submitted": submitted,
        "error": error,
        "form_data": form_data or {},
    }


@router.get("/user/feedback")
async def feedback_page(request: Request, submitted: bool = False):
    if _user_context(request) is None:
        if _admin_is_authenticated(request):
            return RedirectResponse(url="/admin/feedback", status_code=303)
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="feedback.html",
        context=_feedback_form_context(request, submitted=submitted),
    )


@router.post("/user/feedback")
async def submit_feedback(
    request: Request,
    topic: str = Form(...),
    rating: int = Form(...),
    description: str = Form(...),
):
    context = _user_context(request)
    if context is None:
        return RedirectResponse(url="/login", status_code=303)

    form_data = {
        "topic": topic,
        "rating": rating,
        "description": description,
    }
    try:
        create_feedback(
            user_id=context["user_id"],
            user_email=context["user_email"],
            topic=topic,
            rating=rating,
            description=description,
        )
    except FeedbackValidationError as exc:
        return templates.TemplateResponse(
            request=request,
            name="feedback.html",
            context=_feedback_form_context(
                request,
                error=str(exc),
                form_data=form_data,
            ),
            status_code=400,
        )
    except Exception:
        logger.exception(
            "Feedback submission failed for user_id=%s",
            context["user_id"],
        )
        return templates.TemplateResponse(
            request=request,
            name="feedback.html",
            context=_feedback_form_context(
                request,
                error="Feedback could not be submitted. Please try again.",
                form_data=form_data,
            ),
            status_code=503,
        )

    return RedirectResponse(
        url="/user/feedback?submitted=true",
        status_code=303,
    )


@router.get("/admin/feedback")
async def admin_feedback_page(request: Request):
    if not _admin_is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)

    try:
        feedback_rows = list_feedback()
        load_error = None
    except Exception:
        logger.exception("Admin feedback inbox lookup failed")
        feedback_rows = []
        load_error = "Feedback could not be loaded."

    ratings = [
        int(row["rating"])
        for row in feedback_rows
        if row.get("rating") is not None
    ]
    stats = {
        "total": len(feedback_rows),
        "average_rating": (
            round(sum(ratings) / len(ratings), 1)
            if ratings
            else None
        ),
        "five_star": sum(rating == 5 for rating in ratings),
    }
    return templates.TemplateResponse(
        request=request,
        name="user_admin/feedback_list.html",
        context={
            **_admin_context(request),
            "feedback_rows": feedback_rows,
            "stats": stats,
            "load_error": load_error,
        },
    )


@router.get("/admin/feedback/{feedback_id}")
async def admin_feedback_detail(request: Request, feedback_id: UUID):
    if not _admin_is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)

    try:
        feedback = get_feedback(str(feedback_id))
    except Exception as exc:
        logger.exception("Admin feedback detail lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Feedback could not be loaded.",
        ) from exc

    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found.")

    return templates.TemplateResponse(
        request=request,
        name="user_admin/feedback_detail.html",
        context={
            **_admin_context(request),
            "feedback": feedback,
        },
    )
