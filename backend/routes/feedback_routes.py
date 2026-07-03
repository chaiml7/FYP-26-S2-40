"""User feedback and model feedback routes."""
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from backend.database.supabase_client import supabase

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


def _get_layout(role: str) -> str:
    if role == "premium_user":
        return "premium_users/base.html"
    return "free_users/base.html"


@router.get("/user/feedback")
async def feedback_page(request: Request):
    role = request.session.get("user_role")
    if not role:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="feedback.html",
        context={
            "base_layout": _get_layout(role),
            "user_email": request.session.get("user_email", ""),
            "user_initial": request.session.get("user_email", "U")[:1].upper(),
            "user_role": role,
            "success": False,
            "error": False,
        }
    )


@router.post("/user/feedback")
async def submit_feedback(request: Request):
    role = request.session.get("user_role")
    if not role:
        return RedirectResponse(url="/login", status_code=303)

    form = await request.form()
    topic = form.get("topic", "")
    rating = form.get("rating", 0)
    description = form.get("description", "")
    user_id = request.session.get("user_id")
    user_email = request.session.get("user_email", "")

    success = False
    error = False

    try:
        supabase.table("user_feedback").insert({
            "user_id": user_id,
            "username": user_email,
            "topic": topic,
            "rating": int(rating),
            "description": description,
        }).execute()
        success = True
    except Exception as e:
        print(f"Feedback submission error: {e}")
        error = True

    return templates.TemplateResponse(
        request=request,
        name="feedback.html",
        context={
            "base_layout": _get_layout(role),
            "user_email": user_email,
            "user_initial": user_email[:1].upper(),
            "user_role": role,
            "success": success,
            "error": error,
        }
    )


class ModelFeedbackRequest(BaseModel):
    model_type: str
    vote: str


@router.post("/user/model_feedback")
async def submit_model_feedback(request: Request, body: ModelFeedbackRequest):
    user_id = request.session.get("user_id")
    if not user_id:
        return {"error": "Not authenticated"}

    if body.model_type not in ["technical", "sentiment", "financial"]:
        return {"error": "Invalid model type"}

    if body.vote not in ["up", "down"]:
        return {"error": "Invalid vote"}

    try:
        existing = supabase.table("model_feedback").select("id").eq(
            "user_id", user_id
        ).eq("model_type", body.model_type).execute()

        if existing.data:
            supabase.table("model_feedback").update({
                "vote": body.vote
            }).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase.table("model_feedback").insert({
                "user_id": user_id,
                "model_type": body.model_type,
                "vote": body.vote,
            }).execute()

        return {"success": True}
    except Exception as e:
        print(f"Model feedback error: {e}")
        return {"error": str(e)}