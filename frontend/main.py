import os
import sys
import time

from dotenv import load_dotenv
from supabase import create_client, Client

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from backend.routes.stock_routes import router as stock_router
from backend.routes.user_routes import router as user_router
from backend.routes.premium_user_routes import router as premium_user_router
from backend.routes.admin_routes import router as admin_router
from backend.routes.dashboard_routes import router as dashboard_router
from backend.routes.feedback_routes import router as feedback_router
from backend.routes.billing_ui_routes import router as billing_ui_router
from backend.services.billing_service import (
    BillingConfigurationError,
    BillingError,
    create_checkout_session,
    reconcile_user_subscription_role,
)
from backend.services.auth_service import AuthServiceError, create_account
from backend.services.user_profile_service import get_profile
from backend.services.dashboard_service import (
    get_public_market_leaders,
    get_public_model_metrics,
)

# Load the keys from .env file into Python's memory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

dotenv_path = os.path.join(root_dir, "backend", ".env")
load_dotenv(dotenv_path)

# Separate client instance used only for login/logout auth calls.
# Never used for DB queries — keeps the shared service-role singleton uncontaminated.
_supabase_auth = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY"))

app = FastAPI()
app.include_router(stock_router)
# admin_router must be registered before user_router: both are mounted here
# with no prefix, and they both define GET /admin/users/{user_id} (this one
# session-cookie HTML, user_router's a bearer-token JSON endpoint gated on
# the admin role). Route matching is first-registered-wins, so this order
# makes the HTML page win on this app. user_router's JSON variant is
# unaffected on backend.main:app, where it's mounted under /api.
app.include_router(admin_router)
app.include_router(user_router)
app.include_router(premium_user_router)
app.include_router(dashboard_router)
app.include_router(feedback_router)
app.include_router(billing_ui_router)

@app.middleware("http")
async def refresh_cached_user_role(request: Request, call_next):
    """Limit stale Premium access after a webhook changes the database role."""
    user_id = request.session.get("user_id")
    last_checked = float(request.session.get("role_checked_at") or 0)
    if (
        user_id
        and not request.url.path.startswith("/static/")
        and time.time() - last_checked >= 60
    ):
        try:
            profiles = get_profile(str(user_id))
            if (
                isinstance(profiles, list)
                and profiles
                and isinstance(profiles[0], dict)
            ):
                stored_role = str(
                    profiles[0].get("role_id") or "basic_user"
                ).lower()
                request.session["user_role"] = reconcile_user_subscription_role(
                    str(user_id), stored_role
                )
                request.session["role_checked_at"] = time.time()
        except Exception as exc:
            # A temporary database failure must not log the user out. Premium
            # routes retain their existing checks and the refresh retries.
            print(f"Session role refresh failed: {exc}")
    return await call_next(request)


# Add SessionMiddleware after the function middleware so it is the outer
# layer and request.session is populated before role refresh runs.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "my-super-secret-key"),
    https_only=os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true",
    same_site="lax",
)

# Serve CSS assets from the static folder
static_path = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Configure the directory where HTML layouts sit
template_path = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=template_path)

# ==========================================
# Home Page Route
# ==========================================
@app.get("/")
async def home(request: Request):
    try:
        market_leaders = get_public_market_leaders()
    except Exception as exc:
        print(f"Public market leaders lookup failed: {exc}")
        market_leaders = []
    try:
        model_metrics = get_public_model_metrics()
    except Exception as exc:
        print(f"Public model metrics lookup failed: {exc}")
        model_metrics = None
    featured_analysis = next(
        (
            stock
            for stock in market_leaders
            if stock.get("overall_score") is not None
        ),
        None,
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "market_leaders": market_leaders,
            "featured_analysis": featured_analysis,
            "model_metrics": model_metrics,
        },
    )


# ==========================================
# Public Legal Pages
# ==========================================
def _legal_page_context(request: Request) -> dict[str, str | Request]:
    return {
        "request": request,
        "effective_date": "14 August 2026",
        "legal_contact_email": os.getenv("LEGAL_CONTACT_EMAIL", "").strip(),
    }


@app.get("/terms")
async def terms_of_service(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="legal/terms.html",
        context=_legal_page_context(request),
    )


@app.get("/privacy")
async def privacy_policy(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="legal/privacy.html",
        context=_legal_page_context(request),
    )

# ==========================================
# Login / Logout Route + function
# ==========================================

@app.get("/login")
async def show_login(request: Request, deleted: str = None):
    context = {}
    if deleted:
        context["info"] = "Your account has been deleted."
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context=context,
    )

@app.post("/login")
async def process_login(
    request: Request, 
    email: str = Form(...), 
    password: str = Form(...)
):
    try:
        # Secure Authentication
        auth_response = _supabase_auth.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        user_id = auth_response.user.id
        
        # Get role using the service layer
        profile_data = get_profile(user_id)
        
        # Safely extract the role, defaulting to basic_user if the profile is empty
        if profile_data and len(profile_data) > 0:
            stored_role = str(
                profile_data[0].get("role_id", "basic_user")
            ).lower()
            user_role = reconcile_user_subscription_role(
                str(user_id), stored_role
            )
        else:
            user_role = "basic_user"

        # Stamp the session cookie
        request.session["user_email"] = email
        request.session["user_id"] = str(user_id)
        request.session["user_role"] = user_role

        pending_plan = request.session.pop("pending_plan", None)
        if pending_plan == "premium" and user_role == "basic_user":
            base_url = (os.getenv("APP_PUBLIC_URL") or str(request.base_url)).rstrip("/")
            try:
                checkout_url = create_checkout_session(
                    user_id=str(user_id),
                    email=email,
                    success_url=f"{base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
                    cancel_url=f"{base_url}/billing/cancel",
                )
                return RedirectResponse(url=checkout_url, status_code=303)
            except (BillingConfigurationError, BillingError) as exc:
                return templates.TemplateResponse(
                    request=request,
                    name="billing/error.html",
                    context={
                        "base_layout": "free_users/base.html",
                        "user_role": user_role,
                        "user_id": str(user_id),
                        "user_email": email,
                        "user_initial": email[:1].upper(),
                        "billing_error": str(exc),
                    },
                    status_code=503,
                )

        if user_role == "admin":
            return RedirectResponse(url="/admin/user_management", status_code=303)
        else:
            return RedirectResponse(url="/dashboard", status_code=303)

    except Exception as e:
        return templates.TemplateResponse(
            request=request, 
            name="login.html", 
            context={"error": "Invalid email or password. Please try again."}
        )
    
@app.get("/logout")
async def logout(request: Request):
    try:
        # Kill active Auth session
        _supabase_auth.auth.sign_out()
    except Exception as e:
        print(f"Supabase Sign-Out Error: {e}")

    # Shred cookie
    request.session.clear()

    # Bounce user back to login page
    return RedirectResponse(url="/login", status_code=303)

# ==========================================
# Sign up Route
# ==========================================
@app.get("/signup")
async def signup(request: Request, plan: str = "free"):
    selected_plan = "premium" if plan.lower() == "premium" else "free"
    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={"selected_plan": selected_plan},
    )

@app.post("/signup")
async def process_signup(
    request: Request,
    firstName: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    plan: str = Form("free"),
    terms: str | None = Form(None),
):
    selected_plan = "premium" if plan.lower() == "premium" else "free"
    if terms != "accepted":
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error": "You must accept the Terms of Service and Privacy Policy to create an account.",
                "selected_plan": selected_plan,
            },
            status_code=400,
        )
    try:
        auth_response = create_account(email, password, firstName)
    except AuthServiceError as e:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": e.detail, "selected_plan": selected_plan}
        )

    # Supabase only issues a session immediately when email confirmation is
    # disabled; otherwise the response has no access_token and the user must
    # confirm their email before they can log in.
    if not auth_response.get("access_token"):
        if selected_plan == "premium":
            request.session["pending_plan"] = "premium"
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Account created! Please check your email to confirm your address before logging in."}
        )

    user_id = auth_response.get("user", {}).get("id")

    request.session["user_email"] = email
    request.session["user_id"] = str(user_id)
    request.session["user_role"] = "basic_user"

    if selected_plan == "premium":
        base_url = (os.getenv("APP_PUBLIC_URL") or str(request.base_url)).rstrip("/")
        try:
            checkout_url = create_checkout_session(
                user_id=str(user_id),
                email=email,
                success_url=f"{base_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{base_url}/billing/cancel",
            )
            return RedirectResponse(url=checkout_url, status_code=303)
        except (BillingConfigurationError, BillingError) as exc:
            return templates.TemplateResponse(
                request=request,
                name="billing/error.html",
                context={
                    "base_layout": "free_users/base.html",
                    "user_role": "basic_user",
                    "user_id": str(user_id),
                    "user_email": email,
                    "user_initial": email[:1].upper(),
                    "billing_error": str(exc),
                },
                status_code=503,
            )

    return RedirectResponse(url="/dashboard", status_code=303)
