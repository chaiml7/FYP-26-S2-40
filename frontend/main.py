import os
import sys

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
from backend.routes.backend_admin_routes import router as backend_admin_router
from backend.routes.dashboard_routes import router as dashboard_router
from backend.database.supabase_client import supabase

from backend.services.user_profile_service import get_profile

# Load the keys from .env file into Python's memory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

dotenv_path = os.path.join(root_dir, "backend", ".env")
load_dotenv(dotenv_path)

app = FastAPI()
app.include_router(stock_router)
app.include_router(user_router)
app.include_router(premium_user_router)
app.include_router(admin_router)
app.include_router(backend_admin_router)
app.include_router(dashboard_router)

# Session Middleware
app.add_middleware(SessionMiddleware, secret_key="my-super-secret-key")

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
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

# ==========================================
# Login / Logout Route + function
# ==========================================

@app.get("/login")
async def show_login(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="login.html"
    )

@app.post("/login")
async def process_login(
    request: Request, 
    email: str = Form(...), 
    password: str = Form(...)
):
    try:
        # Secure Authentication
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        user_id = auth_response.user.id
        
        # Get role using the service layer
        profile_data = get_profile(user_id)
        
        # Safely extract the role, defaulting to basic_user if the profile is empty
        if profile_data and len(profile_data) > 0:
            user_role = str(profile_data[0].get("role_id", "basic_user")).lower()
        else:
            user_role = "basic_user"

        # Stamp the session cookie
        request.session["user_email"] = email
        request.session["user_id"] = str(user_id)
        request.session["user_role"] = user_role

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
        supabase.auth.sign_out()
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
async def signup(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="signup.html"
    )
