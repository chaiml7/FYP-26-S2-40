from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from backend.database.supabase_client import supabase

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates/user_admin")

@router.get("/admin/user_management")
async def user_management_page(request: Request, filter: str = "all"):
    session_role = request.session.get("user_role")
    if not session_role or session_role != "frontend_admin":
        return RedirectResponse(url="/login", status_code=303)
    
    response = supabase.table("user_profiles").select("*").execute()
    all_users = response.data

    stats = {
        "total_users": len(all_users), 
        "premium_users": sum(1 for u in all_users if str(u.get("role_id")).lower() == "premium_user"),
        "suspended_users": sum(1 for u in all_users if str(u.get("is_active")).lower() == "false"),
        "active_today": 0 
    }

    mapped_users = []
    for u in all_users:
        is_active_str = str(u.get("is_active", "true")).lower()
        role_id_raw = str(u.get("role_id", "basic_user")).lower()
        
        mapped_users.append({
            "id": u.get("id"),
            "username": u.get("username") or u.get("email", "").split("@")[0],
            "full_name": u.get("full_name") or "Unknown User",
            "role_id": role_id_raw,
            "status": "Active" if is_active_str == "true" else "Suspended"
        })
    
    display_users = mapped_users
    if filter == "active":
        display_users = [u for u in mapped_users if u["status"] == "Active"]
    elif filter == "suspended":
        display_users = [u for u in mapped_users if u["status"] == "Suspended"]

    return templates.TemplateResponse(
        request=request,
        name="user_management.html",
        context={
            "request": request, 
            "users": display_users,  
            "stats": stats,          
            "current_filter": filter
        }
    )

@router.get("/admin/roles_management")
async def roles_management_page(request: Request, role: str = "user admin"):
    session_role = request.session.get("user_role")
    if not session_role or session_role != "frontend_admin":
        return RedirectResponse(url="/login", status_code=303)
        
    target_role = role.lower()

    roles_response = supabase.table("roles").select("*").execute()
    roles_data = roles_response.data

    selected_role = next((r for r in roles_data if r["id"].lower() == target_role), roles_data[0])

    role_perms = {
        "basic_user": [
            {"name": "View public pages", "allowed": True}, {"name": "Manage watchlist", "allowed": True},
            {"name": "View financial reports", "allowed": True}, {"name": "Manage social media follow list", "allowed": True},
            {"name": "Upgrade subscription", "allowed": True}, {"name": "Search stock tickers", "allowed": True},
            {"name": "View top gainers & losers", "allowed": True}, {"name": "View news & social feeds", "allowed": True},
            {"name": "Update account password", "allowed": True}, {"name": "Delete account", "allowed": True},
            {"name": "AI prediction breakdown", "allowed": False}, {"name": "Risk-based recommendations", "allowed": False},
            {"name": "Custom model weightages", "allowed": False}
        ],
        "premium_user": [
            {"name": "View public pages", "allowed": True}, {"name": "Manage watchlist", "allowed": True},
            {"name": "View financial reports", "allowed": True}, {"name": "Manage social media follow list", "allowed": True},
            {"name": "Downgrade subscription", "allowed": True}, {"name": "AI prediction breakdown", "allowed": True},
            {"name": "Custom model weightages", "allowed": True}, {"name": "Search stock tickers", "allowed": True},
            {"name": "View top gainers & losers", "allowed": True}, {"name": "View news & social feeds", "allowed": True},
            {"name": "Update account password", "allowed": True}, {"name": "Delete account", "allowed": True},
            {"name": "Risk-based recommendations", "allowed": True}
        ],
        "frontend_admin": [
            {"name": "Access admin dashboard", "allowed": True}, {"name": "Update user account details", "allowed": True},
            {"name": "Manage role assignments", "allowed": True}, {"name": "View user account details", "allowed": True},
            {"name": "Suspend / reinstate users", "allowed": True}, {"name": "Modify stock database", "allowed": False},
            {"name": "Manage sentiment watchlist", "allowed": False}, {"name": "View stock database", "allowed": False},
            {"name": "Update default model weightages", "allowed": False}, {"name": "Generate performance reports", "allowed": False}
        ],
        "backend_admin": [
            {"name": "Access backend admin dashboard", "allowed": True}, {"name": "Remove / delist stocks", "allowed": True},
            {"name": "Add sentiment watchlist sources", "allowed": True}, {"name": "View sentiment watchlist", "allowed": True},
            {"name": "Add new stocks to database", "allowed": True}, {"name": "Update default model weightages", "allowed": True},
            {"name": "Suspend sentiment watchlist sources", "allowed": True}, {"name": "Generate performance reports", "allowed": True},
            {"name": "Manage user accounts", "allowed": False}, {"name": "Suspend users", "allowed": False}
        ]
    }

    permissions_data = role_perms.get(selected_role["id"].lower(), [])

    users_response = supabase.table("user_profiles").select("id, full_name, email, role_id").execute()
    users_data = users_response.data

    display_users = []
    for user in users_data:
        display_users.append({
            "id": user.get("id"),
            "full_name": user.get("full_name") or "Unknown User", 
            "email": user.get("email") or "No Email",
            "role": str(user.get("role_id", "")).lower() 
        })  

    assigned_users = [u for u in display_users if u["role"] == selected_role["id"].lower()]
    unassigned_users = [u for u in display_users if u["role"] == ""]

    return templates.TemplateResponse(
        request=request,
        name="user_roles_management.html",
        context={
            "request": request,
            "roles": roles_data,
            "selected_role": selected_role,
            "permissions": permissions_data,
            "assigned_users": assigned_users, 
            "unassigned_users": unassigned_users
        }
    )