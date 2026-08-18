from fastapi import APIRouter, Request, Form, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from backend.database.supabase_client import supabase
from backend.services.admin_report_service import (
    build_admin_report,
    build_model_accuracy_log,
    build_system_health,
    render_report_csv,
    render_report_pdf,
)
from backend.services.session_context import get_session_context
from backend.services.stock_list_service import get_all_stocks
from backend.services.sentiment_source_service import (
    add_sentiment_source,
    delete_sentiment_source,
    get_all_sentiment_sources,
    set_sentiment_source_active,
    update_sentiment_source,
)
from backend.services.activity_log_service import get_activity_log, log_activity
from backend.services.user_profile_service import get_profile, update_user_status

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")
ADMIN_REPORT_ROLES = {"admin"}


def _admin_report_session(request: Request):
    session = get_session_context(request)
    if session and session["user_role"] in ADMIN_REPORT_ROLES:
        return session
    return None

@router.get("/admin/user_management")
async def user_management_page(request: Request, filter: str = "all", q: str = ""):
    session_role = request.session.get("user_role")
    if not session_role or session_role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

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
            "email": u.get("email") or "",
            "role_id": role_id_raw,
            "status": "Active" if is_active_str == "true" else "Suspended"
        })

    display_users = mapped_users
    if filter == "active":
        display_users = [u for u in display_users if u["status"] == "Active"]
    elif filter == "suspended":
        display_users = [u for u in display_users if u["status"] == "Suspended"]

    search_term = q.strip().lower()
    if search_term:
        display_users = [
            u for u in display_users
            if search_term in str(u["id"]).lower()
            or search_term in u["username"].lower()
            or search_term in u["full_name"].lower()
            or search_term in u["email"].lower()
        ]

    return templates.TemplateResponse(
        request=request,
        name="user_admin/user_management.html",
        context={
            **session,
            "request": request,
            "users": display_users,
            "stats": stats,
            "current_filter": filter,
            "search_query": q,
        }
    )

@router.get("/admin/activity_log")
async def admin_activity_log_page(
    request: Request,
    user: str = "",
    date_from: str = "",
    date_to: str = "",
):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

    entries = get_activity_log(
        email=user.strip() or None,
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
    )

    return templates.TemplateResponse(
        request=request,
        name="user_admin/activity_log.html",
        context={
            **session,
            "request": request,
            "entries": entries,
            "filter_user": user,
            "filter_date_from": date_from,
            "filter_date_to": date_to,
        }
    )


@router.get("/admin/users/{user_id}")
async def admin_user_detail(request: Request, user_id: str):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

    profile = get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")

    raw = profile[0]
    is_active_str = str(raw.get("is_active", "true")).lower()
    detail = {
        "id": raw.get("id"),
        "username": raw.get("username") or (raw.get("email") or "").split("@")[0],
        "full_name": raw.get("full_name") or "Unknown User",
        "email": raw.get("email") or "No Email",
        "role_id": str(raw.get("role_id", "basic_user")).lower(),
        "status": "Active" if is_active_str == "true" else "Suspended",
        "created_at": str(raw.get("created_at", "")),
    }

    return templates.TemplateResponse(
        request=request,
        name="user_admin/user_detail.html",
        context={**session, "request": request, "user": detail}
    )


@router.post("/admin/users/{user_id}/suspend")
async def admin_suspend_user(request: Request, user_id: str):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    if user_id == request.session.get("user_id"):
        return RedirectResponse(url="/admin/user_management", status_code=303)

    updated = update_user_status(user_id, False)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    target_email = updated[0].get("email", user_id)
    admin_email = request.session.get("user_email", "admin")
    log_activity(target_email, "account_suspended", f"Suspended by {admin_email}")
    return RedirectResponse(url="/admin/user_management", status_code=303)


@router.post("/admin/users/{user_id}/unsuspend")
async def admin_unsuspend_user(request: Request, user_id: str):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    updated = update_user_status(user_id, True)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    target_email = updated[0].get("email", user_id)
    admin_email = request.session.get("user_email", "admin")
    log_activity(target_email, "account_reactivated", f"Reactivated by {admin_email}")
    return RedirectResponse(url="/admin/user_management", status_code=303)


@router.get("/admin/roles_management")
async def roles_management_page(request: Request, role: str = "admin"):
    session_role = request.session.get("user_role")
    if not session_role or session_role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

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
        "admin": [
            {"name": "Access admin dashboard", "allowed": True}, {"name": "Update user account details", "allowed": True},
            {"name": "Manage role assignments", "allowed": True}, {"name": "View user account details", "allowed": True},
            {"name": "Suspend / reinstate users", "allowed": True}, {"name": "Modify stock database", "allowed": True},
            {"name": "Manage sentiment watchlist", "allowed": True}, {"name": "View stock database", "allowed": True},
            {"name": "Update default model weightages", "allowed": True}, {"name": "Generate performance reports", "allowed": True}
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
    unassigned_users = [u for u in display_users if u["role"] != selected_role["id"].lower()]

    return templates.TemplateResponse(
        request=request,
        name="user_admin/user_roles_management.html",
        context={
            **session,
            "request": request,
            "roles": roles_data,
            "selected_role": selected_role,
            "permissions": permissions_data,
            "assigned_users": assigned_users,
            "unassigned_users": unassigned_users
        }
    )

@router.get("/admin/stocks")
async def admin_stock_database(request: Request):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

    # 1. Fetch all stocks using your groupmate's existing logic
    raw_stocks = get_all_stocks()

    # 2. THE DATA TRANSFORMATION LAYER
    # Translate the database dictionary keys into exactly what Jinja2 expects
    display_stocks = []
    for stock in raw_stocks:
        display_stocks.append({
            # Map DB 'symbol' to HTML 'ticker', ensuring it's uppercase
            "ticker": stock.get("symbol", "N/A").upper(), 
            
            # Assuming your DB uses 'name' for the company
            "company_name": stock.get("company_name", "Unknown Company"), 
            
            "exchange": stock.get("exchange", "NASDAQ"),
            
            # Convert to string safely so the HTML [:10] slicing doesn't crash
            "created_at": str(stock.get("created_at", "2026-01-01")), 
            
            # Translate the boolean 'is_active' into the exact string the HTML expects
            "status": "Active" if stock.get("is_active") else "Delisted" 
        })

    # 3. Render the page
    return templates.TemplateResponse(
        request=request,
        name="user_admin/stock_database.html",
        context={
            **session,
            "request": request,
            "stocks": display_stocks
        }
    )

@router.get("/admin/weightages")
async def admin_weightages_page(request: Request):
    # Session
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

    try:
        db_response = supabase.table("weightages").select(
            "technical, sentiment, financial"
        ).eq("id", "1").execute()
        
        global_defaults = db_response.data[0] if db_response.data else None
    except Exception as e:
        print(f"Database error fetching platform defaults: {e}")
        global_defaults = None

    return templates.TemplateResponse(
        request=request, 
        name="user_admin/admin_weightages.html",
        context={
            **session,
            "request": request,
            "defaults": global_defaults
        }
    )

@router.post("/admin/weightages")
async def save_admin_weightages(
    request: Request,
    technical: int = Form(...),
    sentiment: int = Form(...),
    financial: int = Form(...) 
):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    if not all(0 <= value <= 100 for value in (technical, sentiment, financial)):
        return RedirectResponse(url="/admin/weightages", status_code=303)

    if technical + sentiment + financial != 100:
        return RedirectResponse(url="/admin/weightages", status_code=303)

    try:
        payload = {
            "technical": technical,
            "sentiment": sentiment,
            "financial": financial
        }
        
        supabase.table("weightages").update(payload).eq("id", "1").execute()
        
    except Exception as e:
        print(f"Database error saving admin defaults: {e}")

    return RedirectResponse(url="/admin/weightages", status_code=303)

@router.get("/admin/sentiment")
async def admin_sentiment_watchlist(request: Request):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

    raw_sources = get_all_sentiment_sources()
    sources = [
        {
            "id": source.get("id"),
            "type": source.get("source_type"),
            "account": source.get("account"),
            "relevance": source.get("relevance") or "—",
            "status": "Active" if source.get("is_active") else "Suspended",
        }
        for source in raw_sources
    ]

    return templates.TemplateResponse(
        request=request,
        name="user_admin/admin_sentiment_watchlist.html",
        context={**session, "request": request, "sources": sources}
    )

@router.post("/admin/sentiment/add")
async def admin_add_sentiment_source(
    request: Request,
    source_type: str = Form(...),
    account: str = Form(...),
    relevance: str = Form(default=""),
):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    add_sentiment_source(source_type.strip(), account.strip(), relevance.strip() or None)
    return RedirectResponse(url="/admin/sentiment", status_code=303)


@router.post("/admin/sentiment/{source_id}/edit")
async def admin_edit_sentiment_source(
    request: Request,
    source_id: str,
    source_type: str = Form(...),
    account: str = Form(...),
    relevance: str = Form(default=""),
):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    update_sentiment_source(source_id, source_type.strip(), account.strip(), relevance.strip() or None)
    return RedirectResponse(url="/admin/sentiment", status_code=303)


@router.post("/admin/sentiment/{source_id}/suspend")
async def admin_suspend_sentiment_source(request: Request, source_id: str):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    set_sentiment_source_active(source_id, False)
    return RedirectResponse(url="/admin/sentiment", status_code=303)


@router.post("/admin/sentiment/{source_id}/reactivate")
async def admin_reactivate_sentiment_source(request: Request, source_id: str):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    set_sentiment_source_active(source_id, True)
    return RedirectResponse(url="/admin/sentiment", status_code=303)


@router.post("/admin/sentiment/{source_id}/delete")
async def admin_delete_sentiment_source(request: Request, source_id: str):
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)

    delete_sentiment_source(source_id)
    return RedirectResponse(url="/admin/sentiment", status_code=303)

@router.get("/admin/reports")
async def admin_reports_page(request: Request):
    session = _admin_report_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="user_admin/performance_reports.html",
        context={
            **session,
            "request": request,
            "report": None,
        },
    )

@router.post("/admin/reports/generate")
async def generate_admin_report_page(
    request: Request,
    date_from: str = Form(""),
    date_to: str = Form(""),
):
    session = _admin_report_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    report = build_admin_report(
        session.get("user_email", ""),
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
    )

    return templates.TemplateResponse(
        request=request,
        name="user_admin/performance_reports.html",
        context={
            **session,
            "request": request,
            "report": report,
        },
    )


@router.get("/admin/reports/export.csv")
async def export_admin_report_csv(
    request: Request,
    date_from: str = "",
    date_to: str = "",
):
    session = _admin_report_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    report = build_admin_report(
        session.get("user_email", ""),
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
    )
    csv_bytes = render_report_csv(report)

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="stocklens_admin_report.csv"'
        },
    )


@router.get("/admin/reports/export.pdf")
async def export_admin_report_pdf(
    request: Request,
    date_from: str = "",
    date_to: str = "",
):
    session = _admin_report_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    report = build_admin_report(
        session.get("user_email", ""),
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
    )
    pdf_bytes = render_report_pdf(report)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="stocklens_admin_report.pdf"'
        },
    )

@router.get("/admin/prediction_logs")
async def admin_prediction_logs_page(
    request: Request,
    model_version: str = "",
    date_from: str = "",
    date_to: str = "",
):
    session = _admin_report_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    log = build_model_accuracy_log(
        model_version=model_version.strip() or None,
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
    )

    return templates.TemplateResponse(
        request=request,
        name="user_admin/prediction_logs.html",
        context={
            **session,
            "request": request,
            "log": log,
            "filter_model_version": model_version,
            "filter_date_from": date_from,
            "filter_date_to": date_to,
        }
    )


@router.get("/admin/system_health")
async def admin_system_health_page(request: Request):
    session = _admin_report_session(request)
    if not session:
        return RedirectResponse(url="/login", status_code=303)

    health = build_system_health()

    return templates.TemplateResponse(
        request=request,
        name="user_admin/system_health.html",
        context={**session, "request": request, "health": health}
    )


@router.get("/admin/stocks/new")
async def add_stock_page(request: Request):
    # Session
    role = request.session.get("user_role")
    if not role or role != "admin":
        return RedirectResponse(url="/login", status_code=303)
    session = get_session_context(request)

    return templates.TemplateResponse(
        request=request,
        name="user_admin/add_stock.html",
        context={**session, "request": request}
    )
