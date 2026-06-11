from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from backend.services.stock_list_service import get_all_stocks
from backend.database.supabase_client import supabase

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates/backend_admin")

@router.get("/backend_admin/stocks")
async def admin_stock_database(request: Request):
    role = request.session.get("user_role")
    if not role or role != "backend_admin":
        return RedirectResponse(url="/login", status_code=303)
    
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
        name="stock_database.html",
        context={
            "request": request,
            "stocks": display_stocks
        }
    )

@router.get("/backend_admin/weightages")
async def admin_weightages_page(request: Request):
    # Session
    role = request.session.get("user_role")
    if not role or role != "backend_admin":
        return RedirectResponse(url="/login", status_code=303)
    
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
        name="admin_weightages.html",
        context={
            "request": request, 
            "defaults": global_defaults
        }
    )

@router.post("/backend_admin/weightages")
async def save_admin_weightages(
    request: Request,
    technical: int = Form(...),
    sentiment: int = Form(...),
    financial: int = Form(...) 
):
    role = request.session.get("user_role")
    if not role or role != "backend_admin":
        return RedirectResponse(url="/login", status_code=303)

    if technical + sentiment + financial != 100:
        return RedirectResponse(url="/backend_admin/weightages", status_code=303)

    try:
        payload = {
            "technical": technical,
            "sentiment": sentiment,
            "financial": financial
        }
        
        supabase.table("weightages").update(payload).eq("id", "1").execute()
        
    except Exception as e:
        print(f"Database error saving admin defaults: {e}")

    return RedirectResponse(url="/backend_admin/weightages", status_code=303)

@router.get("/backend_admin/sentiment")
async def admin_sentiment_watchlist(request: Request):
    # Session
    role = request.session.get("user_role")
    if not role or role != "backend_admin":
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request=request, 
        name="admin_sentiment_watchlist.html"
        # context={"sources": sources_list}
    )

@router.get("/backend_admin/stocks/new")
async def add_stock_page(request: Request):
    # Session
    role = request.session.get("user_role")
    if not role or role != "backend_admin":
        return RedirectResponse(url="/login", status_code=303)
    
    return templates.TemplateResponse(
        request=request,
        name="add_stock.html",
        context={"request": request}
    )