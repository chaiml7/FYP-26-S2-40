from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from datetime import date

from backend.services.stock_list_service import get_active_stocks
from backend.services.prediction_service import get_latest_prediction_by_symbol, get_technical_score, get_financial_score
from backend.services.sentiment.sentiment_aggregator import get_weighted_sentiment_score
from backend.database.supabase_client import supabase

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates/premium_users")

@router.get("/premium/recommendations")
async def premium_recommendations(request: Request):
    # Session
    role = request.session.get("user_role")
    if not role or role != "premium_user":
        return RedirectResponse(url="/login", status_code=303)
    
    active_stocks = get_active_stocks()

    # 2. Build the Data Transformation Layer
    display_recommendations = []
    for stock in active_stocks:
        symbol = stock.get("symbol", "").upper()
        
        # Grab the latest AI prediction for this specific stock
        raw_pred = get_latest_prediction_by_symbol(symbol)
        
        # If a prediction exists in the DB, format it for the HTML
        if raw_pred and len(raw_pred) > 0:
            pred = raw_pred[0]
            display_recommendations.append({
                "ticker": symbol,
                "company_name": stock.get("name", "Unknown Company"),
                "action": pred.get("action", "HOLD").upper(),
                "target_price": f"{float(pred.get('target_price', 0)):.2f}",
                "confidence": pred.get("confidence_score", 0),
                "rationale": pred.get("rationale", "Standard market conditions apply.")
            })

    return templates.TemplateResponse(
        request=request, 
        name="stock_recommendations.html",
        context={
            "request": request,
            "recommendations": display_recommendations
        }
    )

@router.get("/premium/prediction_breakdown")
async def premium_prediction_breakdown(request: Request, symbol: str = "NVDA"):
    role = request.session.get("user_role")
    user_id = request.session.get("user_id")
    if not role or role != "premium_user":
        return RedirectResponse(url="/login", status_code=303)
    
    target_symbol = symbol.upper()

    try:
        user_w_res = supabase.table("weightages").select("technical, sentiment, financial").eq("user_id", user_id).execute()
        if user_w_res.data:
            weights = user_w_res.data[0]
        else:
            admin_w_res = supabase.table("weightages").select("technical, sentiment, financial").eq("id", "1").execute()
            weights = admin_w_res.data[0] if admin_w_res.data else {"technical": 40, "sentiment": 30, "financial": 30}
    except Exception as e:
        print(f"Error matching weight records: {e}")
        weights = {"technical": 40, "sentiment": 30, "financial": 30}

    tech_w = weights.get("technical")
    sent_w = weights.get("sentiment")
    fin_w = weights.get("financial")

    # Fetch Sentiment
    sentiment_date = date(2026, 5, 25)
    sentiment_data = get_weighted_sentiment_score(target_symbol, sentiment_date)
    if sentiment_data and "bullish_score" in sentiment_data:
        raw_sent = int((sentiment_data.get("bullish_score") or 0))
    else:
        raw_sent = 0

    try:
        tech_data = get_technical_score(target_symbol)
        raw_tech = tech_data if isinstance(tech_data, (int, float)) else tech_data.get('score', 0)
    except Exception:
        raw_tech = 0

    try:
        fin_data = get_financial_score(target_symbol)
        raw_fin = fin_data if isinstance(fin_data, (int, float)) else fin_data.get('score', 0)
    except Exception:
        raw_fin = 0

    composite_score = round(
        (raw_tech * (tech_w / 100.0)) +
        (raw_sent * (sent_w / 100.0)) +
        (raw_fin * (fin_w / 100.0)),
        2,
    )

    if composite_score >= 6.5:
        action_label = "BUY"
    elif composite_score <= 3.5:
        action_label = "SELL"
    else:
        action_label = "HOLD"

    display_data = {
        "symbol": target_symbol,
        "action": action_label,
        "composite": composite_score,
        "technical_score": round(float(raw_tech), 2),
        "sentiment_score": round(float(raw_sent), 2),
        "financial_score": round(float(raw_fin), 2),
        "tech_weight": tech_w,
        "sent_weight": sent_w,
        "fin_weight": fin_w
    }

    return templates.TemplateResponse(
        request=request, 
        name="prediction_breakdown.html",
        context={
            "request": request,
            "data": display_data
        }
    )

@router.get("/premium/weightages")
async def premium_user_weightages(request: Request):
    role = request.session.get("user_role")
    user_id = request.session.get("user_id")

    if not role or role != "premium_user":
        return RedirectResponse(url="/login", status_code=303)
    
    admin_defaults = {"technical": 40, "sentiment": 30, "financial": 30}
    try:
        admin_response = supabase.table("weightages").select(
            "technical, sentiment, financial"
        ).eq("id", "1").execute()
        if admin_response.data:
            admin_defaults = admin_response.data[0]
    except Exception as e:
        print(f"Database error fetching admin defaults: {e}")

    try:
        db_response = supabase.table("weightages").select(
            "technical, sentiment, financial"
        ).eq("user_id", user_id).execute()
        
        user_weights = db_response.data[0] if db_response.data else None
    except Exception as e:
        print(f"Database error fetching user weights: {e}")
        user_weights = None

    return templates.TemplateResponse(
        request=request, 
        name="user_model_weightage.html",
        context={"request": request, "weights": user_weights, "defaults": admin_defaults}
    )

@router.post("/premium/weightages")
async def save_premium_weightages(
    request: Request,
    technical: int = Form(...),
    sentiment: int = Form(...),
    financial: int = Form(...),
):
    role = request.session.get("user_role")
    user_id = request.session.get("user_id")

    if not role or role != "premium_user":
        return RedirectResponse(url="/login", status_code=303)

    total = technical + sentiment + financial
    if total != 100:
        return RedirectResponse(url="/premium/weightages", status_code=303)

    try:
        payload = {
            "user_id": user_id,
            "technical": technical,
            "sentiment": sentiment,
            "financial": financial
        }
        
        # This properly creates a row if it doesn't exist, and updates if it does!
        supabase.table("weightages").upsert(payload).execute()
        
    except Exception as e:
        print(f"Database error saving weightages: {e}")

    return RedirectResponse(url="/premium/weightages", status_code=303)
