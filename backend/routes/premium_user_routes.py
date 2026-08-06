import os
from pathlib import Path
from dotenv import load_dotenv
import uuid
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from datetime import date

from pydantic import BaseModel
try:
    from snaptrade_client import SnapTrade, SnapTradeAuth
    from snaptrade_client.exceptions import ApiException
except ImportError:
    SnapTrade = None
    SnapTradeAuth = None

    class ApiException(Exception):
        """Fallback type used when the optional SnapTrade SDK is unavailable."""

from typing import Optional

import asyncio

from backend.services.stock_list_service import get_active_stocks
from backend.services.prediction_service import get_latest_prediction_by_symbol, get_technical_score, get_financial_score
from backend.services.sentiment.sentiment_aggregator import get_weighted_sentiment_score, get_sentiment_summary
from backend.services.user_watchlist_service import (
    add_watchlist_by_symbol,
    get_user_watchlist_symbols,
    remove_watchlist_by_symbol,
)
from backend.database.supabase_client import supabase
from backend.services.session_context import get_session_context

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path, override=True)

client_id = os.environ.get("SNAPTRADE_CLIENT_ID")
consumer_key = os.environ.get("SNAPTRADE_CONSUMER_KEY")

snaptrade = None
if SnapTrade is not None and client_id and consumer_key:
    snaptrade = SnapTrade(
        auth=SnapTradeAuth.commercial_api_key(
            client_id=client_id,
            consumer_key=consumer_key,
        )
    )


def _require_snaptrade_client():
    if snaptrade is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Broker integration is unavailable. Install the SnapTrade Python "
                "SDK and configure SNAPTRADE_CLIENT_ID and SNAPTRADE_CONSUMER_KEY."
            ),
        )
    return snaptrade

class ConnectRequest(BaseModel):
    user_id: str

class TradeRequest(BaseModel):
    user_id: str
    account_id: str
    action: str  
    symbol: str  
    units: float 
    price: Optional[float] = None

@router.get("/premium/recommendations")
async def premium_recommendations(request: Request):
    # Session
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
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
                "company_name": stock.get("company_name", "Unknown Company"),
                "action": pred.get("action", "HOLD").upper(),
                "target_price": f"{float(pred.get('target_price', 0)):.2f}",
                "confidence": pred.get("confidence_score", 0),
                "rationale": pred.get("rationale", "Standard market conditions apply.")
            })

    return templates.TemplateResponse(
        request=request, 
        name="premium_users/stock_recommendations.html",
        context={
            **session,
            "request": request,
            "recommendations": display_recommendations
        }
    )

@router.get("/premium/prediction_breakdown")
async def premium_prediction_breakdown(request: Request, symbol: str = "NVDA"):
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        return RedirectResponse(url="/login", status_code=303)
    user_id = session["user_id"]

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
    sentiment_date = date(2026, 6, 10)
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
        name="premium_users/prediction_breakdown.html",
        context={
            **session,
            "request": request,
            "data": display_data
        }
    )

@router.get("/premium/weightages")
async def premium_user_weightages(request: Request):
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        return RedirectResponse(url="/login", status_code=303)
    user_id = session["user_id"]

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
        name="premium_users/user_model_weightage.html",
        context={**session, "request": request, "weights": user_weights, "defaults": admin_defaults}
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
        
        existing_response = (
            supabase.table("weightages")
            .select("id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if existing_response.data:
            (
                supabase.table("weightages")
                .update(payload)
                .eq("user_id", user_id)
                .execute()
            )
        else:
            supabase.table("weightages").insert(payload).execute()

    except Exception as e:
        print(f"Database error saving weightages: {e}")

    return RedirectResponse(url="/premium/weightages", status_code=303)

@router.get("/premium/news/{symbol}")
async def premium_news_feed(request: Request, symbol: str, days: int = 7):
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        return RedirectResponse(url="/dashboard", status_code=303)

    target_symbol = symbol.upper()
    summary = get_sentiment_summary(target_symbol, days=days)

    # Resolve company name from active stocks list
    stocks = get_active_stocks()
    company_name = next(
        (s.get("company_name", target_symbol) for s in stocks if s.get("symbol", "").upper() == target_symbol),
        target_symbol,
    )

    return templates.TemplateResponse(
        request=request,
        name="premium_users/news_feed.html",
        context={
            **session,
            "request": request,
            "symbol": target_symbol,
            "company_name": company_name,
            "summary": summary,
        },
    )


def _require_premium_session(request: Request) -> str:
    role = request.session.get("user_role")
    user_id = request.session.get("user_id")
    if role != "premium_user":
        raise HTTPException(status_code=403, detail="Premium access required")
    return user_id


@router.post("/premium/watchlist/{symbol}")
async def add_premium_watchlist_stock(symbol: str, request: Request):
    user_id = _require_premium_session(request)

    try:
        add_watchlist_by_symbol(user_id, symbol)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

    return {"symbol": symbol.upper(), "watchlisted": True}


@router.delete("/premium/watchlist/{symbol}")
async def remove_premium_watchlist_stock(symbol: str, request: Request):
    user_id = _require_premium_session(request)

    try:
        remove_watchlist_by_symbol(user_id, symbol)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

    return {"symbol": symbol.upper(), "watchlisted": False}


@router.get("/premium/watchlist/symbols")
async def view_premium_watchlist_symbols(request: Request):
    user_id = _require_premium_session(request)
    return {"symbols": get_user_watchlist_symbols(user_id)}

# ==========================================
# Connect Broker & Trading Routes
# ==========================================

@router.post("/api/broker/connect")
async def generate_connection_link(req: ConnectRequest):
    snaptrade_client = _require_snaptrade_client()
    user_id = req.user_id
    user_secret = None

    # 1. Check if user already has a saved SnapTrade secret in user_profiles
    try:
        user_res = supabase.table("user_profiles").select("snaptrade_secret").eq("user_id", user_id).execute()
        if not user_res.data:
            user_res = supabase.table("user_profiles").select("snaptrade_secret").eq("id", user_id).execute()

        if user_res.data and user_res.data[0].get("snaptrade_secret"):
            user_secret = user_res.data[0]["snaptrade_secret"]
    except Exception as e:
        print(f"Supabase fetch error: {e}")

    # 2. If no secret exists in DB, register user with SnapTrade & save the secret
    if not user_secret:
        try:
            register_res = snaptrade_client.authentication.register_snap_trade_user(
                body={"userId": user_id}
            )
            user_secret = register_res.body["userSecret"]
        except ApiException as e:
            # FAILSAFE: If the user already exists on SnapTrade but not in DB, reset them!
            if e.status == 400:
                snaptrade_client.authentication.delete_snap_trade_user(user_id=user_id)
                register_res = snaptrade_client.authentication.register_snap_trade_user(
                    body={"userId": user_id}
                )
                user_secret = register_res.body["userSecret"]
            else:
                raise HTTPException(status_code=400, detail=f"SnapTrade registration failed: {e.body}")

        # Save the generated secret to user_profiles
        try:
            supabase.table("user_profiles").update({"snaptrade_secret": user_secret}).eq("user_id", user_id).execute()
        except Exception:
            supabase.table("user_profiles").update({"snaptrade_secret": user_secret}).eq("id", user_id).execute()

    # 3. Generate the portal link using the valid user_secret
    try:
        login_res = snaptrade_client.authentication.login_snap_trade_user(
            user_id=user_id,
            user_secret=user_secret,
            connection_type="trade"
        )
        return {"redirect_url": login_res.body["redirectURI"]}
    except ApiException as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate connection link: {e.body}")

@router.get("/api/broker/accounts/{user_id}")
async def get_user_accounts(user_id: str):
    """Fetches all brokerage accounts linked to this user."""
    snaptrade_client = _require_snaptrade_client()
    # Retrieve secret from DB
    try:
        user_res = supabase.table("user_profiles").select("snaptrade_secret").eq("id", user_id).execute()
        if not user_res.data or not user_res.data[0].get("snaptrade_secret"):
            return {"connected": False, "accounts": []}

        user_secret = user_res.data[0]["snaptrade_secret"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    # Query SnapTrade for connected accounts
    try:
        accounts_res = snaptrade_client.account_information.list_user_accounts(
            user_id=user_id,
            user_secret=user_secret
        )
        return {"connected": True, "accounts": accounts_res.body}
    except ApiException as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve accounts: {e.body}")


@router.get("/api/broker/holdings/{user_id}")
async def get_user_holdings(user_id: str):
    """Fetches holdings across connected accounts."""
    snaptrade_client = _require_snaptrade_client()
    try:
        user_res = supabase.table("user_profiles").select("snaptrade_secret").eq("id", user_id).execute()
        if not user_res.data or not user_res.data[0].get("snaptrade_secret"):
            raise HTTPException(status_code=404, detail="No connected broker found.")

        user_secret = user_res.data[0]["snaptrade_secret"]

        accounts_res = snaptrade_client.account_information.list_user_accounts(
            user_id=user_id,
            user_secret=user_secret
        )

        if not accounts_res.body:
            return {"holdings": []}

        account_id = accounts_res.body[0]["id"]
        positions_res = snaptrade.account_information.get_user_account_positions(
            account_id=account_id,
            user_id=user_id,
            user_secret=user_secret
        )
        return positions_res.body
    except ApiException as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve holdings: {e.body}")
    
@router.post("/api/broker/trade")
async def execute_trade(req: TradeRequest):
    # 1. Fetch the user secret from Supabase
    try:
        user_res = supabase.table("user_profiles").select("snaptrade_secret").eq("id", req.user_id).execute()
            
        if not user_res.data or not user_res.data[0].get("snaptrade_secret"):
            raise HTTPException(status_code=400, detail="Broker not connected.")
            
        user_secret = user_res.data[0]["snaptrade_secret"]
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Real DB Error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # 2. Look up the universal_symbol_id for the requested stock
    try:
        symbol_res = snaptrade.reference_data.get_symbols_by_ticker(
            query=req.symbol.upper()
        )
    except ApiException as e:
        # If we hit the short-term burst limit, pause and retry automatically
        if e.status == 429:
            print("SnapTrade burst rate limit hit. Pausing for 2.5 seconds...")
            await asyncio.sleep(2.5) # Sleep just slightly longer than the requested 2 seconds
            
            # Retry the exact same request
            symbol_res = snaptrade.reference_data.get_symbols_by_ticker(
                query=req.symbol.upper()
            )
        else:
            raise HTTPException(status_code=400, detail=f"Symbol Lookup Failed: {str(e.body)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

    # Extract the UUID securely
    if isinstance(symbol_res.body, list) and len(symbol_res.body) > 0:
        u_symbol_id = symbol_res.body[0]["id"]
    elif isinstance(symbol_res.body, dict) and "id" in symbol_res.body:
        u_symbol_id = symbol_res.body["id"]
    else:
        raise HTTPException(status_code=400, detail="Unexpected symbol format returned from SnapTrade")

    # 3. Determine order type
    order_type = "Limit" if req.price else "Market"
    
    # 4. Build the order arguments
    order_kwargs = {
        "user_id": req.user_id,
        "user_secret": user_secret,
        "account_id": req.account_id,
        "action": req.action.upper(),
        "universal_symbol_id": u_symbol_id, 
        "order_type": order_type,
        "time_in_force": "Day",
        "units": req.units
    }
    
    if req.price:
        order_kwargs["price"] = req.price

    # 5. Place the trade
    try:
        trade_res = snaptrade.trading.place_force_order(**order_kwargs)
        return {
            "status": "success", 
            "message": "Order placed successfully!",
            "trade_details": trade_res.body
        }
    except ApiException as e:
        raise HTTPException(status_code=400, detail=f"Trade Failed: {str(e.body)}")

@router.get("/api/broker/position/{user_id}/{symbol}")
async def get_specific_position(user_id: str, symbol: str):
    """Fetches user holding details for a specific stock symbol."""
    try:
        user_res = supabase.table("user_profiles").select("snaptrade_secret").eq("id", user_id).execute()
        if not user_res.data or not user_res.data[0].get("snaptrade_secret"):
            return {"has_position": False}
            
        user_secret = user_res.data[0]["snaptrade_secret"]
        
        accounts_res = snaptrade.account_information.list_user_accounts(
            user_id=user_id,
            user_secret=user_secret
        )
        
        if not accounts_res.body:
            return {"has_position": False}

        # Scan accounts for holdings matching the target symbol
        for account in accounts_res.body:
            account_id = account["id"]
            
            positions_res = snaptrade.account_information.get_user_account_positions(
                account_id=account_id,
                user_id=user_id,
                user_secret=user_secret
            )
            
            positions = positions_res.body
            for pos in positions:
                # 1. Grab the outer 'symbol' dict (Brokerage Symbol)
                brokerage_sym = pos.get("symbol", {})
                
                # 2. Grab the inner 'symbol' dict (Universal Symbol)
                universal_sym = brokerage_sym.get("symbol", {})
                
                # 3. Extract the actual string ticker safely
                ticker = ""
                if isinstance(universal_sym, dict):
                    ticker = universal_sym.get("symbol", "")
                elif isinstance(universal_sym, str):
                    ticker = universal_sym
                
                # 4. Compare the extracted string
                if ticker and ticker.upper() == symbol.upper():
                    return {
                        "has_position": True,
                        "account_name": account.get("name"),
                        "units": pos.get("units"),
                        "average_price": pos.get("average_purchase_price"),
                        "current_price": pos.get("price")
                    }
                    
        return {"has_position": False}
    except Exception as e:
        print(f"Error fetching position: {e}")
        return {"has_position": False}
