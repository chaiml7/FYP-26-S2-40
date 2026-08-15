import os
from pathlib import Path
from urllib.parse import urlencode
from dotenv import load_dotenv
import uuid
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from datetime import date, timedelta
import httpx

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
from backend.services.prediction_service import (
    get_effective_weights,
    build_composite_scorecard,
    get_latest_technical_signals,
    get_latest_direction_outlook,
    risk_tier_from_volatility,
)
from backend.services.stock_history_service import get_recent_stock_history
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
async def premium_recommendations(request: Request, risk: str = None):
    # Session
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        return RedirectResponse(url="/login", status_code=303)

    risk_filter = risk.lower() if risk else None

    # The page shell renders immediately; the recommendation cards are
    # fetched asynchronously from /premium/recommendations/data because
    # scoring every active stock takes several seconds (see that route).
    return templates.TemplateResponse(
        request=request,
        name="premium_users/stock_recommendations.html",
        context={
            **session,
            "request": request,
            "active_risk": risk_filter,
        }
    )


@router.get("/premium/recommendations/data")
async def premium_recommendations_data(request: Request, risk: str = None):
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        raise HTTPException(status_code=403, detail="Premium access required")

    user_id = session["user_id"]
    weights = get_effective_weights(user_id)
    active_stocks = get_active_stocks()
    risk_filter = risk.lower() if risk else None

    # Build each recommendation from the same real technical/sentiment/
    # financial scoring pipeline that Prediction Breakdown uses, rather than
    # the old `predictions` table (which nothing in the ML pipeline writes to).
    display_recommendations = []
    for stock in active_stocks:
        symbol = stock.get("symbol", "").upper()

        scorecard = build_composite_scorecard(symbol, weights)
        signals = get_latest_technical_signals(symbol)
        risk_slug, risk_label = risk_tier_from_volatility(
            signals.get("rolling_volatility_20") if signals else None
        )

        if risk_filter and risk_slug != risk_filter:
            continue

        display_recommendations.append({
            "ticker": symbol,
            "company_name": stock.get("company_name", "Unknown Company"),
            "action": scorecard["action"],
            "composite": scorecard["composite"],
            "rationale": (
                f"Technical {scorecard['technical_score']}/10 · "
                f"Sentiment {scorecard['sentiment_score']}/10 · "
                f"Financial {scorecard['financial_score']}/10 "
                f"(weighted {scorecard['tech_weight']}/{scorecard['sent_weight']}/{scorecard['fin_weight']})"
            ),
            "risk_slug": risk_slug,
            "risk_label": risk_label,
        })

    display_recommendations.sort(key=lambda rec: rec["composite"], reverse=True)

    return {"recommendations": display_recommendations}

def _build_signal_breakdown(signals_row: dict, sentiment_score: float, outlook: dict) -> dict:
    """Format raw indicator/outlook rows into the display strings the
    Signal Breakdown panel needs. Every field here traces back to a real
    stored value; rows with no backing data source were dropped rather than
    filled in with placeholders."""

    def _fmt_pct(value, decimals=0):
        return f"{value:+.{decimals}f}%"

    rsi = signals_row.get("rsi_14") if signals_row else None
    relative_volume = signals_row.get("relative_volume") if signals_row else None
    macd = signals_row.get("macd") if signals_row else None
    macd_signal = signals_row.get("macd_signal") if signals_row else None
    return_5d = signals_row.get("return_5d") if signals_row else None

    if rsi is not None:
        rsi_display = f"{float(rsi):.1f}"
    else:
        rsi_display = "No data"

    if relative_volume is not None:
        volume_pct = (float(relative_volume) - 1) * 100
        volume_display = f"{_fmt_pct(volume_pct)} vs 20D avg"
        volume_positive = volume_pct > 0
    else:
        volume_display = "No data"
        volume_positive = None

    if macd is not None and macd_signal is not None:
        macd_diff = float(macd) - float(macd_signal)
        if macd_diff > 0:
            macd_display = "Bullish crossover"
        elif macd_diff < 0:
            macd_display = "Bearish crossover"
        else:
            macd_display = "Flat"
        macd_positive = macd_diff > 0
    else:
        macd_display = "No data"
        macd_positive = None

    if return_5d is not None:
        momentum_pct = float(return_5d) * 100
        momentum_display = f"{_fmt_pct(momentum_pct, 1)} (5D)"
        momentum_positive = momentum_pct > 0
    else:
        momentum_display = "No data"
        momentum_positive = None

    if sentiment_score is not None:
        if sentiment_score >= 6:
            sentiment_display = f"Bullish ({sentiment_score}/10)"
            sentiment_positive = True
        elif sentiment_score < 4:
            sentiment_display = f"Bearish ({sentiment_score}/10)"
            sentiment_positive = False
        else:
            sentiment_display = f"Neutral ({sentiment_score}/10)"
            sentiment_positive = None
    else:
        sentiment_display = "No data"
        sentiment_positive = None

    if outlook and outlook.get("prediction"):
        prediction_label = outlook["prediction"]
        probabilities = outlook.get("probabilities") or {}
        # The model is a binary up/down classifier — "neutral" just means
        # neither side cleared the confidence threshold, so probabilities["neutral"]
        # is always 0 by construction. Show the bullish probability instead,
        # since that's the number that actually explains the neutral call.
        if prediction_label == "neutral":
            bullish_probability = probabilities.get("bullish")
            if bullish_probability is not None:
                outlook_display = f"Neutral · {bullish_probability * 100:.0f}% bullish probability"
            else:
                outlook_display = "Neutral"
        else:
            confidence = probabilities.get(prediction_label)
            if confidence is not None:
                outlook_display = f"{prediction_label.title()} · {confidence * 100:.0f}% confidence"
            else:
                outlook_display = prediction_label.title()
        if prediction_label == "bullish":
            outlook_positive = True
        elif prediction_label == "bearish":
            outlook_positive = False
        else:
            outlook_positive = None
    else:
        outlook_display = "No data"
        outlook_positive = None

    return {
        "rsi": rsi_display,
        "volume": volume_display,
        "volume_positive": volume_positive,
        "macd": macd_display,
        "macd_positive": macd_positive,
        "momentum": momentum_display,
        "momentum_positive": momentum_positive,
        "sentiment": sentiment_display,
        "sentiment_positive": sentiment_positive,
        "outlook": outlook_display,
        "outlook_positive": outlook_positive,
    }


def _build_price_chart(actual_prices: list) -> dict:
    """SVG polyline `d` string plus per-point coordinates (for hover markers)
    plotting real close prices on a 1000x100 viewBox.

    Returns {"path": None, "points": []} when there isn't enough history to
    draw a meaningful line (the old version drew the exact same fake curve
    regardless of data).
    """
    if not actual_prices or len(actual_prices) < 2:
        return {"path": None, "points": []}

    closes = [point["close"] for point in actual_prices]
    low, high = min(closes), max(closes)
    span = (high - low) or 1

    count = len(closes)
    coords = []
    for index, point in enumerate(actual_prices):
        x = (index / (count - 1)) * 1000
        # SVG y grows downward, so flip the normalized value.
        y = 90 - ((point["close"] - low) / span) * 80
        coords.append({"x": round(x, 1), "y": round(y, 1), "date": point["date"], "close": point["close"]})

    path = "M " + " L ".join(f"{c['x']} {c['y']}" for c in coords)
    return {"path": path, "points": coords}


@router.get("/premium/prediction_breakdown")
async def premium_prediction_breakdown(request: Request, symbol: str = "NVDA"):
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        return RedirectResponse(url="/login", status_code=303)
    user_id = session["user_id"]

    target_symbol = symbol.upper()
    weights = get_effective_weights(user_id)
    scorecard = build_composite_scorecard(target_symbol, weights)

    signals_row = get_latest_technical_signals(target_symbol)
    outlook = get_latest_direction_outlook(target_symbol)
    signals = _build_signal_breakdown(signals_row, scorecard["sentiment_score"], outlook)

    price_history = get_recent_stock_history(target_symbol, limit=30)
    actual_prices = [
        {"date": row["trade_date"], "close": float(row["close"])}
        for row in price_history
        if row.get("close") is not None
    ]

    price_chart = _build_price_chart(actual_prices)

    display_data = {
        **scorecard,
        "signals": signals,
        "actual_prices": actual_prices,
        "actual_path": price_chart["path"],
        "actual_points": price_chart["points"],
    }

    return templates.TemplateResponse(
        request=request,
        name="premium_users/prediction_breakdown.html",
        context={
            **session,
            "request": request,
            "data": display_data,
            "stocks": get_active_stocks(),
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

    # Preserve old bookmarks while using the same filtered feed as the
    # working News & Social page.
    target_symbol = symbol.strip().upper()
    return RedirectResponse(
        url=f"/user/news_social?{urlencode({'symbol': target_symbol})}",
        status_code=303,
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
async def generate_connection_link(req: ConnectRequest, request: Request):

    role = request.session.get("user_role")
    if role != "premium_user":
        raise HTTPException(status_code=403, detail="Trading is restricted to Premium accounts.")
    
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
async def get_user_accounts(user_id: str, request: Request):

    role = request.session.get("user_role")
    if role != "premium_user":
        raise HTTPException(status_code=403, detail="Trading is restricted to Premium accounts.")

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
async def execute_trade(req: TradeRequest, request: Request):
    role = request.session.get("user_role")
    if role != "premium_user":
        raise HTTPException(status_code=403, detail="Trading is restricted to Premium accounts.")
    
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
async def get_specific_position(user_id: str, symbol: str, request: Request):

    role = request.session.get("user_role")
    if role != "premium_user":
        raise HTTPException(status_code=403, detail="Trading is restricted to Premium accounts.")
    
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

@router.get("/premium/earnings_calendar")
async def premium_earnings_calendar(request: Request):
    # 1. Premium Check
    session = get_session_context(request)
    if not session or session["user_role"] != "premium_user":
        return RedirectResponse(url="/dashboard", status_code=303)

    # 2. Fetch API Key
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="FMP API Key is missing from .env")

    # 3. Fetch Earnings Data asynchronously for the next 30 days
    today = date.today()
    end_date = today + timedelta(days=30)

    url = f"https://financialmodelingprep.com/stable/earnings-calendar?from={today}&to={end_date}&apikey={api_key}"
    
    tracked_symbols = {stock.get("symbol", "").upper() for stock in get_active_stocks()}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)

            if response.status_code != 200:
                print(f"--- FMP API FAILED ---")
                print(f"Status Code: {response.status_code}")
                print(f"Error Details: {response.text}")
                print(f"Attempted URL: {url.replace(api_key, 'HIDDEN_KEY')}")
                earnings_data = []
            else:
                # FMP's earnings-calendar endpoint returns the entire market.
                # Scope it to StockLens' tracked stocks instead of showing 50
                # random unrelated tickers, and filter BEFORE truncating so
                # the cap doesn't get exhausted by untracked symbols.
                all_earnings = response.json()
                earnings_data = [
                    item for item in all_earnings
                    if item.get("symbol", "").upper() in tracked_symbols
                ][:50]
        except Exception as e:
            print(f"--- FMP REQUEST CRASHED ---")
            print(f"Error: {e}")
            earnings_data = []

    # 4. Render the template
    return templates.TemplateResponse(
        request=request,
        name="premium_users/earnings_calendar.html",
        context={
            **session,
            "request": request,
            "earnings": earnings_data,
            "tracked_count": len(tracked_symbols),
        }
    )
