"""Notification routes for the bell icon."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from backend.database.supabase_client import supabase
from backend.services.user_watchlist_service import get_user_watchlist_summary

router = APIRouter()


@router.get("/user/notifications")
async def get_notifications(request: Request):
    role = request.session.get("user_role")
    user_id = request.session.get("user_id")
    notifications = []

    if role == "premium_user" and user_id:
        try:
            stocks = get_user_watchlist_summary(user_id)
            for stock in stocks:
                symbol = stock.get("symbol", "")
                signal = (stock.get("prediction_signal") or "NEUTRAL").upper()
                change_percent = stock.get("change_percent")

                # Prediction notification
                if signal in ["BUY", "STRONG BUY"]:
                    notifications.append({
                        "icon": "📈",
                        "title": f"{symbol} prediction is ready",
                        "message": f"Signal: {signal} — Check the prediction breakdown"
                    })
                elif signal in ["SELL", "STRONG SELL"]:
                    notifications.append({
                        "icon": "📉",
                        "title": f"{symbol} prediction is ready",
                        "message": f"Signal: {signal} — Consider reviewing your position"
                    })
                else:
                    notifications.append({
                        "icon": "📊",
                        "title": f"{symbol} prediction is ready",
                        "message": f"Signal: NEUTRAL — Market is uncertain"
                    })

                # Price movement notification
                if change_percent is not None:
                    if change_percent <= -3:
                        notifications.append({
                            "icon": "🔴",
                            "title": f"{symbol} dropped significantly",
                            "message": f"Price dropped {abs(change_percent):.1f}% today"
                        })
                    elif change_percent >= 3:
                        notifications.append({
                            "icon": "🟢",
                            "title": f"{symbol} surged today",
                            "message": f"Price up {change_percent:.1f}% today"
                        })

        except Exception as e:
            print(f"Error loading premium notifications: {e}")

    else:
        # Free user — show general market notifications
        try:
            response = (
                supabase
                .table("stocks")
                .select("symbol")
                .limit(5)
                .execute()
            )
            stocks = response.data or []
            for stock in stocks[:3]:
                symbol = stock.get("symbol", "")
                notifications.append({
                    "icon": "📊",
                    "title": f"{symbol} prediction is ready",
                    "message": "Upgrade to Premium to see your personalized watchlist predictions"
                })

            notifications.append({
                "icon": "⭐",
                "title": "Unlock personalized alerts",
                "message": "Upgrade to Premium to get watchlist notifications"
            })

        except Exception as e:
            print(f"Error loading free user notifications: {e}")

    return JSONResponse({"notifications": notifications})
