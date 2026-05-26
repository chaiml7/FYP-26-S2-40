from fastapi import APIRouter, HTTPException
from services.stock_list_service import (
    get_active_stocks,
    get_stock_by_symbol,
    add_stock,
    deactivate_stock,
    update_last_imported_at
)
from services.yfinance_service import fetch_stock_history
from services.stock_history_service import save_stock_history, get_stock_history
from services.sentiment.sentiment_aggregator import get_sentiment_summary
from services.sentiment.sentiment_pipeline import run_pipeline as run_sentiment_pipeline

router = APIRouter()


@router.get("/stocks")
def view_active_stocks():
    return get_active_stocks()


@router.post("/stocks")
def create_stock(stock_data: dict):
    return add_stock(stock_data)


@router.patch("/stocks/{symbol}/deactivate")
def remove_stock_from_tracking(symbol: str):
    return deactivate_stock(symbol)


@router.post("/stocks/import")
def import_all_active_stocks(period: str = "6mo", interval: str = "1d"):
    stocks = get_active_stocks()

    if len(stocks) == 0:
        raise HTTPException(status_code=404, detail="No active stocks found")

    results = []

    for stock in stocks:
        symbol = stock["symbol"]

        rows = fetch_stock_history(symbol, period, interval)
        result = save_stock_history(rows)

        results.append({
            "symbol": symbol,
            "rows_imported": result["rows_saved"],
            "message": result["message"]
        })

    return {
        "message": "Import completed for active stocks",
        "stocks_processed": len(results),
        "results": results
    }


@router.post("/stocks/import/{symbol}")
def import_one_tracked_stock(symbol: str, period: str = "6mo", interval: str = "1d"):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    if stock[0].get("is_active") is False:
        raise HTTPException(
            status_code=400,
            detail=f"{symbol.upper()} is not active"
        )

    rows = fetch_stock_history(symbol, period, interval)

    if len(rows) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No yfinance data found for {symbol.upper()}"
        )

    result = save_stock_history(rows)

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "rows_imported": result["rows_saved"],
        "message": result["message"]
    }


@router.get("/stocks/{symbol}/history")
def view_stock_history(symbol: str):
    data = get_stock_history(symbol)

    if len(data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for {symbol.upper()}"
        )

    return data


@router.get("/stocks/{symbol}/sentiment")
def get_stock_sentiment(symbol: str):
    data = get_sentiment_summary(symbol)
    if not data["daily_scores"] and not data["headlines"]:
        raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol.upper()}")
    return {"symbol": symbol.upper(), **data}


@router.post("/sentiment/run-pipeline")
def trigger_sentiment_pipeline():
    try:
        return run_sentiment_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/stocks/import/{symbol}")
def import_one_tracked_stock(symbol: str, period: str = "6mo", interval: str = "1d"):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    stock_id = stock[0]["id"]

    rows = fetch_stock_history(stock_id, symbol, period, interval)
    result = save_stock_history(rows)

    if result["success"]:
        update_last_imported_at(symbol)

    return {
        "symbol": symbol.upper(),
        "rows_imported": result["rows_saved"],
        "message": result["message"]
    }