from datetime import date

from fastapi import APIRouter, HTTPException
from services.stock_list_service import (
    get_all_stocks,
    get_active_stocks,
    get_inactive_stocks,
    get_stocks_by_sector,
    get_stock_by_symbol,
    add_stock,
    update_stock,
    deactivate_stock,
    update_last_imported_at
)
from services.yfinance_service import fetch_stock_history
from services.stock_history_service import (
    save_stock_history,
    get_stock_history,
    get_latest_stock_price,
    get_stock_history_by_date_range,
    delete_stock_history
)
from services.sentiment.sentiment_aggregator import (
    get_weighted_sentiment_score,
    save_daily_sentiment_score,
)
from services.sentiment.sentiment_pipeline import run_pipeline as run_sentiment_pipeline
from services.prediction_service import (
    save_prediction,
    get_predictions_by_symbol,
    get_latest_prediction_by_symbol
)
from schemas import StockCreate, StockUpdate, PredictionCreate

router = APIRouter()


def _payload(model, exclude_none: bool = False):
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=exclude_none, mode="json")
    return model.dict(exclude_none=exclude_none)


@router.get("/stocks")
def view_active_stocks():
    return get_active_stocks()


@router.get("/stocks/all")
def view_all_stocks():
    return get_all_stocks()


@router.get("/stocks/inactive")
def view_inactive_stocks():
    return get_inactive_stocks()


@router.get("/stocks/sector/{sector}")
def view_stocks_by_sector(sector: str):
    return get_stocks_by_sector(sector)


@router.post("/stocks")
def create_stock(stock_data: StockCreate):
    return add_stock(_payload(stock_data))


@router.get("/stocks/{symbol}")
def view_stock_by_symbol(symbol: str):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    return stock[0]


@router.patch("/stocks/{symbol}")
def edit_stock(symbol: str, stock_data: StockUpdate):
    payload = _payload(stock_data, exclude_none=True)

    if len(payload) == 0:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    updated = update_stock(symbol, payload)

    if len(updated) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    return updated[0]


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
        stock_id = stock["id"]
        symbol = stock["symbol"]

        rows = fetch_stock_history(stock_id, symbol, period, interval)
        result = save_stock_history(rows)

        if result["success"]:
            update_last_imported_at(symbol)

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

    stock_id = stock[0]["id"]

    rows = fetch_stock_history(stock_id, symbol, period, interval)

    if len(rows) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No yfinance data found for {symbol.upper()}"
        )

    result = save_stock_history(rows)

    if result["success"]:
        update_last_imported_at(symbol)

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


@router.get("/stocks/{symbol}/history/latest")
def view_latest_stock_price(symbol: str):
    data = get_latest_stock_price(symbol)

    if len(data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for {symbol.upper()}"
        )

    return data[0]


@router.get("/stocks/{symbol}/history/range")
def view_stock_history_by_date_range(symbol: str, start_date: date, end_date: date):
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be before or equal to end_date"
        )

    data = get_stock_history_by_date_range(
        symbol,
        start_date.isoformat(),
        end_date.isoformat()
    )

    if len(data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for {symbol.upper()} in the selected date range"
        )

    return data


@router.delete("/stocks/{symbol}/history")
def remove_stock_history(symbol: str):
    deleted = delete_stock_history(symbol)

    return {
        "symbol": symbol.upper(),
        "rows_deleted": len(deleted),
        "message": "Stock history deleted"
    }


@router.post("/stocks/{symbol}/predictions")
def create_stock_prediction(symbol: str, prediction_data: PredictionCreate):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    payload = _payload(prediction_data, exclude_none=True)
    payload["symbol"] = symbol.upper()
    payload["stock_id"] = stock[0]["id"]

    result = save_prediction(payload) or []
    return result[0] if len(result) > 0 else payload


@router.get("/stocks/{symbol}/predictions")
def view_stock_predictions(symbol: str):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    return get_predictions_by_symbol(symbol)


@router.get("/stocks/{symbol}/predictions/latest")
def view_latest_stock_prediction(symbol: str):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    prediction = get_latest_prediction_by_symbol(symbol)

    if len(prediction) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {symbol.upper()}"
        )

    return prediction[0]


@router.get("/stocks/{symbol}/sentiment")
def get_stock_sentiment(symbol: str, score_date: date = None):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    selected_date = score_date or date.today()
    score = get_weighted_sentiment_score(symbol, selected_date)

    if score is None:
        raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol.upper()}")

    return {
        "symbol": symbol.upper(),
        "score_date": selected_date.isoformat(),
        "sentiment": score,
    }


@router.post("/stocks/{symbol}/sentiment/daily-score")
def create_stock_daily_sentiment_score(symbol: str, score_date: date = None):
    stock = get_stock_by_symbol(symbol)

    if len(stock) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the stocks table"
        )

    result = save_daily_sentiment_score(symbol, score_date)

    if result["rows_saved"] == 0:
        raise HTTPException(
            status_code=404,
            detail=result.get("reason", "No sentiment rows found")
        )

    return {
        "symbol": symbol.upper(),
        "score_date": score_date.isoformat() if score_date else date.today().isoformat(),
        **result,
    }


@router.post("/sentiment/run-pipeline")
def trigger_sentiment_pipeline():
    try:
        return run_sentiment_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
