"""
backend/routes/financial_routes.py
API routes for financial predictions.
Exposes endpoints to get latest predictions and history.
"""

from backend.services.prediction_service import (
    get_latest_predictions,
    get_predictions_by_outlook,
    get_predictions_history_by_ticker,
)
from backend.services.financials_service import load_statements_by_ticker


def get_latest():
    """GET /financial/predictions — latest prediction per ticker."""
    df = get_latest_predictions()
    return df.to_dict(orient="records")


def get_by_outlook(prediction: str):
    """GET /financial/predictions/{prediction} — filter by positive/neutral/negative."""
    df = get_predictions_by_outlook(prediction)
    return df.to_dict(orient="records")


def get_history(ticker: str):
    """GET /financial/predictions/history/{ticker} — full history for a ticker."""
    df = get_predictions_history_by_ticker(ticker)
    return df.to_dict(orient="records")


def get_statements(ticker: str):
    """GET /financial/statements/{ticker} — raw financial statements for a ticker."""
    df = load_statements_by_ticker(ticker)
    return df.to_dict(orient="records")
