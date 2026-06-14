import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
MODULE_WEIGHTED = "backend.routes.stock_routes.get_weighted_sentiment_score"
MODULE_PIPE = "backend.routes.stock_routes.run_sentiment_pipeline"
MODULE_STOCK = "backend.routes.stock_routes.get_stock_by_symbol"
MODULE_SAVE_DAILY = "backend.routes.stock_routes.save_daily_sentiment_score"

MOCK_SUMMARY = {
    "id": "score-1",
    "symbol": "AAPL",
    "score_date": "2026-05-24",
    "bullish_score": 7.2,
    "sentiment_label": "bullish",
}

MOCK_PIPELINE_RESULT = {
    "message": "Pipeline complete",
    "symbols_processed": 10,
    "results": [{"symbol": "AAPL", "headlines_scored": 5, "status": "ok"}],
}


@patch(MODULE_WEIGHTED, return_value=MOCK_SUMMARY)
@patch(MODULE_STOCK, return_value=[{"id": 1, "symbol": "AAPL"}])
def test_get_sentiment_200(mock_stock, mock_weighted):
    response = client.get("/api/stocks/AAPL/sentiment?score_date=2026-05-24")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["score_date"] == "2026-05-24"
    assert data["sentiment"]["bullish_score"] == 7.2
    assert "daily_scores" not in data
    assert "headlines" not in data


@patch(MODULE_WEIGHTED, return_value=None)
@patch(MODULE_STOCK, return_value=[{"id": 1, "symbol": "AAPL"}])
def test_get_sentiment_404_when_no_data(mock_stock, mock_weighted):
    response = client.get("/api/stocks/AAPL/sentiment")
    assert response.status_code == 404


@patch(MODULE_WEIGHTED, return_value=MOCK_SUMMARY)
@patch(MODULE_STOCK, return_value=[{"id": 1, "symbol": "AAPL"}])
def test_get_sentiment_symbol_uppercased(mock_stock, mock_weighted):
    response = client.get("/api/stocks/aapl/sentiment?score_date=2026-05-24")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"


@patch(MODULE_WEIGHTED, return_value={**MOCK_SUMMARY, "symbol": "TSLA"})
@patch(MODULE_STOCK, return_value=[{"id": 3, "symbol": "TSLA"}])
def test_get_sentiment_response_includes_symbol(mock_stock, mock_weighted):
    response = client.get("/api/stocks/TSLA/sentiment?score_date=2026-05-24")
    assert response.json()["symbol"] == "TSLA"


@patch(MODULE_SAVE_DAILY, return_value={"rows_saved": 1, "daily_score": {"bullish_score": 7.2}})
@patch(MODULE_STOCK, return_value=[{"id": 1, "symbol": "AAPL"}])
def test_create_daily_sentiment_score_200(mock_stock, mock_save_daily):
    response = client.post("/api/stocks/AAPL/sentiment/daily-score?score_date=2026-05-24")

    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["score_date"] == "2026-05-24"
    assert data["rows_saved"] == 1
    assert data["daily_score"]["bullish_score"] == 7.2


@patch(MODULE_STOCK, return_value=[])
def test_create_daily_sentiment_score_404_unknown_stock(mock_stock):
    response = client.post("/api/stocks/XXXX/sentiment/daily-score?score_date=2026-05-24")

    assert response.status_code == 404


@patch(MODULE_PIPE, return_value=MOCK_PIPELINE_RESULT)
def test_run_pipeline_200(mock_pipe):
    response = client.post("/api/sentiment/run-pipeline")
    assert response.status_code == 200
    data = response.json()
    assert "symbols_processed" in data
    assert "results" in data


@patch(MODULE_PIPE, side_effect=RuntimeError("FinBERT failed to load"))
def test_run_pipeline_500_on_error(mock_pipe):
    response = client.post("/api/sentiment/run-pipeline")
    assert response.status_code == 500
