import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
MODULE_AGG = "routes.stock_routes.get_sentiment_summary"
MODULE_PIPE = "routes.stock_routes.run_sentiment_pipeline"

MOCK_SUMMARY = {
    "daily_scores": [{"date": "2026-05-24", "avg_score": 0.75, "label": "positive", "headline_count": 5}],
    "headlines": [{"headline": "Apple profits up", "source": "finnhub", "published_at": "2026-05-24T09:00:00Z", "label": "positive", "score": 0.91}],
}

MOCK_PIPELINE_RESULT = {
    "message": "Pipeline complete",
    "symbols_processed": 10,
    "results": [{"symbol": "AAPL", "headlines_scored": 5, "status": "ok"}],
}


@patch(MODULE_AGG, return_value=MOCK_SUMMARY)
def test_get_sentiment_200(mock_agg):
    response = client.get("/api/stocks/AAPL/sentiment")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "daily_scores" in data
    assert "headlines" in data


@patch(MODULE_AGG, return_value={"daily_scores": [], "headlines": []})
def test_get_sentiment_404_when_no_data(mock_agg):
    response = client.get("/api/stocks/AAPL/sentiment")
    assert response.status_code == 404


@patch(MODULE_AGG, return_value=MOCK_SUMMARY)
def test_get_sentiment_symbol_uppercased(mock_agg):
    response = client.get("/api/stocks/aapl/sentiment")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"


@patch(MODULE_AGG, return_value=MOCK_SUMMARY)
def test_get_sentiment_response_includes_symbol(mock_agg):
    response = client.get("/api/stocks/TSLA/sentiment")
    assert response.json()["symbol"] == "TSLA"


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
