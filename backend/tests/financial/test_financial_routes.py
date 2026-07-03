from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


@patch("backend.routes.financial_routes.import_financial_statements")
def test_import_financial_statements_route(mock_import):
    mock_import.return_value = {
        "symbol": "AAPL",
        "periods_received": 5,
        "rows_saved": 5,
        "skipped_periods": [],
    }

    response = client.post("/api/financial/statements/import/AAPL")

    assert response.status_code == 200
    assert response.json()["rows_saved"] == 5
    mock_import.assert_called_once_with("AAPL")


@patch("backend.routes.financial_routes.import_all_financial_statements")
def test_import_all_financial_statements_route(mock_import):
    mock_import.return_value = {
        "stocks_processed": 11,
        "stocks_imported": 10,
        "results": [],
    }

    response = client.post("/api/financial/statements/import")

    assert response.status_code == 200
    assert response.json()["stocks_processed"] == 11


@patch("backend.routes.financial_routes.train_financial_model")
def test_train_financial_model_route(mock_train):
    mock_train.return_value = {
        "saved_model": {
            "model_version": "xgboost_financial_binary_v1",
            "labels": ["bearish", "bullish"],
        },
        "activated_model": {"is_active": True},
    }

    response = client.post(
        "/api/financial/model/train",
        json={"top_n": 3},
    )

    assert response.status_code == 200
    assert response.json()["saved_model"]["labels"] == ["bearish", "bullish"]
    mock_train.assert_called_once_with(top_n=3)


@patch("backend.routes.financial_routes.tune_financial_model")
def test_tune_financial_model_route(mock_tune):
    mock_tune.return_value = {
        "dataset_rows": 413,
        "candidate_count": 20,
        "best_result": {
            "strategy": "strong_only",
            "latest_holdout_metrics": {"accuracy": 0.65},
        },
    }

    response = client.post(
        "/api/financial/model/tune",
        json={
            "top_n": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["candidate_count"] == 20
    mock_tune.assert_called_once_with(top_n=3)


@patch("backend.routes.financial_routes.generate_financial_prediction")
def test_generate_financial_prediction_route(mock_generate):
    mock_generate.return_value = {
        "ticker": "AAPL",
        "prediction": "neutral",
        "binary_prediction": "bullish",
        "fundamental_score": 5.4,
        "prediction_horizon": "next_quarter",
        "model_type": "binary_financial",
    }

    response = client.post(
        "/api/financial/predictions/aapl"
        "?model_version=xgboost_financial_binary_20260626T120000000000Z"
    )

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert response.json()["fundamental_score"] == 5.4
    assert response.json()["binary_prediction"] == "bullish"
    assert response.json()["prediction_horizon"] == "next_quarter"
    mock_generate.assert_called_once_with(
        "aapl",
        "xgboost_financial_binary_20260626T120000000000Z",
    )


@patch("backend.routes.financial_routes.generate_financial_prediction")
def test_generate_prediction_requires_trained_model(mock_generate):
    mock_generate.side_effect = FileNotFoundError("Train the model first.")

    response = client.post("/api/financial/predictions/AAPL")

    assert response.status_code == 409


@patch("backend.routes.financial_routes.read_latest_financial_prediction", return_value=None)
def test_latest_prediction_returns_404_when_missing(mock_latest):
    response = client.get("/api/financial/predictions/AAPL/latest")

    assert response.status_code == 404


@patch("backend.routes.financial_routes.read_latest_financial_prediction")
def test_latest_prediction_route(mock_latest):
    mock_latest.return_value = {
        "ticker": "AAPL",
        "prediction": "positive",
        "fundamental_score": 6.2,
        "model_type": "binary_financial",
    }

    response = client.get("/api/financial/predictions/AAPL/latest")

    assert response.status_code == 200
    assert response.json()["model_type"] == "binary_financial"
@patch("backend.routes.financial_routes.read_financial_prediction_history")
def test_prediction_history_route(mock_history):
    mock_history.return_value = [
        {
            "ticker": "AAPL",
            "prediction": "neutral",
            "period": "2026-03-31",
        }
    ]

    response = client.get("/api/financial/predictions/AAPL")

    assert response.status_code == 200
    assert response.json()[0]["prediction"] == "neutral"


@patch("backend.routes.financial_routes.read_financial_model_versions")
def test_list_model_versions_route(mock_versions):
    mock_versions.return_value = [
        {
            "model_version": "xgboost_financial_v2",
            "parent_version": "xgboost_financial_v1",
            "metrics": {"macro_f1": 0.55},
            "is_active": True,
        },
        {
            "model_version": "xgboost_financial_v1",
            "parent_version": None,
            "metrics": {"macro_f1": 0.43},
            "is_active": False,
        },
    ]

    response = client.get("/api/financial/model/versions")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["is_active"] is True


@patch("backend.routes.financial_routes.set_active_financial_model")
def test_activate_model_version_route(mock_activate):
    mock_activate.return_value = {
        "model_version": "xgboost_financial_v1",
        "is_active": True,
    }

    response = client.post(
        "/api/financial/model/versions/xgboost_financial_v1/activate"
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is True
