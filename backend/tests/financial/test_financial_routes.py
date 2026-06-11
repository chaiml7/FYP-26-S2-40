from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@patch("routes.financial_routes.import_financial_statements")
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


@patch("routes.financial_routes.import_all_financial_statements")
def test_import_all_financial_statements_route(mock_import):
    mock_import.return_value = {
        "stocks_processed": 11,
        "stocks_imported": 10,
        "results": [],
    }

    response = client.post("/api/financial/statements/import")

    assert response.status_code == 200
    assert response.json()["stocks_processed"] == 11


@patch("routes.financial_routes.train_financial_model")
def test_train_financial_model_route(mock_train):
    mock_train.return_value = {
        "model_version": "xgboost_financial_v1",
        "training_rows": 44,
        "holdout_rows": 11,
        "metrics": {"macro_f1": 0.5},
    }

    response = client.post(
        "/api/financial/model/train",
        json={"training_mode": "fresh"},
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "xgboost_financial_v1"
    mock_train.assert_called_once_with("fresh", None)


@patch("routes.financial_routes.train_financial_model")
def test_continue_training_route(mock_train):
    mock_train.return_value = {
        "model_version": "xgboost_financial_v2",
        "parent_version": "xgboost_financial_v1",
        "training_mode": "continue",
    }

    response = client.post(
        "/api/financial/model/train",
        json={
            "training_mode": "continue",
            "base_version": "xgboost_financial_v1",
        },
    )

    assert response.status_code == 200
    mock_train.assert_called_once_with(
        "continue",
        "xgboost_financial_v1",
    )


@patch("routes.financial_routes.generate_financial_prediction")
def test_generate_financial_prediction_route(mock_generate):
    mock_generate.return_value = {
        "ticker": "AAPL",
        "prediction": "positive",
        "confidence": 72.5,
        "fundamental_score": 7.8,
        "prediction_horizon": "next_quarter",
    }

    response = client.post(
        "/api/financial/predictions/aapl?model_version=xgboost_financial_v1"
    )

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert response.json()["fundamental_score"] == 7.8
    assert response.json()["prediction_horizon"] == "next_quarter"
    mock_generate.assert_called_once_with("aapl", "xgboost_financial_v1")


@patch("routes.financial_routes.generate_financial_prediction")
def test_generate_prediction_requires_trained_model(mock_generate):
    mock_generate.side_effect = FileNotFoundError("Train the model first.")

    response = client.post("/api/financial/predictions/AAPL")

    assert response.status_code == 409


@patch("routes.financial_routes.read_latest_financial_prediction", return_value=None)
def test_latest_prediction_returns_404_when_missing(mock_latest):
    response = client.get("/api/financial/predictions/AAPL/latest")

    assert response.status_code == 404


@patch("routes.financial_routes.read_financial_prediction_history")
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


@patch("routes.financial_routes.read_financial_model_versions")
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


@patch("routes.financial_routes.set_active_financial_model")
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
