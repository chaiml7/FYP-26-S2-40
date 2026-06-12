from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


@patch("backend.routes.technical_routes.train_technical_model")
def test_train_model_does_not_require_authorization(mock_train):
    mock_train.return_value = {"model_version": "technical_test"}

    response = client.post("/api/technical/model/train")

    assert response.status_code == 200
    assert response.json()["model_version"] == "technical_test"


@patch("backend.routes.technical_routes.generate_technical_prediction")
def test_create_prediction_does_not_require_authorization(mock_predict):
    mock_predict.return_value = {
        "symbol": "AAPL",
        "technical_score": 6.2,
    }

    response = client.post("/api/technical/predictions/AAPL")

    assert response.status_code == 200
    assert response.json()["technical_score"] == 6.2
