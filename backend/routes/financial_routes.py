"""Manual financial model training and prediction endpoints."""

from fastapi import APIRouter, HTTPException

from backend.schemas import FinancialModelTrainRequest
from backend.services.financial.financial_service import (
    generate_all_financial_predictions,
    generate_financial_prediction,
    import_all_financial_statements,
    import_financial_statements,
    read_financial_model_version,
    read_financial_model_versions,
    read_financial_prediction_history,
    read_latest_financial_prediction,
    set_active_financial_model,
    train_financial_model,
)


router = APIRouter(prefix="/financial", tags=["financial"])


def _raise_financial_error(exc: Exception):
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/statements/import")
def import_all_statements():
    try:
        return import_all_financial_statements()
    except Exception as exc:
        _raise_financial_error(exc)


@router.post("/statements/import/{symbol}")
def import_statements(symbol: str):
    try:
        return import_financial_statements(symbol)
    except Exception as exc:
        _raise_financial_error(exc)


@router.post("/model/train")
def train_model(request: FinancialModelTrainRequest = None):
    try:
        request = request or FinancialModelTrainRequest()
        return train_financial_model(
            request.training_mode,
            request.base_version,
        )
    except Exception as exc:
        _raise_financial_error(exc)


@router.get("/model/versions")
def view_model_versions():
    return read_financial_model_versions()


@router.get("/model/versions/{model_version}")
def view_model_version(model_version: str):
    version = read_financial_model_version(model_version)
    if version is None:
        raise HTTPException(status_code=404, detail="Financial model version not found.")
    return version


@router.post("/model/versions/{model_version}/activate")
def activate_model(model_version: str):
    try:
        return set_active_financial_model(model_version)
    except Exception as exc:
        _raise_financial_error(exc)


@router.post("/predictions")
def create_all_predictions(model_version: str = None):
    try:
        return generate_all_financial_predictions(model_version)
    except Exception as exc:
        _raise_financial_error(exc)


@router.post("/predictions/{symbol}")
def create_prediction(symbol: str, model_version: str = None):
    try:
        return generate_financial_prediction(symbol, model_version)
    except Exception as exc:
        _raise_financial_error(exc)


@router.get("/predictions/{symbol}/latest")
def view_latest_prediction(symbol: str):
    prediction = read_latest_financial_prediction(symbol)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"No financial prediction found for {symbol.upper()}.",
        )
    return prediction


@router.get("/predictions/{symbol}")
def view_prediction_history(symbol: str):
    return read_financial_prediction_history(symbol)
