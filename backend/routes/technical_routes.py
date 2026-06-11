"""Manual technical data, model, and prediction endpoints."""

from fastapi import APIRouter, Header, HTTPException, Query

from backend.services.auth_service import AuthServiceError, get_auth_user
from backend.services.user_profile_service import get_profile
from backend.services.technical.indicator_service import (
    get_technical_indicators_from_supabase,
)
from backend.services.technical.price_service import get_stock_by_symbol
from backend.services.technical.technical_service import (
    generate_all_technical_predictions,
    generate_technical_prediction,
    import_all_technical_prices,
    import_technical_prices,
    read_latest_prediction,
    read_model_version,
    read_model_versions,
    read_prediction_history,
    set_active_technical_model,
    train_technical_model,
)


router = APIRouter(prefix="/technical", tags=["technical"])


def _require_backend_admin(authorization: str = None) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    try:
        user = get_auth_user(token)
    except AuthServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc

    profile = get_profile(user["id"])
    if not profile or profile[0].get("role_id") != "backend_admin":
        raise HTTPException(
            status_code=403,
            detail="Backend admin access required",
        )


def _raise_technical_error(exc: Exception):
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prices/import")
def import_all_prices(
    period: str = Query(default="10y"),
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    try:
        return import_all_technical_prices(period)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/prices/import/{symbol}")
def import_prices(
    symbol: str,
    period: str = Query(default="10y"),
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    try:
        return import_technical_prices(symbol, period)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/model/train")
def train_model(authorization: str = Header(default=None)):
    _require_backend_admin(authorization)
    try:
        return train_technical_model()
    except Exception as exc:
        _raise_technical_error(exc)


@router.get("/model/versions")
def view_model_versions():
    return read_model_versions()


@router.get("/model/versions/{model_version}")
def view_model_version(model_version: str):
    version = read_model_version(model_version)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail="Technical model version not found.",
        )
    return version


@router.post("/model/versions/{model_version}/activate")
def activate_model(
    model_version: str,
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    try:
        return set_active_technical_model(model_version)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/predictions")
def create_all_predictions(
    model_version: str = None,
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    try:
        return generate_all_technical_predictions(model_version)
    except Exception as exc:
        _raise_technical_error(exc)


@router.post("/predictions/{symbol}")
def create_prediction(
    symbol: str,
    model_version: str = None,
    authorization: str = Header(default=None),
):
    _require_backend_admin(authorization)
    try:
        return generate_technical_prediction(symbol, model_version)
    except Exception as exc:
        _raise_technical_error(exc)


@router.get("/predictions/{symbol}/latest")
def view_latest_prediction(symbol: str):
    prediction = read_latest_prediction(symbol)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"No technical prediction found for {symbol.upper()}.",
        )
    return prediction


@router.get("/predictions/{symbol}")
def view_prediction_history(symbol: str):
    return read_prediction_history(symbol)


@router.get("/indicators/{symbol}")
def view_indicators(
    symbol: str,
    limit: int = Query(default=30, ge=1, le=500),
):
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol.upper()} is not in the active stocks table.",
        )
    indicators = get_technical_indicators_from_supabase(
        stock["id"],
        stock["symbol"],
    )
    if indicators.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No technical indicators found for {symbol.upper()}.",
        )
    return indicators.tail(limit).iloc[::-1].to_dict(orient="records")
