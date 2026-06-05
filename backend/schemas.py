from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StockCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    company_name: str = Field(..., min_length=1)
    exchange: Optional[str] = None
    sector: Optional[str] = None
    is_active: bool = True


class StockUpdate(BaseModel):
    company_name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    is_active: Optional[bool] = None


class PredictionCreate(BaseModel):
    prediction_date: Optional[date] = None
    target_date: Optional[date] = None
    predicted_close: float
    signal: Literal["buy", "sell", "hold"]
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    model_version: Optional[str] = None
