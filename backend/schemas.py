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


class AccountCreate(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None


class PasswordUpdate(BaseModel):
    new_password: str = Field(..., min_length=8)


class EmailUpdate(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class WatchlistAdd(BaseModel):
    stock_id: Optional[int] = None
    symbol: Optional[str] = None


class UserRoleUpdate(BaseModel):
    role_id: Literal["basic_user", "premium_user", "frontend_admin", "backend_admin"]


class UserStatusUpdate(BaseModel):
    is_active: bool
