from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.services.auth_service import (
    PASSWORD_COMPLEXITY_MESSAGE,
    password_meets_complexity,
)


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


class FinancialModelTrainRequest(BaseModel):
    top_n: int = Field(default=5, ge=1, le=20)


class FinancialModelTuneRequest(BaseModel):
    top_n: int = Field(default=5, ge=1, le=20)


class AccountCreate(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        if not password_meets_complexity(value):
            raise ValueError(PASSWORD_COMPLEXITY_MESSAGE)
        return value


class LoginRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None


class PasswordUpdate(BaseModel):
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def _validate_new_password(cls, value: str) -> str:
        if not password_meets_complexity(value):
            raise ValueError(PASSWORD_COMPLEXITY_MESSAGE)
        return value


class EmailUpdate(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class WatchlistAdd(BaseModel):
    stock_id: Optional[int] = None
    symbol: Optional[str] = None


class UserRoleUpdate(BaseModel):
    role_id: Literal["basic_user", "premium_user", "admin"]


class UserStatusUpdate(BaseModel):
    is_active: bool


class NotificationPreferenceUpdate(BaseModel):
    analysis_ready_email: bool
