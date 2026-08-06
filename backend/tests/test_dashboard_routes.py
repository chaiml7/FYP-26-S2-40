import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.routes.dashboard_routes import stock_detail
from backend.routes.premium_user_routes import (
    premium_user_weightages,
    save_premium_weightages,
)


@pytest.mark.parametrize(
    ("role", "expected_weight_user_id", "include_indicators"),
    [
        ("free_user", None, False),
        ("frontend_admin", None, False),
        ("premium_user", "user-123", True),
    ],
)
def test_stock_detail_uses_role_appropriate_weightages(
    role,
    expected_weight_user_id,
    include_indicators,
):
    request = MagicMock()
    session = {"user_id": "user-123", "user_role": role}
    stock = {"symbol": "AAPL"}

    with (
        patch(
            "backend.routes.dashboard_routes._session_context",
            return_value=session,
        ),
        patch(
            "backend.routes.dashboard_routes.get_stock_dashboard",
            return_value=stock,
        ) as mock_dashboard,
        patch(
            "backend.routes.dashboard_routes.get_user_watchlist_symbols",
            return_value=[],
        ),
        patch(
            "backend.routes.dashboard_routes.templates.TemplateResponse",
            return_value="rendered",
        ),
    ):
        result = asyncio.run(stock_detail(request, "aapl", None))

    assert result == "rendered"
    mock_dashboard.assert_called_once_with(
        "aapl",
        None,
        include_technical_indicators=include_indicators,
        weight_user_id=expected_weight_user_id,
    )


@pytest.mark.parametrize("role", ["free_user", "frontend_admin"])
def test_personal_model_weightage_page_is_premium_only(role):
    request = MagicMock()
    with patch(
        "backend.routes.premium_user_routes.get_session_context",
        return_value={"user_id": "user-123", "user_role": role},
    ):
        response = asyncio.run(premium_user_weightages(request))

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@patch("backend.routes.premium_user_routes.supabase")
def test_premium_weightage_save_updates_existing_user_default(mock_supabase):
    request = MagicMock()
    request.session = {"user_id": "user-123", "user_role": "premium_user"}
    table = mock_supabase.table.return_value
    table.select.return_value.eq.return_value.limit.return_value.execute.return_value = (
        SimpleNamespace(data=[{"id": 7}])
    )

    response = asyncio.run(save_premium_weightages(request, 50, 30, 20))

    table.update.assert_called_once_with({
        "user_id": "user-123",
        "technical": 50,
        "sentiment": 30,
        "financial": 20,
    })
    table.update.return_value.eq.assert_called_once_with("user_id", "user-123")
    table.insert.assert_not_called()
    assert response.status_code == 303
    assert response.headers["location"] == "/premium/weightages"
