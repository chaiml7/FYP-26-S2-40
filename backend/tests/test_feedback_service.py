from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.feedback_service import (
    FeedbackValidationError,
    _validate_feedback,
    create_feedback,
    get_feedback,
    list_feedback,
)


def _query_for(mock_supabase):
    query = MagicMock()
    mock_supabase.table.return_value = query
    query.insert.return_value = query
    query.select.return_value = query
    query.eq.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    return query


def test_feedback_validation_normalizes_valid_input():
    result = _validate_feedback(
        "Dashboard",
        4,
        "  The dashboard is easy to understand.  ",
    )

    assert result == {
        "topic": "Dashboard",
        "rating": 4,
        "description": "The dashboard is easy to understand.",
    }


@pytest.mark.parametrize(
    ("topic", "rating", "description"),
    [
        ("Unknown", 4, "This description is long enough."),
        ("Dashboard", 0, "This description is long enough."),
        ("Dashboard", 6, "This description is long enough."),
        ("Dashboard", 4, "Too short"),
    ],
)
def test_feedback_validation_rejects_invalid_input(
    topic,
    rating,
    description,
):
    with pytest.raises(FeedbackValidationError):
        _validate_feedback(topic, rating, description)


@patch("backend.services.feedback_service.supabase")
def test_create_feedback_inserts_authenticated_user_submission(mock_supabase):
    query = _query_for(mock_supabase)
    stored = {
        "id": "feedback-id",
        "user_id": "user-id",
        "username": "user@example.com",
        "topic": "Predictions",
        "rating": 5,
        "description": "The prediction breakdown is useful.",
    }
    query.execute.return_value = SimpleNamespace(data=[stored])

    result = create_feedback(
        "user-id",
        "user@example.com",
        "Predictions",
        5,
        "The prediction breakdown is useful.",
    )

    assert result == stored
    query.insert.assert_called_once_with({
        "user_id": "user-id",
        "username": "user@example.com",
        "topic": "Predictions",
        "rating": 5,
        "description": "The prediction breakdown is useful.",
    })


@patch("backend.services.feedback_service.supabase")
def test_list_feedback_orders_latest_first_and_limits_result(mock_supabase):
    query = _query_for(mock_supabase)
    query.execute.return_value = SimpleNamespace(data=[{"id": "latest"}])

    assert list_feedback(limit=50) == [{"id": "latest"}]
    query.order.assert_called_once_with("created_at", desc=True)
    query.limit.assert_called_once_with(50)


@patch("backend.services.feedback_service.supabase")
def test_get_feedback_returns_none_when_id_is_missing(mock_supabase):
    query = _query_for(mock_supabase)
    query.execute.return_value = SimpleNamespace(data=[])

    assert get_feedback("missing-id") is None
    query.eq.assert_called_once_with("id", "missing-id")
