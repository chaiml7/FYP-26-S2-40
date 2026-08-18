from unittest.mock import patch

from backend.services.activity_log_service import get_activity_log, log_activity

MODULE = "backend.services.activity_log_service"


@patch(f"{MODULE}.supabase")
def test_log_activity_inserts_row(mock_supa):
    log_activity("user@example.com", "login", "detail text")

    mock_supa.table.assert_called_with("activity_log")
    insert_call = mock_supa.table.return_value.insert.call_args
    assert insert_call.args[0] == {
        "email": "user@example.com",
        "action": "login",
        "detail": "detail text",
    }


@patch(f"{MODULE}.supabase")
def test_log_activity_swallows_errors(mock_supa):
    mock_supa.table.return_value.insert.return_value.execute.side_effect = Exception("boom")

    # Should not raise.
    log_activity("user@example.com", "login")


@patch(f"{MODULE}.supabase")
def test_get_activity_log_returns_data(mock_supa):
    mock_supa.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"email": "user@example.com", "action": "login", "detail": None}
    ]

    result = get_activity_log()

    mock_supa.table.assert_called_with("activity_log")
    assert result[0]["action"] == "login"


@patch(f"{MODULE}.supabase")
def test_get_activity_log_applies_email_and_date_filters(mock_supa):
    query = mock_supa.table.return_value.select.return_value
    query.ilike.return_value = query
    query.gte.return_value = query
    query.lte.return_value = query
    query.order.return_value.limit.return_value.execute.return_value.data = []

    get_activity_log(email="user@example.com", date_from="2026-08-01", date_to="2026-08-10")

    query.ilike.assert_called_with("email", "%user@example.com%")
    query.gte.assert_called_with("created_at", "2026-08-01T00:00:00")
    query.lte.assert_called_with("created_at", "2026-08-10T23:59:59")


@patch(f"{MODULE}.supabase")
def test_get_activity_log_returns_empty_list_when_no_data(mock_supa):
    mock_supa.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = None

    result = get_activity_log()

    assert result == []
