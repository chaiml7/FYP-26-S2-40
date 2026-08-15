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
def test_get_activity_log_returns_empty_list_when_no_data(mock_supa):
    mock_supa.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = None

    result = get_activity_log()

    assert result == []
