from unittest.mock import patch

from backend.services.sentiment_source_service import (
    add_sentiment_source,
    delete_sentiment_source,
    get_all_sentiment_sources,
    set_sentiment_source_active,
    update_sentiment_source,
)

MODULE = "backend.services.sentiment_source_service"


@patch(f"{MODULE}.supabase")
def test_get_all_sentiment_sources_returns_data(mock_supa):
    mock_supa.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
        {"id": "1", "source_type": "RSS", "account": "Reuters", "relevance": "Market-wide", "is_active": True}
    ]

    result = get_all_sentiment_sources()

    mock_supa.table.assert_called_with("sentiment_sources")
    assert result[0]["account"] == "Reuters"


@patch(f"{MODULE}.supabase")
def test_get_all_sentiment_sources_returns_empty_list_when_no_data(mock_supa):
    mock_supa.table.return_value.select.return_value.order.return_value.execute.return_value.data = None

    result = get_all_sentiment_sources()

    assert result == []


@patch(f"{MODULE}.supabase")
def test_add_sentiment_source_inserts_row(mock_supa):
    add_sentiment_source("RSS", "Reuters", "Market-wide")

    insert_call = mock_supa.table.return_value.insert.call_args
    assert insert_call.args[0] == {
        "source_type": "RSS",
        "account": "Reuters",
        "relevance": "Market-wide",
    }


@patch(f"{MODULE}.supabase")
def test_update_sentiment_source_updates_row(mock_supa):
    update_sentiment_source("abc-123", "RSS", "Bloomberg", "Market-wide")

    mock_supa.table.return_value.update.assert_called_with({
        "source_type": "RSS",
        "account": "Bloomberg",
        "relevance": "Market-wide",
    })
    mock_supa.table.return_value.update.return_value.eq.assert_called_with("id", "abc-123")


@patch(f"{MODULE}.supabase")
def test_set_sentiment_source_active_updates_row(mock_supa):
    set_sentiment_source_active("abc-123", False)

    mock_supa.table.return_value.update.assert_called_with({"is_active": False})
    mock_supa.table.return_value.update.return_value.eq.assert_called_with("id", "abc-123")


@patch(f"{MODULE}.supabase")
def test_delete_sentiment_source_deletes_row(mock_supa):
    delete_sentiment_source("abc-123")

    mock_supa.table.return_value.delete.assert_called_once()
    mock_supa.table.return_value.delete.return_value.eq.assert_called_with("id", "abc-123")
