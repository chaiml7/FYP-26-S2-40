import pytest
import requests
from unittest.mock import patch, MagicMock, call
from datetime import date
from services.sentiment.finnhub_service import fetch_news

MODULE = "services.sentiment.finnhub_service"

SAMPLE_FINNHUB_RESPONSE = [
    {"headline": "Apple Q2 earnings beat expectations", "datetime": 1716537600},
    {"headline": "Apple launches new product line", "datetime": 1716624000},
]


def mock_response(status_code=200, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data if json_data is not None else SAMPLE_FINNHUB_RESPONSE
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


@patch(f"{MODULE}.requests.get")
def test_fetch_returns_headlines(mock_get):
    mock_get.return_value = mock_response()
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert result[0]["source"] == "finnhub"
    assert "headline" in result[0]
    assert "published_at" in result[0]


@patch(f"{MODULE}.requests.get")
def test_fetch_empty_response(mock_get):
    mock_get.return_value = mock_response(json_data=[])
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.requests.get")
def test_fetch_malformed_json(mock_get):
    mock_get.return_value = mock_response(json_data={"error": "bad"})
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert result == []


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_retry_on_429(mock_get, mock_sleep):
    mock_get.side_effect = [mock_response(429), mock_response(200)]
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once_with(2)


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_retry_exhausted_raises(mock_get, mock_sleep):
    mock_get.return_value = mock_response(429)
    with pytest.raises(Exception, match="rate limit"):
        fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert mock_get.call_count == 3


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_retry_on_timeout(mock_get, mock_sleep):
    mock_get.side_effect = [requests.Timeout(), mock_response(200)]
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 2
    assert mock_get.call_count == 2


@patch(f"{MODULE}.time.sleep")
@patch(f"{MODULE}.requests.get")
def test_timeout_exhausted_returns_empty(mock_get, mock_sleep):
    mock_get.side_effect = requests.Timeout()
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert result == []
    assert mock_get.call_count == 3


@patch(f"{MODULE}.requests.get")
def test_published_at_is_iso_string(mock_get):
    mock_get.return_value = mock_response()
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    from datetime import datetime
    datetime.fromisoformat(result[0]["published_at"].replace("Z", "+00:00"))


@patch(f"{MODULE}.requests.get")
def test_symbol_uppercased_in_params(mock_get):
    mock_get.return_value = mock_response()
    fetch_news("aapl", from_date=date(2026, 5, 23))
    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["params"]["symbol"] == "AAPL"


@patch(f"{MODULE}.requests.get")
def test_headline_missing_filtered_out(mock_get):
    mock_get.return_value = mock_response(json_data=[
        {"headline": "Valid headline", "datetime": 1716537600},
        {"headline": "", "datetime": 1716537700},
        {"datetime": 1716537800},
    ])
    result = fetch_news("AAPL", from_date=date(2026, 5, 23))
    assert len(result) == 1
    assert result[0]["headline"] == "Valid headline"
