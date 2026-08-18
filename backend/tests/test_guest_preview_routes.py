from unittest.mock import patch

from fastapi.testclient import TestClient

from frontend.main import app

client = TestClient(app)

PREVIEW = {
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "overall_score": 7.2,
    "tone": "bullish",
}


def _fresh_client():
    """A separate client so guest-preview session state doesn't leak
    between tests via the shared cookie jar."""
    return TestClient(app)


def test_guest_preview_requires_no_login():
    client = _fresh_client()
    response = client.get("/guest/preview")

    assert response.status_code == 200


def test_guest_preview_blank_symbol_prompts_for_input():
    client = _fresh_client()
    response = client.get("/guest/preview")

    assert "Enter a ticker symbol to preview." in response.text


@patch("frontend.main.get_public_stock_preview", return_value=None)
def test_guest_preview_unknown_symbol_shows_no_data_message(_mock_preview):
    client = _fresh_client()
    response = client.get("/guest/preview?symbol=ZZZZ")

    assert "No data available for" in response.text
    assert "ZZZZ" in response.text


@patch("frontend.main.get_public_stock_preview", return_value=PREVIEW)
def test_guest_preview_shows_prediction_and_decrements_counter(_mock_preview):
    client = _fresh_client()
    response = client.get("/guest/preview?symbol=aapl")

    assert response.status_code == 200
    assert "AAPL" in response.text
    assert "Apple Inc." in response.text
    assert "2 of 3 free previews left today." in response.text


@patch("frontend.main.get_public_stock_preview", return_value=PREVIEW)
def test_guest_preview_blocks_after_three_previews_same_day(_mock_preview):
    client = _fresh_client()
    for _ in range(3):
        client.get("/guest/preview?symbol=aapl")

    response = client.get("/guest/preview?symbol=aapl")

    assert response.status_code == 200
    assert "You have reached your daily preview limit. Sign up for a free account to continue." in response.text
    assert 'href="/signup"' in response.text


@patch("frontend.main.get_public_stock_preview", return_value=PREVIEW)
def test_home_page_shows_remaining_guest_preview_count(_mock_preview):
    client = _fresh_client()
    client.get("/guest/preview?symbol=aapl")

    response = client.get("/")

    assert response.status_code == 200
    assert "2 of 3 free previews left today." in response.text
