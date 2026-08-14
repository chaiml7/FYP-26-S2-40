from unittest.mock import patch

from fastapi.testclient import TestClient

from frontend.main import app


client = TestClient(app)


def test_terms_page_is_public_and_contains_service_terms():
    response = client.get("/terms")

    assert response.status_code == 200
    assert "Terms of Service" in response.text
    assert "Brokerage connections and orders" in response.text
    assert 'href="/privacy"' in response.text


def test_privacy_page_is_public_and_names_current_processors():
    response = client.get("/privacy")

    assert response.status_code == 200
    assert "Privacy Policy" in response.text
    assert "Supabase" in response.text
    assert "Stripe" in response.text
    assert "SnapTrade" in response.text
    assert "SendGrid" in response.text


@patch("frontend.main.get_public_model_metrics", return_value=None)
@patch("frontend.main.get_public_market_leaders", return_value=[])
def test_homepage_has_small_legal_footer(_mock_leaders, _mock_metrics):
    response = client.get("/")

    assert response.status_code == 200
    assert 'aria-label="Legal information"' in response.text
    assert 'href="/terms"' in response.text
    assert 'href="/privacy"' in response.text


@patch("frontend.main.create_account")
def test_signup_rejects_missing_legal_acceptance(mock_create_account):
    response = client.post(
        "/signup",
        data={
            "firstName": "Test User",
            "email": "test@example.com",
            "password": "password123",
            "plan": "free",
        },
    )

    assert response.status_code == 400
    assert "must accept the Terms of Service" in response.text
    mock_create_account.assert_not_called()
