from unittest.mock import patch

from fastapi.testclient import TestClient

from frontend.main import app


client = TestClient(app)


@patch("frontend.main.get_profile", return_value=[])
def test_premium_signup_link_preselects_premium_plan(_mock_profile):
    response = client.get("/signup?plan=premium")

    assert response.status_code == 200
    assert 'name="plan" value="premium" checked' in response.text


def test_billing_checkout_requires_login():
    response = client.post("/billing/checkout", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
