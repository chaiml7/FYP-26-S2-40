from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.services.billing_service import BillingError
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


def _free_user_context():
    return {
        "user_role": "basic_user",
        "user_id": "user-id",
        "user_email": "user@example.com",
        "user_initial": "U",
        "base_layout": "free_users/base.html",
    }


@patch("backend.routes.billing_ui_routes.create_checkout_session")
@patch("backend.routes.billing_ui_routes._authenticated_customer")
def test_checkout_renders_safe_error_page(mock_customer, mock_checkout):
    mock_customer.return_value = _free_user_context()
    mock_checkout.side_effect = BillingError("Stripe Checkout could not be started.")

    response = client.post("/billing/checkout")

    assert response.status_code == 400
    assert "Stripe Checkout could not be started." in response.text


@patch("backend.routes.billing_ui_routes.create_checkout_session")
@patch("backend.routes.billing_ui_routes._authenticated_customer")
def test_checkout_converts_unexpected_failure_to_safe_error_page(
    mock_customer, mock_checkout
):
    mock_customer.return_value = _free_user_context()
    mock_checkout.side_effect = RuntimeError("sensitive diagnostic")

    response = client.post("/billing/checkout")

    assert response.status_code == 500
    assert "Billing is temporarily unavailable" in response.text
    assert "sensitive diagnostic" not in response.text
