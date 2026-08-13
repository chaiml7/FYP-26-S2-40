from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.billing_service import BillingConfigurationError


client = TestClient(app)


@patch("backend.routes.billing_routes.process_stripe_webhook")
def test_webhook_forwards_raw_body_and_signature(mock_process):
    mock_process.return_value = {
        "received": True,
        "duplicate": False,
        "event_type": "invoice.paid",
    }

    response = client.post(
        "/api/billing/stripe/webhook",
        content=b'{"id":"evt_test"}',
        headers={
            "Content-Type": "application/json",
            "Stripe-Signature": "test-signature",
        },
    )

    assert response.status_code == 200
    assert response.json()["received"] is True
    mock_process.assert_called_once_with(b'{"id":"evt_test"}', "test-signature")


@patch("backend.routes.billing_routes.process_stripe_webhook")
def test_webhook_returns_service_unavailable_for_missing_configuration(mock_process):
    mock_process.side_effect = BillingConfigurationError(
        "STRIPE_WEBHOOK_SECRET is not configured."
    )

    response = client.post(
        "/api/billing/stripe/webhook",
        content=b"{}",
        headers={"Stripe-Signature": "test-signature"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "STRIPE_WEBHOOK_SECRET is not configured."
