from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.services import billing_service


def _query(response_data):
    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
    query.limit.return_value = query
    query.update.return_value = query
    query.upsert.return_value = query
    query.insert.return_value = query
    query.execute.return_value = SimpleNamespace(data=response_data)
    return query


def test_identifier_supports_stripe_v15_objects():
    class StripeV15LikeObject:
        def to_dict(self):
            return {"id": "cus_test"}

        def __getitem__(self, key):
            # StripeObject is not a normal Mapping; dict(value) tries index 0
            # and raises this error in stripe-python 15.x.
            raise KeyError(key)

    assert billing_service._identifier(StripeV15LikeObject()) == "cus_test"


@patch("backend.services.billing_service._upsert_customer")
@patch("backend.services.billing_service.get_user_subscription", return_value=None)
@patch("backend.services.billing_service._premium_price_id", return_value="price_test")
@patch("backend.services.billing_service._stripe_client")
def test_checkout_reuses_stocklens_identity_metadata(
    mock_stripe_client,
    _mock_price,
    _mock_subscription,
    mock_upsert_customer,
):
    stripe = MagicMock()
    stripe.Customer.create.return_value = {"id": "cus_test"}
    stripe.checkout.Session.create.return_value = SimpleNamespace(
        url="https://checkout.stripe.test/session"
    )
    mock_stripe_client.return_value = stripe

    result = billing_service.create_checkout_session(
        user_id="user-id",
        email="user@example.com",
        success_url="http://localhost:8001/billing/success",
        cancel_url="http://localhost:8001/billing/cancel",
    )

    assert result == "https://checkout.stripe.test/session"
    mock_upsert_customer.assert_called_once_with("user-id", "cus_test")
    customer_args = stripe.Customer.create.call_args.kwargs
    assert customer_args["idempotency_key"] == "stocklens-customer-user-id"
    stripe.checkout.Session.create.assert_called_once()
    checkout_args = stripe.checkout.Session.create.call_args.kwargs
    assert checkout_args["mode"] == "subscription"
    assert checkout_args["client_reference_id"] == "user-id"
    assert checkout_args["line_items"] == [{"price": "price_test", "quantity": 1}]
    assert checkout_args["subscription_data"]["metadata"]["stocklens_user_id"] == "user-id"


@patch(
    "backend.services.billing_service._upsert_customer",
    side_effect=RuntimeError("temporary database failure"),
)
@patch("backend.services.billing_service.get_user_subscription", return_value=None)
@patch("backend.services.billing_service._premium_price_id", return_value="price_test")
@patch("backend.services.billing_service._stripe_client")
def test_checkout_continues_when_preliminary_customer_cache_write_fails(
    mock_stripe_client,
    _mock_price,
    _mock_subscription,
    _mock_upsert_customer,
):
    stripe = MagicMock()
    stripe.Customer.create.return_value = {"id": "cus_test"}
    stripe.checkout.Session.create.return_value = SimpleNamespace(
        url="https://checkout.stripe.test/session"
    )
    mock_stripe_client.return_value = stripe

    result = billing_service.create_checkout_session(
        user_id="user-id",
        email="user@example.com",
        success_url="http://localhost:8001/billing/success",
        cancel_url="http://localhost:8001/billing/cancel",
    )

    assert result == "https://checkout.stripe.test/session"
    stripe.checkout.Session.create.assert_called_once()


@patch("backend.services.billing_service._premium_price_id", return_value="price_test")
@patch("backend.services.billing_service._set_managed_role")
def test_active_subscription_is_saved_and_grants_premium(
    mock_set_role, _mock_price
):
    subscription_select = _query([])
    subscription_upsert = _query([{"status": "active"}])
    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = [subscription_select, subscription_upsert]

    subscription = {
        "id": "sub_test",
        "customer": "cus_test",
        "status": "active",
        "current_period_end": 1_800_000_000,
        "cancel_at_period_end": False,
        "metadata": {"stocklens_user_id": "user-id"},
        "items": {"data": [{"price": {"id": "price_test"}}]},
    }

    with patch.object(billing_service, "supabase", mock_supabase):
        result = billing_service.synchronize_subscription(subscription)

    assert result["status"] == "active"
    payload = subscription_upsert.upsert.call_args.args[0]
    assert payload["stripe_subscription_id"] == "sub_test"
    assert payload["stripe_price_id"] == "price_test"
    mock_set_role.assert_called_once_with("user-id", premium=True)


@patch("backend.services.billing_service._set_managed_role")
def test_old_cancellation_cannot_downgrade_new_active_subscription(mock_set_role):
    active_row = {
        "user_id": "user-id",
        "stripe_subscription_id": "sub_new",
        "stripe_customer_id": "cus_test",
        "status": "active",
    }
    subscription_select = _query([active_row])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = subscription_select

    old_cancellation = {
        "id": "sub_old",
        "customer": "cus_test",
        "status": "canceled",
        "metadata": {"stocklens_user_id": "user-id"},
        "items": {"data": []},
    }

    with patch.object(billing_service, "supabase", mock_supabase):
        result = billing_service.synchronize_subscription(old_cancellation)

    assert result == active_row
    mock_set_role.assert_not_called()
    subscription_select.upsert.assert_not_called()


@patch("backend.services.billing_service._premium_price_id", return_value="price_premium")
@patch("backend.services.billing_service._set_managed_role")
def test_active_subscription_for_another_price_does_not_grant_premium(
    mock_set_role, _mock_price
):
    subscription_select = _query([])
    subscription_upsert = _query([{"status": "active"}])
    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = [subscription_select, subscription_upsert]

    subscription = {
        "id": "sub_other",
        "customer": "cus_test",
        "status": "active",
        "metadata": {"stocklens_user_id": "user-id"},
        "items": {"data": [{"price": {"id": "price_other"}}]},
    }

    with patch.object(billing_service, "supabase", mock_supabase):
        billing_service.synchronize_subscription(subscription)

    mock_set_role.assert_called_once_with("user-id", premium=False)


def test_invoice_subscription_id_supports_new_invoice_parent_shape():
    invoice = {
        "parent": {
            "subscription_details": {
                "subscription": "sub_parent",
            }
        }
    }

    assert billing_service._invoice_subscription_id(invoice) == "sub_parent"
