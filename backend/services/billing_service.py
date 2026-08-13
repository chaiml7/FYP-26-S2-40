"""Stripe subscription billing and Supabase entitlement synchronization."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from backend.database.supabase_client import supabase


PREMIUM_SUBSCRIPTION_STATUSES = {"active", "trialing"}
MANAGED_USER_ROLES = {"basic_user", "premium_user"}


class BillingError(Exception):
    """Base error safe for a route to present without leaking secrets."""


class BillingConfigurationError(BillingError):
    pass


class BillingWebhookError(BillingError):
    pass


class AlreadyPremiumError(BillingError):
    pass


def _stripe_module():
    try:
        import stripe
    except ImportError as exc:
        raise BillingConfigurationError(
            "Stripe support is not installed. Install backend requirements first."
        ) from exc
    return stripe


def _stripe_client(*, require_webhook_secret: bool = False):
    secret_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret_key:
        raise BillingConfigurationError("STRIPE_SECRET_KEY is not configured.")
    if require_webhook_secret and not os.getenv("STRIPE_WEBHOOK_SECRET", "").strip():
        raise BillingConfigurationError("STRIPE_WEBHOOK_SECRET is not configured.")

    stripe = _stripe_module()
    stripe.api_key = secret_key
    return stripe


def _premium_price_id() -> str:
    price_id = os.getenv("STRIPE_PREMIUM_PRICE_ID", "").strip()
    if not price_id:
        raise BillingConfigurationError("STRIPE_PREMIUM_PRICE_ID is not configured.")
    return price_id


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict_recursive"):
        return value.to_dict_recursive()
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _identifier(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    data = _as_dict(value)
    identifier = data.get("id")
    return str(identifier) if identifier else None


def _first_row(response: Any) -> dict[str, Any] | None:
    data = getattr(response, "data", None) or []
    return data[0] if data else None


def get_user_subscription(user_id: str) -> dict[str, Any] | None:
    response = (
        supabase.table("user_subscriptions")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return _first_row(response)


def _find_subscription_owner(
    *, subscription_id: str | None, customer_id: str | None
) -> str | None:
    query = supabase.table("user_subscriptions").select("user_id")
    if subscription_id:
        query = query.eq("stripe_subscription_id", subscription_id)
    elif customer_id:
        query = query.eq("stripe_customer_id", customer_id)
    else:
        return None
    row = _first_row(query.limit(1).execute())
    return str(row["user_id"]) if row and row.get("user_id") else None


def _upsert_customer(user_id: str, customer_id: str) -> None:
    existing = get_user_subscription(user_id) or {}
    payload = {
        "user_id": user_id,
        "stripe_customer_id": customer_id,
        "status": existing.get("status") or "inactive",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    supabase.table("user_subscriptions").upsert(
        payload, on_conflict="user_id"
    ).execute()


def create_checkout_session(
    *, user_id: str, email: str, success_url: str, cancel_url: str
) -> str:
    stripe = _stripe_client()
    price_id = _premium_price_id()
    existing = get_user_subscription(user_id) or {}

    if (
        str(existing.get("status", "")).lower() in PREMIUM_SUBSCRIPTION_STATUSES
        and existing.get("stripe_price_id") == price_id
    ):
        raise AlreadyPremiumError("This account already has an active Premium subscription.")

    customer_id = existing.get("stripe_customer_id")
    if not customer_id:
        try:
            customer = stripe.Customer.create(
                email=email,
                metadata={"stocklens_user_id": user_id},
            )
        except Exception as exc:
            raise BillingError(
                "Stripe could not create the billing customer. Please try again."
            ) from exc
        customer_id = _identifier(customer)
        if not customer_id:
            raise BillingError("Stripe did not return a customer identifier.")
        _upsert_customer(user_id, customer_id)

    try:
        checkout = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            client_reference_id=user_id,
            line_items=[{"price": price_id, "quantity": 1}],
            metadata={"stocklens_user_id": user_id},
            subscription_data={"metadata": {"stocklens_user_id": user_id}},
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except Exception as exc:
        raise BillingError(
            "Stripe Checkout could not be started. Please try again."
        ) from exc
    checkout_url = getattr(checkout, "url", None) or _as_dict(checkout).get("url")
    if not checkout_url:
        raise BillingError("Stripe did not return a Checkout URL.")
    return str(checkout_url)


def create_customer_portal_session(*, user_id: str, return_url: str) -> str:
    stripe = _stripe_client()
    subscription = get_user_subscription(user_id) or {}
    customer_id = subscription.get("stripe_customer_id")
    if not customer_id:
        raise BillingError("No Stripe billing account is connected to this user.")

    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
    except Exception as exc:
        raise BillingError(
            "The Stripe billing portal could not be opened. Please try again."
        ) from exc
    portal_url = getattr(portal, "url", None) or _as_dict(portal).get("url")
    if not portal_url:
        raise BillingError("Stripe did not return a Customer Portal URL.")
    return str(portal_url)


def _period_end(subscription: dict[str, Any]) -> str | None:
    timestamp = subscription.get("current_period_end")
    if timestamp is None:
        items = _as_dict(subscription.get("items")).get("data") or []
        item_periods = [
            _as_dict(item).get("current_period_end")
            for item in items
            if _as_dict(item).get("current_period_end") is not None
        ]
        timestamp = max(item_periods) if item_periods else None
    if timestamp is None:
        return None
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def _price_from_subscription(subscription: dict[str, Any]) -> str | None:
    items = _as_dict(subscription.get("items")).get("data") or []
    if not items:
        return None
    price = _as_dict(_as_dict(items[0]).get("price"))
    return _identifier(price)


def _set_managed_role(user_id: str, *, premium: bool) -> None:
    profile_response = (
        supabase.table("user_profiles")
        .select("role_id")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    profile = _first_row(profile_response)
    if not profile:
        raise BillingError("The subscription user profile could not be found.")

    current_role = str(profile.get("role_id") or "basic_user").lower()
    if current_role not in MANAGED_USER_ROLES:
        return
    desired_role = "premium_user" if premium else "basic_user"
    if current_role != desired_role:
        (
            supabase.table("user_profiles")
            .update({"role_id": desired_role})
            .eq("id", user_id)
            .execute()
        )


def synchronize_subscription(
    subscription_value: Any, *, fallback_user_id: str | None = None
) -> dict[str, Any]:
    subscription = _as_dict(subscription_value)
    subscription_id = _identifier(subscription)
    customer_id = _identifier(subscription.get("customer"))
    metadata = _as_dict(subscription.get("metadata"))
    user_id = (
        metadata.get("stocklens_user_id")
        or fallback_user_id
        or _find_subscription_owner(
            subscription_id=subscription_id, customer_id=customer_id
        )
    )
    if not user_id:
        raise BillingWebhookError("Stripe subscription is not linked to a StockLens user.")

    status = str(subscription.get("status") or "inactive").lower()
    existing = get_user_subscription(str(user_id)) or {}
    existing_subscription_id = existing.get("stripe_subscription_id")
    existing_status = str(existing.get("status") or "").lower()

    # Stripe does not guarantee event order. An old canceled subscription must
    # not downgrade a newer active subscription for the same customer.
    if (
        subscription_id
        and existing_subscription_id
        and subscription_id != existing_subscription_id
        and existing_status in PREMIUM_SUBSCRIPTION_STATUSES
        and status not in PREMIUM_SUBSCRIPTION_STATUSES
    ):
        return existing

    price_id = _price_from_subscription(subscription) or existing.get(
        "stripe_price_id"
    )
    payload = {
        "user_id": str(user_id),
        "stripe_customer_id": customer_id or existing.get("stripe_customer_id"),
        "stripe_subscription_id": subscription_id,
        "stripe_price_id": price_id,
        "status": status,
        "current_period_end": _period_end(subscription),
        "cancel_at_period_end": bool(subscription.get("cancel_at_period_end", False)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    response = (
        supabase.table("user_subscriptions")
        .upsert(payload, on_conflict="user_id")
        .execute()
    )
    _set_managed_role(
        str(user_id),
        premium=(
            status in PREMIUM_SUBSCRIPTION_STATUSES
            and price_id == _premium_price_id()
        ),
    )
    return _first_row(response) or payload


def _retrieve_subscription(subscription_id: str):
    stripe = _stripe_client()
    return stripe.Subscription.retrieve(subscription_id)


def _invoice_subscription_id(invoice_value: Any) -> str | None:
    invoice = _as_dict(invoice_value)
    direct = _identifier(invoice.get("subscription"))
    if direct:
        return direct
    parent = _as_dict(invoice.get("parent"))
    details = _as_dict(parent.get("subscription_details"))
    return _identifier(details.get("subscription"))


def _begin_webhook_event(event_id: str, event_type: str) -> bool:
    existing = _first_row(
        supabase.table("stripe_webhook_events")
        .select("processing_status, attempts")
        .eq("event_id", event_id)
        .limit(1)
        .execute()
    )
    if existing and existing.get("processing_status") in {"processing", "processed"}:
        return False
    if existing:
        supabase.table("stripe_webhook_events").update(
            {
                "processing_status": "processing",
                "attempts": int(existing.get("attempts") or 1) + 1,
                "error_message": None,
            }
        ).eq("event_id", event_id).execute()
        return True

    supabase.table("stripe_webhook_events").insert(
        {"event_id": event_id, "event_type": event_type}
    ).execute()
    return True


def _finish_webhook_event(
    event_id: str, *, succeeded: bool, error_message: str | None = None
) -> None:
    payload = {
        "processing_status": "processed" if succeeded else "failed",
        "error_message": error_message[:1000] if error_message else None,
        "processed_at": datetime.now(timezone.utc).isoformat() if succeeded else None,
    }
    (
        supabase.table("stripe_webhook_events")
        .update(payload)
        .eq("event_id", event_id)
        .execute()
    )


def process_stripe_webhook(payload: bytes, signature: str | None) -> dict[str, Any]:
    stripe = _stripe_client(require_webhook_secret=True)
    if not signature:
        raise BillingWebhookError("Missing Stripe-Signature header.")
    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=signature,
            secret=os.environ["STRIPE_WEBHOOK_SECRET"],
        )
    except Exception as exc:
        raise BillingWebhookError("Invalid Stripe webhook signature or payload.") from exc

    event_data = _as_dict(event)
    event_id = str(event_data.get("id") or "")
    event_type = str(event_data.get("type") or "")
    if not event_id or not event_type:
        raise BillingWebhookError("Stripe webhook is missing its event ID or type.")
    if not _begin_webhook_event(event_id, event_type):
        return {"received": True, "duplicate": True, "event_type": event_type}

    try:
        event_object = _as_dict(_as_dict(event_data.get("data")).get("object"))
        if event_type == "checkout.session.completed":
            subscription_id = _identifier(event_object.get("subscription"))
            user_id = event_object.get("client_reference_id") or _as_dict(
                event_object.get("metadata")
            ).get("stocklens_user_id")
            customer_id = _identifier(event_object.get("customer"))
            if user_id and customer_id:
                _upsert_customer(str(user_id), customer_id)
            if subscription_id:
                synchronize_subscription(
                    _retrieve_subscription(subscription_id),
                    fallback_user_id=str(user_id) if user_id else None,
                )
        elif event_type in {
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
            "customer.subscription.paused",
            "customer.subscription.resumed",
        }:
            synchronize_subscription(event_object)
        elif event_type in {"invoice.paid", "invoice.payment_failed"}:
            subscription_id = _invoice_subscription_id(event_object)
            if subscription_id:
                synchronize_subscription(_retrieve_subscription(subscription_id))
    except Exception as exc:
        _finish_webhook_event(event_id, succeeded=False, error_message=str(exc))
        raise

    _finish_webhook_event(event_id, succeeded=True)
    return {"received": True, "duplicate": False, "event_type": event_type}
