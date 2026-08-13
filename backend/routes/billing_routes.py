"""Public Stripe webhook endpoint served by the backend API."""

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from backend.services.billing_service import (
    BillingConfigurationError,
    BillingWebhookError,
    process_stripe_webhook,
)


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/billing/stripe/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
):
    payload = await request.body()
    try:
        return process_stripe_webhook(payload, stripe_signature)
    except BillingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BillingWebhookError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Stripe webhook processing failed")
        raise HTTPException(
            status_code=500,
            detail="Stripe webhook processing failed and will be retried.",
        ) from exc
