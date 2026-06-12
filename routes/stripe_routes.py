"""Stripe payment routes — handles subscription checkouts, billing portal sessions, and webhook listeners."""

import os
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, Header

logger = logging.getLogger(__name__)

# Dynamically import stripe if installed, otherwise log warning
try:
    import stripe
except ImportError:
    stripe = None
    logger.warning("stripe package is not installed. Stripe routes will fall back to simulation/mock mode.")

def setup_stripe_routes() -> APIRouter:
    router = APIRouter(tags=["stripe"])

    # Initialize Stripe API Key if available
    stripe_key = os.environ.get("STRIPE_API_KEY") or os.environ.get("STRIPE_SECRET_KEY")
    if stripe_key and stripe:
        stripe.api_key = stripe_key
        logger.info("Stripe SDK initialized with API key.")
    else:
        logger.info("Stripe SDK in Simulation/Mock mode (STRIPE_API_KEY/STRIPE_SECRET_KEY environment variable not set).")

    @router.post("/api/stripe/checkout-session")
    async def create_checkout_session(request: Request) -> Dict[str, Any]:
        """Creates a Stripe Checkout Session for subscription tier selection."""
        try:
            body = await request.json()
            tier = body.get("tier", "Pro")  # Free, Pro, Enterprise
            success_url = body.get("success_url", f"{request.base_url}?payment=success")
            cancel_url = body.get("cancel_url", f"{request.base_url}?payment=cancelled")
            
            if not stripe or not stripe_key:
                # Simulation Mode
                logger.info(f"[Simulation] Created checkout session for {tier} tier.")
                return {
                    "id": f"sess_mock_{os.urandom(8).hex()}",
                    "url": success_url,
                    "simulated": True
                }

            # Map tiers to Stripe Price IDs (from environment variables)
            price_id = os.environ.get(f"STRIPE_PRICE_ID_{tier.upper()}")
            if not price_id:
                raise HTTPException(status_code=400, detail=f"Price ID for tier '{tier}' is not configured.")

            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
            )
            return {"id": session.id, "url": session.url, "simulated": False}
        except Exception as e:
            logger.error(f"Error creating checkout session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/stripe/webhook")
    async def stripe_webhook(request: Request, stripe_signature: str = Header(None)) -> Dict[str, Any]:
        """Listens for Stripe Webhook events and manages subscription states."""
        payload = await request.body()
        event = None

        if not stripe or not stripe_key:
            # Simulation Webhook
            try:
                data = await request.json()
                logger.info(f"[Simulation] Received webhook event: {data.get('type')}")
                return {"status": "success", "event_type": data.get("type"), "simulated": True}
            except Exception:
                return {"status": "success", "simulated": True}

        endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, endpoint_secret
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Handle key events
        event_type = event['type']
        logger.info(f"Stripe Webhook event received: {event_type}")

        if event_type == "checkout.session.completed":
            session = event['data']['object']
            logger.info(f"Checkout completed for session: {session.get('id')}")
            # Update user level in database here
        elif event_type in ["customer.subscription.created", "customer.subscription.updated"]:
            subscription = event['data']['object']
            logger.info(f"Subscription updated: {subscription.get('id')}")
        elif event_type == "customer.subscription.deleted":
            subscription = event['data']['object']
            logger.info(f"Subscription cancelled: {subscription.get('id')}")
        elif event_type in ["invoice.payment_succeeded", "invoice.payment_failed"]:
            invoice = event['data']['object']
            logger.info(f"Payment invoice: {invoice.get('id')} - State: {event_type}")

        return {"status": "success", "event_type": event_type, "simulated": False}

    @router.get("/api/stripe/portal")
    async def billing_portal(request: Request) -> Dict[str, Any]:
        """Generates a billing portal link for self-service subscription management."""
        try:
            return_url = f"{request.base_url}"
            if not stripe or not stripe_key:
                return {"url": return_url, "simulated": True}

            # Retrieve stripe customer ID from session/user table
            customer_id = os.environ.get("MOCK_CUSTOMER_ID", "cus_default")
            
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            return {"url": session.url, "simulated": False}
        except Exception as e:
            logger.error(f"Error creating billing portal session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
