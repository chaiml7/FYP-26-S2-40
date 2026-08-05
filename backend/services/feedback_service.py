"""Persistence and validation for user feedback."""

from backend.database.supabase_client import supabase


FEEDBACK_TOPICS = (
    "Dashboard",
    "Predictions",
    "Watchlist",
    "News & Social",
    "Market Overview",
    "Financials",
    "General",
)

FEEDBACK_COLUMNS = (
    "id,user_id,username,topic,rating,description,created_at"
)


class FeedbackValidationError(ValueError):
    """Raised when submitted feedback does not meet form requirements."""


def _validate_feedback(topic: str, rating: int, description: str) -> dict:
    clean_topic = str(topic or "").strip()
    clean_description = str(description or "").strip()

    if clean_topic not in FEEDBACK_TOPICS:
        raise FeedbackValidationError("Please select a valid feedback topic.")

    try:
        clean_rating = int(rating)
    except (TypeError, ValueError) as exc:
        raise FeedbackValidationError(
            "Please select a rating from 1 to 5."
        ) from exc

    if clean_rating < 1 or clean_rating > 5:
        raise FeedbackValidationError("Please select a rating from 1 to 5.")
    if len(clean_description) < 10:
        raise FeedbackValidationError(
            "Feedback must contain at least 10 characters."
        )
    if len(clean_description) > 2000:
        raise FeedbackValidationError(
            "Feedback cannot exceed 2,000 characters."
        )

    return {
        "topic": clean_topic,
        "rating": clean_rating,
        "description": clean_description,
    }


def create_feedback(
    user_id: str,
    user_email: str,
    topic: str,
    rating: int,
    description: str,
) -> dict:
    if not user_id:
        raise FeedbackValidationError("You must be signed in to send feedback.")

    validated = _validate_feedback(topic, rating, description)
    payload = {
        "user_id": user_id,
        "username": str(user_email or "").strip(),
        **validated,
    }
    response = (
        supabase.table("user_feedback")
        .insert(payload)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise RuntimeError("Supabase did not return the submitted feedback.")
    return rows[0]


def list_feedback(limit: int = 200) -> list:
    safe_limit = max(1, min(int(limit), 500))
    response = (
        supabase.table("user_feedback")
        .select(FEEDBACK_COLUMNS)
        .order("created_at", desc=True)
        .limit(safe_limit)
        .execute()
    )
    return response.data or []


def get_feedback(feedback_id: str) -> dict | None:
    response = (
        supabase.table("user_feedback")
        .select(FEEDBACK_COLUMNS)
        .eq("id", str(feedback_id))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None
