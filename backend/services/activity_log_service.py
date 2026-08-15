from backend.database.supabase_client import supabase


def log_activity(email: str, action: str, detail: str = None):
    try:
        supabase.table("activity_log").insert({
            "email": email or "unknown",
            "action": action,
            "detail": detail,
        }).execute()
    except Exception as exc:
        # Logging must never break the action it's recording.
        print(f"Activity log write failed: {exc}")


def get_activity_log(limit: int = 200):
    response = (
        supabase
        .table("activity_log")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data or []
