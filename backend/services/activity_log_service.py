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


def get_activity_log(
    limit: int = 200,
    email: str = None,
    date_from: str = None,
    date_to: str = None,
):
    query = supabase.table("activity_log").select("*")

    if email:
        query = query.ilike("email", f"%{email}%")
    if date_from:
        query = query.gte("created_at", f"{date_from}T00:00:00")
    if date_to:
        query = query.lte("created_at", f"{date_to}T23:59:59")

    response = query.order("created_at", desc=True).limit(limit).execute()

    return response.data or []
