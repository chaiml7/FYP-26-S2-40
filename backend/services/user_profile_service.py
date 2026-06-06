from database.supabase_client import supabase


def get_profiles():
    response = (
        supabase
        .table("user_profiles")
        .select("*, roles(*)")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


def get_profile(user_id: str):
    response = (
        supabase
        .table("user_profiles")
        .select("*, roles(*)")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )

    return response.data


def update_profile(user_id: str, profile_data: dict):
    response = (
        supabase
        .table("user_profiles")
        .update(profile_data)
        .eq("id", user_id)
        .execute()
    )

    return response.data


def update_user_role(user_id: str, role_id: str):
    response = (
        supabase
        .table("user_profiles")
        .update({"role_id": role_id})
        .eq("id", user_id)
        .execute()
    )

    return response.data


def update_user_status(user_id: str, is_active: bool):
    response = (
        supabase
        .table("user_profiles")
        .update({"is_active": is_active})
        .eq("id", user_id)
        .execute()
    )

    return response.data
