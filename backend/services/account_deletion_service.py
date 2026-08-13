"""Deletes a user's data across every table that references them, then
removes the Supabase Auth user itself. No in-repo schema defines ON DELETE
behavior for these foreign keys (the base schema lives only in the live
Supabase project), so each table is cleared explicitly rather than relying
on cascade deletes. Children are deleted before user_profiles/the auth user
to avoid FK violations.
"""

from backend.database.supabase_client import supabase
from backend.services.auth_service import admin_delete_user


def delete_user_account(user_id: str) -> None:
    supabase.table("user_watchlists").delete().eq("user_id", user_id).execute()
    supabase.table("user_notification_preferences").delete().eq("user_id", user_id).execute()
    supabase.table("notification_deliveries").delete().eq("user_id", user_id).execute()
    supabase.table("weightages").delete().eq("user_id", user_id).execute()
    supabase.table("user_subscriptions").delete().eq("user_id", user_id).execute()
    supabase.table("user_profiles").delete().eq("id", user_id).execute()
    admin_delete_user(user_id)
