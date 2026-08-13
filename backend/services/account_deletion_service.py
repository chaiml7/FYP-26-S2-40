"""Deletes a user's data across every table that references them, then
removes the Supabase Auth user itself. No in-repo schema defines ON DELETE
behavior for these foreign keys (the base schema lives only in the live
Supabase project), so each table is cleared explicitly rather than relying
on cascade deletes. Children are deleted before user_profiles/the auth user
to avoid FK violations.

Each child-table delete is isolated: a single table being missing/misconfigured
(e.g. a stale PostgREST schema cache) must not block deleting the rest of the
user's data or the Auth user itself, which is the part that actually matters
for account deletion.
"""

import logging

from backend.database.supabase_client import supabase
from backend.services.auth_service import admin_delete_user


logger = logging.getLogger(__name__)

CHILD_TABLES = (
    ("user_watchlists", "user_id"),
    ("user_notification_preferences", "user_id"),
    ("notification_deliveries", "user_id"),
    ("weightages", "user_id"),
    ("user_subscriptions", "user_id"),
    ("user_profiles", "id"),
)


def delete_user_account(user_id: str) -> None:
    for table, column in CHILD_TABLES:
        try:
            supabase.table(table).delete().eq(column, user_id).execute()
        except Exception:
            logger.exception(
                "Account deletion: could not clear %s for user_id=%s, continuing", table, user_id
            )

    admin_delete_user(user_id)
