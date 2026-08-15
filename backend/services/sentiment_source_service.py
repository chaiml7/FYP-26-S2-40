from backend.database.supabase_client import supabase


def get_all_sentiment_sources():
    response = (
        supabase
        .table("sentiment_sources")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


def add_sentiment_source(source_type: str, account: str, relevance: str = None):
    response = (
        supabase
        .table("sentiment_sources")
        .insert({
            "source_type": source_type,
            "account": account,
            "relevance": relevance,
        })
        .execute()
    )

    return response.data


def update_sentiment_source(source_id: str, source_type: str, account: str, relevance: str = None):
    response = (
        supabase
        .table("sentiment_sources")
        .update({
            "source_type": source_type,
            "account": account,
            "relevance": relevance,
        })
        .eq("id", source_id)
        .execute()
    )

    return response.data


def set_sentiment_source_active(source_id: str, is_active: bool):
    response = (
        supabase
        .table("sentiment_sources")
        .update({"is_active": is_active})
        .eq("id", source_id)
        .execute()
    )

    return response.data


def delete_sentiment_source(source_id: str):
    response = (
        supabase
        .table("sentiment_sources")
        .delete()
        .eq("id", source_id)
        .execute()
    )

    return response.data
