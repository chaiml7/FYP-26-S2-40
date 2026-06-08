from database.supabase_client import supabase


def save_financial_forecast(forecast_data: dict):
    response = (
        supabase
        .table("financial_forecasts")
        .insert(forecast_data)
        .execute()
    )

    return response.data


def get_financial_forecasts_by_symbol(symbol: str):
    response = (
        supabase
        .table("financial_forecasts")
        .select("*")
        .eq("symbol", symbol.upper())
        .order("created_at", desc=True)
        .execute()
    )

    return response.data
