import numpy as np
import pandas as pd
import yfinance as yf


STATEMENT_CATEGORIES = {
    "income_statement": "quarterly_financials",
    "balance_sheet": "quarterly_balance_sheet",
    "cash_flow": "quarterly_cashflow",
}


def _normalize_statement_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [pd.to_datetime(col) for col in df.columns]
    df = df.transpose()
    df.index = df.index.to_period("Q")
    df.index = df.index.astype(str)
    return df


def fetch_company_quarterly_statements(symbol: str) -> dict:
    ticker = yf.Ticker(symbol.upper())
    statements = {}

    for category, attr_name in STATEMENT_CATEGORIES.items():
        df = getattr(ticker, attr_name, None)
        statements[category] = _normalize_statement_dataframe(df)

    return statements


def _forecast_series(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().astype(float)
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values.iloc[-1]), 0.25

    y = values.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)

    # Trend forecast using a linear fit
    slope, intercept = np.polyfit(x, y, 1)
    trend_pred = slope * len(y) + intercept

    # Quarterly growth forecast
    prev = y[:-1]
    growth = np.divide(y[1:] - prev, np.where(prev == 0, np.nan, prev))
    finite_growth = growth[np.isfinite(growth)]
    avg_growth = float(np.nanmean(finite_growth)) if finite_growth.size > 0 else 0.0

    seasonal_pred = y[-1] * (1.0 + avg_growth)
    prediction = float(max(0.0, (trend_pred + seasonal_pred) / 2.0))

    volatility = float(np.nanstd(finite_growth) if finite_growth.size > 0 else 0.0)
    confidence = 0.15 + min(0.8, 0.1 * len(y)) - min(0.6, abs(volatility) * 2.5)
    confidence = float(max(0.05, min(0.99, confidence)))

    return prediction, confidence


def _upcoming_quarter_label(last_quarter: pd.Period) -> str:
    current_upcoming = pd.Period(pd.Timestamp.today(), freq="Q") + 1
    next_history = last_quarter + 1
    return str(current_upcoming if current_upcoming > next_history else next_history)


def forecast_quarterly_statements(statements: dict) -> dict:
    forecast = {}

    for category, df in statements.items():
        if df.empty:
            forecast[category] = {
                "next_quarter": None,
                "items": {},
            }
            continue

        last_quarter = pd.Period(df.index[-1], freq="Q")
        next_quarter = _upcoming_quarter_label(last_quarter)

        rows = {}
        for item in df.columns:
            history = df[item]
            predicted_value, confidence = _forecast_series(history)
            rows[item] = {
                "predicted_value": predicted_value,
                "confidence": confidence,
                "history_quarters": history.dropna().index.tolist(),
                "history_values": [float(v) for v in history.dropna().to_numpy(dtype=float)],
            }

        forecast[category] = {
            "next_quarter": next_quarter,
            "items": rows,
        }

    return forecast


def summarize_forecast(forecast: dict, top_n: int = 10) -> dict:
    summary = {}
    for category, data in forecast.items():
        items = sorted(
            data["items"].items(),
            key=lambda kv: abs(kv[1]["predicted_value"]),
            reverse=True
        )[:top_n]
        summary[category] = {
            "next_quarter": data["next_quarter"],
            "top_items": [
                {
                    "item": item,
                    "predicted_value": values["predicted_value"],
                    "confidence": values["confidence"],
                }
                for item, values in items
            ]
        }
    return summary
