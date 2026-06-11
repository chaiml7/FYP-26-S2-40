import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from database.supabase_client import supabase

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]
PRICE_UPSERT_BATCH_SIZE = 500
DAILY_OHLCV_UPSERT_BATCH_SIZE = 500
SUPABASE_SELECT_BATCH_SIZE = 1000

MARKET_CONTEXT_COLUMNS = [
    "market_spy_return_1d",
    "market_spy_return_5d",
    "market_spy_above_sma_200",
    "market_qqq_return_1d",
    "market_qqq_return_5d",
    "market_qqq_above_sma_200",
    "market_vix_level",
    "market_vix_return_1d",
    "market_vix_return_5d",
    "market_sector_return_1d",
    "market_sector_return_5d",
    "market_sector_above_sma_200",
]

MARKET_CONTEXT_INTEGER_COLUMNS = {
    "market_spy_above_sma_200",
    "market_qqq_above_sma_200",
    "market_sector_above_sma_200",
}

MARKET_CONTEXT_SYMBOLS = {
    "SPY",
    "QQQ",
    "^VIX",
    "XLK",
    "XLC",
    "XLY",
    "XLF",
    "XLV",
    "XLE",
    "KWEB",
}

SECTOR_ETF_BY_SYMBOL = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMD": "XLK",
    "AVGO": "XLK",
    "ORCL": "XLK",
    "CRM": "XLK",
    "ADBE": "XLK",
    "INTC": "XLK",
    "CSCO": "XLK",
    "GOOG": "XLC",
    "GOOGL": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "HD": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "SBUX": "XLY",
    "JPM": "XLF",
    "BAC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "JNJ": "XLV",
    "PFE": "XLV",
    "MRK": "XLV",
    "UNH": "XLV",
    "XOM": "XLE",
    "CVX": "XLE",
    "BABA": "KWEB",
}


def get_stocks_from_supabase(
    include_market_context_symbols: bool = False,
) -> list[dict[str, Any]]:
    """Return tracked stocks with usable ticker symbols."""
    response = supabase.table("stocks").select("id, symbol").execute()
    rows = response.data or []

    stocks = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        stock_id = row.get("id")
        if not include_market_context_symbols and symbol in MARKET_CONTEXT_SYMBOLS:
            continue
        if symbol and stock_id is not None:
            stocks.append({"id": stock_id, "symbol": symbol})

    return stocks


def get_stock_by_symbol(symbol: str) -> dict[str, Any] | None:
    """Look up one stock row by ticker symbol."""
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        return None

    response = (
        supabase.table("stocks")
        .select("id, symbol")
        .eq("symbol", clean_symbol)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return None

    row = rows[0]
    return {"id": row.get("id"), "symbol": str(row.get("symbol") or clean_symbol).upper()}


def fetch_price_history(
    symbol: str,
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch daily OHLCV history from yfinance and normalize column names."""
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    try:
        raw_df = yf.Ticker(clean_symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", clean_symbol, exc)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    missing_columns = [column for column in PRICE_COLUMNS if column not in df.columns]
    if missing_columns:
        logger.warning("yfinance data for %s missing columns: %s", clean_symbol, missing_columns)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = df[PRICE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df


def upsert_stock_prices(stock_id: int, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    """Upsert technical-analysis-ready price rows into Supabase."""
    if df is None or df.empty:
        return {"rows_saved": 0}

    rows = []
    clean_symbol = symbol.upper()
    for _, row in df.iterrows():
        rows.append(
            {
                "stock_id": _to_json_value(stock_id),
                "symbol": clean_symbol,
                "date": _to_json_value(row["date"]),
                "open": _to_json_value(row["open"]),
                "high": _to_json_value(row["high"]),
                "low": _to_json_value(row["low"]),
                "close": _to_json_value(row["close"]),
                "adj_close": _to_json_value(row["adj_close"]),
                "volume": _to_json_value(row["volume"]),
            }
        )
        for column in MARKET_CONTEXT_COLUMNS:
            if column in row:
                rows[-1][column] = _to_json_value(
                    _to_int_flag(row[column])
                    if column in MARKET_CONTEXT_INTEGER_COLUMNS
                    else row[column]
                )

    saved_rows = 0
    for batch in _chunks(rows, PRICE_UPSERT_BATCH_SIZE):
        response = (
            supabase.table("stock_prices")
            .upsert(batch, on_conflict="stock_id,date")
            .execute()
        )
        saved_rows += len(response.data or batch)

    return {"rows_saved": saved_rows}


def upsert_daily_ohlcv(stock_id: int, symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    """Save raw yfinance daily OHLCV rows into the existing daily_ohlcv table."""
    if df is None or df.empty:
        return {"rows_saved": 0}

    clean_symbol = symbol.upper()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "stock_id": _to_json_value(stock_id),
                "symbol": clean_symbol,
                "trade_date": _to_json_value(row["date"]),
                "open": _to_json_value(row["open"]),
                "high": _to_json_value(row["high"]),
                "low": _to_json_value(row["low"]),
                "close": _to_json_value(row["close"]),
                "volume": _to_json_value(row["volume"]),
                "source": "yfinance",
            }
        )

    saved_rows = 0
    for batch in _chunks(rows, DAILY_OHLCV_UPSERT_BATCH_SIZE):
        response = (
            supabase.table("daily_ohlcv")
            .upsert(batch, on_conflict="stock_id,trade_date")
            .execute()
        )
        saved_rows += len(response.data or batch)

    return {"rows_saved": saved_rows}


def get_daily_ohlcv_from_supabase(stock_id: int, symbol: str | None = None) -> pd.DataFrame:
    """Read raw daily prices from Supabase and normalize them for indicators."""
    rows = []
    offset = 0
    while True:
        response = (
            supabase.table("daily_ohlcv")
            .select("stock_id, symbol, trade_date, open, high, low, close, volume")
            .eq("stock_id", stock_id)
            .order("trade_date", desc=False)
            .range(offset, offset + SUPABASE_SELECT_BATCH_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < SUPABASE_SELECT_BATCH_SIZE:
            break
        offset += SUPABASE_SELECT_BATCH_SIZE

    if not rows:
        logger.warning("No daily_ohlcv rows found for stock_id=%s symbol=%s", stock_id, symbol)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = pd.DataFrame(rows).rename(columns={"trade_date": "date"})
    df["adj_close"] = df["close"]
    return _normalize_price_dataframe(df)


def get_stock_prices_from_supabase(stock_id: int, symbol: str | None = None) -> pd.DataFrame:
    """Read technical-ready stock_prices rows from Supabase for model training."""
    rows = []
    offset = 0
    while True:
        response = (
            supabase.table("stock_prices")
            .select("*")
            .eq("stock_id", stock_id)
            .order("date", desc=False)
            .range(offset, offset + SUPABASE_SELECT_BATCH_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < SUPABASE_SELECT_BATCH_SIZE:
            break
        offset += SUPABASE_SELECT_BATCH_SIZE

    if not rows:
        logger.warning("No stock_prices rows found for stock_id=%s symbol=%s", stock_id, symbol)
        return pd.DataFrame(columns=PRICE_COLUMNS + MARKET_CONTEXT_COLUMNS)

    df = pd.DataFrame(rows)
    return _normalize_price_dataframe(df, extra_columns=MARKET_CONTEXT_COLUMNS)


def add_market_context_features(
    df: pd.DataFrame,
    symbol: str | None = None,
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Merge broad-market and sector context without using future rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date
    result = result.dropna(subset=["date"])
    result["date"] = result["date"].astype(str)
    result = result.sort_values("date", ascending=True).reset_index(drop=True)

    context_sources = [
        ("SPY", "market_spy", False),
        ("QQQ", "market_qqq", False),
        ("^VIX", "market_vix", True),
    ]

    sector_etf = _sector_etf_for_symbol(symbol)
    if sector_etf:
        context_sources.append((sector_etf, "market_sector", False))

    for ticker, prefix, include_level in context_sources:
        context_df = _build_context_frame(ticker, prefix, period, interval, include_level)
        if context_df.empty:
            logger.warning("Market context unavailable for %s", ticker)
            continue
        result = result.merge(context_df, on="date", how="left")

    result = result.sort_values("date", ascending=True).reset_index(drop=True)
    for column in MARKET_CONTEXT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].ffill()

    neutral_defaults = {
        "market_spy_return_1d": 0.0,
        "market_spy_return_5d": 0.0,
        "market_spy_above_sma_200": 0.0,
        "market_qqq_return_1d": 0.0,
        "market_qqq_return_5d": 0.0,
        "market_qqq_above_sma_200": 0.0,
        "market_vix_level": 0.0,
        "market_vix_return_1d": 0.0,
        "market_vix_return_5d": 0.0,
        "market_sector_return_1d": 0.0,
        "market_sector_return_5d": 0.0,
        "market_sector_above_sma_200": 0.0,
    }
    result = result.fillna(value=neutral_defaults)
    for column in MARKET_CONTEXT_INTEGER_COLUMNS:
        result[column] = result[column].fillna(0).astype(int)
    return result


def _normalize_price_dataframe(
    df: pd.DataFrame,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    result = df.copy()
    if "adj_close" not in result.columns and "close" in result.columns:
        result["adj_close"] = result["close"]

    missing_columns = [column for column in PRICE_COLUMNS if column not in result.columns]
    if missing_columns:
        logger.warning("Supabase price data missing columns: %s", missing_columns)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    extra_columns = extra_columns or []
    available_extra_columns = [column for column in extra_columns if column in result.columns]
    result = result[PRICE_COLUMNS + available_extra_columns].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.date

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in available_extra_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    result["date"] = result["date"].astype(str)
    return result.sort_values("date", ascending=True).reset_index(drop=True)


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _build_context_frame(
    ticker: str,
    prefix: str,
    period: str,
    interval: str,
    include_level: bool,
) -> pd.DataFrame:
    context_period = _market_context_period(period)
    history = fetch_price_history(ticker, period=context_period, interval=interval)
    if history.empty and context_period != period:
        history = fetch_price_history(ticker, period=period, interval=interval)
    if history.empty:
        return pd.DataFrame()

    close = pd.to_numeric(history["close"], errors="coerce")
    context = pd.DataFrame({"date": history["date"]})
    context[f"{prefix}_return_1d"] = close.pct_change()
    context[f"{prefix}_return_5d"] = close.pct_change(periods=5)

    if include_level:
        context[f"{prefix}_level"] = close
    else:
        sma_200 = close.rolling(window=200, min_periods=200).mean()
        context[f"{prefix}_above_sma_200"] = (close > sma_200).astype(int)

    return context.replace([np.inf, -np.inf], np.nan)


def _sector_etf_for_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    return SECTOR_ETF_BY_SYMBOL.get(symbol.upper())


def _market_context_period(period: str) -> str:
    period_text = str(period or "").strip().lower()
    if period_text == "max":
        return "max"

    if period_text.endswith("y") and period_text[:-1].isdigit():
        return f"{int(period_text[:-1]) + 1}y"

    if period_text.endswith("mo") and period_text[:-2].isdigit():
        return f"{int(period_text[:-2]) + 12}mo"

    return period


def _to_int_flag(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _to_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
