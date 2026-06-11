"""
Sync yfinance prices into Supabase for the technical-analysis pipeline.

Run from repo root:
    python scripts/sync_technical_prices.py --symbol NVDA
    python scripts/sync_technical_prices.py --all

Flow:
    yfinance
    -> daily_ohlcv
    -> read daily_ohlcv from Supabase
    -> stock_prices
    -> read stock_prices from Supabase
    -> technical_indicators
    -> read technical_indicators from Supabase

This script intentionally stops before model training and prediction.
"""
import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.indicator_service import (
    add_technical_indicators,
    get_technical_indicators_from_supabase,
    upsert_technical_indicators,
)
from services.technical.price_service import (
    add_market_context_features,
    fetch_price_history,
    get_daily_ohlcv_from_supabase,
    get_stock_by_symbol,
    get_stock_prices_from_supabase,
    get_stocks_from_supabase,
    upsert_daily_ohlcv,
    upsert_stock_prices,
)


def main() -> int:
    args = parse_args()
    stocks = get_target_stocks(args)

    if not stocks:
        print("No stocks found to sync.")
        return 1

    results = []
    for stock in stocks:
        try:
            result = sync_stock_prices(
                stock_id=stock["id"],
                symbol=stock["symbol"],
                period=args.period,
                interval=args.interval,
            )
        except Exception as exc:
            result = {
                "stock_id": stock.get("id"),
                "symbol": stock.get("symbol"),
                "status": "error",
                "reason": str(exc),
            }
        results.append(result)
        print_result(result)

    ok_count = len([result for result in results if result["status"] == "ok"])
    print(f"\nCompleted: {ok_count}/{len(results)} stocks synced successfully.")
    return 0 if ok_count == len(results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch yfinance data, store daily_ohlcv, read it back, store "
            "stock_prices, then calculate and store technical_indicators."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--symbol",
        help="Single stock symbol to sync. The symbol must exist in the stocks table.",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Sync every prediction-target stock in the stocks table.",
    )
    parser.add_argument(
        "--period",
        default="10y",
        help="yfinance lookback period. Default: 10y",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval. Default: 1d",
    )
    return parser.parse_args()


def get_target_stocks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all:
        return get_stocks_from_supabase()

    stock = get_stock_by_symbol(args.symbol)
    if stock is None:
        print(f"{args.symbol.upper()} was not found in the stocks table.")
        return []
    return [stock]


def sync_stock_prices(
    stock_id: int,
    symbol: str,
    period: str = "10y",
    interval: str = "1d",
) -> dict[str, Any]:
    clean_symbol = symbol.upper()

    yfinance_df = fetch_price_history(clean_symbol, period=period, interval=interval)
    if yfinance_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No yfinance rows returned",
        }

    daily_result = upsert_daily_ohlcv(stock_id, clean_symbol, yfinance_df)

    supabase_price_df = get_daily_ohlcv_from_supabase(stock_id, clean_symbol)
    if supabase_price_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No daily_ohlcv rows found after upsert",
            "yfinance_rows": len(yfinance_df),
            "daily_ohlcv_rows_saved": daily_result["rows_saved"],
        }

    stock_prices_df = add_market_context_features(
        supabase_price_df,
        symbol=clean_symbol,
        period=period,
        interval=interval,
    )
    stock_prices_result = upsert_stock_prices(stock_id, clean_symbol, stock_prices_df)

    training_price_df = get_stock_prices_from_supabase(stock_id, clean_symbol)
    if training_price_df.empty:
        return {
            "stock_id": stock_id,
            "symbol": clean_symbol,
            "status": "no_data",
            "reason": "No stock_prices rows found after upsert",
            "yfinance_rows": len(yfinance_df),
            "daily_ohlcv_rows_saved": daily_result["rows_saved"],
            "daily_ohlcv_rows_read": len(supabase_price_df),
            "stock_prices_rows_saved": stock_prices_result["rows_saved"],
        }

    indicator_df = add_technical_indicators(training_price_df)
    indicator_result = upsert_technical_indicators(stock_id, clean_symbol, indicator_df)
    stored_indicator_df = get_technical_indicators_from_supabase(stock_id, clean_symbol)

    return {
        "stock_id": stock_id,
        "symbol": clean_symbol,
        "status": "ok",
        "period": period,
        "interval": interval,
        "yfinance_rows": len(yfinance_df),
        "yfinance_date_range": _date_range(yfinance_df),
        "daily_ohlcv_rows_read": len(supabase_price_df),
        "daily_ohlcv_rows_saved": daily_result["rows_saved"],
        "daily_ohlcv_date_range": _date_range(supabase_price_df),
        "stock_prices_rows_saved": stock_prices_result["rows_saved"],
        "stock_prices_rows_read": len(training_price_df),
        "stock_prices_date_range": _date_range(training_price_df),
        "technical_indicator_rows_calculated": len(indicator_df),
        "technical_indicator_rows_saved": indicator_result["rows_saved"],
        "technical_indicator_rows_read": len(stored_indicator_df),
        "technical_indicators_date_range": _date_range(stored_indicator_df),
    }


def print_result(result: dict[str, Any]) -> None:
    symbol = result.get("symbol", "UNKNOWN")
    status = result.get("status")
    if status != "ok":
        print(f"[{symbol}] {status}: {result.get('reason', 'unknown reason')}")
        return

    print(
        f"[{symbol}] ok | "
        f"yfinance={result['yfinance_rows']} | "
        f"daily_ohlcv_saved={result['daily_ohlcv_rows_saved']} | "
        f"daily_ohlcv_read={result['daily_ohlcv_rows_read']} | "
        f"stock_prices_saved={result['stock_prices_rows_saved']} | "
        f"stock_prices_read={result['stock_prices_rows_read']} | "
        f"indicators_saved={result['technical_indicator_rows_saved']} | "
        f"indicators_read={result['technical_indicator_rows_read']} | "
        f"dates={result['technical_indicators_date_range']}"
    )


def _date_range(df: Any) -> str:
    if df is None or df.empty or "date" not in df:
        return "n/a"

    dates = df["date"].dropna().astype(str)
    if dates.empty:
        return "n/a"

    return f"{dates.min()} to {dates.max()}"


if __name__ == "__main__":
    raise SystemExit(main())
