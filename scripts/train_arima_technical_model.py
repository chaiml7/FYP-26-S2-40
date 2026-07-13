"""
Train ARIMA close-price forecasting models and save them locally.

Run from repo root after syncing indicators:
    python scripts/train_arima_technical_model.py --all
    python scripts/train_arima_technical_model.py --symbol NVDA
    python scripts/train_arima_technical_model.py --symbol NVDA --order 5,1,0
"""
import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.arima_model_service import (  # noqa: E402
    ARIMA_ARTIFACT_PATH,
    DEFAULT_ARIMA_MIN_TRAIN_ROWS,
    DEFAULT_ARIMA_PREDICTION_LENGTH,
    DEFAULT_ARIMA_VALIDATION_WINDOWS,
    TARGET_RETURN_THRESHOLD,
    train_arima_artifact,
)
from services.technical.indicator_service import (  # noqa: E402
    get_all_technical_indicators_from_supabase,
    get_technical_indicators_from_supabase,
)
from services.technical.price_service import get_stock_by_symbol  # noqa: E402


def main() -> int:
    args = parse_args()
    indicator_df = load_indicator_data(args.symbol)
    if indicator_df.empty:
        symbol_text = args.symbol.upper() if args.symbol else "all target tickers"
        print("status: no_data")
        print(f"reason: No technical_indicators rows found for {symbol_text}")
        return 1

    result = train_arima_artifact(
        indicator_df=indicator_df,
        model_scope="single_ticker" if args.symbol else "global_all_tickers",
        trained_symbol=args.symbol.upper() if args.symbol else None,
        order=parse_order(args.order) if args.order else None,
        prediction_length=args.prediction_length,
        threshold=args.threshold,
        train_before_date=args.train_before_date,
        validation_windows=args.validation_windows,
        min_train_rows=args.min_train_rows,
        artifact_path=args.output_path,
    )
    print_result(result)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate ARIMA close-price forecasting models. Forecasted "
            "close is converted into up vs down/equal direction."
        )
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Train one ARIMA model per prediction-target ticker. This is the default.",
    )
    target_group.add_argument(
        "--symbol",
        help="Train only one ticker's stored technical_indicators close history.",
    )
    parser.add_argument(
        "--order",
        help="Optional ARIMA order as p,d,q. Example: 5,1,0",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=DEFAULT_ARIMA_PREDICTION_LENGTH,
        help="Forecast horizon in trading rows. Default: 1",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Return threshold for predicted/actual up direction. Default: 0.002",
    )
    parser.add_argument(
        "--train-before-date",
        help=(
            "Only train on indicator rows before this date. "
            "Example: 2026-06-04 excludes 2026-06-04 itself."
        ),
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        default=DEFAULT_ARIMA_VALIDATION_WINDOWS,
        help="Number of most recent rolling windows to validate. Default: 30",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=DEFAULT_ARIMA_MIN_TRAIN_ROWS,
        help="Minimum rows before validation starts. Default: 200",
    )
    parser.add_argument(
        "--output-path",
        default=str(ARIMA_ARTIFACT_PATH),
        help="Where to save the ARIMA .joblib artifact.",
    )
    return parser.parse_args()


def load_indicator_data(symbol: str | None):
    if symbol:
        clean_symbol = symbol.upper()
        stock = get_stock_by_symbol(clean_symbol)
        if stock is None:
            return empty_frame()
        return get_technical_indicators_from_supabase(stock["id"], stock["symbol"])

    return get_all_technical_indicators_from_supabase()


def empty_frame():
    import pandas as pd

    return pd.DataFrame()


def parse_order(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--order must be in p,d,q format")
    return tuple(int(part) for part in parts)


def print_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status != "ok":
        print(f"status: {status}")
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    metrics = result.get("metrics", {})
    print("ARIMA technical model training complete")
    print(f"model_scope: {result['model_scope']}")
    if result.get("trained_symbol"):
        print(f"trained_symbol: {result['trained_symbol']}")
    if result.get("train_before_date"):
        print(f"train_before_date: {result['train_before_date']}")
    print(
        "training_date_range: "
        f"{result.get('training_start_date')} to {result.get('training_end_date')}"
    )
    print(f"symbols_trained_count: {result['symbols_trained_count']}")
    print(f"prediction_length: {result['prediction_length']}")
    print(f"direction_threshold: {result['direction_threshold']:.4f}")
    print(f"validation_windows: {result['validation_windows']}")
    print(f"min_train_rows: {result['min_train_rows']}")
    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"window_count: {metrics.get('window_count')}")
    print(f"accuracy: {format_metric(metrics.get('accuracy'))}")
    print(f"balanced_accuracy: {format_metric(metrics.get('balanced_accuracy'))}")
    print(f"precision: {format_metric(metrics.get('precision'))}")
    print(f"recall: {format_metric(metrics.get('recall'))}")
    print(f"f1_score: {format_metric(metrics.get('f1_score'))}")
    print(f"mcc: {format_metric(metrics.get('mcc'))}")
    print(f"mae: {format_metric(metrics.get('mae'))}")
    print(f"rmse: {format_metric(metrics.get('rmse'))}")
    print("symbols:")
    for item in result.get("symbols_detail", []):
        print(
            f"  {item.get('symbol')}: {item.get('status')} "
            f"order={item.get('order')} windows={item.get('window_count')}"
        )


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
