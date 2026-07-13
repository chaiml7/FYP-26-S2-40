"""
Backtest a zero-shot Chronos or TimesFM close-price forecast locally.

Run from repo root after syncing indicators:
    python scripts/backtest_foundation_forecast_model.py --model chronos --symbol NVDA
    python scripts/backtest_foundation_forecast_model.py --model timesfm --symbol NVDA --max-windows 20
    python scripts/backtest_foundation_forecast_model.py --model chronos --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04

Optional dependencies:
    pip install -r backend/requirements-foundation-models.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.foundation_forecast_service import (  # noqa: E402
    DEFAULT_BACKTEST_STRIDE,
    DEFAULT_BACKTEST_WINDOWS,
    DEFAULT_CHRONOS_MODEL_ID,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_FREQ,
    DEFAULT_FOUNDATION_MODEL,
    DEFAULT_MIN_CONTEXT_ROWS,
    DEFAULT_PREDICTION_LENGTH,
    DEFAULT_TIMESFM_MODEL_ID,
    TARGET_RETURN_THRESHOLD,
    default_model_id,
    normalize_model_type,
    run_foundation_backtest,
)
from services.technical.indicator_service import get_technical_indicators_from_supabase  # noqa: E402
from services.technical.price_service import get_stock_by_symbol  # noqa: E402


def main() -> int:
    args = parse_args()
    try:
        result = run_backtest(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print_result(result, output_path)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest a zero-shot Chronos or TimesFM forecast over historical "
            "technical_indicators close rows. No Supabase writes."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_FOUNDATION_MODEL,
        choices=["chronos", "chronon", "chronons", "timesfm"],
        help="Foundation model family. Default: chronos",
    )
    parser.add_argument(
        "--model-id",
        help=(
            "Hugging Face model id. Defaults: "
            f"{DEFAULT_CHRONOS_MODEL_ID} for Chronos, "
            f"{DEFAULT_TIMESFM_MODEL_ID} for TimesFM."
        ),
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Ticker symbol to backtest.",
    )
    parser.add_argument(
        "--start-date",
        help="First as-of date to test.",
    )
    parser.add_argument(
        "--end-date",
        help="Last as-of date to test.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_BACKTEST_WINDOWS,
        help=(
            "Maximum historical windows to test, using the most recent windows. "
            "Default: 30. Use 0 to test every available window."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_BACKTEST_STRIDE,
        help="Evaluate every Nth eligible as-of date. Default: 1",
    )
    parser.add_argument(
        "--min-context-rows",
        type=int,
        default=DEFAULT_MIN_CONTEXT_ROWS,
        help="Minimum historical rows required before an as-of date. Default: 128",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=DEFAULT_PREDICTION_LENGTH,
        help="Forecast horizon in trading rows. Default: 1",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Number of recent close rows to pass as context. Default: 512",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Return threshold for predicted/actual up direction. Default: 0.002",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Chronos device_map value, e.g. cpu or cuda. Default: cpu",
    )
    parser.add_argument(
        "--freq",
        default=DEFAULT_FORECAST_FREQ,
        help=(
            "Chronos pandas frequency. Default: B for business-day stock data. "
            "Use D for calendar-day data."
        ),
    )
    parser.add_argument(
        "--output",
        default="technical_analysis/foundation_forecast_backtest.json",
        help=(
            "Local JSON output path. Default: "
            "technical_analysis/foundation_forecast_backtest.json"
        ),
    )
    return parser.parse_args()


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    model_type = normalize_model_type(args.model)
    stock = get_stock_by_symbol(args.symbol.upper())
    if stock is None:
        return {
            "status": "no_data",
            "reason": f"{args.symbol.upper()} was not found in the stocks table",
        }

    indicator_df = get_technical_indicators_from_supabase(stock["id"], stock["symbol"])
    if indicator_df.empty:
        return {
            "status": "no_data",
            "reason": f"No technical_indicators rows found for {stock['symbol']}",
        }

    result = run_foundation_backtest(
        indicator_df=indicator_df,
        model_type=model_type,
        model_id=args.model_id or default_model_id(model_type),
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        threshold=args.threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        max_windows=args.max_windows,
        stride=args.stride,
        min_context_rows=args.min_context_rows,
        device=args.device,
        freq=args.freq,
    )
    result["requested_symbol"] = stock["symbol"]
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def print_result(result: dict[str, Any], output_path: Path) -> None:
    print(f"status: {result['status']}")
    if result["status"] != "ok":
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    metrics = result.get("metrics", {})
    print(f"model_type: {result['model_type']}")
    print(f"model_id: {result['model_id']}")
    if result.get("freq"):
        print(f"freq: {result['freq']}")
    print(f"symbol: {result.get('symbol') or result.get('requested_symbol')}")
    print(f"window_count: {result['window_count']}")
    print(f"date_range: {result['date_range']['start']} to {result['date_range']['end']}")
    print(f"prediction_length: {result['prediction_length']}")
    print(f"context_length: {result['context_length']}")
    print(f"direction_threshold: {result['direction_threshold']:.4f}")
    print(f"accuracy: {format_metric(metrics.get('accuracy'))}")
    print(f"balanced_accuracy: {format_metric(metrics.get('balanced_accuracy'))}")
    print(f"precision: {format_metric(metrics.get('precision'))}")
    print(f"recall: {format_metric(metrics.get('recall'))}")
    print(f"f1_score: {format_metric(metrics.get('f1_score'))}")
    print(f"mcc: {format_metric(metrics.get('mcc'))}")
    print("confusion_matrix:")
    print("          pred 0  pred 1")
    matrix = result.get("confusion_matrix", [[0, 0], [0, 0]])
    print(f"actual 0  {matrix[0][0]:>6}  {matrix[0][1]:>6}")
    print(f"actual 1  {matrix[1][0]:>6}  {matrix[1][1]:>6}")
    print(f"output: {output_path}")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
