"""
Run a zero-shot Chronos or TimesFM close-price forecast and write local results.

Run from repo root after syncing indicators:
    python scripts/predict_foundation_forecast_model.py --model chronos --symbol NVDA
    python scripts/predict_foundation_forecast_model.py --model timesfm --symbol NVDA
    python scripts/predict_foundation_forecast_model.py --model chronos --symbol NVDA --as-of-date 2026-06-04

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
    DEFAULT_CHRONOS_MODEL_ID,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_FREQ,
    DEFAULT_FOUNDATION_MODEL,
    DEFAULT_PREDICTION_LENGTH,
    DEFAULT_TIMESFM_MODEL_ID,
    TARGET_RETURN_THRESHOLD,
    default_model_id,
    normalize_model_type,
    run_foundation_forecast,
)
from services.technical.indicator_service import get_technical_indicators_from_supabase  # noqa: E402
from services.technical.price_service import get_stock_by_symbol  # noqa: E402


def main() -> int:
    args = parse_args()
    try:
        result = run_prediction(args)
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
            "Run a zero-shot time-series foundation model forecast using stored "
            "technical_indicators close history. No Supabase writes."
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
        help="Ticker symbol to forecast.",
    )
    parser.add_argument(
        "--as-of-date",
        help="Use history up to this completed trading day.",
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
        help="Return threshold for predicted up direction. Default: 0.002",
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
        default="technical_analysis/foundation_forecast_predictions.json",
        help=(
            "Local JSON output path. Default: "
            "technical_analysis/foundation_forecast_predictions.json"
        ),
    )
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
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

    result = run_foundation_forecast(
        indicator_df=indicator_df,
        model_type=model_type,
        model_id=args.model_id or default_model_id(model_type),
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        threshold=args.threshold,
        as_of_date=args.as_of_date,
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

    print(f"model_type: {result['model_type']}")
    print(f"model_id: {result['model_id']}")
    if result.get("freq"):
        print(f"freq: {result['freq']}")
    print(f"symbol: {result.get('symbol') or result.get('requested_symbol')}")
    print(f"as_of_date: {result['as_of_date']}")
    print(f"as_of_close: {result['as_of_close']:.4f}")
    print(f"predicted_close: {result['predicted_close']:.4f}")
    print(f"predicted_return: {result['predicted_return']:.4f}")
    print(f"predicted_direction: {result['predicted_direction']}")
    if result.get("actual_date"):
        print(f"actual_date: {result['actual_date']}")
        print(f"actual_return: {result['actual_return']:.4f}")
        print(f"actual_direction: {result['actual_direction']}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
