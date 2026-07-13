"""
Use the saved ARIMA technical model and write local prediction results.

Run from repo root after training:
    python scripts/predict_arima_technical_model.py --symbol NVDA
    python scripts/predict_arima_technical_model.py --symbol NVDA --as-of-date 2026-06-04
"""
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.arima_model_service import (  # noqa: E402
    ARIMA_ARTIFACT_PATH,
    load_arima_artifact,
    predict_with_arima_artifact,
)
from services.technical.indicator_service import (  # noqa: E402
    get_all_technical_indicators_from_supabase,
    get_technical_indicators_from_supabase,
)
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
            "Load saved ARIMA close-price models, read stored indicators from "
            "Supabase, and write local next-direction forecasts."
        )
    )
    parser.add_argument(
        "--symbol",
        help="Only predict this ticker. Default: all artifact symbols.",
    )
    parser.add_argument(
        "--as-of-date",
        help="Refit ARIMA using history up to this completed trading day.",
    )
    parser.add_argument(
        "--model-path",
        default=str(ARIMA_ARTIFACT_PATH),
        help="Path to saved ARIMA .joblib artifact.",
    )
    parser.add_argument(
        "--output",
        default="technical_analysis/arima_predictions.json",
        help="Local JSON output path. Default: technical_analysis/arima_predictions.json",
    )
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_arima_artifact(args.model_path)
    metadata = artifact.get("metadata", {})
    trained_symbol = metadata.get("trained_symbol")
    requested_symbol = args.symbol.upper() if args.symbol else None
    if trained_symbol and requested_symbol and requested_symbol != str(trained_symbol).upper():
        return {
            "status": "wrong_model_scope",
            "reason": (
                f"Saved ARIMA model was trained for {trained_symbol}; "
                f"train an all-ticker model or a {requested_symbol} model first."
            ),
            "model_artifact_path": str(Path(args.model_path)),
        }

    indicator_df = load_indicator_data(requested_symbol or trained_symbol)
    if indicator_df.empty:
        return {
            "status": "no_data",
            "reason": "No technical_indicators rows found for prediction",
            "model_artifact_path": str(Path(args.model_path)),
        }

    predictions = predict_with_arima_artifact(
        artifact,
        indicator_df=indicator_df,
        symbol=requested_symbol or trained_symbol,
        as_of_date=args.as_of_date,
    )

    return {
        "status": "ok" if predictions else "no_data",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "model_artifact_path": str(Path(args.model_path)),
        "model_scope": metadata.get("model_scope"),
        "trained_symbol": trained_symbol,
        "requested_symbol": requested_symbol,
        "as_of_date": normalize_date(args.as_of_date) if args.as_of_date else None,
        "prediction_length": metadata.get("prediction_length"),
        "direction_threshold": metadata.get("direction_threshold"),
        "metrics": metadata.get("metrics", {}),
        "predictions": predictions,
    }


def load_indicator_data(symbol: str | None) -> pd.DataFrame:
    if symbol:
        stock = get_stock_by_symbol(str(symbol).upper())
        if stock is None:
            return pd.DataFrame()
        return get_technical_indicators_from_supabase(stock["id"], stock["symbol"])

    return get_all_technical_indicators_from_supabase()


def normalize_date(value: str) -> str:
    return pd.to_datetime(value, errors="raise").date().isoformat()


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

    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"output: {output_path}")
    for prediction in result.get("predictions", []):
        print(
            f"{prediction.get('symbol')} {prediction.get('as_of_date')} "
            f"{prediction.get('predicted_direction')} "
            f"predicted_close={format_optional_float(prediction.get('predicted_close'))} "
            f"predicted_return={format_optional_float(prediction.get('predicted_return'))}"
        )
        if prediction.get("actual_direction"):
            print(
                f"  actual={prediction.get('actual_direction')} "
                f"actual_return={format_optional_float(prediction.get('actual_return'))}"
            )


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
