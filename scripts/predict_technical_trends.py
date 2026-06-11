"""
Use the saved technical-analysis model to predict next-day trends.

Run from repo root after training:
    python scripts/predict_technical_trends.py
    python scripts/predict_technical_trends.py --symbol NVDA
"""
import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.prediction_pipeline import predict_trends_with_saved_model


def main() -> int:
    args = parse_args()
    result = predict_trends_with_saved_model(
        symbol=args.symbol.upper() if args.symbol else None,
    )
    print_result(result)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved shared technical model, predict latest next-day "
            "directions, and upsert rows into direction_predictions."
        )
    )
    parser.add_argument(
        "--symbol",
        help="Only save a prediction for this symbol. Default: all target stocks.",
    )
    return parser.parse_args()


def print_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status != "ok":
        print(f"status: {status}")
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    print("Technical trend prediction complete")
    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"model_used: {result['model_used']}")
    print(f"decision_threshold: {result['decision_threshold']:.4f}")
    print(f"target_return_threshold: {result['target_return_threshold']:.4f}")
    print(f"prediction_rows_saved: {result['prediction_rows_saved']}")

    for prediction in result.get("predictions", []):
        print(
            f"{prediction['symbol']} {prediction['latest_date']} "
            f"{prediction['predicted_direction']} "
            f"confidence={prediction['confidence']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
