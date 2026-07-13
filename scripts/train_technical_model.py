"""
Train the shared technical-analysis model and save it locally.

Run from repo root after syncing indicators:
    python scripts/train_technical_model.py
    python scripts/train_technical_model.py --all
    python scripts/train_technical_model.py --symbol NVDA
    python scripts/train_technical_model.py --all --threshold 0.005
    python scripts/train_technical_model.py --all --train-before-date 2026-06-04

Use --sync-first to run the Supabase price/indicator sync before training.
"""
import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.model_service import TARGET_RETURN_THRESHOLD
from services.technical.prediction_pipeline import train_global_technical_model


def main() -> int:
    args = parse_args()
    result = train_global_technical_model(
        symbol=args.symbol.upper() if args.symbol else None,
        sync_first=args.sync_first,
        period=args.period,
        interval=args.interval,
        train_before_date=args.train_before_date,
        target_return_threshold=args.threshold,
    )

    print_result(result)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ML targets, train/evaluate the shared LightGBM technical "
            "model, and save it locally."
        )
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Train on every prediction-target ticker. This is the default.",
    )
    target_group.add_argument(
        "--symbol",
        help="Train only on one ticker's stored technical_indicators rows.",
    )
    parser.add_argument(
        "--sync-first",
        action="store_true",
        help="Run yfinance -> Supabase sync before training.",
    )
    parser.add_argument(
        "--period",
        default="10y",
        help="yfinance lookback period used only with --sync-first. Default: 10y",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval used only with --sync-first. Default: 1d",
    )
    parser.add_argument(
        "--train-before-date",
        help=(
            "Only train on indicator rows before this date. "
            "Example: 2026-06-04 excludes 2026-06-04 itself."
        ),
    )
    parser.add_argument(
        "--threshold",
        "--target-threshold",
        dest="threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help=(
            "Return threshold for class 1/up. "
            "Use 0.002 for 0.20%%, 0.005 for 0.50%%. Default: 0.002"
        ),
    )
    return parser.parse_args()


def print_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status != "ok":
        print(f"status: {status}")
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    print("Technical model training complete")
    print(f"model_scope: {result['model_scope']}")
    if result.get("trained_symbol"):
        print(f"trained_symbol: {result['trained_symbol']}")
    if result.get("train_before_date"):
        print(f"train_before_date: {result['train_before_date']}")
    if result.get("training_start_date") or result.get("training_end_date"):
        print(
            "training_date_range: "
            f"{result.get('training_start_date')} to {result.get('training_end_date')}"
        )
    print(f"symbols_trained_count: {result['symbols_trained_count']}")
    if result.get("raw_indicator_rows") is not None:
        print(f"raw_indicator_rows: {result['raw_indicator_rows']}")
    print(f"indicator_rows: {result['indicator_rows']}")
    print(f"clean_training_rows: {result['clean_training_rows']}")
    print(f"model_used: {result['model_used']}")
    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"target_return_threshold: {result['target_return_threshold']:.4f}")
    print(f"decision_threshold: {result['decision_threshold']:.4f}")
    print(f"accuracy: {format_metric(result.get('accuracy'))}")
    print(f"precision: {format_metric(result.get('precision'))}")
    print(f"recall: {format_metric(result.get('recall'))}")
    print(f"f1_score: {format_metric(result.get('f1_score'))}")
    print(f"roc_auc: {format_metric(result.get('roc_auc'))}")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
