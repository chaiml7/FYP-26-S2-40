"""
Train the XGBoost technical-analysis model and save it locally.

Run from repo root after syncing indicators:
    python scripts/train_xgboost_technical_model.py --all
    python scripts/train_xgboost_technical_model.py --symbol NVDA
    python scripts/train_xgboost_technical_model.py --all --train-before-date 2026-06-04
"""
import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.indicator_service import (  # noqa: E402
    get_all_technical_indicators_from_supabase,
    get_technical_indicators_from_supabase,
)
from services.technical.price_service import get_stock_by_symbol  # noqa: E402
from services.technical.xgboost_model_service import (  # noqa: E402
    DEFAULT_LOOKAHEAD_DAYS,
    DEFAULT_XGBOOST_MAX_FEATURES,
    TARGET_RETURN_THRESHOLD,
    XGBOOST_ARTIFACT_PATH,
    train_xgboost_artifact,
)


def main() -> int:
    args = parse_args()
    indicator_df = load_indicator_data(args.symbol)
    if indicator_df.empty:
        symbol_text = args.symbol.upper() if args.symbol else "all target tickers"
        print(f"status: no_data")
        print(f"reason: No technical_indicators rows found for {symbol_text}")
        return 1

    result = train_xgboost_artifact(
        indicator_df=indicator_df,
        model_scope="single_ticker" if args.symbol else "global_all_tickers",
        trained_symbol=args.symbol.upper() if args.symbol else None,
        lookahead_days=args.lookahead_days,
        threshold=args.threshold,
        train_before_date=args.train_before_date,
        n_splits=args.n_splits,
        max_features=args.max_features,
        artifact_path=args.output_path,
    )
    print_result(result)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a separate XGBoost technical model. The classifier "
            "predicts neutral/down/up direction and the regressor predicts "
            "next OHLC returns."
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
        "--lookahead-days",
        type=int,
        default=DEFAULT_LOOKAHEAD_DAYS,
        help="Future trading-day horizon for labels. Default: 1",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Return threshold for up/down classes. Default: 0.002",
    )
    parser.add_argument(
        "--train-before-date",
        help=(
            "Only train on indicator rows before this date. "
            "Example: 2026-06-04 excludes 2026-06-04 itself."
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Chronological TimeSeriesSplit fold count. Default: 5",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_XGBOOST_MAX_FEATURES,
        help="Top XGBoost feature-importance features to keep. Default: 45",
    )
    parser.add_argument(
        "--output-path",
        default=str(XGBOOST_ARTIFACT_PATH),
        help="Where to save the XGBoost .joblib artifact.",
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


def print_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status != "ok":
        print(f"status: {status}")
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    class_metrics = result.get("classification_metrics", {})
    regression_metrics = result.get("regression_metrics", {})
    thresholds = result.get("class_probability_thresholds", {})

    print("XGBoost technical model training complete")
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
    print(f"indicator_rows: {result['indicator_rows']}")
    print(f"clean_training_rows: {result['clean_training_rows']}")
    print(f"lookahead_days: {result['lookahead_days']}")
    print(f"direction_threshold: {result['direction_threshold']:.4f}")
    print(f"selected_feature_count: {result['selected_feature_count']}")
    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"class_threshold_down: {thresholds.get('down')}")
    print(f"class_threshold_up: {thresholds.get('up')}")
    print(f"classification_accuracy: {format_metric(class_metrics.get('accuracy'))}")
    print(f"classification_balanced_accuracy: {format_metric(class_metrics.get('balanced_accuracy'))}")
    print(f"classification_f1_macro: {format_metric(class_metrics.get('f1_macro'))}")
    print(f"classification_precision_macro: {format_metric(class_metrics.get('precision_macro'))}")
    print(f"classification_recall_macro: {format_metric(class_metrics.get('recall_macro'))}")
    print(f"classification_mcc: {format_metric(class_metrics.get('mcc'))}")
    print(f"regression_mae: {format_metric(regression_metrics.get('mae'))}")
    print(f"regression_mse: {format_metric(regression_metrics.get('mse'))}")
    print(f"regression_r2: {format_metric(regression_metrics.get('r2'))}")
    print(f"regression_explained_variance: {format_metric(regression_metrics.get('explained_variance'))}")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
