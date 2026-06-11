"""
Evaluate the technical-analysis direction classifier for one ticker.

Run from repo root:
    python scripts/evaluate_technical_model.py --symbol AAPL

By default this script writes the yfinance price data into Supabase, stores
stock_prices and technical_indicators rows, then evaluates from stored
technical_indicators data. Use --no-supabase for a local-only experiment.
"""
import argparse
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.indicator_service import (
    add_technical_indicators,
    get_technical_indicators_from_supabase,
    upsert_technical_indicators,
)
from services.technical.model_service import (
    DEFAULT_MAX_FEATURES,
    FEATURES,
    TARGET_RETURN_THRESHOLD,
    get_feature_importance,
    get_model,
    prepare_training_data,
    train_final_model,
    tune_lightgbm_params,
    walk_forward_validation,
)
from services.technical.price_service import (
    add_market_context_features,
    fetch_price_history,
    get_daily_ohlcv_from_supabase,
    get_stock_by_symbol,
    get_stock_prices_from_supabase,
    upsert_daily_ohlcv,
    upsert_stock_prices,
)


METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "baseline_accuracy",
    "majority_baseline_accuracy",
]


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()

    print_header(f"Technical Model Evaluation: {symbol}")
    print(f"Period: {args.period}")
    print(f"Interval: {args.interval}")
    print(f"Walk-forward test window: {args.test_size} trading days")
    print(f"Target threshold: next-day return > {args.target_threshold:.2%}")

    price_df = fetch_price_history(symbol, period=args.period, interval=args.interval)
    if price_df.empty:
        print(f"\nNo yfinance data returned for {symbol}.")
        return 1

    indicator_df = prepare_indicator_data(args, symbol, price_df)
    X, y, clean_df = prepare_training_data(
        indicator_df,
        target_return_threshold=args.target_threshold,
    )
    if X.empty or y.empty:
        print("\nNot enough clean rows to evaluate the model.")
        print("Try a longer --period, such as --period 10y.")
        return 1

    tuning_result = {"best_params": {}, "evaluations": []}
    if not args.no_tuning:
        print_header("Tuning")
        print("Running chronological LightGBM parameter search...")
        tuning_result = tune_lightgbm_params(
            indicator_df,
            initial_train_size=args.initial_train_size,
            test_size=max(args.test_size, 60),
            target_return_threshold=args.target_threshold,
        )
        print(f"Best params: {tuning_result['best_params']}")
        print(f"Best tuning accuracy: {format_metric(tuning_result.get('best_accuracy'))}")

    metrics = walk_forward_validation(
        indicator_df,
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        model_params=tuning_result["best_params"],
        tune_threshold=not args.no_threshold_tuning,
        use_feature_selection=not args.no_feature_selection,
        max_features=args.max_features,
        target_return_threshold=args.target_threshold,
    )
    final_model, _, model_used = train_final_model(
        indicator_df,
        model_params=tuning_result["best_params"],
        target_return_threshold=args.target_threshold,
        use_feature_selection=not args.no_feature_selection,
        max_features=args.max_features,
    )

    print_data_summary(price_df, indicator_df, clean_df, y)
    print_model_summary(model_used, metrics)
    print_metrics(metrics)
    print_confusion_matrix(metrics)
    print_feature_importance(final_model)

    if args.show_windows:
        print_window_metrics(metrics.get("windows", []))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the StockLens technical-analysis classifier."
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Ticker symbol to evaluate. Default: AAPL",
    )
    parser.add_argument(
        "--period",
        default="10y",
        help="yfinance lookback period. Examples: 1y, 2y, 5y, 10y, max. Default: 10y",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval. The model is designed for 1d. Default: 1d",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=30,
        help="Number of trading days in each walk-forward test window. Default: 30",
    )
    parser.add_argument(
        "--initial-train-size",
        type=int,
        default=None,
        help="Optional first training window size. Default: 70%% of clean rows",
    )
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="Print metrics for every walk-forward validation window.",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Minimum next-day return required for target=1. Default: 0.002",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Maximum features kept by LightGBM importance selection. Default: 40",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip LightGBM parameter tuning.",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Use decision threshold 0.50 instead of tuning it on validation windows.",
    )
    parser.add_argument(
        "--no-feature-selection",
        action="store_true",
        help="Use all features instead of selecting by LightGBM feature importance.",
    )
    parser.add_argument(
        "--no-supabase",
        action="store_true",
        help="Skip Supabase sync/readback and evaluate directly from yfinance data.",
    )
    return parser.parse_args()


def prepare_indicator_data(
    args: argparse.Namespace,
    symbol: str,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    if args.no_supabase:
        print("\nUsing local yfinance data directly because --no-supabase was provided.")
        training_price_df = add_market_context_features(
            price_df,
            symbol=symbol,
            period=args.period,
            interval=args.interval,
        )
        return add_technical_indicators(training_price_df)

    stock = get_stock_by_symbol(symbol)
    if stock is None:
        print(f"\n{symbol} was not found in the stocks table; falling back to local yfinance data.")
        training_price_df = add_market_context_features(
            price_df,
            symbol=symbol,
            period=args.period,
            interval=args.interval,
        )
        return add_technical_indicators(training_price_df)

    stock_id = stock["id"]
    raw_result = upsert_daily_ohlcv(stock_id, symbol, price_df)
    raw_price_df = get_daily_ohlcv_from_supabase(stock_id, symbol)
    enriched_price_df = add_market_context_features(
        raw_price_df,
        symbol=symbol,
        period=args.period,
        interval=args.interval,
    )
    stock_prices_result = upsert_stock_prices(stock_id, symbol, enriched_price_df)
    training_price_df = get_stock_prices_from_supabase(stock_id, symbol)
    indicator_df = add_technical_indicators(training_price_df)
    indicator_result = upsert_technical_indicators(stock_id, symbol, indicator_df)
    stored_indicator_df = get_technical_indicators_from_supabase(stock_id, symbol)

    print_header("Supabase Source")
    print(f"daily_ohlcv rows upserted: {raw_result['rows_saved']}")
    print(f"stock_prices rows upserted: {stock_prices_result['rows_saved']}")
    print(f"stock_prices rows read for training: {len(training_price_df)}")
    print(f"technical_indicators rows upserted: {indicator_result['rows_saved']}")
    print(f"technical_indicators rows read for training: {len(stored_indicator_df)}")

    return stored_indicator_df


def print_data_summary(
    price_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    y: pd.Series,
) -> None:
    print_header("Data")
    print(f"Fetched price rows: {len(price_df)}")
    print(f"Indicator rows: {len(indicator_df)}")
    print(f"Clean train/eval rows: {len(clean_df)}")
    print(f"Feature count: {len(FEATURES)}")
    print(f"Date range: {clean_df['date'].iloc[0]} to {clean_df['date'].iloc[-1]}")

    class_counts = y.value_counts().sort_index().to_dict()
    down_count = int(class_counts.get(0, 0))
    up_count = int(class_counts.get(1, 0))
    total = max(1, down_count + up_count)
    print(f"Target distribution: down/equal={down_count} ({down_count / total:.1%}), up={up_count} ({up_count / total:.1%})")


def print_model_summary(model_used: str, metrics: dict[str, Any]) -> None:
    model = get_model()
    print_header("Model")
    print(f"Classifier: {model.__class__.__name__}")
    print(f"Final model: {model_used}")
    print(f"Target: 1 when next-day return is greater than {metrics.get('target_return_threshold', TARGET_RETURN_THRESHOLD):.2%}, else 0")
    print(f"Decision threshold: {format_metric(metrics.get('decision_threshold'))}")
    print("Validation: chronological expanding-window walk-forward split")


def print_metrics(metrics: dict[str, Any]) -> None:
    print_header("Average Metrics")
    for key in METRIC_KEYS:
        print(f"{key}: {format_metric(metrics.get(key))}")

    accuracy = metrics.get("accuracy")
    baseline = metrics.get("baseline_accuracy")
    majority_baseline = metrics.get("majority_baseline_accuracy")
    if accuracy is not None and baseline is not None:
        print(f"vs previous-direction baseline: {accuracy - baseline:+.4f}")
    if accuracy is not None and majority_baseline is not None:
        print(f"vs majority-class baseline: {accuracy - majority_baseline:+.4f}")


def print_confusion_matrix(metrics: dict[str, Any]) -> None:
    windows = metrics.get("windows", [])
    matrix = np.zeros((2, 2), dtype=int)
    for window in windows:
        matrix += np.array(window.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=int)

    print_header("Aggregate Confusion Matrix")
    print("Rows are actual class, columns are predicted class.")
    print("Classes: 0 = down/equal, 1 = up")
    print(f"          pred 0  pred 1")
    print(f"actual 0  {matrix[0, 0]:6d}  {matrix[0, 1]:6d}")
    print(f"actual 1  {matrix[1, 0]:6d}  {matrix[1, 1]:6d}")


def print_feature_importance(model: Any) -> None:
    print_header("Top Features")
    importance = get_feature_importance(model, limit=15)
    if not importance:
        print("Feature importance is not available for this model.")
        return

    for index, item in enumerate(importance, start=1):
        print(f"{index:02d}. {item['feature']}: {item['importance']:.4f}")


def print_window_metrics(windows: list[dict[str, Any]]) -> None:
    print_header("Walk-Forward Windows")
    if not windows:
        print("No validation windows were created.")
        return

    for index, window in enumerate(windows, start=1):
        print(
            f"{index:02d}. "
            f"{window.get('test_start_date')} to {window.get('test_end_date')} | "
            f"train={window.get('train_rows')} test={window.get('test_rows')} | "
            f"accuracy={format_metric(window.get('accuracy'))} "
            f"precision={format_metric(window.get('precision'))} "
            f"recall={format_metric(window.get('recall'))} "
            f"f1={format_metric(window.get('f1_score'))} "
            f"roc_auc={format_metric(window.get('roc_auc'))}"
        )


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{float(value):.4f}"


def print_header(label: str) -> None:
    print(f"\n{'=' * 72}")
    print(label)
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
