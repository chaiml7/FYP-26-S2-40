"""
Train a binary XGBoost technical-analysis direction model and save it locally.

Run from repo root after syncing indicators:
    python scripts/train_binary_xgboost_technical_model.py --all
    python scripts/train_binary_xgboost_technical_model.py --symbol NVDA
    python scripts/train_binary_xgboost_technical_model.py --symbols AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO
    python scripts/train_binary_xgboost_technical_model.py --all --train-before-date 2026-06-04
"""
import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from services.technical.binary_xgboost_model_service import (  # noqa: E402
    BINARY_XGBOOST_ARTIFACT_PATH,
    DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
    DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
    DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE,
    DEFAULT_LOOKAHEAD_DAYS,
    DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    DEFAULT_XGBOOST_MAX_FEATURES,
    TARGET_RETURN_THRESHOLD,
    train_binary_xgboost_artifact,
)
from services.technical.indicator_service import (  # noqa: E402
    get_all_technical_indicators_from_supabase,
    get_technical_indicators_from_supabase,
)
from services.technical.price_service import get_stock_by_symbol  # noqa: E402


def main() -> int:
    args = parse_args()
    selected_symbols = parse_symbols(args.symbols)
    indicator_df = load_indicator_data(args.symbol, selected_symbols)
    if indicator_df.empty:
        symbol_text = selected_symbol_text(args.symbol, selected_symbols)
        print("status: no_data")
        print(f"reason: No technical_indicators rows found for {symbol_text}")
        return 1

    model_scope = model_scope_for_args(args.symbol, selected_symbols)
    result = train_binary_xgboost_artifact(
        indicator_df=indicator_df,
        model_scope=model_scope,
        trained_symbol=args.symbol.upper() if args.symbol else None,
        lookahead_days=args.lookahead_days,
        threshold=args.threshold,
        train_before_date=args.train_before_date,
        n_splits=args.n_splits,
        max_features=args.max_features,
        ensemble_size=args.ensemble_size,
        use_sample_weighting=not args.no_sample_weighting,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        early_stopping_rounds=args.early_stopping_rounds,
        artifact_path=args.output_path,
    )
    print_result(result)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a binary XGBoost technical model. Target is "
            "1 when future close return is greater than the threshold, else 0."
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
    target_group.add_argument(
        "--symbols",
        nargs="+",
        help=(
            "Train one shared model on only these tickers. Accepts space-separated "
            "or comma-separated symbols. Example: --symbols AAPL MSFT NVDA"
        ),
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
        help="Return threshold for class 1/up. Default: 0.002",
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
        "--ensemble-size",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE,
        help="Number of seeded XGBoost models to average. Default: 3",
    )
    parser.add_argument(
        "--no-sample-weighting",
        action="store_true",
        help="Disable liquidity/outlier sample weighting.",
    )
    parser.add_argument(
        "--purge-days",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
        help="Days to purge before each validation fold. Default: 1",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
        help="Days to embargo after each validation fold. Default: 0",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
        help="Early stopping rounds for internal training eval slices. Default: 50",
    )
    parser.add_argument(
        "--output-path",
        default=str(BINARY_XGBOOST_ARTIFACT_PATH),
        help="Where to save the binary XGBoost .joblib artifact.",
    )
    return parser.parse_args()


def parse_symbols(value: list[str] | None) -> list[str]:
    if not value:
        return []

    symbols = []
    for item in value:
        for symbol in item.split(","):
            clean_symbol = symbol.strip().upper()
            if clean_symbol and clean_symbol not in symbols:
                symbols.append(clean_symbol)
    return symbols


def load_indicator_data(symbol: str | None, symbols: list[str] | None = None):
    if symbol:
        clean_symbol = symbol.upper()
        stock = get_stock_by_symbol(clean_symbol)
        if stock is None:
            return empty_frame()
        return get_technical_indicators_from_supabase(stock["id"], stock["symbol"])

    if symbols:
        frames = []
        missing_symbols = []
        for clean_symbol in symbols:
            stock = get_stock_by_symbol(clean_symbol)
            if stock is None:
                missing_symbols.append(clean_symbol)
                continue

            indicator_df = get_technical_indicators_from_supabase(
                stock["id"],
                stock["symbol"],
            )
            if indicator_df.empty:
                missing_symbols.append(clean_symbol)
                continue
            frames.append(indicator_df)

        if missing_symbols:
            print(
                "warning: no technical_indicators rows found for "
                f"{', '.join(missing_symbols)}"
            )
        if not frames:
            return empty_frame()

        import pandas as pd

        return pd.concat(frames, ignore_index=True)

    return get_all_technical_indicators_from_supabase()


def selected_symbol_text(symbol: str | None, symbols: list[str]) -> str:
    if symbol:
        return symbol.upper()
    if symbols:
        return ", ".join(symbols)
    return "all target tickers"


def model_scope_for_args(symbol: str | None, symbols: list[str]) -> str:
    if symbol:
        return "single_ticker"
    if symbols:
        return "selected_tickers"
    return "global_all_tickers"


def empty_frame():
    import pandas as pd

    return pd.DataFrame()


def print_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status != "ok":
        print(f"status: {status}")
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    metrics = result.get("metrics", {})
    distribution = result.get("target_distribution", {})
    print("Binary XGBoost technical model training complete")
    print(f"model_scope: {result['model_scope']}")
    if result.get("trained_symbol"):
        print(f"trained_symbol: {result['trained_symbol']}")
    if result.get("symbols_trained"):
        print(f"symbols_trained: {', '.join(result['symbols_trained'])}")
    if result.get("train_before_date"):
        print(f"train_before_date: {result['train_before_date']}")
    print(
        "training_date_range: "
        f"{result.get('training_start_date')} to {result.get('training_end_date')}"
    )
    print(f"symbols_trained_count: {result['symbols_trained_count']}")
    print(f"indicator_rows: {result['indicator_rows']}")
    print(f"clean_training_rows: {result['clean_training_rows']}")
    print(f"target_distribution: {distribution}")
    print(f"lookahead_days: {result['lookahead_days']}")
    print(f"direction_threshold: {result['direction_threshold']:.4f}")
    print(f"decision_threshold: {result['decision_threshold']:.4f}")
    print(f"selected_feature_count: {result['selected_feature_count']}")
    print(f"ensemble_size: {result.get('ensemble_size')}")
    print(f"sample_weighting: {result.get('sample_weighting')}")
    print(f"purge_days: {result.get('purge_days')}")
    print(f"embargo_days: {result.get('embargo_days')}")
    print(f"early_stopping_rounds: {result.get('early_stopping_rounds')}")
    print(f"model_artifact_path: {result['model_artifact_path']}")
    print(f"accuracy: {format_metric(metrics.get('accuracy'))}")
    print(f"balanced_accuracy: {format_metric(metrics.get('balanced_accuracy'))}")
    print(f"precision: {format_metric(metrics.get('precision'))}")
    print(f"recall: {format_metric(metrics.get('recall'))}")
    print(f"f1_score: {format_metric(metrics.get('f1_score'))}")
    print(f"roc_auc: {format_metric(metrics.get('roc_auc'))}")
    print(f"mcc: {format_metric(metrics.get('mcc'))}")
    print(f"majority_baseline_accuracy: {format_metric(metrics.get('majority_baseline_accuracy'))}")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
