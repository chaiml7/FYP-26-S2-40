"""
Train one binary XGBoost model per ticker and write a metrics report.

Example:
    python scripts/train_binary_xgboost_per_ticker_report.py --start-date 2026-01-01 --end-date 2026-06-04

The script trains each ticker separately using rows before the test window,
tests each saved ticker-specific artifact on the same requested window, and
writes one Markdown report plus one JSON report.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

import compare_technical_models as comparison  # noqa: E402
from services.technical.binary_xgboost_model_service import (  # noqa: E402
    DEFAULT_BINARY_XGBOOST_EARLY_STOPPING_ROUNDS,
    DEFAULT_BINARY_XGBOOST_EMBARGO_DAYS,
    DEFAULT_BINARY_XGBOOST_ENSEMBLE_SIZE,
    DEFAULT_BINARY_XGBOOST_PURGE_DAYS,
    DEFAULT_LOOKAHEAD_DAYS,
    DEFAULT_XGBOOST_MAX_FEATURES,
    TARGET_RETURN_THRESHOLD,
    train_binary_xgboost_artifact,
)
from services.technical.indicator_service import (  # noqa: E402
    get_technical_indicators_from_supabase,
)
from services.technical.price_service import get_stock_by_symbol  # noqa: E402

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "TSLA",
    "AMD",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "PLTR",
    "AVGO",
]


def main() -> int:
    args = parse_args()
    result = run_report(args)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(result), encoding="utf-8")
    output_json.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")

    print_result(result, output_md, output_json)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one binary XGBoost model per ticker, test each on the same "
            "date window, and write accuracy/balanced-accuracy reports."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=(
            "Tickers to train separately. Accepts space-separated or comma-separated "
            "symbols. Default: the 10 FYP tickers."
        ),
    )
    parser.add_argument(
        "--start-date",
        default="2026-01-01",
        help="First as-of date in the test window. Default: 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2026-06-04",
        help="Last as-of date in the test window. Default: 2026-06-04",
    )
    parser.add_argument(
        "--train-before-date",
        help=(
            "Training cutoff. Defaults to --start-date, so the test window is "
            "out-of-sample."
        ),
    )
    parser.add_argument(
        "--threshold",
        "--target-threshold",
        dest="threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Direction target threshold. Default: 0.002",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=DEFAULT_LOOKAHEAD_DAYS,
        help="Future trading-day horizon for labels. Default: 1",
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
        help="Number of seeded XGBoost models to average per ticker. Default: 3",
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
        "--rows",
        type=int,
        default=120,
        help="Comparison rows if --start-date is omitted in future use. Default: 120",
    )
    parser.add_argument(
        "--artifact-dir",
        default="backend/artifacts/per_ticker_binary_xgboost",
        help="Directory for per-ticker .joblib artifacts.",
    )
    parser.add_argument(
        "--output-md",
        default="technical_analysis/per_ticker_binary_xgboost_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--output-json",
        default="technical_analysis/per_ticker_binary_xgboost_report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail the run if any ticker cannot be trained or tested.",
    )
    return parser.parse_args()


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    train_before_date = normalize_date(args.train_before_date or args.start_date)
    artifact_dir = Path(args.artifact_dir)
    rows = []
    failures = {}

    for symbol in symbols:
        print("")
        print(f"=== {symbol} ===")
        try:
            row = train_and_test_symbol(
                symbol=symbol,
                args=args,
                train_before_date=train_before_date,
                artifact_dir=artifact_dir,
            )
        except Exception as exc:
            row = {
                "symbol": symbol,
                "status": "error",
                "reason": str(exc),
            }

        if row.get("status") == "ok":
            rows.append(row)
            print(
                f"{symbol}: accuracy={format_metric(row['test_metrics'].get('accuracy'))}, "
                f"balanced_accuracy={format_metric(row['test_metrics'].get('balanced_accuracy'))}"
            )
        else:
            failures[symbol] = row
            print(f"{symbol}: {row.get('status')} - {row.get('reason')}")
            if args.require_all:
                break

    status = "ok" if rows and not (args.require_all and failures) else "error"
    if not rows:
        status = "no_data"

    return {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "binary-xgboost",
        "training_mode": "one_model_per_ticker",
        "symbols_requested": symbols,
        "symbols_evaluated": [row["symbol"] for row in rows],
        "train_before_date": train_before_date,
        "test_start_date": normalize_date(args.start_date),
        "test_end_date": normalize_date(args.end_date),
        "direction_threshold": float(args.threshold),
        "lookahead_days": int(args.lookahead_days),
        "n_splits": int(args.n_splits),
        "max_features": int(args.max_features),
        "ensemble_size": int(args.ensemble_size),
        "sample_weighting": not args.no_sample_weighting,
        "purge_days": int(args.purge_days),
        "embargo_days": int(args.embargo_days),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "artifact_dir": str(artifact_dir),
        "rows": rows,
        "failures": failures,
        "overall_test_metrics": aggregate_overall_metrics(rows),
    }


def train_and_test_symbol(
    symbol: str,
    args: argparse.Namespace,
    train_before_date: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    stock = get_stock_by_symbol(symbol)
    if stock is None:
        return {
            "symbol": symbol,
            "status": "no_data",
            "reason": f"{symbol} was not found in the stocks table",
        }

    indicator_df = get_technical_indicators_from_supabase(stock["id"], stock["symbol"])
    if indicator_df.empty:
        return {
            "symbol": symbol,
            "status": "no_data",
            "reason": f"No technical_indicators rows found for {symbol}",
        }

    artifact_path = artifact_dir / f"technical_xgboost_binary_{symbol}.joblib"
    training_result = train_binary_xgboost_artifact(
        indicator_df=indicator_df,
        model_scope="single_ticker",
        trained_symbol=symbol,
        lookahead_days=args.lookahead_days,
        threshold=args.threshold,
        train_before_date=train_before_date,
        n_splits=args.n_splits,
        max_features=args.max_features,
        ensemble_size=args.ensemble_size,
        use_sample_weighting=not args.no_sample_weighting,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        early_stopping_rounds=args.early_stopping_rounds,
        artifact_path=artifact_path,
    )
    if training_result.get("status") != "ok":
        return {
            "symbol": symbol,
            "status": training_result.get("status", "training_failed"),
            "reason": training_result.get("reason", "Training failed"),
            "training_result": training_result,
        }

    previous_binary_path = comparison.MODEL_PATHS["binary-xgboost"]
    comparison.MODEL_PATHS["binary-xgboost"] = artifact_path
    try:
        comparison_result = comparison.run_comparison(
            Namespace(
                symbol=symbol,
                models="binary-xgboost",
                start_date=normalize_date(args.start_date),
                end_date=normalize_date(args.end_date),
                rows=args.rows,
                threshold=float(args.threshold),
                require_all=True,
                output_html=None,
                output_json=None,
                max_table_rows=0,
            )
        )
    finally:
        comparison.MODEL_PATHS["binary-xgboost"] = previous_binary_path

    if comparison_result.get("status") != "ok":
        return {
            "symbol": symbol,
            "status": comparison_result.get("status", "test_failed"),
            "reason": comparison_result.get("reason", "Testing failed"),
            "training_result": training_result,
            "comparison_result": comparison_result,
            "artifact_path": str(artifact_path),
        }

    test_metrics = comparison_result["summary"]["binary-xgboost"]
    training_metrics = training_result.get("metrics", {})
    return {
        "symbol": symbol,
        "status": "ok",
        "artifact_path": str(artifact_path),
        "training_start_date": training_result.get("training_start_date"),
        "training_end_date": training_result.get("training_end_date"),
        "test_start_date": comparison_result.get("common_window_start"),
        "test_end_date": comparison_result.get("common_window_end"),
        "test_rows": comparison_result.get("common_rows"),
        "clean_training_rows": training_result.get("clean_training_rows"),
        "target_distribution": training_result.get("target_distribution"),
        "decision_threshold": training_result.get("decision_threshold"),
        "training_metrics": training_metrics,
        "test_metrics": test_metrics,
    }


def parse_symbols(values: list[str]) -> list[str]:
    symbols = []
    for value in values:
        for symbol in value.split(","):
            clean_symbol = symbol.strip().upper()
            if clean_symbol and clean_symbol not in symbols:
                symbols.append(clean_symbol)
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def normalize_date(value: str) -> str:
    return pd.to_datetime(value, errors="raise").date().isoformat()


def aggregate_overall_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = []
    for row in rows:
        metrics = row.get("test_metrics", {})
        predictions.append(
            {
                "symbol": row["symbol"],
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "roc_auc": metrics.get("roc_auc"),
                "mcc": metrics.get("mcc"),
                "test_rows": row.get("test_rows"),
            }
        )

    if not predictions:
        return {}

    return {
        "ticker_count": len(predictions),
        "mean_accuracy": average_metric(predictions, "accuracy"),
        "mean_balanced_accuracy": average_metric(predictions, "balanced_accuracy"),
        "mean_precision": average_metric(predictions, "precision"),
        "mean_recall": average_metric(predictions, "recall"),
        "mean_f1_score": average_metric(predictions, "f1_score"),
        "mean_roc_auc": average_metric(predictions, "roc_auc"),
        "mean_mcc": average_metric(predictions, "mcc"),
        "total_test_rows": int(
            sum(int(item["test_rows"] or 0) for item in predictions)
        ),
    }


def average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Per-Ticker Binary XGBoost Report",
        "",
        f"Generated at: `{result.get('generated_at')}`",
        "",
        "## Setup",
        "",
        f"- Model: `{result.get('model')}`",
        f"- Training mode: `{result.get('training_mode')}`",
        f"- Train before date: `{result.get('train_before_date')}`",
        f"- Test window: `{result.get('test_start_date')}` to `{result.get('test_end_date')}`",
        f"- Direction threshold: `{result.get('direction_threshold')}`",
        f"- Lookahead days: `{result.get('lookahead_days')}`",
        f"- Ensemble size: `{result.get('ensemble_size')}`",
        f"- Sample weighting: `{result.get('sample_weighting')}`",
        f"- Purge days: `{result.get('purge_days')}`",
        f"- Embargo days: `{result.get('embargo_days')}`",
        f"- Early stopping rounds: `{result.get('early_stopping_rounds')}`",
        "",
        "The accuracy and balanced accuracy below are from the out-of-sample test window. "
        "The internal training metrics are chronological validation metrics from rows before the test window.",
        "",
        "## Overall Mean Metrics",
        "",
    ]
    overall = result.get("overall_test_metrics", {})
    if overall:
        lines.extend(
            [
                f"- Ticker count: `{overall.get('ticker_count')}`",
                f"- Total test rows: `{overall.get('total_test_rows')}`",
                f"- Mean accuracy: `{format_metric(overall.get('mean_accuracy'))}`",
                f"- Mean balanced accuracy: `{format_metric(overall.get('mean_balanced_accuracy'))}`",
                f"- Mean F1 score: `{format_metric(overall.get('mean_f1_score'))}`",
                f"- Mean ROC AUC: `{format_metric(overall.get('mean_roc_auc'))}`",
                "",
            ]
        )
    else:
        lines.extend(["No successful ticker evaluations.", ""])

    lines.extend(
        [
            "## Per-Ticker Test Metrics",
            "",
            "| Symbol | Test Rows | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC AUC | MCC | Correct | Train Rows | Artifact |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(result.get("rows", []), key=lambda item: item["symbol"]):
        metrics = row.get("test_metrics", {})
        lines.append(
            "| "
            f"{row['symbol']} | "
            f"{row.get('test_rows')} | "
            f"{format_metric(metrics.get('accuracy'))} | "
            f"{format_metric(metrics.get('balanced_accuracy'))} | "
            f"{format_metric(metrics.get('precision'))} | "
            f"{format_metric(metrics.get('recall'))} | "
            f"{format_metric(metrics.get('f1_score'))} | "
            f"{format_metric(metrics.get('roc_auc'))} | "
            f"{format_metric(metrics.get('mcc'))} | "
            f"{metrics.get('correct_predictions')}/{metrics.get('total_predictions')} | "
            f"{row.get('clean_training_rows')} | "
            f"`{row.get('artifact_path')}` |"
        )

    if result.get("failures"):
        lines.extend(
            [
                "",
                "## Failed Or Skipped Tickers",
                "",
                "| Symbol | Status | Reason |",
                "| --- | --- | --- |",
            ]
        )
        for symbol, failure in sorted(result["failures"].items()):
            lines.append(
                f"| {symbol} | {failure.get('status')} | {failure.get('reason')} |"
            )

    lines.append("")
    return "\n".join(lines)


def print_result(result: dict[str, Any], output_md: Path, output_json: Path) -> None:
    print("")
    print(f"status: {result['status']}")
    overall = result.get("overall_test_metrics", {})
    if overall:
        print(f"mean_accuracy: {format_metric(overall.get('mean_accuracy'))}")
        print(
            "mean_balanced_accuracy: "
            f"{format_metric(overall.get('mean_balanced_accuracy'))}"
        )
    print(f"markdown_report: {output_md}")
    print(f"json_report: {output_json}")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


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


if __name__ == "__main__":
    raise SystemExit(main())
