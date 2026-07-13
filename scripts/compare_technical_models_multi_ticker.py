"""
Compare saved technical-analysis models across multiple tickers in one report.

Examples:
    python scripts/compare_technical_models_multi_ticker.py --start-date 2026-01-01 --end-date 2026-06-04
    python scripts/compare_technical_models_multi_ticker.py --models binary-xgboost --symbols AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO --start-date 2026-01-01 --end-date 2026-06-04

The script reads saved local artifacts and Supabase technical_indicators. It
does not train models and does not write to Supabase.
"""
import argparse
import html
import json
import sys
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compare_technical_models import (  # noqa: E402
    DEFAULT_MODELS,
    TARGET_RETURN_THRESHOLD,
    calculate_metrics,
    parse_model_names,
    run_comparison,
)

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
    try:
        result = run_multi_ticker_comparison(args)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    output_html = Path(args.output_html or "technical_analysis/selected_tickers_model_comparison.html")
    output_json = Path(args.output_json or "technical_analysis/selected_tickers_model_comparison.json")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_html(result), encoding="utf-8")
    output_json.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    print_result(result, output_html, output_json)
    return 0 if result.get("status") == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved technical-analysis models across multiple tickers "
            "and write one combined HTML/JSON report."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=(
            "Tickers to include. Accepts space-separated or comma-separated "
            "symbols. Default: the 10 FYP tickers."
        ),
    )
    parser.add_argument(
        "--models",
        default="binary-xgboost",
        help=(
            "Comma-separated models to compare. Default: binary-xgboost. "
            f"Use all models with: {','.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--start-date",
        help="First as-of feature date to test. Example: 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        help="Last as-of feature date to test. Example: 2026-06-04",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=120,
        help="Most recent eligible rows to test when --start-date is omitted. Default: 120",
    )
    parser.add_argument(
        "--threshold",
        "--target-threshold",
        dest="threshold",
        type=float,
        default=TARGET_RETURN_THRESHOLD,
        help="Common actual next-day return threshold for class 1/up. Default: 0.002",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested ticker or model cannot be evaluated.",
    )
    parser.add_argument(
        "--output-html",
        help="Local HTML comparison report path.",
    )
    parser.add_argument(
        "--output-json",
        help="Local JSON comparison report path.",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=400,
        help="Maximum per-ticker rows shown in the HTML table. Default: 400",
    )
    return parser.parse_args()


def run_multi_ticker_comparison(args: argparse.Namespace) -> dict[str, Any]:
    symbols = parse_symbols(args.symbols)
    model_names = parse_model_names(args.models)
    per_symbol = {}
    failures = {}

    for symbol in symbols:
        comparison_args = Namespace(
            symbol=symbol,
            models=",".join(model_names),
            start_date=args.start_date,
            end_date=args.end_date,
            rows=args.rows,
            threshold=float(args.threshold),
            require_all=args.require_all,
            output_html=None,
            output_json=None,
            max_table_rows=args.max_table_rows,
        )
        result = run_comparison(comparison_args)
        if result.get("status") == "ok":
            per_symbol[symbol] = result
        else:
            failures[symbol] = result

    if args.require_all and failures:
        return {
            "status": "error",
            "reason": "At least one requested ticker/model could not be evaluated",
            "symbols_requested": symbols,
            "models_requested": model_names,
            "failures": failures,
        }

    if not per_symbol:
        return {
            "status": "no_data",
            "reason": "No requested tickers could be evaluated",
            "symbols_requested": symbols,
            "models_requested": model_names,
            "failures": failures,
        }

    overall_summary = aggregate_overall_summary(per_symbol, model_names)
    return {
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "supabase_technical_indicators",
        "supabase_writes": False,
        "symbols_requested": symbols,
        "symbols_evaluated": sorted(per_symbol),
        "models_requested": model_names,
        "comparison_threshold": float(args.threshold),
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "requested_rows": int(args.rows),
        "overall_summary": overall_summary,
        "per_symbol": per_symbol,
        "failures": failures,
        "max_table_rows": int(args.max_table_rows),
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


def aggregate_overall_summary(
    per_symbol: dict[str, dict[str, Any]],
    model_names: list[str],
) -> dict[str, Any]:
    summary = {}
    for model_name in model_names:
        predictions = []
        tickers = []
        for symbol, symbol_result in per_symbol.items():
            model_payload = symbol_result.get("models", {}).get(model_name)
            if not model_payload:
                continue
            model_predictions = model_payload.get("predictions", [])
            if not model_predictions:
                continue
            tickers.append(symbol)
            predictions.extend(model_predictions)

        if not predictions:
            continue
        summary[model_name] = {
            **calculate_metrics(predictions),
            "ticker_count": len(tickers),
            "tickers": tickers,
        }
    return summary


def render_html(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return minimal_html(
            "Multi-Ticker Model Comparison",
            f"<p>{escape_text(result.get('reason', 'No comparison generated'))}</p>",
        )

    body = f"""
    <header>
      <h1>Multi-Ticker Model Comparison</h1>
      <p class="muted">
        Symbols: {escape_text(", ".join(result["symbols_evaluated"]))}
        | Models: {escape_text(", ".join(result["models_requested"]))}
        | Target threshold: {result["comparison_threshold"]:.4f}
      </p>
    </header>
    <section class="panel">
      <h2>Overall Metrics Across All Evaluated Tickers</h2>
      {render_overall_table(result)}
    </section>
    <section class="panel">
      <h2>Per-Ticker Metrics</h2>
      {render_per_ticker_table(result)}
    </section>
    {render_failures(result.get("failures", {}))}
    """
    return minimal_html("Multi-Ticker Model Comparison", body)


def render_overall_table(result: dict[str, Any]) -> str:
    rows = []
    for model_name, metrics in sorted(
        result.get("overall_summary", {}).items(),
        key=lambda item: item[1].get("balanced_accuracy") or -1,
        reverse=True,
    ):
        rows.append(
            "<tr>"
            f"<th>{escape_text(model_name)}</th>"
            f"<td>{metrics.get('ticker_count')}</td>"
            f"<td>{metrics.get('total_predictions')}</td>"
            f"<td>{format_metric(metrics.get('accuracy'))}</td>"
            f"<td>{format_metric(metrics.get('balanced_accuracy'))}</td>"
            f"<td>{format_metric(metrics.get('precision'))}</td>"
            f"<td>{format_metric(metrics.get('recall'))}</td>"
            f"<td>{format_metric(metrics.get('f1_score'))}</td>"
            f"<td>{format_metric(metrics.get('roc_auc'))}</td>"
            f"<td>{format_metric(metrics.get('mcc'))}</td>"
            f"<td>{metrics.get('correct_predictions')}/{metrics.get('total_predictions')}</td>"
            "</tr>"
        )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Tickers</th>
          <th>Rows</th>
          <th>Accuracy</th>
          <th>Balanced Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>ROC AUC</th>
          <th>MCC</th>
          <th>Correct</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_per_ticker_table(result: dict[str, Any]) -> str:
    rows = []
    for symbol in sorted(result.get("per_symbol", {})):
        symbol_result = result["per_symbol"][symbol]
        for model_name, metrics in sorted(symbol_result.get("summary", {}).items()):
            rows.append(
                "<tr>"
                f"<th>{escape_text(symbol)}</th>"
                f"<td>{escape_text(model_name)}</td>"
                f"<td>{escape_text(symbol_result.get('common_window_start'))}</td>"
                f"<td>{escape_text(symbol_result.get('common_window_end'))}</td>"
                f"<td>{symbol_result.get('common_rows')}</td>"
                f"<td>{format_metric(metrics.get('accuracy'))}</td>"
                f"<td>{format_metric(metrics.get('balanced_accuracy'))}</td>"
                f"<td>{format_metric(metrics.get('precision'))}</td>"
                f"<td>{format_metric(metrics.get('recall'))}</td>"
                f"<td>{format_metric(metrics.get('f1_score'))}</td>"
                f"<td>{format_metric(metrics.get('roc_auc'))}</td>"
                f"<td>{metrics.get('correct_predictions')}/{metrics.get('total_predictions')}</td>"
                "</tr>"
            )
    return f"""
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Model</th>
          <th>Window Start</th>
          <th>Window End</th>
          <th>Rows</th>
          <th>Accuracy</th>
          <th>Balanced Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>ROC AUC</th>
          <th>Correct</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_failures(failures: dict[str, Any]) -> str:
    if not failures:
        return ""

    rows = []
    for symbol, payload in failures.items():
        rows.append(
            "<tr>"
            f"<th>{escape_text(symbol)}</th>"
            f"<td>{escape_text(payload.get('status'))}</td>"
            f"<td>{escape_text(payload.get('reason'))}</td>"
            "</tr>"
        )
    return f"""
    <section class="panel">
      <h2>Skipped Tickers</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Status</th><th>Reason</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def minimal_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape_text(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #172033;
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header, section {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      padding: 28px 20px 8px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .muted {{
      color: #667085;
    }}
    .panel {{
      margin: 16px auto;
      padding: 18px;
      background: #ffffff;
      border: 1px solid #d7dde8;
      border-radius: 8px;
      box-sizing: border-box;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #d7dde8;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f8fafc;
      color: #344054;
      font-weight: 700;
    }}
    @media (max-width: 780px) {{
      header, .panel {{
        margin-left: 14px;
        margin-right: 14px;
      }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def escape_text(value: Any) -> str:
    if value is None:
        return "n/a"
    return html.escape(str(value), quote=True)


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


def print_result(result: dict[str, Any], output_html: Path, output_json: Path) -> None:
    print(f"status: {result['status']}")
    if result["status"] != "ok":
        print(f"reason: {result.get('reason', 'unknown reason')}")
        return

    print(f"symbols_evaluated: {', '.join(result['symbols_evaluated'])}")
    print(f"models_requested: {', '.join(result['models_requested'])}")
    print(f"comparison_threshold: {result['comparison_threshold']:.4f}")
    print("")
    print("Overall metrics across all evaluated tickers:")
    for model_name, metrics in sorted(
        result.get("overall_summary", {}).items(),
        key=lambda item: item[1].get("balanced_accuracy") or -1,
        reverse=True,
    ):
        print(
            f"{model_name}: "
            f"accuracy={format_metric(metrics.get('accuracy'))}, "
            f"balanced_accuracy={format_metric(metrics.get('balanced_accuracy'))}, "
            f"f1={format_metric(metrics.get('f1_score'))}, "
            f"roc_auc={format_metric(metrics.get('roc_auc'))}, "
            f"correct={metrics.get('correct_predictions')}/{metrics.get('total_predictions')}"
        )
    if result.get("failures"):
        print("")
        print("Skipped tickers:")
        for symbol, payload in result["failures"].items():
            print(f"{symbol}: {payload.get('status')} - {payload.get('reason')}")
    print("")
    print(f"html_report: {output_html}")
    print(f"json_report: {output_json}")


if __name__ == "__main__":
    raise SystemExit(main())
