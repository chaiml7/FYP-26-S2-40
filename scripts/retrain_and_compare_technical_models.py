"""
Retrain selected technical-analysis models, then compare them on one window.

Clean out-of-sample example:
    python scripts/retrain_and_compare_technical_models.py --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04
    python scripts/retrain_and_compare_technical_models.py --symbol NVDA --models binary-xgboost --train-symbols AAPL MSFT TSLA AMD AMZN GOOGL META NVDA PLTR AVGO --start-date 2026-01-01 --end-date 2026-06-04

Intentional leakage / overfit check:
    python scripts/retrain_and_compare_technical_models.py --symbol NVDA --start-date 2026-01-01 --end-date 2026-06-04 --mode leaky-window

The script runs existing training scripts, then runs compare_technical_models.py
only for the models that successfully retrained. It does not write to Supabase.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_TRAIN_SCRIPTS = {
    "lightgbm": "train_technical_model.py",
    "xgboost": "train_xgboost_technical_model.py",
    "binary-xgboost": "train_binary_xgboost_technical_model.py",
    "catboost": "train_catboost_technical_model.py",
    "lstm": "train_lstm_technical_model.py",
    "arima": "train_arima_technical_model.py",
}

DEFAULT_MODELS = [
    "lightgbm",
    "xgboost",
    "binary-xgboost",
    "catboost",
    "lstm",
    "arima",
]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    selected_models = parse_model_names(args.models)
    cutoff_date = training_cutoff_date(args)
    run_log = {
        "status": "running",
        "mode": args.mode,
        "symbol": args.symbol.upper(),
        "train_scope": args.train_scope,
        "train_symbols": parse_symbols(args.train_symbols),
        "models_requested": selected_models,
        "start_date": normalize_date(args.start_date),
        "end_date": normalize_date(args.end_date) if args.end_date else None,
        "training_cutoff_date": cutoff_date,
        "threshold": float(args.threshold),
        "started_at": pd.Timestamp.utcnow().isoformat(),
        "training_runs": {},
        "comparison": None,
    }

    print_mode_note(args, cutoff_date)
    successful_models = []
    for model_name in selected_models:
        command = build_train_command(
            repo_root=repo_root,
            model_name=model_name,
            args=args,
            cutoff_date=cutoff_date,
        )
        print("")
        print(f"=== Training {model_name} ===")
        if args.dry_run:
            print(format_command(command))
            run_log["training_runs"][model_name] = {
                "status": "dry_run",
                "command": command,
            }
            successful_models.append(model_name)
            continue

        run_result = run_command(command, cwd=repo_root)
        run_log["training_runs"][model_name] = {
            "status": "ok" if run_result["returncode"] == 0 else "error",
            **run_result,
        }
        if run_result["returncode"] == 0:
            successful_models.append(model_name)
        elif args.require_all:
            run_log["status"] = "error"
            run_log["reason"] = f"{model_name} training failed"
            save_run_log(args, run_log)
            return 1
        else:
            print(f"Skipping {model_name} in comparison because retraining failed.")

    if not successful_models:
        run_log["status"] = "error"
        run_log["reason"] = "No models retrained successfully"
        save_run_log(args, run_log)
        print("No models retrained successfully, so comparison was not run.")
        return 1

    compare_command = build_compare_command(
        repo_root=repo_root,
        args=args,
        models=successful_models,
    )
    print("")
    print("=== Comparing successful models on the same window ===")
    if args.dry_run:
        print(format_command(compare_command))
        run_log["comparison"] = {
            "status": "dry_run",
            "command": compare_command,
        }
        run_log["status"] = "dry_run"
        save_run_log(args, run_log)
        return 0

    compare_result = run_command(compare_command, cwd=repo_root)
    run_log["comparison"] = {
        "status": "ok" if compare_result["returncode"] == 0 else "error",
        **compare_result,
    }
    run_log["status"] = "ok" if compare_result["returncode"] == 0 else "error"
    run_log["finished_at"] = pd.Timestamp.utcnow().isoformat()
    log_path = save_run_log(args, run_log)
    print("")
    print(f"run_log: {log_path}")
    return 0 if compare_result["returncode"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain selected technical models with one cutoff, then compare "
            "them on the same requested historical window."
        )
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol to test.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated models to retrain and compare. Options: "
            f"{', '.join(MODEL_TRAIN_SCRIPTS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="First as-of date in the test window. Example: 2026-01-01",
    )
    parser.add_argument(
        "--end-date",
        help="Last as-of date in the test window. Example: 2026-06-04",
    )
    parser.add_argument(
        "--mode",
        choices=["out-of-sample", "leaky-window"],
        default="out-of-sample",
        help=(
            "out-of-sample trains before --start-date. leaky-window trains "
            "through --end-date and is only for overfit/leakage checks."
        ),
    )
    parser.add_argument(
        "--train-scope",
        choices=["symbol", "all"],
        default="symbol",
        help="Train each artifact on only --symbol or on all tickers. Default: symbol",
    )
    parser.add_argument(
        "--train-symbols",
        nargs="+",
        help=(
            "Train supported models on only these tickers. Accepts space-separated "
            "or comma-separated symbols. Currently supported for binary-xgboost."
        ),
    )
    parser.add_argument(
        "--threshold",
        "--target-threshold",
        dest="threshold",
        type=float,
        default=0.002,
        help="Direction target threshold passed to train and compare. Default: 0.002",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=120,
        help="Comparison rows if --start-date is removed in future use. Default: 120",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        help="Override LSTM epochs to speed up or deepen retraining.",
    )
    parser.add_argument(
        "--arima-validation-windows",
        type=int,
        help="Override ARIMA validation windows during retraining.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Stop if any requested model fails to retrain.",
    )
    parser.add_argument(
        "--comparison-output-html",
        help="HTML path to pass to compare_technical_models.py.",
    )
    parser.add_argument(
        "--comparison-output-json",
        help="JSON path to pass to compare_technical_models.py.",
    )
    parser.add_argument(
        "--output-log",
        help="Local JSON run log path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training or comparison.",
    )
    return parser.parse_args()


def parse_model_names(value: str) -> list[str]:
    names = []
    for item in value.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in MODEL_TRAIN_SCRIPTS:
            valid = ", ".join(MODEL_TRAIN_SCRIPTS)
            raise ValueError(f"Unknown model '{name}'. Valid models: {valid}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one model must be selected")
    return names


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


def training_cutoff_date(args: argparse.Namespace) -> str:
    if args.mode == "out-of-sample":
        return normalize_date(args.start_date)

    if not args.end_date:
        raise ValueError("--mode leaky-window requires --end-date")
    end_date = pd.to_datetime(args.end_date, errors="raise").date()
    return (end_date + timedelta(days=1)).isoformat()


def normalize_date(value: str) -> str:
    return pd.to_datetime(value, errors="raise").date().isoformat()


def build_train_command(
    repo_root: Path,
    model_name: str,
    args: argparse.Namespace,
    cutoff_date: str,
) -> list[str]:
    script_path = repo_root / "scripts" / MODEL_TRAIN_SCRIPTS[model_name]
    command = [sys.executable, str(script_path)]
    train_symbols = parse_symbols(args.train_symbols)
    if train_symbols and model_name == "binary-xgboost":
        command.append("--symbols")
        command.extend(train_symbols)
    elif train_symbols:
        raise ValueError(
            "--train-symbols is currently supported for binary-xgboost only. "
            f"Cannot pass it to {model_name}."
        )
    elif args.train_scope == "all":
        command.append("--all")
    else:
        command.extend(["--symbol", args.symbol.upper()])

    command.extend(["--threshold", str(float(args.threshold))])
    command.extend(["--train-before-date", cutoff_date])

    if model_name == "lstm" and args.lstm_epochs is not None:
        command.extend(["--epochs", str(int(args.lstm_epochs))])
    if model_name == "arima" and args.arima_validation_windows is not None:
        command.extend(["--validation-windows", str(int(args.arima_validation_windows))])

    return command


def build_compare_command(
    repo_root: Path,
    args: argparse.Namespace,
    models: list[str],
) -> list[str]:
    script_path = repo_root / "scripts" / "compare_technical_models.py"
    command = [
        sys.executable,
        str(script_path),
        "--symbol",
        args.symbol.upper(),
        "--models",
        ",".join(models),
        "--threshold",
        str(float(args.threshold)),
        "--start-date",
        normalize_date(args.start_date),
        "--rows",
        str(int(args.rows)),
    ]
    if args.end_date:
        command.extend(["--end-date", normalize_date(args.end_date)])
    if args.comparison_output_html:
        command.extend(["--output-html", args.comparison_output_html])
    if args.comparison_output_json:
        command.extend(["--output-json", args.comparison_output_json])
    return command


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    output_lines = []
    print(format_command(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    returncode = process.wait()
    duration = time.perf_counter() - start
    return {
        "command": command,
        "returncode": int(returncode),
        "duration_seconds": float(duration),
        "output": "".join(output_lines),
    }


def format_command(command: list[str]) -> str:
    return " ".join(quote_part(part) for part in command)


def quote_part(value: str) -> str:
    if not value:
        return '""'
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def print_mode_note(args: argparse.Namespace, cutoff_date: str) -> None:
    if args.mode == "out-of-sample":
        print(
            "Clean test: training rows are restricted to dates before "
            f"{cutoff_date}, then testing starts at {normalize_date(args.start_date)}."
        )
        return

    print(
        "Leaky overfit check: training includes rows through the requested test "
        f"window cutoff {cutoff_date}. Metrics from this mode are not a valid "
        "out-of-sample estimate."
    )


def save_run_log(args: argparse.Namespace, run_log: dict[str, Any]) -> Path:
    path = Path(
        args.output_log
        or f"technical_analysis/{args.symbol.upper()}_retrain_compare_run.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(run_log), indent=2), encoding="utf-8")
    return path


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
