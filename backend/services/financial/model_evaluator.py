"""
backend/services/financial/model_evaluator.py
Proper model evaluation:
  - Train/test split (80/20)
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - Logs results to LOG.md
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from .feature_engineering import FEATURE_COLS, prepare_training_data

LOG_PATH = Path(__file__).resolve().parents[3] / "LOG.md"


def evaluate(df) -> dict:
    """
    Full model evaluation pipeline.
    Returns dict with all metrics.
    Appends results to LOG.md.
    """
    X, y, df_clean = prepare_training_data(df)
    
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("\n" + "─" * 50)
    print("  Model Evaluation Report")
    print("─" * 50)
    print(f"  Total samples  : {len(df_clean)}")
    print(f"  Features       : {len(FEATURE_COLS)}")
    print(f"  Classes        : {list(le.classes_)}")
    print(f"  Distribution   : {dict(zip(le.classes_, np.bincount(y_enc)))}")

    base_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=2)
    pipeline   = Pipeline([("scaler", StandardScaler()), ("model", calibrated)])

    results = {}

    # ── 1. Train / Test Split (80/20) ─────────────────────────
    print("\n  [1] Train/Test Split (80/20)")
    if len(df_clean) >= 15:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        pipeline.fit(X_train, y_train)
        y_pred   = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results["train_test_accuracy"] = round(float(accuracy), 4)
        print(f"  Accuracy       : {accuracy:.2%}")
        print(f"  Train samples  : {len(X_train)}")
        print(f"  Test samples   : {len(X_test)}")
    else:
        print("  Not enough samples for 80/20 split")
        results["train_test_accuracy"] = None

    # ── 2. Stratified K-Fold Cross Validation ─────────────────
    print("\n  [2] Stratified K-Fold Cross Validation")
    min_class = min(np.bincount(y_enc))
    n_folds   = min(5, min_class)

    if n_folds >= 2:
        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(pipeline, X, y_enc, cv=skf)
        cv_acc = accuracy_score(y_enc, y_pred_cv)
        results["cv_folds"]    = n_folds
        results["cv_accuracy"] = round(float(cv_acc), 4)
        print(f"  Folds          : {n_folds}")
        print(f"  CV Accuracy    : {cv_acc:.2%}")
    else:
        print("  Not enough samples per class for cross-validation")
        results["cv_folds"]    = None
        results["cv_accuracy"] = None
        y_pred_cv = None

    # ── 3. Classification Report ───────────────────────────────
    print("\n  [3] Classification Report")
    if y_pred_cv is not None:
        report = classification_report(
            y_enc, y_pred_cv,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True
        )
        report_str = classification_report(
            y_enc, y_pred_cv,
            target_names=le.classes_
        )
        print(report_str)
        results["classification_report"] = report
    elif results.get("train_test_accuracy") is not None:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True
        )
        report_str = classification_report(
            y_test, y_pred,
            target_names=le.classes_
        )
        print(report_str)
        results["classification_report"] = report

    # ── 4. Confusion Matrix ────────────────────────────────────
    print("\n  [4] Confusion Matrix")
    if y_pred_cv is not None:
        cm = confusion_matrix(y_enc, y_pred_cv)
    elif results.get("train_test_accuracy") is not None:
        pipeline.fit(X_train, y_train)
        cm = confusion_matrix(y_test, pipeline.predict(X_test))
    else:
        cm = None

    if cm is not None:
        results["confusion_matrix"] = cm.tolist()
        classes = le.classes_
        col_w   = 12
        print(f"  {'':15}", end="")
        for c in classes:
            print(f"  pred_{c[:3]:4}", end="")
        print()
        for i, c in enumerate(classes):
            print(f"  actual_{c[:3]:8}", end="")
            for j in range(len(classes)):
                print(f"  {cm[i][j]:10}", end="")
            print()

    # ── 5. Summary ────────────────────────────────────────────
    results["evaluated_at"]   = datetime.utcnow().isoformat()
    results["total_samples"]  = len(df_clean)
    results["label_dist"]     = dict(zip(le.classes_, np.bincount(y_enc).tolist()))

    best_acc = results.get("cv_accuracy") or results.get("train_test_accuracy")
    print("\n" + "─" * 50)
    if best_acc:
        print(f"  Best accuracy  : {best_acc:.2%}")
    print("─" * 50)

    # ── 6. Write to LOG.md ────────────────────────────────────
    _write_log(results, le.classes_)

    return results


def _write_log(results: dict, classes):
    """Append evaluation results to LOG.md."""
    now       = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    best_acc  = results.get("cv_accuracy") or results.get("train_test_accuracy")
    acc_str   = f"{best_acc:.2%}" if best_acc else "N/A"
    dist      = results.get("label_dist", {})
    cm        = results.get("confusion_matrix")

    entry = f"""
## Run — {now}

| Metric | Value |
|--------|-------|
| Samples | {results.get('total_samples', 'N/A')} |
| CV Folds | {results.get('cv_folds', 'N/A')} |
| CV Accuracy | {f"{results['cv_accuracy']:.2%}" if results.get('cv_accuracy') else 'N/A'} |
| Train/Test Accuracy | {f"{results['train_test_accuracy']:.2%}" if results.get('train_test_accuracy') else 'N/A'} |
| Best Accuracy | {acc_str} |

**Label distribution:** {dist}

"""

    if cm and results.get("classification_report"):
        report = results["classification_report"]
        entry += "**Per-class metrics:**\n\n"
        entry += "| Class | Precision | Recall | F1-score | Support |\n"
        entry += "|-------|-----------|--------|----------|---------|\n"
        for cls in classes:
            r = report.get(cls, {})
            entry += f"| {cls} | {r.get('precision', 0):.2%} | {r.get('recall', 0):.2%} | {r.get('f1-score', 0):.2%} | {int(r.get('support', 0))} |\n"

    entry += "\n---\n"

    # Create or append
    if LOG_PATH.exists():
        with open(LOG_PATH, "a") as f:
            f.write(entry)
    else:
        with open(LOG_PATH, "w") as f:
            f.write("# Financial ML — Model Evaluation Log\n")
            f.write("Each run appends a new entry below.\n\n---\n")
            f.write(entry)

    print(f"\n  Log saved → {LOG_PATH}")
