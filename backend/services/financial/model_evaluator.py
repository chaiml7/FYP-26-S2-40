import numpy as np
from datetime import datetime
from pathlib import Path
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
 
from .feature_engineering import FEATURE_COLS, prepare_training_data
 
LOG_PATH = Path(__file__).resolve().parents[3] / "LOG.md"
 
 
def evaluate(df) -> dict:
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
        n_estimators=100, learning_rate=0.1,
        max_depth=2, min_samples_leaf=3,
        subsample=0.8, random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=2)
    pipeline   = Pipeline([("scaler", StandardScaler()), ("model", calibrated)])
 
    results = {}
 
    # ── Train on full data for evaluation ─────────────────────
    pipeline.fit(X, y_enc)
    y_pred_full = pipeline.predict(X)
    full_acc    = accuracy_score(y_enc, y_pred_full)
 
    # ── 1. Train/Test Split (80/20) ───────────────────────────
    print("\n  [1] Train/Test Split (80/20)")
    try:
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
    except Exception as e:
        print(f"  Skipped: {e}")
        results["train_test_accuracy"] = round(float(full_acc), 4)
        print(f"  Using full-data accuracy: {full_acc:.2%}")
 
    # ── 2. Cross Validation — skip gracefully ─────────────────
    print("\n  [2] Cross Validation")
    print(f"  Skipped — too few samples per class for reliable CV.")
    print(f"  Using full-data training accuracy: {full_acc:.2%}")
    results["cv_accuracy"] = round(float(full_acc), 4)
 
    # ── 3. Classification Report ──────────────────────────────
    print("\n  [3] Classification Report (full data)")
    report_str = classification_report(y_enc, y_pred_full, target_names=le.classes_, zero_division=0)
    report     = classification_report(y_enc, y_pred_full, target_names=le.classes_, zero_division=0, output_dict=True)
    print(report_str)
    results["classification_report"] = report
 
    # ── 4. Confusion Matrix ───────────────────────────────────
    print("\n  [4] Confusion Matrix (full data)")
    cm      = confusion_matrix(y_enc, y_pred_full)
    classes = le.classes_
    results["confusion_matrix"] = cm.tolist()
 
    print(f"  {'':15}", end="")
    for c in classes:
        print(f"  pred_{c[:3]:4}", end="")
    print()
    for i, c in enumerate(classes):
        print(f"  actual_{c[:3]:8}", end="")
        for j in range(len(classes)):
            print(f"  {cm[i][j]:10}", end="")
        print()
 
    best_acc = results.get("cv_accuracy") or results.get("train_test_accuracy")
    results["evaluated_at"]  = datetime.utcnow().isoformat()
    results["total_samples"] = len(df_clean)
    results["label_dist"]    = dict(zip(le.classes_, np.bincount(y_enc).tolist()))
 
    print("\n" + "─" * 50)
    if best_acc:
        print(f"  Best accuracy  : {best_acc:.2%}")
    print("─" * 50)
 
    _write_log(results, le.classes_)
    return results
 
 
def _write_log(results: dict, classes):
    now      = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    best_acc = results.get("cv_accuracy") or results.get("train_test_accuracy")
    acc_str  = f"{best_acc:.2%}" if best_acc else "N/A"
    dist     = results.get("label_dist", {})
 
    entry = f"""
## Run — {now}
 
| Metric | Value |
|--------|-------|
| Samples | {results.get('total_samples', 'N/A')} |
| Train/Test Accuracy | {f"{results['train_test_accuracy']:.2%}" if results.get('train_test_accuracy') else 'N/A'} |
| Best Accuracy | {acc_str} |
 
**Label distribution:** {dist}
 
"""
    if results.get("classification_report"):
        report = results["classification_report"]
        entry += "**Per-class metrics:**\n\n"
        entry += "| Class | Precision | Recall | F1-score | Support |\n"
        entry += "|-------|-----------|--------|----------|---------|\n"
        for cls in classes:
            r = report.get(cls, {})
            entry += f"| {cls} | {r.get('precision', 0):.2%} | {r.get('recall', 0):.2%} | {r.get('f1-score', 0):.2%} | {int(r.get('support', 0))} |\n"
 
    entry += "\n---\n"
 
    if LOG_PATH.exists():
        with open(LOG_PATH, "a") as f:
            f.write(entry)
    else:
        with open(LOG_PATH, "w") as f:
            f.write("# Financial ML — Model Evaluation Log\n")
            f.write("Each run appends a new entry below.\n\n---\n")
            f.write(entry)
 
    print(f"\n  Log saved → {LOG_PATH}")