#!/usr/bin/env python3
"""
FinBERT Fine-Tuning Script for StockLens FYP
============================================
Self-contained script for Google Colab with GPU.
Demonstrates fine-tuning methodology for academic assessment.

Dataset: Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment)
- 11,932 samples from Twitter + news sources
- Labels: 0=negative, 1=neutral, 2=positive
- Modern Parquet format (no loading script issues)

Runs 4 models:
- Baseline: No fine-tuning (ProsusAI/finbert as-is)
- Experiment 1: Standard fine-tune
- Experiment 2: Lower learning rate
- Experiment 3: Frozen early layers

All results saved to /content/drive/MyDrive/finbert_results/
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ============================================================================
# STEP 0: Check GPU and Install Dependencies
# ============================================================================

def check_gpu():
    """Verify GPU availability before starting training."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("=" * 70)
            print("⚠️  WARNING: No GPU detected!")
            print("=" * 70)
            print("Fine-tuning FinBERT without GPU will be extremely slow.")
            print("Please enable GPU in Colab:")
            print("  Runtime → Change runtime type → Hardware accelerator → GPU")
            print("=" * 70)
            sys.exit(1)

        gpu_name = torch.cuda.get_device_name(0)
        print("=" * 70)
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        print("=" * 70)
        return True
    except ImportError:
        print("PyTorch not installed yet, will install dependencies first.")
        return False

def install_dependencies():
    """Install required packages if not already present."""
    packages = [
        "transformers",
        "datasets",
        "torch",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "accelerate",  # Required by Trainer API
    ]

    print("\n" + "=" * 70)
    print("Installing dependencies...")
    print("=" * 70)

    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", package
            ])
            print(f"✓ {package} installed")

    print("=" * 70)
    print("All dependencies ready!")
    print("=" * 70 + "\n")

# Install dependencies first
install_dependencies()

# Now import everything
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)

# Check GPU now that torch is imported
check_gpu()

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("/content/drive/MyDrive/finbert_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "ProsusAI/finbert"
RANDOM_STATE = 42
MAX_LENGTH = 128

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"Results will be saved to: {OUTPUT_DIR}\n")

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_and_split_data():
    """
    Load Twitter Financial News Sentiment dataset and split into train/val/test.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("=" * 70)
    print("Loading Twitter Financial News Sentiment Dataset")
    print("=" * 70)

    try:
        # Load dataset from HuggingFace (modern Parquet format, no loading script)
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

        # The dataset comes with train/validation splits
        # We'll combine and re-split for our desired 70/15/15 distribution
        train_data = dataset["train"]
        val_data = dataset["validation"]

        # Combine all data first
        from datasets import concatenate_datasets
        full_dataset = concatenate_datasets([train_data, val_data])

        print(f"✓ Loaded {len(full_dataset)} financial news samples")
        print(f"✓ Label distribution:")

        # Check label distribution
        labels = full_dataset["label"]
        unique, counts = np.unique(labels, return_counts=True)
        label_names = ["Negative", "Neutral", "Positive"]
        for label_id, count in zip(unique, counts):
            print(f"  {label_names[label_id]}: {count} ({count/len(labels)*100:.1f}%)")

        # Split: 70% train, 15% validation, 15% test
        train_test = full_dataset.train_test_split(test_size=0.3, seed=RANDOM_STATE)
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_STATE)

        train_dataset = train_test["train"]
        val_dataset = val_test["train"]
        test_dataset = val_test["test"]

        print(f"\n✓ Split complete:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        print("=" * 70 + "\n")

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Please check your internet connection and HuggingFace access.")
        sys.exit(1)

def tokenize_dataset(dataset, tokenizer):
    """Tokenize text data for model input."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],  # Twitter dataset uses "text" field, not "sentence"
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    return dataset.map(tokenize_function, batched=True)

# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 for evaluation.
    Used by Trainer API during training.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
    }

def evaluate_model_detailed(
    model,
    tokenizer,
    test_dataset,
    model_name: str
) -> Dict[str, any]:
    """
    Comprehensive evaluation with per-class metrics and confusion matrix.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for preprocessing
        test_dataset: Test dataset
        model_name: Name for saving results

    Returns:
        Dictionary containing all metrics and predictions
    """
    print(f"\nEvaluating {model_name}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    # Run inference
    with torch.no_grad():
        for example in test_dataset:
            inputs = tokenizer(
                example["text"],  # Twitter dataset uses "text" field
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)

            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()

            all_predictions.append(prediction)
            all_labels.append(example["label"])

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    # Per-class F1 scores
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_negative": float(f1_per_class[0]),
        "f1_neutral": float(f1_per_class[1]),
        "f1_positive": float(f1_per_class[2]),
        "confusion_matrix": cm.tolist(),
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Macro: {f1:.4f}")
    print(f"  F1 per class: Neg={f1_per_class[0]:.4f}, "
          f"Neu={f1_per_class[1]:.4f}, Pos={f1_per_class[2]:.4f}")

    return results

def plot_confusion_matrix(cm, model_name: str, save_path: Path):
    """Generate and save confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive'],
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved confusion matrix to {save_path.name}")

# ============================================================================
# STEP 1: Baseline Evaluation (No Fine-Tuning)
# ============================================================================

def evaluate_baseline(test_dataset):
    """
    Evaluate pretrained FinBERT without any fine-tuning.
    This establishes our baseline performance.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Baseline Evaluation (No Fine-Tuning)")
    print("=" * 70)
    print("Loading base ProsusAI/finbert model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True  # Handle label mismatch if any
    )

    results = evaluate_model_detailed(model, tokenizer, test_dataset, "Baseline")

    # Save metrics
    metrics_path = OUTPUT_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved metrics to {metrics_path.name}")

    # Plot confusion matrix
    cm_path = OUTPUT_DIR / "baseline_confusion_matrix.png"
    plot_confusion_matrix(
        np.array(results["confusion_matrix"]),
        "Baseline (No Fine-Tuning)",
        cm_path
    )

    print("=" * 70)
    return results, tokenizer

# ============================================================================
# STEP 2: Fine-Tuning Experiments
# ============================================================================

def freeze_model_layers(model, num_layers_to_freeze: int = 6):
    """
    Freeze the first N transformer layers.
    Only the top layers and classification head will be trained.
    """
    # Freeze embedding layer
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze first N encoder layers
    for i in range(num_layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.1f}%)")

def run_fine_tuning_experiment(
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer,
    experiment_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    freeze_layers: bool,
    num_frozen: int = 6
):
    """
    Run a single fine-tuning experiment.

    Returns:
        Dictionary with training history and final test metrics
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Freeze Layers: {freeze_layers}")

    # Load fresh model from base weights
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # Freeze layers if requested
    if freeze_layers:
        print(f"  Freezing first {num_frozen} layers...")
        freeze_model_layers(model, num_frozen)

    # Tokenize datasets
    print("  Tokenizing datasets...")
    train_encoded = tokenize_dataset(train_dataset, tokenizer)
    val_encoded = tokenize_dataset(val_dataset, tokenizer)

    # Training arguments
    exp_slug = experiment_name.lower().replace(" ", "_")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"{exp_slug}_checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",  # Updated parameter name in transformers v4.40+
        save_strategy="epoch",
        save_total_limit=1,  # Only keep best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(OUTPUT_DIR / f"{exp_slug}_logs"),
        logging_steps=50,
        report_to="none",  # Disable wandb/tensorboard
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        compute_metrics=compute_metrics,
    )

    # Train
    print("  Starting training...")
    train_result = trainer.train()

    # Extract training history
    history = trainer.state.log_history

    # Organize per-epoch metrics
    epochs_data = []
    for i in range(num_epochs):
        epoch_logs = [log for log in history if log.get("epoch") == i + 1]

        train_loss = next((log["loss"] for log in epoch_logs if "loss" in log), None)
        eval_metrics = next((log for log in epoch_logs if "eval_loss" in log), None)

        if eval_metrics:
            epochs_data.append({
                "epoch": i + 1,
                "train_loss": train_loss,
                "eval_loss": eval_metrics["eval_loss"],
                "eval_accuracy": eval_metrics["eval_accuracy"],
                "eval_f1_macro": eval_metrics["eval_f1_macro"],
            })

    # Evaluate on test set
    print("  Evaluating on test set...")
    test_results = evaluate_model_detailed(
        trainer.model, tokenizer, test_dataset, experiment_name
    )

    # Save model
    model_save_path = OUTPUT_DIR / f"{exp_slug}_model"
    trainer.save_model(str(model_save_path))
    print(f"  ✓ Saved model to {model_save_path.name}/")

    # Save metrics
    results = {
        "experiment_name": experiment_name,
        "config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "freeze_layers": freeze_layers,
        },
        "training_history": epochs_data,
        "test_results": test_results,
    }

    metrics_path = OUTPUT_DIR / f"{exp_slug}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved metrics to {metrics_path.name}")

    # Plot loss curve
    plot_loss_curve(epochs_data, experiment_name, OUTPUT_DIR / f"{exp_slug}_loss_curve.png")

    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(test_results["confusion_matrix"]),
        experiment_name,
        OUTPUT_DIR / f"{exp_slug}_confusion_matrix.png"
    )

    print("=" * 70)
    return results

def plot_loss_curve(epochs_data: List[Dict], experiment_name: str, save_path: Path):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["eval_loss"] for e in epochs_data]

    ax.plot(epochs, train_loss, marker='o', linewidth=2, label='Training Loss', color='#2E86AB')
    ax.plot(epochs, val_loss, marker='s', linewidth=2, label='Validation Loss', color='#A23B72')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'Training Progress - {experiment_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved loss curve to {save_path.name}")

# ============================================================================
# STEP 3: Visualization - Comparison Charts
# ============================================================================

def plot_model_comparison(all_results: Dict[str, Dict], save_path: Path):
    """
    Grouped bar chart comparing accuracy and F1 across all models.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    models = list(all_results.keys())
    accuracy_scores = [all_results[m]["accuracy"] for m in models]
    f1_scores = [all_results[m]["f1_macro"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy',
                   color='#06D6A0', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Macro',
                   color='#EF476F', edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison chart to {save_path.name}")

def plot_per_class_f1(all_results: Dict[str, Dict], save_path: Path):
    """
    Bar chart showing per-class F1 scores across all models.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    models = list(all_results.keys())
    classes = ['Negative', 'Neutral', 'Positive']

    # Prepare data
    data = {cls: [] for cls in classes}
    for model_name in models:
        data['Negative'].append(all_results[model_name]['f1_negative'])
        data['Neutral'].append(all_results[model_name]['f1_neutral'])
        data['Positive'].append(all_results[model_name]['f1_positive'])

    x = np.arange(len(models))
    width = 0.25
    colors = ['#E63946', '#F1FAEE', '#06D6A0']

    # Plot bars for each class
    for idx, (cls, color) in enumerate(zip(classes, colors)):
        offset = width * (idx - 1)
        bars = ax.bar(x + offset, data[cls], width, label=cls,
                     color=color, edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11, frameon=True, shadow=True, title='Sentiment Class')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved per-class F1 chart to {save_path.name}")

# ============================================================================
# STEP 4: Results Summary CSV
# ============================================================================

def save_results_summary(all_results: Dict[str, Dict], baseline_results: Dict):
    """Save comprehensive results table as CSV."""

    summary_data = []

    # Add baseline
    summary_data.append({
        "Model": "Baseline",
        "Learning Rate": "N/A",
        "Batch Size": "N/A",
        "Epochs": "N/A",
        "Frozen Layers": "N/A",
        "Accuracy": f"{baseline_results['accuracy']:.4f}",
        "F1 Macro": f"{baseline_results['f1_macro']:.4f}",
        "F1 Positive": f"{baseline_results['f1_positive']:.4f}",
        "F1 Neutral": f"{baseline_results['f1_neutral']:.4f}",
        "F1 Negative": f"{baseline_results['f1_negative']:.4f}",
    })

    # Add experiments
    for exp_name, exp_data in all_results.items():
        if exp_name == "Baseline":
            continue

        config = exp_data.get("config", {})
        test_results = exp_data.get("test_results", exp_data)

        summary_data.append({
            "Model": exp_name,
            "Learning Rate": config.get("learning_rate", "N/A"),
            "Batch Size": config.get("batch_size", "N/A"),
            "Epochs": config.get("num_epochs", "N/A"),
            "Frozen Layers": "Yes" if config.get("freeze_layers") else "No",
            "Accuracy": f"{test_results['accuracy']:.4f}",
            "F1 Macro": f"{test_results['f1_macro']:.4f}",
            "F1 Positive": f"{test_results['f1_positive']:.4f}",
            "F1 Neutral": f"{test_results['f1_neutral']:.4f}",
            "F1 Negative": f"{test_results['f1_negative']:.4f}",
        })

    df = pd.DataFrame(summary_data)
    csv_path = OUTPUT_DIR / "results_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"✓ Saved results summary to {csv_path.name}")
    return df

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution pipeline."""

    print("\n" + "=" * 70)
    print("  FinBERT Fine-Tuning Pipeline for StockLens FYP")
    print("=" * 70)
    print(f"Start Time: {pd.Timestamp.now()}")
    print("=" * 70)

    # Load and split data
    train_dataset, val_dataset, test_dataset = load_and_split_data()

    # Step 1: Baseline evaluation
    baseline_results, tokenizer = evaluate_baseline(test_dataset)

    # Step 2: Fine-tuning experiments
    experiment_configs = [
        {
            "name": "Experiment 1",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 3,
            "freeze_layers": False,
        },
        {
            "name": "Experiment 2",
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 3,
            "freeze_layers": False,
        },
        {
            "name": "Experiment 3",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 3,
            "freeze_layers": True,
        },
    ]

    all_results = {"Baseline": baseline_results}

    for config in experiment_configs:
        exp_results = run_fine_tuning_experiment(
            train_dataset,
            val_dataset,
            test_dataset,
            tokenizer,
            experiment_name=config["name"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            num_epochs=config["epochs"],
            freeze_layers=config["freeze_layers"],
        )

        # Store test results under experiment name
        all_results[config["name"]] = {
            **exp_results["test_results"],
            "config": exp_results["config"]
        }

    # Step 3: Generate comparison visualizations
    print("\n" + "=" * 70)
    print("STEP 3: Generating Comparison Visualizations")
    print("=" * 70)

    plot_model_comparison(all_results, OUTPUT_DIR / "model_comparison.png")
    plot_per_class_f1(all_results, OUTPUT_DIR / "per_class_f1_comparison.png")

    # Find best model
    best_model = max(
        [k for k in all_results.keys() if k != "Baseline"],
        key=lambda k: all_results[k]["f1_macro"]
    )
    print(f"\n✓ Best model: {best_model} "
          f"(F1={all_results[best_model]['f1_macro']:.4f})")

    # Step 4: Save results summary
    print("\n" + "=" * 70)
    print("STEP 4: Saving Results Summary")
    print("=" * 70)

    summary_df = save_results_summary(all_results, baseline_results)

    # Step 5: Final console summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print("\n📊 Performance Results:")
    print("-" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Macro':<12} {'Improvement':<12}")
    print("-" * 70)

    baseline_acc = baseline_results["accuracy"]
    baseline_f1 = baseline_results["f1_macro"]

    print(f"{'Baseline':<20} {baseline_acc:<12.4f} {baseline_f1:<12.4f} {'—':<12}")

    for exp_name in ["Experiment 1", "Experiment 2", "Experiment 3"]:
        exp_acc = all_results[exp_name]["accuracy"]
        exp_f1 = all_results[exp_name]["f1_macro"]
        improvement = exp_f1 - baseline_f1

        print(f"{exp_name:<20} {exp_acc:<12.4f} {exp_f1:<12.4f} "
              f"{'+' if improvement > 0 else ''}{improvement:<11.4f}")

    print("-" * 70)

    best_f1 = all_results[best_model]["f1_macro"]
    best_acc = all_results[best_model]["accuracy"]
    acc_improvement = best_acc - baseline_acc
    f1_improvement = best_f1 - baseline_f1

    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Accuracy: {best_acc:.4f} (+{acc_improvement:.4f} from baseline)")
    print(f"   F1 Macro: {best_f1:.4f} (+{f1_improvement:.4f} from baseline)")

    print(f"\n📁 All Results Saved To:")
    print(f"   {OUTPUT_DIR}/")
    print("\n   Files generated:")
    print("   - baseline_metrics.json & confusion_matrix.png")
    print("   - experiment_1_metrics.json, loss_curve.png, confusion_matrix.png, model/")
    print("   - experiment_2_metrics.json, loss_curve.png, confusion_matrix.png, model/")
    print("   - experiment_3_metrics.json, loss_curve.png, confusion_matrix.png, model/")
    print("   - model_comparison.png")
    print("   - per_class_f1_comparison.png")
    print("   - results_summary.csv")

    print("\n" + "=" * 70)
    print(f"  Pipeline Complete! | End Time: {pd.Timestamp.now()}")
    print("=" * 70)

    print("\n✅ You can now use these results in your FYP documentation.")
    print("   The trained models are ready for inference on new financial text.")

if __name__ == "__main__":
    main()
