"""
evaluate.py — Model evaluation on the held-out test set.

Usage:
    python -m src.evaluate                      # evaluate custom CNN
    python -m src.evaluate --model transfer     # evaluate transfer model
    python -m src.evaluate --model both         # evaluate both

Outputs saved to outputs/:
    figures/confusion_matrix_{name}.png
    figures/per_class_metrics_{name}.png
    figures/sample_predictions_{name}.png
    metrics/model_metrics.csv
    reports/model_evaluation.md
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_utils import get_datasets, get_true_labels


def load_class_names() -> list:
    with open(config.CLASS_NAMES_PATH) as f:
        return json.load(f)


def get_predictions(model, test_ds):
    """Return (y_true, y_pred, y_prob) over the full test set."""
    # Rebuild a fresh iterator to avoid consuming the dataset
    _, _, test_ds_fresh = get_datasets()
    probs = model.predict(test_ds_fresh, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    # get_true_labels returns integer labels directly (label_mode='int')
    _, _, test_ds_labels = get_datasets()
    y_true = get_true_labels(test_ds_labels)
    return y_true, y_pred, probs


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix — {model_name}\n(counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=35)

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"Confusion Matrix — {model_name}\n(% per true class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    path = os.path.join(config.FIG_DIR, f"confusion_matrix_{model_name}.png")
    os.makedirs(config.FIG_DIR, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")
    return cm


def plot_per_class_metrics(report_dict, class_names, model_name: str):
    classes = [c for c in class_names if c in report_dict]
    precision = [report_dict[c]["precision"] for c in classes]
    recall    = [report_dict[c]["recall"]    for c in classes]
    f1        = [report_dict[c]["f1-score"]  for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class Metrics — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(config.FIG_DIR, f"per_class_metrics_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Per-class metrics chart saved → {path}")


def plot_sample_predictions(model, test_ds, class_names, model_name: str,
                             n_correct: int = 6, n_wrong: int = 6):
    """Show sample correct and incorrect predictions."""
    probs  = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = get_true_labels(test_ds)

    # Collect file paths from raw dataset (before map transforms)
    _, _, test_ds_raw = get_datasets()
    # image_dataset_from_directory stores file_paths as dataset attribute
    # We need the raw dataset for file paths
    test_dir = Path(config.PROC_DIR) / "test"
    # Rebuild file list in the same order (shuffle=False)
    raw_ds = keras.utils.image_dataset_from_directory(
        str(test_dir),
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    file_paths = raw_ds.file_paths

    correct_idx = np.where(y_pred == y_true)[0][:n_correct]
    wrong_idx   = np.where(y_pred != y_true)[0][:n_wrong]

    n_cols = max(n_correct, n_wrong)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 8))

    for row, idx_list in enumerate([correct_idx, wrong_idx]):
        label = "CORRECT" if row == 0 else "WRONG"
        for col, idx in enumerate(idx_list):
            ax = axes[row][col]
            img = plt.imread(file_paths[idx])
            ax.imshow(img)
            true_cls = class_names[y_true[idx]]
            pred_cls = class_names[y_pred[idx]]
            conf     = probs[idx][y_pred[idx]]
            color    = "green" if row == 0 else "red"
            ax.set_title(
                f"[{label}]\nTrue: {true_cls}\nPred: {pred_cls}\n{conf:.1%}",
                fontsize=7, color=color
            )
            ax.axis("off")
        for col in range(len(idx_list), n_cols):
            axes[row][col].axis("off")

    plt.suptitle(f"Sample Predictions — {model_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(config.FIG_DIR, f"sample_predictions_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Sample predictions saved → {path}")


def compute_metrics(y_true, y_pred, class_names, model_name: str) -> dict:
    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec   = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    print(f"\n{'='*50}")
    print(f"Results — {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"  Macro Precision : {macro_prec:.4f}")
    print(f"  Macro Recall    : {macro_rec:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return {
        "model":            model_name,
        "accuracy":         round(acc, 4),
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(weighted_f1, 4),
        "macro_precision":  round(macro_prec, 4),
        "macro_recall":     round(macro_rec, 4),
        "report":           report,
    }


def save_metrics_csv(results_list: list):
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    rows = [
        {k: r[k] for k in ("model", "accuracy", "macro_f1", "weighted_f1",
                             "macro_precision", "macro_recall")}
        for r in results_list
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(config.METRICS_DIR, "model_metrics.csv")
    df.to_csv(path, index=False)
    print(f"\nMetrics CSV saved → {path}")
    return df


def write_evaluation_report(results_list: list, class_names: list):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    lines = [
        "# AutoClaim AI — Model Evaluation Report\n",
        f"**Dataset**: CarDD  |  **Classes**: {', '.join(class_names)}\n\n---\n",
    ]

    for r in results_list:
        lines += [
            f"## {r['model']}\n\n",
            "| Metric | Value |\n|--------|-------|\n",
            f"| Accuracy       | {r['accuracy']:.4f} |\n",
            f"| Macro F1       | {r['macro_f1']:.4f} |\n",
            f"| Weighted F1    | {r['weighted_f1']:.4f} |\n",
            f"| Macro Precision| {r['macro_precision']:.4f} |\n",
            f"| Macro Recall   | {r['macro_recall']:.4f} |\n\n",
            "### Per-class results\n\n",
            "| Class | Precision | Recall | F1 | Support |\n",
            "|-------|-----------|--------|----|---------|\n",
        ]
        rep = r["report"]
        for cls in class_names:
            if cls in rep:
                c = rep[cls]
                lines.append(
                    f"| {cls} | {c['precision']:.3f} | {c['recall']:.3f} "
                    f"| {c['f1-score']:.3f} | {int(c['support'])} |\n"
                )
        lines.append("\n")

    lines += [
        "---\n\n## Business Interpretation\n\n",
        "- **Scratch / dent**: easiest (most common, fast-track eligible).\n",
        "- **Crack / tire flat**: hardest (few samples, visually subtle).\n",
        "- Confusing *dent* ↔ *scratch* is low-risk (same triage path).\n",
        "- Confusing *crack* → *scratch* is high-risk: cracks are structural.\n",
        "- Confidence thresholds in `config.py` route borderline cases to human review.\n",
    ]

    path = os.path.join(config.REPORTS_DIR, "model_evaluation.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Evaluation report saved → {path}")


def evaluate_model(model_path: str, model_name: str, class_names: list) -> dict | None:
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path} — skipping {model_name}.")
        return None

    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)

    _, _, test_ds = get_datasets()
    y_true, y_pred, probs = get_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, class_names, model_name)
    plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    plot_per_class_metrics(metrics["report"], class_names, model_name)
    try:
        plot_sample_predictions(model, test_ds, class_names, model_name)
    except Exception as e:
        print(f"  (Could not generate sample predictions: {e})")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AutoClaim AI Evaluation")
    parser.add_argument("--model", choices=["custom", "transfer", "both"], default="custom")
    args = parser.parse_args()

    if not os.path.exists(config.CLASS_NAMES_PATH):
        print("class_names.json not found. Run `python -m src.train` first.")
        sys.exit(1)

    class_names = load_class_names()
    results = []

    if args.model in ("custom", "both"):
        r = evaluate_model(config.CUSTOM_MODEL_PATH, "custom_cnn", class_names)
        if r:
            results.append(r)

    if args.model in ("transfer", "both"):
        r = evaluate_model(config.TRANSFER_MODEL_PATH, "transfer_mobilenetv2", class_names)
        if r:
            results.append(r)

    if results:
        save_metrics_csv(results)
        write_evaluation_report(results, class_names)


if __name__ == "__main__":
    main()
