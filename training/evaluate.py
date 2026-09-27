"""
BioPrint Vision — Evaluation Script

Evaluates the trained model on the held-out test set and generates:
  - evaluation/metrics.json        — Machine-readable metrics
  - evaluation/confusion_matrix.png — Confusion matrix visualization
  - evaluation/classification_report.txt — Per-class report

Usage:
    python -m training.evaluate --data-dir data/raw --model-path models/best_model.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.image_preprocess import get_val_transform
from training.train import build_model, set_seed


def evaluate_model(model, dataloader, device, class_names):
    """
    Run full evaluation on a dataloader.

    Returns:
        all_preds: list of predicted class indices
        all_labels: list of true class indices
        all_probs: list of probability arrays
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_preds, all_labels, all_probs


def compute_metrics(all_preds, all_labels, all_probs, class_names):
    """Compute comprehensive evaluation metrics."""
    metrics = {}

    # Overall metrics
    metrics["accuracy"] = round(accuracy_score(all_labels, all_preds) * 100, 2)
    metrics["macro_precision"] = round(
        precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2
    )
    metrics["macro_recall"] = round(
        recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2
    )
    metrics["macro_f1"] = round(
        f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2
    )
    metrics["weighted_f1"] = round(
        f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 2
    )

    # Per-class metrics
    per_class_precision = precision_score(
        all_labels, all_preds, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        all_labels, all_preds, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0
    )

    metrics["per_class"] = {}
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": round(float(per_class_precision[i]) * 100, 2),
            "recall": round(float(per_class_recall[i]) * 100, 2),
            "f1": round(float(per_class_f1[i]) * 100, 2),
        }

    # Top-2 accuracy
    probs_array = np.array(all_probs)
    top2_correct = 0
    for i in range(len(all_labels)):
        top2_indices = np.argsort(probs_array[i])[-2:]
        if all_labels[i] in top2_indices:
            top2_correct += 1
    metrics["top2_accuracy"] = round(100.0 * top2_correct / len(all_labels), 2)

    # Confidence distribution
    confidences = [max(p) * 100 for p in all_probs]
    metrics["confidence_stats"] = {
        "mean": round(float(np.mean(confidences)), 2),
        "median": round(float(np.median(confidences)), 2),
        "std": round(float(np.std(confidences)), 2),
        "min": round(float(np.min(confidences)), 2),
        "max": round(float(np.max(confidences)), 2),
    }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def plot_confusion_matrix(cm, class_names, output_path):
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("BioPrint Vision — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BioPrint Vision model")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth")
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  BIOPRINT VISION — EVALUATION")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model_path}")
    print(f"  Data Dir:   {args.data_dir}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"  ERROR: Model not found at '{args.model_path}'")
        print(f"  Run training first: python -m training.train")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    img_size = checkpoint.get("img_size", 224)

    print(f"  Classes:    {class_names}")
    print(f"  Image Size: {img_size}")
    print(f"  Best epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val Acc:    {checkpoint.get('val_acc', '?'):.2f}%")

    # ── Build and load model ──────────────────────────────────────────────
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Load test split ───────────────────────────────────────────────────
    split_path = Path(args.model_path).parent / "split_indices.json"
    if not split_path.exists():
        print(f"  ERROR: Split indices not found at '{split_path}'")
        sys.exit(1)

    with open(split_path) as f:
        split_info = json.load(f)
    test_idx = split_info["test_indices"]

    full_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=get_val_transform(img_size),
    )
    test_set = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"  Test set size: {len(test_set)}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, class_names
    )

    metrics = compute_metrics(all_preds, all_labels, all_probs, class_names)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:         {metrics['accuracy']:.2f}%")
    print(f"  Top-2 Accuracy:   {metrics['top2_accuracy']:.2f}%")
    print(f"  Macro Precision:  {metrics['macro_precision']:.2f}%")
    print(f"  Macro Recall:     {metrics['macro_recall']:.2f}%")
    print(f"  Macro F1:         {metrics['macro_f1']:.2f}%")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.2f}%")

    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':>6s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}")
    print(f"  {'-'*35}")
    for name in class_names:
        pc = metrics["per_class"][name]
        print(f"  {name:>6s}  {pc['precision']:>8.2f}%  {pc['recall']:>6.2f}%  {pc['f1']:>6.2f}%")

    print(f"\n  Confidence Distribution:")
    cs = metrics["confidence_stats"]
    print(f"    Mean:   {cs['mean']:.2f}%")
    print(f"    Median: {cs['median']:.2f}%")
    print(f"    Std:    {cs['std']:.2f}%")
    print(f"    Range:  [{cs['min']:.2f}%, {cs['max']:.2f}%]")

    # ── Save outputs ──────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {output_dir / 'metrics.json'}")

    # Save confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    # Save classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write("BioPrint Vision — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"  Classification report saved to: {output_dir / 'classification_report.txt'}")

    # ── Identify misclassified samples ────────────────────────────────────
    misclassified = []
    for i in range(len(all_preds)):
        if all_preds[i] != all_labels[i]:
            misclassified.append({
                "index": test_idx[i],
                "true_class": class_names[all_labels[i]],
                "pred_class": class_names[all_preds[i]],
                "confidence": round(max(all_probs[i]) * 100, 2),
            })

    # Save top misclassified examples
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    with open(output_dir / "misclassified_samples.json", "w") as f:
        json.dump(misclassified[:50], f, indent=2)
    print(f"  Top misclassified samples saved ({len(misclassified)} total errors)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
