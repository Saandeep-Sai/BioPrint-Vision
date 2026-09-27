"""
BioPrint Vision — Training Script

Trains a ResNet-50 classifier for 8-class blood-group prediction from
fingerprint images using transfer learning from ImageNet.

Usage:
    python -m training.train --data-dir data/raw --epochs 30 --batch-size 32

Outputs:
    models/best_model.pth    — Best model checkpoint
    models/class_names.json  — Ordered class names
    models/training_log.json — Training history
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.image_preprocess import get_train_transform, get_val_transform, IMG_SIZE


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model Architecture ───────────────────────────────────────────────────────
def build_model(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-50 with custom classification head for blood-group prediction.

    Architecture follows the reference project's approach:
    - Frozen early layers (conv1, bn1, layer1, layer2)
    - Trainable layer3 and layer4
    - Custom MLP head: 2048 → 1024 → 512 → 128 → 64 → num_classes
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Freeze early feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3 and layer4 for fine-tuning
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Custom classification head
    num_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )

    return model


# ── Dataset Splitting ────────────────────────────────────────────────────────
def create_splits(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  seed=42):
    """
    Create stratified train/val/test splits.

    NOTE: Subject IDs are NOT available in this dataset, so we cannot
    guarantee subject-disjoint splits. This is a documented limitation.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    targets = [dataset.targets[i] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=targets, random_state=seed
    )

    # Second split: train vs val
    train_val_targets = [targets[i] for i in train_val_idx]
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val,
        stratify=train_val_targets, random_state=seed
    )

    return train_idx, val_idx, test_idx


# ── Training Loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="  Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model, return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── Main Training Function ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train BioPrint Vision model")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use class weights to handle imbalance")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 for Windows compatibility)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  BIOPRINT VISION -- TRAINING")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Data Dir:   {args.data_dir}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Patience:   {args.patience}")
    print(f"  Seed:       {args.seed}")
    print(f"{'='*70}\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    # Use val transform for initial loading; we'll apply train transforms
    # via separate dataset wrapper
    full_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=get_val_transform(args.img_size),
    )

    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Total images: {len(full_dataset)}")

    # ── Create splits ─────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = create_splits(full_dataset, seed=args.seed)
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ── Create datasets with appropriate transforms ───────────────────────
    # Training set with augmentation
    train_dataset_aug = datasets.ImageFolder(
        root=args.data_dir,
        transform=get_train_transform(args.img_size),
    )

    train_set = Subset(train_dataset_aug, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    # ── Class weights / balanced sampling ─────────────────────────────────
    train_targets = [full_dataset.targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    print(f"  Train class distribution: {dict(zip(class_names, class_counts.tolist()))}")

    sampler = None
    class_weights_tensor = None

    if args.use_class_weights:
        # Weighted random sampler for balanced training
        sample_weights = 1.0 / class_counts[train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_idx),
            replacement=True,
        )
        # Also use class weights in the loss
        class_weights = 1.0 / class_counts.astype(np.float64)
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        print(f"  Using class weights: {class_weights.round(3).tolist()}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Build model ───────────────────────────────────────────────────────
    print(f"\n  Building ResNet-50 model...")
    model = build_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_log = []

    print(f"\n  Starting training...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
        }
        training_log.append(log_entry)

        print(
            f"  Epoch {epoch:>3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "img_size": args.img_size,
                "num_classes": num_classes,
            }
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"    ↳ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")

    # ── Save class names ──────────────────────────────────────────────────
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"  Class names saved to {output_dir / 'class_names.json'}")

    # ── Save training log ─────────────────────────────────────────────────
    training_config = {
        "model": "ResNet-50",
        "pretrained": True,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "seed": args.seed,
        "epochs_trained": len(training_log),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_time_s": round(total_time, 1),
        "device": str(device),
        "dataset_size": len(full_dataset),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "class_names": class_names,
        "class_weights_used": args.use_class_weights,
        "data_leakage_note": (
            "Subject IDs are NOT available in this dataset. "
            "Split is stratified random by image, NOT by subject. "
            "Fingerprints from the same person may appear in both "
            "train and test sets."
        ),
        "timestamp": datetime.now().isoformat(),
    }

    log_output = {
        "config": training_config,
        "history": training_log,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_output, f, indent=2)
    print(f"  Training log saved to {output_dir / 'training_log.json'}")

    # ── Quick test evaluation ─────────────────────────────────────────────
    print(f"\n  Running quick test evaluation...")
    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    print(f"\n  Run 'python -m training.evaluate' for full evaluation metrics.\n")

    # Save test indices for evaluation
    split_info = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "seed": args.seed,
    }
    with open(output_dir / "split_indices.json", "w") as f:
        json.dump(split_info, f)


if __name__ == "__main__":
    main()
