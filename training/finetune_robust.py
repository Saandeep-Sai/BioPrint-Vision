"""
BioPrint Vision - Fine-Tuning Script with Aggressive Augmentation

Retrains the existing model checkpoint with domain-randomizing augmentation
to improve robustness against real-world scanner images (noise, blur,
contrast variation, resolution differences).

Usage:
    cd d:\\BioPrint-Vision
    .venv\\Scripts\\python.exe training/finetune_robust.py

    Or with custom parameters:
    .venv\\Scripts\\python.exe training/finetune_robust.py --epochs 15 --lr 5e-5
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train import build_model, set_seed


# ── Aggressive Augmentation Pipeline ─────────────────────────────────────────
class GaussianNoise:
    """Add random Gaussian noise to simulate scanner sensor noise."""
    def __init__(self, mean=0.0, std_range=(0.01, 0.08)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomGrayscale3ch:
    """Convert to grayscale (R=G=B) with given probability, matching training data."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gray = img.convert("L")
            return gray.convert("RGB")
        return img


def get_robust_train_transform(img_size=224):
    """
    Training transform with aggressive domain-randomizing augmentation.
    Simulates real-world scanner conditions the model will encounter.
    """
    return transforms.Compose([
        # Resolution jitter: resize to random size, then to target
        # Simulates different DPI scanners
        transforms.RandomChoice([
            transforms.Compose([
                transforms.Resize((random.randint(80, 120), random.randint(80, 120))),
                transforms.Resize((img_size, img_size)),
            ]),
            transforms.Resize((img_size, img_size)),
        ]),

        # Geometric augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.85, 1.15),
            shear=5,
        ),

        # Grayscale conversion (training data is grayscale as RGB)
        RandomGrayscale3ch(p=0.3),

        # Aggressive color/intensity augmentation
        transforms.ColorJitter(
            brightness=(0.3, 1.8),    # Wide range: dark scanner to bright
            contrast=(0.3, 2.0),      # Wide range: low to high contrast
            saturation=(0.0, 0.3),    # Minimal saturation (fingerprints are grayscale)
            hue=0.02,                 # Tiny hue variation
        ),

        # Random blur (simulates out-of-focus, motion blur)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5)),
        ], p=0.3),

        # Random inversion (some scanners have opposite polarity)
        transforms.RandomInvert(p=0.15),

        # Random posterize (reduces bit depth, simulates low-quality sensors)
        transforms.RandomPosterize(bits=6, p=0.1),

        # Random auto-contrast (simulates CLAHE-like preprocessing)
        transforms.RandomAutocontrast(p=0.2),

        # Random equalize (simulates histogram equalization)
        transforms.RandomEqualize(p=0.15),

        # Standard conversion
        transforms.ToTensor(),

        # Add Gaussian noise AFTER ToTensor (simulates sensor noise)
        GaussianNoise(mean=0.0, std_range=(0.0, 0.06)),

        # ImageNet normalization (must match original training)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform(img_size=224):
    """Standard validation transform (unchanged from original)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Resolution-jittering transform (needs custom dataset) ───────────────────
class RobustTrainTransform:
    """Wraps the augmentation with per-call random resolution jitter."""
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.base_transforms = transforms.Compose([
            # Geometric
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=5,
            ),

            # Grayscale (like training data)
            RandomGrayscale3ch(p=0.3),

            # Aggressive brightness/contrast
            transforms.ColorJitter(
                brightness=(0.3, 1.8),
                contrast=(0.3, 2.0),
                saturation=(0.0, 0.3),
                hue=0.02,
            ),

            # Blur
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5)),
            ], p=0.3),

            # Polarity
            transforms.RandomInvert(p=0.15),

            # Low-quality sensor simulation
            transforms.RandomPosterize(bits=6, p=0.1),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomEqualize(p=0.15),
        ])

    def __call__(self, img):
        # Resolution jitter: randomly downscale then upscale
        if random.random() < 0.4:
            jitter_size = random.randint(64, 140)
            img = img.resize((jitter_size, jitter_size))

        # Resize to model input
        img = img.resize((self.img_size, self.img_size))

        # Apply augmentation
        img = self.base_transforms(img)

        # ToTensor + Noise + Normalize
        tensor = transforms.ToTensor()(img)
        tensor = GaussianNoise(mean=0.0, std_range=(0.0, 0.06))(tensor)
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(tensor)

        return tensor


def finetune(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BIOPRINT VISION - ROBUST FINE-TUNING")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load existing checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = checkpoint.get("num_classes", 8)
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Previous val_acc: {checkpoint.get('val_acc', '?')}")
    else:
        state_dict = checkpoint
        num_classes = 8

    # Build model and load weights
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)


    # Unfreeze more layers for fine-tuning (layer2 + layer3 + layer4 + fc)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["layer2", "layer3", "layer4", "fc"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    # Dataset
    train_transform = RobustTrainTransform(img_size=224)
    val_transform = get_val_transform(img_size=224)

    full_dataset = datasets.ImageFolder(args.data_dir)

    # Validate all files actually exist on disk (prune broken entries)
    valid_indices = []
    removed = 0
    for i in range(len(full_dataset)):
        path, _ = full_dataset.samples[i]
        if os.path.isfile(path):
            valid_indices.append(i)
        else:
            removed += 1
    if removed > 0:
        print(f"  [Warning] Pruned {removed} missing file(s) from dataset index")
        full_dataset.samples = [full_dataset.samples[i] for i in valid_indices]
        full_dataset.targets = [full_dataset.targets[i] for i in valid_indices]
        full_dataset.imgs = full_dataset.samples

    # Handle class mismatch: remap dataset labels to original 8-class model indices
    original_classes = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    disk_classes = list(full_dataset.classes)
    print(f"  Classes on disk: {disk_classes}")

    if disk_classes != original_classes:
        label_remap = {}
        for ds_idx, cls_name in enumerate(disk_classes):
            if cls_name in original_classes:
                label_remap[ds_idx] = original_classes.index(cls_name)
            else:
                print(f"  [ERROR] Unknown class '{cls_name}' not in model contract")
                sys.exit(1)

        missing = [c for c in original_classes if c not in disk_classes]
        if missing:
            print(f"  Missing classes (retain original weights): {missing}")
        print(f"  Label remap: {label_remap}")

        for i in range(len(full_dataset)):
            old_label = full_dataset.targets[i]
            new_label = label_remap[old_label]
            full_dataset.targets[i] = new_label
            path, _ = full_dataset.samples[i]
            full_dataset.samples[i] = (path, new_label)
        full_dataset.imgs = full_dataset.samples
        class_names = original_classes
        print(f"  Labels remapped to 8-class model indices.")
    else:
        class_names = original_classes

    print(f"  Total images: {len(full_dataset)}")

    # Stratified split
    targets = [full_dataset.targets[i] for i in range(len(full_dataset))]
    indices = list(range(len(full_dataset)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=targets
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=temp_targets
    )

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Apply transforms via wrapper datasets
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = TransformSubset(full_dataset, train_idx, train_transform)
    val_dataset = TransformSubset(full_dataset, val_idx, val_transform)
    test_dataset = TransformSubset(full_dataset, test_idx, val_transform)

    # Weighted sampler for class imbalance
    train_targets = [targets[i] for i in train_idx]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    # Avoid division by zero for missing classes
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero].astype(np.float64)
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total_samples:.4f}")

        train_loss = running_loss / total_samples
        train_acc = correct / total_samples

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "img_size": 224,
                "class_names": class_names,
                "finetune_date": datetime.now().isoformat(),
                "augmentation": "robust_domain_randomization",
            }, str(save_path))
            print(f"  >> Saved best model (val_acc={val_acc:.4f})")

    # Test evaluation
    model.load_state_dict(
        torch.load(str(save_path), map_location=device, weights_only=False)["model_state_dict"]
    )
    model.eval()

    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING COMPLETE")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val_acc:   {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Saved to:       {save_path}")
    print(f"{'='*60}\n")

    # Save training log
    log_path = save_path.parent / "finetune_log.json"
    with open(str(log_path), "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "history": history,
        }, f, indent=2)
    print(f"  Training log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioPrint Vision - Robust Fine-Tuning")
    parser.add_argument("--checkpoint", default="models/best_model.pth", help="Source checkpoint")
    parser.add_argument("--data-dir", default="data/raw", help="Training data directory")
    parser.add_argument("--save-path", default="models/best_model_robust.pth", help="Output checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (lower for fine-tuning)")
    args = parser.parse_args()

    finetune(args)
