"""
BioPrint Vision - Definitive Retraining Pipeline

WHY PREPROCESSING ALONE CANNOT FIX THIS:
  The model was trained on 5000 tiny grayscale BMPs (96x103).
  Real-world fingerprints (phone camera, ink on paper) have fundamentally
  different spatial features, textures, and distributions.
  No amount of histogram matching, Gabor filtering, or CLAHE can
  teach the model features it never learned during training.

THIS SCRIPT:
  1. Takes the existing trained model as starting point
  2. Creates aggressive real-world augmentation that simulates:
     - Blue/black ink on white paper
     - Phone camera photos (JPEG artifacts, blur, lighting)
     - Scanner captures (noise, resolution changes)
     - Gabor-processed versions (so the model learns ridge features)
  3. Incorporates any real labeled images (in data/real_world/)
  4. Fine-tunes with mixed data: original + augmented + real images
  5. Saves to models/best_model.pth (overwrites) for production use

USAGE:
  cd d:\\BioPrint-Vision
  .venv\\Scripts\\python.exe training/retrain_definitive.py

  With custom params:
  .venv\\Scripts\\python.exe training/retrain_definitive.py --epochs 15 --lr 3e-5
"""

import os
import sys
import json
import random
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train import build_model, set_seed


# ============================================================================
# REAL-WORLD SIMULATION AUGMENTATIONS
# ============================================================================

class SimulateInkPrint:
    """
    Simulate the look of a real-world ink fingerprint on paper.
    Converts grayscale fingerprint to look like blue/black ink on white paper.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # Convert to grayscale first
        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32)

        # Normalize to 0-1
        arr = arr / 255.0

        # Create ink-on-paper look: dark ridges on white background
        # Invert if needed (training data has light background, dark ridges)
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        # Add white paper background with slight texture
        paper = np.ones_like(arr) * random.uniform(0.92, 1.0)
        paper += np.random.normal(0, 0.01, arr.shape).astype(np.float32)

        # Blend: where there are ridges (dark), show ink color
        ink_strength = random.uniform(0.4, 0.9)
        blend = paper * (1 - arr * ink_strength)
        blend = np.clip(blend, 0, 1)

        # Choose ink color
        ink_type = random.choice(['blue', 'black', 'dark_blue', 'gray'])
        if ink_type == 'blue':
            r = blend * random.uniform(0.3, 0.6)
            g = blend * random.uniform(0.4, 0.7)
            b = blend * random.uniform(0.7, 1.0)
        elif ink_type == 'dark_blue':
            r = blend * random.uniform(0.2, 0.5)
            g = blend * random.uniform(0.3, 0.6)
            b = blend * random.uniform(0.6, 0.9)
        elif ink_type == 'black':
            r = blend
            g = blend
            b = blend
        else:  # gray
            factor = random.uniform(0.6, 0.9)
            r = blend * factor
            g = blend * factor
            b = blend * factor

        rgb = np.stack([
            (r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8)
        ], axis=-1)

        return Image.fromarray(rgb)


class SimulateCameraCapture:
    """
    Simulate phone camera photo artifacts:
    - Slight perspective distortion
    - JPEG compression artifacts
    - Uneven lighting (vignetting)
    - Slight blur
    """
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        arr = np.array(img)

        # Slight brightness gradient (uneven lighting)
        if random.random() < 0.5:
            h, w = arr.shape[:2]
            gradient = np.linspace(
                random.uniform(0.7, 0.95),
                random.uniform(1.0, 1.3),
                w
            ).reshape(1, w, 1)
            arr = np.clip(arr * gradient, 0, 255).astype(np.uint8)

        # Random slight perspective warp
        if random.random() < 0.3:
            h, w = arr.shape[:2]
            margin = int(min(h, w) * 0.05)
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([
                [random.randint(0, margin), random.randint(0, margin)],
                [w - random.randint(0, margin), random.randint(0, margin)],
                [random.randint(0, margin), h - random.randint(0, margin)],
                [w - random.randint(0, margin), h - random.randint(0, margin)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            arr = cv2.warpPerspective(arr, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(arr)


class GaborAugmentation:
    """
    Apply Gabor filter bank as augmentation.
    This teaches the model to recognize ridge-enhanced features.
    """
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        gray = np.array(img.convert('L'), dtype=np.uint8)
        gabor_accum = np.zeros_like(gray, dtype=np.float64)

        n_orient = random.choice([4, 6, 8])
        ksize = random.choice([15, 21, 31])
        sigma = random.uniform(3.0, 5.0)
        lambd = random.uniform(6.0, 10.0)

        for theta in np.arange(0, np.pi, np.pi / n_orient):
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_64F
            )
            filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kern)
            gabor_accum += np.abs(filtered)

        if gabor_accum.max() > 0:
            gabor_norm = ((gabor_accum / gabor_accum.max()) * 255).astype(np.uint8)
        else:
            gabor_norm = gray

        rgb = cv2.cvtColor(gabor_norm, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)


class GaussianNoise:
    """Add Gaussian noise to tensor."""
    def __init__(self, mean=0.0, std_range=(0.01, 0.08)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomGrayscale3ch:
    """Convert to grayscale RGB with given probability."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.convert("L").convert("RGB")
        return img


# ============================================================================
# DATASET WITH MIXED SOURCES
# ============================================================================

class MixedFingerprintDataset(Dataset):
    """
    Dataset that combines:
    1. Original training images from data/raw/
    2. Real-world labeled images from data/real_world/
    With aggressive real-world simulation augmentation.
    """
    def __init__(self, samples, transform, img_size=224):
        self.samples = samples  # List of (filepath, label_index)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Return a blank image if file is corrupted
            img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================================
# TRAINING TRANSFORMS
# ============================================================================

class DefinitiveTrainTransform:
    """
    The definitive training transform that simulates ALL real-world conditions.
    """
    def __init__(self, img_size=224):
        self.img_size = img_size

        # Real-world simulation (applied to PIL images)
        self.real_world_augs = [
            SimulateInkPrint(p=0.4),
            SimulateCameraCapture(p=0.3),
            GaborAugmentation(p=0.25),
        ]

        # Standard augmentations
        self.geometric = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, translate=(0.08, 0.08),
                scale=(0.85, 1.15), shear=5,
            ),
        ])

        self.color_aug = transforms.Compose([
            RandomGrayscale3ch(p=0.25),
            transforms.ColorJitter(
                brightness=(0.3, 1.8),
                contrast=(0.3, 2.0),
                saturation=(0.0, 0.3),
                hue=0.02,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5)),
            ], p=0.3),
            transforms.RandomInvert(p=0.1),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomEqualize(p=0.15),
        ])

    def __call__(self, img):
        # Ensure RGB
        img = img.convert('RGB')

        # Resolution jitter (simulate different DPI)
        if random.random() < 0.4:
            jitter_size = random.randint(64, 160)
            img = img.resize((jitter_size, jitter_size))

        # Resize to target
        img = img.resize((self.img_size, self.img_size))

        # Apply real-world simulation augmentations
        for aug in self.real_world_augs:
            img = aug(img)

        # Geometric augmentation
        img = self.geometric(img)

        # Color augmentation
        img = self.color_aug(img)

        # To tensor + noise + normalize
        tensor = transforms.ToTensor()(img)
        tensor = GaussianNoise(mean=0.0, std_range=(0.0, 0.06))(tensor)
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(tensor)

        return tensor


def get_val_transform(img_size=224):
    """Standard validation transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def retrain(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  BIOPRINT VISION - DEFINITIVE RETRAINING")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Class contract
    original_classes = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    num_classes = len(original_classes)

    # ── Collect training samples ──────────────────────────────────────────
    all_samples = []  # List of (filepath, label_index)

    # Source 1: Original training data from data/raw/
    data_dir = Path(args.data_dir)
    disk_classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"  Classes on disk: {disk_classes}")

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in original_classes:
            print(f"  [WARNING] Unknown class '{cls_name}', skipping")
            continue

        label_idx = original_classes.index(cls_name)
        for f in cls_dir.iterdir():
            if f.suffix.upper() in ('.BMP', '.PNG', '.JPG', '.JPEG'):
                if f.is_file():
                    all_samples.append((str(f), label_idx))

    print(f"  Original training images: {len(all_samples)}")

    # Source 2: Real-world labeled images from data/real_world/
    real_dir = Path(args.real_world_dir)
    real_count = 0
    if real_dir.is_dir():
        for cls_dir in sorted(real_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            if cls_name not in original_classes:
                print(f"  [WARNING] Unknown real-world class '{cls_name}', skipping")
                continue

            label_idx = original_classes.index(cls_name)
            for f in cls_dir.iterdir():
                if f.suffix.upper() in ('.BMP', '.PNG', '.JPG', '.JPEG'):
                    if f.is_file():
                        # Add real-world images with HIGHER weight by duplicating
                        for _ in range(args.real_world_oversample):
                            all_samples.append((str(f), label_idx))
                        real_count += 1

        print(f"  Real-world images: {real_count} (x{args.real_world_oversample} oversample = {real_count * args.real_world_oversample})")
    else:
        print(f"  No real-world data directory found at {real_dir}")
        print(f"  To add real images: create data/real_world/<BloodGroup>/ folders")
        print(f"  Example: data/real_world/AB-/my_print.jpeg")

    # Validate files exist
    valid_samples = [(p, l) for p, l in all_samples if os.path.isfile(p)]
    removed = len(all_samples) - len(valid_samples)
    if removed > 0:
        print(f"  [WARNING] Pruned {removed} missing files")
    all_samples = valid_samples

    print(f"  Total training samples: {len(all_samples)}")

    # Count per class
    class_counts_dict = {}
    for _, label in all_samples:
        cls = original_classes[label]
        class_counts_dict[cls] = class_counts_dict.get(cls, 0) + 1
    for cls in original_classes:
        count = class_counts_dict.get(cls, 0)
        print(f"    {cls}: {count}")

    # ── Load existing checkpoint ──────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"  [ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Previous val_acc: {checkpoint.get('val_acc', '?')}")
    else:
        state_dict = checkpoint

    # Build model and load weights
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)

    # Unfreeze layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on depth setting
    if args.unfreeze == "head":
        layers_to_unfreeze = ["fc"]
    elif args.unfreeze == "shallow":
        layers_to_unfreeze = ["layer4", "fc"]
    elif args.unfreeze == "medium":
        layers_to_unfreeze = ["layer3", "layer4", "fc"]
    else:  # "deep"
        layers_to_unfreeze = ["layer2", "layer3", "layer4", "fc"]

    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfreezing: {layers_to_unfreeze}")
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    # ── Data splits ───────────────────────────────────────────────────────
    targets = [s[1] for s in all_samples]
    indices = list(range(len(all_samples)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.25, random_state=42, stratify=targets
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=temp_targets
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # ── Datasets and loaders ──────────────────────────────────────────────
    train_transform = DefinitiveTrainTransform(img_size=224)
    val_transform = get_val_transform(img_size=224)

    train_dataset = MixedFingerprintDataset(train_samples, train_transform)
    val_dataset = MixedFingerprintDataset(val_samples, val_transform)
    test_dataset = MixedFingerprintDataset(test_samples, val_transform)

    # Weighted sampler for class balance
    train_targets = [s[1] for s in train_samples]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero].astype(np.float64)
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Optimizer and scheduler ───────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing model
    if save_path.exists() and args.backup:
        backup_path = save_path.with_name(f"best_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        shutil.copy2(str(save_path), str(backup_path))
        print(f"  Backed up existing model to: {backup_path}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    patience_counter = 0

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

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "img_size": 224,
                "class_names": original_classes,
                "retrain_date": datetime.now().isoformat(),
                "augmentation": "definitive_real_world_simulation",
                "real_world_images": real_count,
            }, str(save_path))
            print(f"  >> Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  >> Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ── Test evaluation ───────────────────────────────────────────────────
    best_ckpt = torch.load(str(save_path), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            for i in range(labels.size(0)):
                gt = labels[i].item()
                per_class_total[gt] += 1
                if predicted[i].item() == gt:
                    per_class_correct[gt] += 1

    test_acc = test_correct / test_total

    print(f"\n{'='*60}")
    print(f"  RETRAINING COMPLETE")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val_acc:   {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Saved to:       {save_path}")
    print(f"\n  Per-class test accuracy:")
    for i, cls in enumerate(original_classes):
        if per_class_total[i] > 0:
            acc = per_class_correct[i] / per_class_total[i]
            print(f"    {cls}: {int(per_class_correct[i])}/{int(per_class_total[i])} ({acc*100:.1f}%)")
        else:
            print(f"    {cls}: no test samples")
    print(f"{'='*60}\n")

    # Save training log
    log_path = save_path.parent / "retrain_log.json"
    with open(str(log_path), "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "real_world_images": real_count,
            "real_world_oversample": args.real_world_oversample,
            "unfreeze": args.unfreeze,
            "history": history,
        }, f, indent=2)
    print(f"  Training log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioPrint Vision - Definitive Retraining")
    parser.add_argument("--checkpoint", default="models/best_model.pth", help="Source checkpoint")
    parser.add_argument("--data-dir", default="data/raw", help="Original training data")
    parser.add_argument("--real-world-dir", default="data/real_world", help="Real-world labeled images")
    parser.add_argument("--save-path", default="models/best_model.pth", help="Output checkpoint (overwrites)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--unfreeze", default="deep", choices=["head", "shallow", "medium", "deep"],
                        help="How many layers to unfreeze")
    parser.add_argument("--real-world-oversample", type=int, default=20,
                        help="Oversample real-world images N times (compensates for scarcity)")
    parser.add_argument("--no-backup", dest="backup", action="store_false",
                        help="Skip backing up existing model")
    parser.set_defaults(backup=True)
    args = parser.parse_args()

    retrain(args)
