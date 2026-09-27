"""
BioPrint Vision — Dataset Audit Script

Audits the raw fingerprint dataset and generates:
  - dataset_audit.json  (machine-readable)
  - terminal summary    (human-readable)

Usage:
    python -m training.audit_dataset --data-dir data/raw
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

from PIL import Image

# ── Supported extensions ──────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}


def md5_file(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(data_dir: str) -> dict:
    """
    Comprehensive audit of the fingerprint dataset.

    Returns a dict with all audit findings.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    # ── Scan directory structure ──────────────────────────────────────────
    class_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not class_dirs:
        print(f"ERROR: No class subdirectories found in '{data_dir}'.")
        sys.exit(1)

    class_names = [d.name for d in class_dirs]
    print(f"Found {len(class_names)} classes: {class_names}")

    # ── Scan all images ──────────────────────────────────────────────────
    audit = {
        "data_dir": str(data_path.resolve()),
        "class_count": len(class_names),
        "class_names": class_names,
        "total_images": 0,
        "images_per_class": {},
        "file_extensions": defaultdict(int),
        "dimensions": defaultdict(int),
        "channels": defaultdict(int),
        "corrupt_images": [],
        "duplicate_files": [],
        "filename_pattern_analysis": {},
        "subject_id_available": False,
        "subject_id_notes": "",
    }

    all_hashes = {}  # hash -> list of file paths
    all_dims = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = []

        for f in sorted(class_dir.iterdir()):
            ext = f.suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            audit["file_extensions"][ext] += 1

            # Try to open and verify the image
            try:
                img = Image.open(f)
                img.verify()
                # Reopen after verify
                img = Image.open(f)
                w, h = img.size
                mode = img.mode
                audit["dimensions"][f"{w}x{h}"] += 1
                audit["channels"][mode] += 1
                all_dims.append((w, h))
                images.append(str(f))
            except Exception as e:
                audit["corrupt_images"].append({
                    "path": str(f),
                    "error": str(e),
                })
                continue

            # Hash for duplicate detection
            file_hash = md5_file(str(f))
            if file_hash not in all_hashes:
                all_hashes[file_hash] = []
            all_hashes[file_hash].append(str(f))

        audit["images_per_class"][class_name] = len(images)
        audit["total_images"] += len(images)
        print(f"  {class_name}: {len(images)} images")

    # ── Duplicate detection ──────────────────────────────────────────────
    duplicates = []
    for h, paths in all_hashes.items():
        if len(paths) > 1:
            duplicates.append(paths)
    audit["duplicate_files"] = duplicates
    audit["duplicate_count"] = sum(len(g) - 1 for g in duplicates)

    # ── Filename pattern analysis ────────────────────────────────────────
    # Check if filenames encode subject IDs
    # Pattern: cluster_X_YYYY.BMP
    # cluster_X = class cluster, YYYY = image sequence number
    sample_files = []
    for class_dir in class_dirs:
        for f in list(class_dir.iterdir())[:5]:
            sample_files.append(f.name)

    audit["filename_pattern_analysis"] = {
        "sample_filenames": sample_files,
        "pattern": "cluster_<cluster_id>_<sequence_number>.BMP",
        "notes": (
            "Filenames follow pattern 'cluster_X_YYYY.BMP' where X is "
            "a cluster ID (0-7) and YYYY is a sequence number. "
            "The sequence number does NOT appear to encode a subject/person ID. "
            "Multiple fingerprints could be from the same person but there is "
            "no metadata to confirm or deny this."
        ),
    }

    # ── Subject ID analysis ──────────────────────────────────────────────
    audit["subject_id_available"] = False
    audit["subject_id_notes"] = (
        "No subject/person identifiers are available in the filename or "
        "directory structure. The dataset uses cluster_X_YYYY.BMP naming "
        "where the sequence number cannot be confirmed as a subject ID. "
        "This means we CANNOT guarantee subject-disjoint splits. "
        "Multiple fingerprints from the same individual may exist in both "
        "train and test sets. This is a known limitation."
    )

    # ── Class imbalance ──────────────────────────────────────────────────
    counts = list(audit["images_per_class"].values())
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        audit["class_imbalance"] = {
            "max_class_count": max_count,
            "min_class_count": min_count,
            "imbalance_ratio": round(max_count / min_count, 2) if min_count > 0 else float("inf"),
            "mean_per_class": round(sum(counts) / len(counts), 1),
        }

    # Convert defaultdicts for JSON serialization
    audit["file_extensions"] = dict(audit["file_extensions"])
    audit["dimensions"] = dict(audit["dimensions"])
    audit["channels"] = dict(audit["channels"])

    return audit


def print_summary(audit: dict):
    """Print a human-readable summary of the audit."""
    print("\n" + "=" * 70)
    print("  BIOPRINT VISION -- DATASET AUDIT SUMMARY")
    print("=" * 70)
    print(f"\n  Data Directory:    {audit['data_dir']}")
    print(f"  Total Images:      {audit['total_images']}")
    print(f"  Class Count:       {audit['class_count']}")
    print(f"  Class Names:       {audit['class_names']}")

    print(f"\n  Images per Class:")
    for cls, count in sorted(audit["images_per_class"].items()):
        bar = "#" * (count // 20)
        print(f"    {cls:>4s}: {count:>5d}  {bar}")

    if audit.get("class_imbalance"):
        ci = audit["class_imbalance"]
        print(f"\n  Class Imbalance:")
        print(f"    Max class count: {ci['max_class_count']}")
        print(f"    Min class count: {ci['min_class_count']}")
        print(f"    Imbalance ratio: {ci['imbalance_ratio']}:1")
        print(f"    Mean per class:  {ci['mean_per_class']}")

    print(f"\n  File Extensions:   {audit['file_extensions']}")
    print(f"  Image Dimensions:  {audit['dimensions']}")
    print(f"  Image Modes:       {audit['channels']}")
    print(f"  Corrupt Images:    {len(audit['corrupt_images'])}")
    print(f"  Duplicate Files:   {audit['duplicate_count']} "
          f"({len(audit['duplicate_files'])} groups)")

    print(f"\n  Subject IDs Available: {audit['subject_id_available']}")
    print(f"  Subject ID Notes:")
    for line in audit["subject_id_notes"].split(". "):
        print(f"    - {line.strip('.')}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit the fingerprint dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Path to raw dataset directory with class subdirectories"
    )
    parser.add_argument(
        "--output", type=str, default="dataset_audit.json",
        help="Output path for machine-readable audit JSON"
    )
    args = parser.parse_args()

    audit = audit_dataset(args.data_dir)
    print_summary(audit)

    # Save machine-readable audit
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Audit saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
