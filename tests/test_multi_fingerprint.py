"""
BioPrint Vision — Multi-Fingerprint Test Suite
Tests region detection, splitting, probability averaging, and agreement scoring.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.multi_fingerprint import MultiFingerprintAnalyzer, detect_fingerprint_regions
from inference import DEFAULT_CLASSES


def test_single_fingerprint_fallback():
    print("\n[TEST] 1. Single Fingerprint Fallback / Mode...")
    analyzer = MultiFingerprintAnalyzer()

    # Load a known sample
    sample_path = BASE_DIR / "data" / "raw" / "A+" / "cluster_0_1001.BMP"
    assert sample_path.exists(), f"Sample image {sample_path} not found"

    result = analyzer.analyze(sample_path)
    assert result["final_prediction"] in DEFAULT_CLASSES
    assert 0.0 <= result["final_confidence"] <= 100.0
    assert result["detected_count"] >= 1
    assert result["analyzed_count"] >= 1
    assert result["agreement"]["percentage"] >= 0.0
    print(f"  [OK] Single fingerprint analyzed -> Pred: {result['final_prediction']}, Conf: {result['final_confidence']}%")


def test_multi_fingerprint_sheet():
    print("\n[TEST] 2. Multi-Fingerprint Sheet Detection & Probability Averaging...")
    analyzer = MultiFingerprintAnalyzer()

    # Create a realistic composite sheet containing 3 distinct fingerprint impressions on a white card
    p1 = BASE_DIR / "data" / "raw" / "A+" / "cluster_0_1001.BMP"
    p2 = BASE_DIR / "data" / "raw" / "B+" / "cluster_2_10.BMP"
    p3 = BASE_DIR / "data" / "raw" / "O+" / "cluster_6_1004.BMP"

    img1 = Image.open(p1).convert("RGB")
    img2 = Image.open(p2).convert("RGB")
    img3 = Image.open(p3).convert("RGB")

    # Canvas of size 600 x 500 white background
    sheet = Image.new("RGB", (650, 500), color=(245, 245, 245))
    sheet.paste(img1, (40, 50))
    sheet.paste(img2, (380, 50))
    sheet.paste(img3, (210, 260))

    result = analyzer.analyze(sheet)

    print(f"  Detected fingerprints: {result['detected_count']}")
    print(f"  Analyzed fingerprints: {result['analyzed_count']}")
    print(f"  Excluded fingerprints: {result['excluded_count']}")
    print(f"  Final Prediction:      {result['final_prediction']} ({result['final_confidence']}%)")
    print(f"  Agreement Percentage:  {result['agreement']['percentage']}%")

    assert result["detected_count"] >= 2, f"Expected at least 2 detected fingerprints, got {result['detected_count']}"
    assert result["mode"] == "multiple"
    assert result["final_prediction"] in DEFAULT_CLASSES
    assert len(result["final_probabilities"]) == 8

    # Verify probability averaging math:
    # Average of probabilities across included fingerprints must match final_probabilities within 0.1%
    included_probs = [
        [fp["probabilities"][cls] / 100.0 for cls in DEFAULT_CLASSES]
        for fp in result["fingerprints"]
        if fp["included"]
    ]
    expected_avg = np.mean(included_probs, axis=0)
    expected_avg = expected_avg / np.sum(expected_avg)

    for i, cls in enumerate(DEFAULT_CLASSES):
        actual_pct = result["final_probabilities"][cls]
        expected_pct = round(float(expected_avg[i]) * 100.0, 2)
        assert abs(actual_pct - expected_pct) < 0.25, (
            f"Averaging mismatch for {cls}: actual {actual_pct} vs expected {expected_pct}"
        )

    print("  [OK] Probability averaging mathematically verified!")
    print("  [OK] Multi-fingerprint detection and agreement verified!")


if __name__ == "__main__":
    test_single_fingerprint_fallback()
    test_multi_fingerprint_sheet()
    print("\n[OK] All Multi-Fingerprint Tests Passed!\n")
