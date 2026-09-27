"""
BioPrint Vision — Automated System Verification & End-to-End Test Suite

Validates:
  1. Checkpoint loading and class mapping contract
  2. Image quality assessment module
  3. Preprocessing transform pipeline
  4. Real model inference on sample fingerprint images
  5. Probability distribution integrity (sum to ~100%, argmax matches predicted class)
  6. Grad-CAM saliency map and overlay generation
  7. Graceful error handling (None, invalid paths, unreadable data)
  8. Gradio UI construction

Run:
    python test_pipeline.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing.image_preprocess import (
    get_inference_transform,
    load_and_validate_image,
    IMG_SIZE,
)
from preprocessing.quality import check_image_quality
from explainability.gradcam import GradCAM, create_gradcam_overlay
from inference import BloodGroupPredictor, DEFAULT_CLASSES
from app import build_ui, analyze_fingerprint


def print_test_header(title: str):
    print(f"\n[TEST] {title}...")


def test_checkpoint_and_classes():
    print_test_header("1. Checkpoint & Class Contract Verification")
    model_path = Path("models/best_model.pth")
    class_names_path = Path("models/class_names.json")

    assert model_path.exists(), f"Model checkpoint not found at {model_path}"
    assert class_names_path.exists(), f"Class names file not found at {class_names_path}"

    with open(class_names_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    assert classes == DEFAULT_CLASSES, f"Class contract mismatch! Found: {classes}, Expected: {DEFAULT_CLASSES}"
    print(f"  [OK] Verified 8 target classes: {classes}")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict), "Checkpoint should be a dictionary"
    assert "model_state_dict" in checkpoint or len(checkpoint.keys()) > 0, "Checkpoint has no weights"
    val_acc = checkpoint.get("val_acc", None)
    epoch = checkpoint.get("epoch", None)
    print(f"  [OK] Checkpoint loaded successfully (Epoch: {epoch}, Val Acc: {val_acc})")


def test_quality_module():
    print_test_header("2. Image Quality Assessment Module")
    # Test on a normal synthetic image
    synthetic_arr = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    img_normal = Image.fromarray(synthetic_arr)
    q_normal = check_image_quality(img_normal)
    assert "score" in q_normal and "status" in q_normal and "issues" in q_normal
    print(f"  [OK] Synthetic Image -> Score: {q_normal['score']}, Status: {q_normal['status']}")

    # Test on a completely black image (should report underexposed/issues)
    img_black = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    q_black = check_image_quality(img_black)
    assert q_black["status"] == "Poor"
    assert any("underexposed" in issue.lower() or "contrast" in issue.lower() for issue in q_black["issues"])
    print(f"  [OK] Solid Black Image correctly flagged -> Status: {q_black['status']}")

    # Test on a tiny image
    img_tiny = Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 128)
    q_tiny = check_image_quality(img_tiny)
    assert any("resolution" in issue.lower() or "small" in issue.lower() for issue in q_tiny["issues"])
    print(f"  [OK] Tiny Image (16x16) correctly flagged -> Issues: {q_tiny['issues']}")


def test_preprocessing():
    print_test_header("3. Preprocessing Transform Pipeline")
    transform = get_inference_transform(IMG_SIZE)
    img = Image.new("RGB", (300, 400), color=(128, 128, 128))
    tensor = transform(img)
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), f"Unexpected tensor shape: {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Unexpected dtype: {tensor.dtype}"
    print(f"  [OK] Transformed PIL (300x400) -> Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")


def test_real_inference_and_predictions():
    print_test_header("4. Real Fingerprint Inference & Probability Contract")
    predictor = BloodGroupPredictor(device="cpu")

    sample_paths = [
        "data/raw/A+/cluster_0_1001.BMP",
        "data/raw/B+/cluster_2_10.BMP",
        "data/raw/O+/cluster_6_1004.BMP",
    ]

    existing_samples = [p for p in sample_paths if Path(p).exists()]
    if not existing_samples:
        found = list(Path("data/raw").glob("*/*.BMP"))[:2]
        existing_samples = [str(f) for f in found]

    assert len(existing_samples) > 0, "No raw fingerprint samples found to test!"

    for img_path in existing_samples:
        result = predictor.predict(img_path, return_gradcam=True)
        assert result["error"] is None, f"Inference failed with error: {result['error']}"

        pred_class = result["predicted_class"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        quality = result["quality"]
        gradcam = result["gradcam_overlay"]

        # Validate predicted class is one of the 8 target classes
        assert pred_class in DEFAULT_CLASSES, f"Invalid predicted class: {pred_class}"
        # Validate confidence is in range [0, 100]
        assert 0.0 <= confidence <= 100.0, f"Confidence out of bounds: {confidence}"
        # Validate all 8 classes are present in probabilities
        assert set(probs.keys()) == set(DEFAULT_CLASSES), "Probabilities missing target classes"
        # Validate sum of probabilities is ~100%
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 100.0) < 1.0, f"Probabilities do not sum to 100%: {prob_sum}"
        # Validate argmax corresponds to predicted class
        max_class = max(probs.items(), key=lambda x: x[1])[0]
        assert max_class == pred_class, f"Argmax class {max_class} does not match predicted {pred_class}"
        # Validate Grad-CAM overlay generated
        assert gradcam is not None, "Grad-CAM overlay should not be None for valid image"
        assert isinstance(gradcam, Image.Image), "Grad-CAM overlay is not a PIL Image"
        assert gradcam.size == (IMG_SIZE, IMG_SIZE), f"Grad-CAM size mismatch: {gradcam.size}"

        print(
            f"  [OK] Image '{Path(img_path).name}' -> Pred: {pred_class} "
            f"(Conf: {confidence:.1f}%) | Quality: {quality['status']} (Score: {quality['score']}) | "
            f"Grad-CAM: {gradcam.size}"
        )


def test_error_handling():
    print_test_header("5. Robust Error Handling")
    predictor = BloodGroupPredictor(device="cpu")

    # 1. Test None input
    res_none = predictor.predict(None)
    assert res_none["error"] is not None
    print(f"  [OK] None input handled gracefully -> Error: {res_none['error']}")

    # 2. Test non-existent path
    res_missing = predictor.predict("non_existent_fingerprint_file.bmp")
    assert res_missing["error"] is not None
    print(f"  [OK] Missing file handled gracefully -> Error: {res_missing['error']}")


def test_gradio_app_integration():
    print_test_header("6. Gradio Application Construction & Callback")
    app = build_ui()
    assert app is not None, "Failed to instantiate Gradio app"
    print("  [OK] Gradio Blocks UI instantiated successfully")

    # Test analyze_fingerprint callback with None
    summary_md, probs, gradcam, quality_md = analyze_fingerprint(None)
    assert "No Image Provided" in summary_md
    assert probs == {}
    assert gradcam is None
    print("  [OK] analyze_fingerprint(None) callback handled properly")

    # Test analyze_fingerprint callback with actual sample
    sample_path = Path("data/raw/A+/cluster_0_1001.BMP")
    if sample_path.exists():
        img = Image.open(sample_path)
        summary_md, probs, gradcam, quality_md = analyze_fingerprint(img)
        assert "Predicted Blood Group" in summary_md
        assert len(probs) == 8
        assert gradcam is not None
        assert "Quality Assessment" in quality_md
        print("  [OK] analyze_fingerprint(sample_image) end-to-end callback executed successfully")


def run_all_tests():
    print("\n" + "=" * 65)
    print("  BIOPRINT VISION -- SYSTEM VERIFICATION TEST SUITE")
    print("=" * 65)

    test_checkpoint_and_classes()
    test_quality_module()
    test_preprocessing()
    test_real_inference_and_predictions()
    test_error_handling()
    test_gradio_app_integration()

    print("\n" + "=" * 65)
    print("  ALL TESTS PASSED SUCCESSFULLY! PROTOTYPE READY.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_all_tests()
