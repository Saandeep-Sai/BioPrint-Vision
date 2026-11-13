"""
BioPrint Vision -- Standalone Inference Module

Provides the BloodGroupPredictor class that encapsulates:
  1. Checkpoint loading (ResNet-50 with custom classification head)
  2. Input validation and image quality assessment
  3. Preprocessing (matching training resolution & normalization)
  4. Multi-strategy ensemble inference for real-world images
  5. Saliency visualization via Grad-CAM
  6. Structured, deterministic result output

Usage via CLI:
    python inference.py --image data/raw/A+/cluster_0_1001.BMP
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Union, Dict, Any, Optional, List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing.image_preprocess import (
    get_inference_transform,
    load_and_validate_image,
    IMG_SIZE,
)
from preprocessing.quality import check_image_quality
from preprocessing.domain_adapt import (
    adapt_if_needed,
    is_scanner_image,
    get_ensemble_preprocessed,
)
from explainability.gradcam import GradCAM, create_gradcam_overlay
from training.train import build_model

# Authoritative 8-class contract ordering
DEFAULT_CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


class BloodGroupPredictor:
    """
    Production-ready predictor for fingerprint-based blood-group classification.
    Loads the trained PyTorch checkpoint once into memory for rapid inference.

    For real-world images (camera photos, scanner captures, ink prints):
      Uses multi-strategy ensemble + TTA to maximize accuracy.
    For training-like images:
      Uses direct single-pass inference (already high accuracy).
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/best_model.pth",
        class_names_path: Union[str, Path] = "models/class_names.json",
        device: Optional[str] = None,
    ):
        # 1. Determine compute device (GPU when available, else CPU fallback)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load and verify class mapping contract
        self.class_names = list(DEFAULT_CLASSES)
        if os.path.exists(str(class_names_path)):
            try:
                with open(class_names_path, "r", encoding="utf-8") as f:
                    loaded_classes = json.load(f)
                    if isinstance(loaded_classes, list) and len(loaded_classes) == 8:
                        self.class_names = loaded_classes
            except Exception as e:
                print(f"[Warning] Could not load class names from {class_names_path}: {e}. Using default contract.")

        # 3. Verify checkpoint existence
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{model_path.resolve()}'. "
                "Ensure 'models/best_model.pth' is present."
            )

        # 4. Load checkpoint
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)

        # Extract state dict and metadata
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.img_size = checkpoint.get("img_size", IMG_SIZE)
            num_classes = checkpoint.get("num_classes", len(self.class_names))
            self.epoch = checkpoint.get("epoch", None)
            self.val_acc = checkpoint.get("val_acc", None)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            self.img_size = IMG_SIZE
            num_classes = len(self.class_names)
            self.epoch = None
            self.val_acc = None
        else:
            raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # 5. Instantiate ResNet-50 architecture and load weights
        self.model = build_model(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 6. Initialize deterministic inference transform (matching validation)
        self.transform = get_inference_transform(self.img_size)

        # 7. Initialize Grad-CAM on the final residual block
        try:
            self.gradcam = GradCAM(self.model, target_layer=self.model.layer4[-1])
            self.gradcam_available = True
        except Exception as e:
            print(f"[Warning] Failed to initialize Grad-CAM: {e}")
            self.gradcam = None
            self.gradcam_available = False

    def _forward_single(self, pil_img: Image.Image) -> np.ndarray:
        """
        Run a single forward pass on a preprocessed PIL image.
        Returns raw softmax probability vector (shape: [num_classes]).
        """
        pil_img = pil_img.convert("RGB")
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs[0], dim=0)
        return probs.cpu().numpy()

    def _ensemble_predict(self, pil_img: Image.Image) -> np.ndarray:
        """
        Multi-strategy ensemble prediction for real-world images.

        Runs multiple preprocessing strategies + TTA augmentations,
        averages all softmax vectors, and returns the ensemble probability.

        This is the KEY to robust real-world prediction:
        - No single strategy works for all images
        - But the correct class tends to get moderate probability across
          multiple strategies, while wrong classes are inconsistent
        - Averaging washes out the noise and amplifies the signal
        """
        all_probs = []

        # Get all preprocessing strategies
        preprocessed_list = get_ensemble_preprocessed(pil_img)

        for strategy_name, processed_img in preprocessed_list:
            # Run base prediction
            probs = self._forward_single(processed_img)
            all_probs.append(probs)

            # TTA: horizontal flip for each strategy
            flipped = ImageOps.mirror(processed_img)
            probs_flip = self._forward_single(flipped)
            all_probs.append(probs_flip)

        if not all_probs:
            # Fallback: just run the original image
            return self._forward_single(pil_img)

        # Average all probability vectors
        ensemble_probs = np.mean(all_probs, axis=0)

        # Renormalize to ensure sum=1
        total = ensemble_probs.sum()
        if total > 0:
            ensemble_probs = ensemble_probs / total

        return ensemble_probs

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        return_gradcam: bool = True,
        domain_adapt: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end inference on a single fingerprint image.

        For external/real-world images: uses multi-strategy ensemble
        For training-like images: uses direct single-pass inference

        Args:
            image_input: File path, PIL Image, or NumPy array
            return_gradcam: Whether to compute and return Grad-CAM visualization
            domain_adapt: Whether to apply domain-adaptive preprocessing for
                          scanner/external images (recommended for non-training data)

        Returns:
            Dict containing:
                - 'predicted_class': str (e.g. 'B+')
                - 'confidence': float (percentage 0.0 - 100.0)
                - 'probabilities': Dict[str, float] (per-class percentage)
                - 'quality': Dict[str, Any] (quality score, status, issues)
                - 'gradcam_overlay': Optional[PIL.Image.Image]
                - 'error': Optional[str]
        """
        result: Dict[str, Any] = {
            "predicted_class": None,
            "confidence": 0.0,
            "probabilities": {},
            "quality": None,
            "gradcam_overlay": None,
            "error": None,
        }

        # -- Step 1: Input Validation & Image Loading
        if image_input is None:
            result["error"] = "No image provided. Please supply a fingerprint image."
            return result

        try:
            pil_img = load_and_validate_image(image_input)
        except ValueError as e:
            result["error"] = f"Invalid image: {e}"
            return result
        except Exception as e:
            result["error"] = f"Failed to open image file: {e}"
            return result

        # -- Step 2: Image Quality Assessment
        try:
            quality = check_image_quality(pil_img)
            result["quality"] = quality
        except Exception as e:
            result["quality"] = {
                "score": 0.50,
                "status": "Fair",
                "summary": "Quality assessment unavailable",
                "valid": True,
                "issues": [f"Quality analysis error: {e}"],
                "stats": {},
            }

        # -- Step 3: Domain-Adaptive Preprocessing & Inference
        try:
            is_external = is_scanner_image(pil_img)

            if domain_adapt and is_external:
                # ENSEMBLE: Multi-strategy preprocessing for real-world images
                probs = self._ensemble_predict(pil_img)
            else:
                # DIRECT: Single-pass for training-like images
                probs = self._forward_single(pil_img)

            pred_idx = int(np.argmax(probs))
            predicted_class = self.class_names[pred_idx]
            confidence_pct = round(float(probs[pred_idx]) * 100.0, 2)

            result["predicted_class"] = predicted_class
            result["confidence"] = confidence_pct

            # Populate exact 8-class probability dictionary
            probs_dict = {}
            for i, name in enumerate(self.class_names):
                probs_dict[name] = round(float(probs[i]) * 100.0, 2)
            result["probabilities"] = probs_dict

        except Exception as e:
            result["error"] = f"Model execution error: {e}"
            return result

        # -- Step 4: Explainability (Grad-CAM)
        # Use the primary adapted image for Grad-CAM visualization
        if return_gradcam and self.gradcam_available:
            try:
                # Use gabor-adapted image for Grad-CAM on external images
                if domain_adapt and is_external:
                    gradcam_img = adapt_if_needed(pil_img, force=True)
                else:
                    gradcam_img = pil_img

                gradcam_img = gradcam_img.convert("RGB")
                tensor = self.transform(gradcam_img).unsqueeze(0).to(self.device)
                cam = self.gradcam.generate(tensor, pred_idx)
                overlay = create_gradcam_overlay(
                    pil_img, cam, target_size=(self.img_size, self.img_size)
                )
                result["gradcam_overlay"] = overlay
            except Exception as e:
                # Grad-CAM failure does not compromise core classification
                result["gradcam_overlay"] = None

        return result


def main():
    parser = argparse.ArgumentParser(description="BioPrint Vision -- CLI Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input fingerprint image")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Checkpoint path")
    parser.add_argument("--classes", type=str, default="models/class_names.json", help="Class names JSON")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  BIOPRINT VISION -- INFERENCE EVALUATION")
    print("=" * 60)

    try:
        predictor = BloodGroupPredictor(
            model_path=args.model,
            class_names_path=args.classes,
            device=args.device,
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        sys.exit(1)

    result = predictor.predict(args.image, return_gradcam=True)

    if result["error"]:
        print(f"\n[ERROR] {result['error']}\n")
        sys.exit(1)

    print(f"\nInput Image:      {args.image}")
    print(f"Predicted Class:  {result['predicted_class']}")
    print(f"Model Confidence: {result['confidence']:.2f}%")

    if result["quality"]:
        q = result["quality"]
        print(f"Image Quality:    {q['status']} (Score: {q['score']:.2f})")
        if q["issues"]:
            print(f"Quality Notes:    {'; '.join(q['issues'])}")

    print("\nClass Probability Distribution:")
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        bar = "#" * int(round(prob / 2.5))
        marker = "<-- PREDICTED" if cls_name == result["predicted_class"] else ""
        print(f"  {cls_name:>4s}: {prob:>6.2f}%  {bar:<40s} {marker}")

    if result["gradcam_overlay"] is not None:
        print("\nGrad-CAM:         Generated successfully.")

    print("\n" + "-" * 60)
    print("NOTE: Experimental AI prediction for research/prototype purposes.")
    print("This screening tool does not replace laboratory blood typing.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
