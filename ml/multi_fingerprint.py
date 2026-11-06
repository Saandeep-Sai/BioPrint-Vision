"""
BioPrint Vision — Multi-Fingerprint Detection & Aggregation Module

Implements V2 specifications:
  1. Automated detection and splitting of individual fingerprint regions
     from a single photograph containing multiple fingerprints.
  2. Bounding box extraction with spatial ordering (F1, F2, F3...).
  3. Visual annotated image generation with overlaid bounding boxes.
  4. Per-crop quality assessment and deterministic exclusion tracking.
  5. Per-crop ResNet-50 inference and per-crop Grad-CAM saliency maps.
  6. Raw probability vector averaging:
         P_final(class) = (P1(class) + P2(class) + ... + PN(class)) / N
  7. Final class prediction via argmax(P_final) and aggregated confidence.
  8. Agreement and consistency calculation across detected fingerprints.
  9. Seamless backward compatibility for single-fingerprint images.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from inference import BloodGroupPredictor, DEFAULT_CLASSES
from preprocessing.image_preprocess import load_and_validate_image, IMG_SIZE
from preprocessing.quality import check_image_quality


def detect_fingerprint_regions(
    pil_img: Image.Image,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.85,
) -> Tuple[List[Tuple[int, int, int, int]], Image.Image]:
    """
    Detect individual fingerprint bounding boxes in a photograph using
    computer vision techniques (contrast enhancement, adaptive thresholding,
    morphological operations, contour analysis, and non-max suppression).

    Args:
        pil_img: Input PIL Image (photograph)
        min_area_ratio: Minimum bounding box area relative to image
        max_area_ratio: Maximum bounding box area relative to image

    Returns:
        (bounding_boxes, annotated_pil_image)
        where bounding_boxes is a list of (x, y, w, h)
    """
    img_rgb = pil_img.convert("RGB")
    img_np = np.array(img_rgb)
    h_img, w_img, _ = img_np.shape
    total_area = w_img * h_img

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. Contrast Enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gaussian Blur to reduce high-frequency scanner/paper noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 3. Multi-strategy thresholding (Otsu & Adaptive)
    # Fingerprints usually appear darker than the paper background
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to connect ridge islands within the same fingerprint impression
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / max(1, h)

        # Filter by area and aspect ratio (fingerprints are roughly 0.4 - 2.5 aspect ratio)
        if min_area <= area <= max_area and 0.35 <= aspect_ratio <= 2.8:
            if w >= 36 and h >= 36:
                # Add 8% padding around the bounding box to prevent minutiae clipping
                pad_x = int(w * 0.08)
                pad_y = int(h * 0.08)
                x_pad = max(0, x - pad_x)
                y_pad = max(0, y - pad_y)
                w_pad = min(w_img - x_pad, w + 2 * pad_x)
                h_pad = min(h_img - y_pad, h + 2 * pad_y)
                candidate_boxes.append((x_pad, y_pad, w_pad, h_pad))

    # 5. Non-Maximum Suppression / Overlap Merging
    merged_boxes = []
    if candidate_boxes:
        # Sort by area descending
        candidate_boxes = sorted(candidate_boxes, key=lambda b: b[2] * b[3], reverse=True)
        for box in candidate_boxes:
            x1, y1, w1, h1 = box
            keep = True
            for m in merged_boxes:
                mx, my, mw, mh = m
                # Calculate Intersection over Union (IoU)
                ix = max(x1, mx)
                iy = max(y1, my)
                iw = max(0, min(x1 + w1, mx + mw) - ix)
                ih = max(0, min(y1 + h1, my + mh) - iy)
                intersection = iw * ih
                union = (w1 * h1) + (mw * mh) - intersection
                iou = float(intersection) / max(1, union)
                if iou > 0.30:  # significant overlap
                    keep = False
                    break
            if keep:
                merged_boxes.append(box)

    # 6. Fallback: If no discrete regions found, treat entire image as 1 fingerprint
    if not merged_boxes:
        merged_boxes = [(0, 0, w_img, h_img)]

    # 7. Sort bounding boxes spatially: primary by y (rows) and secondary by x (columns)
    # Group boxes that have roughly similar y-coordinates
    merged_boxes = sorted(merged_boxes, key=lambda b: (b[1] // 50, b[0]))

    # 8. Create visual annotated photograph
    annotated_np = img_np.copy()
    for idx, (x, y, w, h) in enumerate(merged_boxes, start=1):
        # Draw bounding rectangle (Emerald green)
        cv2.rectangle(annotated_np, (x, y), (x + w, y + h), (16, 185, 129), 3)

        # Draw header badge
        label = f"F{idx}"
        badge_w, badge_h = 45, 26
        cv2.rectangle(annotated_np, (x, max(0, y - badge_h)), (x + badge_w, y), (16, 185, 129), -1)
        cv2.putText(
            annotated_np,
            label,
            (x + 6, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    annotated_pil = Image.fromarray(annotated_np)
    return merged_boxes, annotated_pil


class MultiFingerprintAnalyzer:
    """
    End-to-end multi-fingerprint analysis engine wrapping:
      - Region detection & splitting
      - Individual crop quality assessment
      - ResNet-50 inference per crop
      - Raw probability vector averaging
      - Consistency & agreement scoring
    """

    def __init__(self, predictor: Optional[BloodGroupPredictor] = None):
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = BloodGroupPredictor()

    def analyze(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        return_gradcam: bool = True,
        force_mode: Optional[str] = None,  # 'single', 'multiple', or None (auto-detect)
    ) -> Dict[str, Any]:
        """
        Execute full multi-fingerprint analysis on the input image.

        Args:
            image_input: File path, PIL Image, or NumPy array
            return_gradcam: Whether to compute Grad-CAM for each crop
            force_mode: Optional override ('single' or 'multiple')

        Returns:
            Structured dictionary matching PRD Section 17 specifications.
        """
        # Load and validate source image
        pil_img = load_and_validate_image(image_input)
        w_img, h_img = pil_img.size

        # Detect fingerprint regions
        if force_mode == "single":
            bboxes = [(0, 0, w_img, h_img)]
            annotated_img = pil_img.copy()
        else:
            bboxes, annotated_img = detect_fingerprint_regions(pil_img)

        mode = "multiple" if len(bboxes) > 1 else "single"
        detected_count = len(bboxes)

        fingerprints_results = []
        usable_probs_list = []
        usable_predictions = []

        # Process each individual crop
        for idx, (x, y, w, h) in enumerate(bboxes, start=1):
            crop = pil_img.crop((x, y, x + w, y + h))

            # 1. Quality check on crop
            quality = check_image_quality(crop)

            # 2. Exclusion rule: if crop is severely defective, exclude from aggregation
            # Never silently exclude: record reason clearly
            included = True
            exclusion_reason = None

            if not quality["valid"]:
                included = False
                exclusion_reason = "Unusable image quality or extreme noise"
            elif quality["status"] == "Poor" and len(bboxes) > 1:
                # If we have multiple fingerprints, we exclude poor ones to improve accuracy
                included = False
                exclusion_reason = (
                    "; ".join(quality["issues"][:2]) if quality["issues"] else "Substandard ridge clarity"
                )

            # 3. Inference on crop
            crop_result = self.predictor.predict(crop, return_gradcam=return_gradcam)
            prediction = crop_result["predicted_class"]
            confidence = crop_result["confidence"]
            probabilities = crop_result["probabilities"]
            gradcam_overlay = crop_result.get("gradcam_overlay")

            fp_entry = {
                "id": idx,
                "bbox": [x, y, w, h],
                "crop": crop,
                "quality": quality,
                "included": included,
                "exclusion_reason": exclusion_reason,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "gradcam": gradcam_overlay,
            }
            fingerprints_results.append(fp_entry)

            if included and probabilities:
                # Raw probability values in [0.0, 1.0]
                raw_vector = [probabilities[cls] / 100.0 for cls in self.predictor.class_names]
                usable_probs_list.append(raw_vector)
                usable_predictions.append(prediction)

        # Fallback if all detected were excluded: include best-quality one
        if not usable_probs_list and fingerprints_results:
            best_fp = max(fingerprints_results, key=lambda f: f["quality"]["score"])
            best_fp["included"] = True
            best_fp["exclusion_reason"] = "Retained as single best available impression"
            raw_vector = [best_fp["probabilities"][cls] / 100.0 for cls in self.predictor.class_names]
            usable_probs_list.append(raw_vector)
            usable_predictions.append(best_fp["prediction"])

        analyzed_count = len(usable_probs_list)
        excluded_count = detected_count - analyzed_count

        # 4. Mandatory Aggregation Algorithm: Raw Probability Averaging
        # P_final(c) = (P1(c) + P2(c) + ... + PN(c)) / N
        probs_array = np.array(usable_probs_list, dtype=np.float64)  # Shape: (N, 8)
        avg_probs = np.mean(probs_array, axis=0)                     # Shape: (8,)

        # Guarantee exact normalization
        sum_p = np.sum(avg_probs)
        if sum_p > 0:
            avg_probs = avg_probs / sum_p

        # Final prediction = argmax(avg_probs)
        final_idx = int(np.argmax(avg_probs))
        final_prediction = self.predictor.class_names[final_idx]
        final_confidence = round(float(avg_probs[final_idx]) * 100.0, 2)

        # Final probability dictionary (percentages)
        final_probabilities = {
            cls: round(float(avg_probs[i]) * 100.0, 2)
            for i, cls in enumerate(self.predictor.class_names)
        }

        # 5. Agreement & Consistency Analysis
        supporting_count = sum(1 for pred in usable_predictions if pred == final_prediction)
        agreement_pct = (
            round((float(supporting_count) / max(1, analyzed_count)) * 100.0, 1)
            if analyzed_count > 0 else 100.0
        )

        return {
            "mode": mode,
            "detected_count": detected_count,
            "analyzed_count": analyzed_count,
            "excluded_count": excluded_count,
            "annotated_image": annotated_img,
            "fingerprints": fingerprints_results,
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "final_probabilities": final_probabilities,
            "agreement": {
                "supporting_count": supporting_count,
                "total_count": analyzed_count,
                "percentage": agreement_pct,
            },
        }
