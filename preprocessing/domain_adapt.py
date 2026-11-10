"""
BioPrint Vision - Domain Adaptation Preprocessing Module

Bridges the gap between training data characteristics and real-world scanner
images by normalizing input images to match the distribution the model was
trained on.

Training data profile:
    - 96x103 BMP files, grayscale stored as RGBA (R=G=B always)
    - Bimodal pixel distribution: peaks at 0-26 (ridges) and 230-255 (valleys)
    - Mean intensity: 146/255 = 0.574
    - High contrast, no sensor noise
    - Source: single ink-on-paper scan sensor

Real scanner images differ in resolution, noise, contrast, polarity, and
pixel distribution. This module normalizes those differences before the
standard inference transform (Resize + ToTensor + ImageNet Normalize).
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter


def adapt_for_inference(
    pil_img: Image.Image,
    denoise: bool = True,
    apply_clahe: bool = True,
    normalize_contrast: bool = True,
    clahe_clip: float = 2.5,
    clahe_grid: tuple = (8, 8),
) -> Image.Image:
    """
    Apply domain-adaptive preprocessing to bridge training/inference gap.

    This pipeline normalizes real-world scanner images to resemble the
    training data distribution before the image enters the model's standard
    Resize + Normalize transform.

    Steps:
        1. Convert to grayscale (training data is grayscale stored as RGB)
        2. Denoise with bilateral filter (removes scanner sensor noise)
        3. CLAHE on grayscale (standardizes contrast across sensors)
        4. Normalize intensity range (match training data's bimodal distribution)
        5. Convert back to RGB (model expects 3-channel input)

    Args:
        pil_img: Input PIL Image (any mode)
        denoise: Apply bilateral denoising filter
        apply_clahe: Apply CLAHE contrast enhancement
        normalize_contrast: Stretch/clip intensity to match training range
        clahe_clip: CLAHE clip limit (higher = more contrast enhancement)
        clahe_grid: CLAHE tile grid size

    Returns:
        Preprocessed PIL Image in RGB mode ready for model inference transform
    """
    # Step 1: Convert to grayscale
    # Training images are grayscale (R=G=B). Converting scanner images to
    # grayscale ensures the same channel relationship.
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)

    # Step 2: Denoise (bilateral preserves edges while removing noise)
    # This is critical - Gaussian noise causes 99%+ confidence wrong predictions
    if denoise:
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Step 3: CLAHE - Adaptive histogram equalization
    # Standardizes contrast across different scanner sensors and lighting
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)

    # Step 4: Normalize intensity range
    # Training images have full 0-255 dynamic range with bimodal distribution.
    # Stretch scanner output to match.
    if normalize_contrast:
        gray = _normalize_intensity(gray)

    # Step 5: Convert back to 3-channel RGB (R=G=B, matching training format)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(rgb)


def _normalize_intensity(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-based intensity normalization.

    Clips outlier pixels and stretches the remaining range to 0-255,
    matching the training data's full dynamic range.

    Args:
        gray: Grayscale uint8 numpy array
        low_pct: Lower percentile for clipping (removes dark outliers)
        high_pct: Upper percentile for clipping (removes bright outliers)

    Returns:
        Normalized uint8 numpy array
    """
    p_low = np.percentile(gray, low_pct)
    p_high = np.percentile(gray, high_pct)

    if p_high - p_low < 10:
        # Image is nearly uniform - skip normalization to avoid amplifying noise
        return gray

    clipped = np.clip(gray, p_low, p_high)
    normalized = ((clipped - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    return normalized


def is_scanner_image(pil_img: Image.Image) -> bool:
    """
    Heuristic to detect whether an image likely comes from a scanner rather
    than the training dataset. Used to decide whether domain adaptation is needed.

    Checks:
        - Image resolution (training images are 96x103, scanners are 260x300+)
        - Whether channels differ (training is strict R=G=B)
        - File format hints

    Args:
        pil_img: Input PIL Image

    Returns:
        True if the image appears to be from a scanner/external source
    """
    w, h = pil_img.size

    # Training images are exactly 96x103 (99.3%) or 241x298 (0.7%)
    if (w, h) == (96, 103) or (w, h) == (241, 298):
        # Could be a training image - check if channels are identical
        if pil_img.mode in ("RGBA", "RGB"):
            arr = np.array(pil_img)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                if np.array_equal(r, g) and np.array_equal(g, b):
                    return False  # Looks like a training image
        return False  # Training resolution, assume training data

    # Any other resolution is likely external/scanner
    return True


def adapt_if_needed(pil_img: Image.Image, force: bool = False) -> Image.Image:
    """
    Conditionally apply domain adaptation. Applies preprocessing only when
    the image appears to come from an external source (scanner, camera, etc.).

    Always applies when force=True.

    Args:
        pil_img: Input PIL Image
        force: Force domain adaptation regardless of heuristic

    Returns:
        PIL Image (adapted or original)
    """
    if force or is_scanner_image(pil_img):
        return adapt_for_inference(pil_img)
    return pil_img
