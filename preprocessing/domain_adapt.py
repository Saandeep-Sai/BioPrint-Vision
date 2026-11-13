"""
BioPrint Vision - Domain Adaptation with Multi-Strategy Ensemble

Bridges the gap between:
  - Training data: tiny 96x103 BMP, grayscale (R=G=B), bimodal distribution
  - Real ink prints: larger JPEG/PNG, COLORED ink (blue/black), on white paper
  - Scanner captures: grayscale, sensor noise, different contrast

KEY INSIGHT: No single preprocessing strategy works for ALL real-world images.
Different ink colors, paper textures, lighting conditions, and camera angles
each need different handling. The solution is to run MULTIPLE preprocessing
strategies and ensemble their predictions.

Validated approach:
  1. Run 3 complementary preprocessing strategies on each external image
  2. Each strategy produces a softmax probability vector
  3. Average all vectors -> final prediction = argmax(average)
  4. This is robust because different strategies fail in different ways,
     but the correct class tends to appear with moderate probability
     across most strategies even when it isn't the top-1 for any single one.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Optional, List, Tuple


# Pre-computed training data reference distribution
_TRAINING_MEAN = 146.32
_TRAINING_STD = 104.11


def _compute_reference_histogram(data_dir: str = "data/raw") -> Optional[np.ndarray]:
    """
    Compute the average grayscale histogram from training images.
    Returns a 256-bin normalized histogram (PDF) of training data.
    Cached after first computation.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return None

    hist_accum = np.zeros(256, dtype=np.float64)
    count = 0

    for cls_dir in sorted(data_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.iterdir())
        # Sample every 5th file for speed
        for f in files[::5]:
            if f.suffix.upper() in ('.BMP', '.PNG', '.JPG', '.JPEG'):
                try:
                    img = Image.open(f).convert('L')
                    arr = np.array(img)
                    h, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
                    hist_accum += h.astype(np.float64)
                    count += 1
                except Exception:
                    pass

    if count == 0:
        return None

    hist_accum /= hist_accum.sum()
    return hist_accum


# Lazily cached reference histogram
_ref_histogram = None


def _get_reference_histogram():
    global _ref_histogram
    if _ref_histogram is None:
        _ref_histogram = _compute_reference_histogram()
    return _ref_histogram


def histogram_match_to_training(gray: np.ndarray) -> np.ndarray:
    """
    Match the histogram of a grayscale image to the training data distribution
    using CDF-based histogram specification.
    """
    ref_hist = _get_reference_histogram()
    if ref_hist is None:
        return _normalize_intensity(gray)

    src_hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    src_pdf = src_hist.astype(np.float64) / max(src_hist.sum(), 1)
    src_cdf = np.cumsum(src_pdf)
    ref_cdf = np.cumsum(ref_hist)

    lut = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lut[src_val] = np.argmin(diff)

    return lut[gray]


def _normalize_intensity(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile-based intensity normalization as fallback."""
    p_low = np.percentile(gray, low_pct)
    p_high = np.percentile(gray, high_pct)
    if p_high - p_low < 10:
        return gray
    clipped = np.clip(gray, p_low, p_high)
    normalized = ((clipped - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    return normalized


# ============================================================================
# PREPROCESSING STRATEGIES
# Each takes a PIL image and returns a preprocessed PIL image in RGB mode.
# ============================================================================

def _strategy_gabor_hm(pil_img: Image.Image) -> Image.Image:
    """
    Strategy 1: Gabor ridge enhancement + histogram matching.
    Best for high-contrast ink prints where ridge structure is clear.
    """
    gray = np.array(pil_img.convert('L'), dtype=np.uint8)

    gabor_accum = np.zeros_like(gray, dtype=np.float64)
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(
            (21, 21), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_64F
        )
        filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kern)
        gabor_accum += np.abs(filtered)

    if gabor_accum.max() > 0:
        gabor_norm = ((gabor_accum / gabor_accum.max()) * 255).astype(np.uint8)
    else:
        gabor_norm = gray

    matched = histogram_match_to_training(gabor_norm)
    rgb = cv2.cvtColor(matched, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _strategy_clahe_hm(pil_img: Image.Image) -> Image.Image:
    """
    Strategy 2: Bilateral denoise + CLAHE + histogram matching.
    Better for faint or low-contrast prints, scanner-like inputs.
    """
    gray = np.array(pil_img.convert('L'), dtype=np.uint8)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    matched = histogram_match_to_training(gray)
    rgb = cv2.cvtColor(matched, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _strategy_adaptive_thresh_hm(pil_img: Image.Image) -> Image.Image:
    """
    Strategy 3: Adaptive thresholding + histogram matching.
    Forces a bimodal distribution similar to training data.
    Good for very noisy or low-quality camera photos.
    """
    gray = np.array(pil_img.convert('L'), dtype=np.uint8)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive threshold creates a clean binary-like ridge map
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=10
    )

    matched = histogram_match_to_training(thresh)
    rgb = cv2.cvtColor(matched, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _strategy_gabor_large(pil_img: Image.Image) -> Image.Image:
    """
    Strategy 4: Gabor with larger kernel + histogram matching.
    Captures coarser ridge patterns, complements strategy 1.
    """
    gray = np.array(pil_img.convert('L'), dtype=np.uint8)

    gabor_accum = np.zeros_like(gray, dtype=np.float64)
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(
            (31, 31), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_64F
        )
        filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kern)
        gabor_accum += np.abs(filtered)

    if gabor_accum.max() > 0:
        gabor_norm = ((gabor_accum / gabor_accum.max()) * 255).astype(np.uint8)
    else:
        gabor_norm = gray

    matched = histogram_match_to_training(gabor_norm)
    rgb = cv2.cvtColor(matched, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def _strategy_invert_gabor_hm(pil_img: Image.Image) -> Image.Image:
    """
    Strategy 5: Invert + Gabor + histogram matching.
    Handles cases where ink polarity is inverted (light ridges on dark).
    """
    gray = np.array(pil_img.convert('L'), dtype=np.uint8)
    gray = 255 - gray  # Invert

    gabor_accum = np.zeros_like(gray, dtype=np.float64)
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(
            (21, 21), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_64F
        )
        filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kern)
        gabor_accum += np.abs(filtered)

    if gabor_accum.max() > 0:
        gabor_norm = ((gabor_accum / gabor_accum.max()) * 255).astype(np.uint8)
    else:
        gabor_norm = gray

    matched = histogram_match_to_training(gabor_norm)
    rgb = cv2.cvtColor(matched, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


# All available strategies for ensemble inference
ALL_STRATEGIES = [
    ("gabor_hm", _strategy_gabor_hm),
    ("clahe_hm", _strategy_clahe_hm),
    ("adaptive_thresh_hm", _strategy_adaptive_thresh_hm),
    ("gabor_large", _strategy_gabor_large),
    ("invert_gabor_hm", _strategy_invert_gabor_hm),
]


def get_ensemble_preprocessed(
    pil_img: Image.Image,
    strategies: Optional[List[str]] = None,
) -> List[Tuple[str, Image.Image]]:
    """
    Apply multiple preprocessing strategies to an external image.

    Args:
        pil_img: Input PIL image (any mode/size)
        strategies: List of strategy names to use (None = all)

    Returns:
        List of (strategy_name, preprocessed_pil_image) tuples
    """
    results = []
    for name, fn in ALL_STRATEGIES:
        if strategies and name not in strategies:
            continue
        try:
            processed = fn(pil_img)
            results.append((name, processed))
        except Exception:
            pass  # Skip failed strategies
    return results


def adapt_for_inference(
    pil_img: Image.Image,
    use_gabor: bool = True,
    histogram_match: bool = True,
) -> Image.Image:
    """
    Single-strategy domain adaptation (backward compatible).
    For ensemble inference, use get_ensemble_preprocessed() instead.
    """
    if use_gabor:
        return _strategy_gabor_hm(pil_img)
    else:
        return _strategy_clahe_hm(pil_img)


def is_scanner_image(pil_img: Image.Image) -> bool:
    """
    Heuristic to detect whether an image comes from a scanner/external
    source rather than the training dataset.

    Returns True for ANY image that doesn't match training data characteristics.
    """
    w, h = pil_img.size

    # Training images are exactly 96x103 (99.3%) or 241x298 (0.7%)
    is_training_res = (w, h) in [(96, 103), (241, 298)]

    if not is_training_res:
        return True

    # Check if channels are identical (training data is strict R=G=B)
    if pil_img.mode in ("RGBA", "RGB"):
        arr = np.array(pil_img)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            if not (np.array_equal(r, g) and np.array_equal(g, b)):
                return True

    return False


def adapt_if_needed(pil_img: Image.Image, force: bool = False) -> Image.Image:
    """
    Conditionally apply domain adaptation.
    For external images: applies Gabor ridge enhancement + histogram matching.
    For training-like images: passes through unchanged.
    """
    if force or is_scanner_image(pil_img):
        return adapt_for_inference(pil_img, use_gabor=True, histogram_match=True)
    return pil_img
