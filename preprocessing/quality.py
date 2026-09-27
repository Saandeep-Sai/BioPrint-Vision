"""
BioPrint Vision — Image Quality Assessment Module

Evaluates fingerprint image characteristics before inference:
  - Blur detection (Laplacian variance)
  - Contrast assessment (intensity standard deviation)
  - Brightness / exposure (mean intensity)
  - Dimensions / resolution
  - Dynamic range and background saturation

Returns a normalized quality score (0.0 to 1.0), status ('Good', 'Fair', 'Poor'),
and descriptive reasons. Note: this is a lightweight screening check and does not
claim to be a certified biometric standard.
"""

import cv2
import numpy as np
from PIL import Image


def check_image_quality(pil_img: Image.Image) -> dict:
    """
    Run comprehensive lightweight quality checks on a fingerprint image.

    Args:
        pil_img: PIL Image object

    Returns:
        dict containing:
            - 'score': float (0.00 to 1.00)
            - 'status': str ('Good', 'Fair', or 'Poor')
            - 'summary': str brief human-readable summary
            - 'valid': bool (False only if critically unusable)
            - 'issues': list of str describing detected problems
            - 'stats': dict of raw computed metrics
    """
    issues = []
    w, h = pil_img.size

    # Convert to grayscale numpy array for analysis
    img_gray = np.array(pil_img.convert("L"))

    mean_val = float(np.mean(img_gray))
    std_val = float(np.std(img_gray))
    min_val = int(np.min(img_gray))
    max_val = int(np.max(img_gray))
    dynamic_range = max_val - min_val

    # 1. Blur assessment via Laplacian variance
    # Low variance indicates low high-frequency content (blur)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Extreme pixel saturation (pure white or pure black background percentage)
    black_pixels = float(np.sum(img_gray <= 5)) / (w * h)
    white_pixels = float(np.sum(img_gray >= 250)) / (w * h)

    stats = {
        "width": w,
        "height": h,
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "min_intensity": min_val,
        "max_intensity": max_val,
        "dynamic_range": dynamic_range,
        "blur_laplacian_var": round(blur_score, 2),
        "black_ratio": round(black_pixels, 4),
        "white_ratio": round(white_pixels, 4),
    }

    # Deductions scoring (starts at 1.0)
    score = 1.0

    # Dimension checks
    if w < 32 or h < 32:
        issues.append(f"Image resolution too small ({w}x{h}). Minimum 32x32 required.")
        score -= 0.50
    elif w < 64 or h < 64:
        issues.append(f"Low image resolution ({w}x{h}). Recommended at least 96x96.")
        score -= 0.15

    # Contrast checks
    if std_val < 8.0:
        issues.append(f"Very low contrast (std={std_val:.1f}). Ridges may be imperceptible.")
        score -= 0.40
    elif std_val < 18.0:
        issues.append(f"Low contrast (std={std_val:.1f}). Ridge clarity is diminished.")
        score -= 0.15

    # Brightness / Exposure checks
    if mean_val < 15:
        issues.append("Image is severely underexposed (nearly solid black).")
        score -= 0.45
    elif mean_val < 40:
        issues.append("Image is dark / underexposed.")
        score -= 0.10
    elif mean_val > 240:
        issues.append("Image is severely overexposed (nearly solid white).")
        score -= 0.45
    elif mean_val > 220:
        issues.append("Image is washed out / overexposed.")
        score -= 0.10

    # Blur checks
    if blur_score < 15.0:
        issues.append(f"Heavy blur detected (sharpness variance={blur_score:.1f}).")
        score -= 0.35
    elif blur_score < 45.0:
        issues.append(f"Moderate blur detected (sharpness variance={blur_score:.1f}).")
        score -= 0.15

    # Dynamic range checks
    if dynamic_range < 25:
        issues.append(f"Severely restricted dynamic range ({dynamic_range}/255).")
        score -= 0.30
    elif dynamic_range < 60:
        issues.append(f"Narrow dynamic range ({dynamic_range}/255).")
        score -= 0.10

    # Extreme saturation checks
    if black_pixels > 0.85 or white_pixels > 0.85:
        issues.append("Excessive background saturation (>85% uniform pixels).")
        score -= 0.30

    # Clamp score to [0.0, 1.0]
    final_score = round(max(0.0, min(1.0, score)), 2)

    # Determine status
    if final_score >= 0.70:
        status = "Good"
    elif final_score >= 0.40:
        status = "Fair"
    else:
        status = "Poor"

    # Critical unsuitability (e.g. completely solid or <16px)
    valid = not (w < 16 or h < 16 or dynamic_range < 10 or (mean_val < 5 and std_val < 2))

    if status == "Good":
        summary = f"Score: {final_score:.2f} | Status: Good"
    elif status == "Fair":
        reasons = "; ".join(issues[:2]) if issues else "Suboptimal contrast or sharpness"
        summary = f"Score: {final_score:.2f} | Status: Fair ({reasons})"
    else:
        reasons = "; ".join(issues[:2]) if issues else "Low quality input"
        summary = f"Score: {final_score:.2f} | Status: Poor ({reasons})"

    return {
        "score": final_score,
        "status": status,
        "summary": summary,
        "valid": valid,
        "issues": issues,
        "stats": stats,
    }
