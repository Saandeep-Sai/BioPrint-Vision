"""
BioPrint Vision — Image Preprocessing Module

Provides deterministic, reproducible preprocessing for both training and inference.
The exact same transforms (minus augmentation) are applied across train, test,
and inference pipelines to avoid train/test distribution shift.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Constants (ImageNet Standard) ─────────────────────────────────────────────
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Training transform with conservative augmentation safe for fingerprint images.
    Preserves dermatoglyphic ridge-valley flow without unrealistic distortions.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Validation / inference transform — deterministic resizing and normalization.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Standard alias
get_inference_transform = get_val_transform


def load_and_validate_image(
    path_or_pil: Union[str, Path, Image.Image, np.ndarray],
    convert_rgb: bool = True,
) -> Image.Image:
    """
    Safely load and validate an image from a file path, PIL Image, or NumPy array.
    Guarantees pixel data is fully loaded into memory and converted to RGB.

    Args:
        path_or_pil: File path (str or Path), PIL.Image.Image, or NumPy array
        convert_rgb: Whether to convert image to RGB (required for ResNet)

    Returns:
        PIL.Image.Image in RGB mode with dimensions verified

    Raises:
        ValueError: If the image cannot be read, is corrupted, or is undersized
    """
    if path_or_pil is None:
        raise ValueError("Image input is None.")

    # 1. Handle file path (str or Path)
    if isinstance(path_or_pil, (str, Path)):
        file_path = Path(path_or_pil)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        try:
            with Image.open(file_path) as raw:
                raw.load()
                img = raw.copy()
        except Exception as e:
            raise ValueError(f"Cannot read image file '{file_path}': {e}")

    # 2. Handle existing PIL Image
    elif isinstance(path_or_pil, Image.Image):
        try:
            path_or_pil.load()
            img = path_or_pil.copy()
        except Exception as e:
            raise ValueError(f"Corrupted or invalid PIL image: {e}")

    # 3. Handle NumPy ndarray
    elif isinstance(path_or_pil, np.ndarray):
        try:
            if path_or_pil.ndim < 2:
                raise ValueError("NumPy array must be at least 2-dimensional.")
            img = Image.fromarray(path_or_pil)
        except Exception as e:
            raise ValueError(f"Invalid NumPy array for image conversion: {e}")

    else:
        raise ValueError(f"Unsupported input type: {type(path_or_pil)}")

    # 4. Check minimum dimension constraints
    w, h = img.size
    if w < 16 or h < 16:
        raise ValueError(f"Image too small ({w}x{h}). Minimum supported size is 16x16.")

    # 5. Convert to RGB
    if convert_rgb and img.mode != "RGB":
        try:
            img = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to convert image to RGB: {e}")

    return img


def apply_clahe(
    pil_img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel
    in LAB color space to enhance ridge-valley contrast.

    Args:
        pil_img: RGB PIL Image
        clip_limit: CLAHE threshold for contrast limiting
        tile_grid_size: Grid size for histogram equalization

    Returns:
        Enhanced RGB PIL Image
    """
    img_np = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb_enhanced)
