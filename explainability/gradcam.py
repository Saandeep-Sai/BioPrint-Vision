"""
BioPrint Vision — Grad-CAM Explainability Module

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for
ResNet-50 to visualize which regions of the fingerprint the model focuses on.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

NOTE:
    Grad-CAM is an explainability / saliency visualization tool indicating
    model attention. It does NOT establish biological causation or prove
    that highlighted minutiae are clinical blood-group biomarkers.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    Grad-CAM visualization for convolutional neural networks.

    Hooks into the final residual block (e.g. model.layer4[-1]) to capture
    forward feature activations and backward gradients, producing a
    class-discriminative spatial attention heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate a normalized 2D Grad-CAM heatmap for a given input and class.

        Args:
            input_tensor: 4D preprocessed tensor [1, C, H, W]
            class_idx: target class index for the saliency map

        Returns:
            cam: 2D numpy array [H, W] normalized to [0.0, 1.0]
        """
        self.model.zero_grad()

        # Detach and enable gradient tracking on input
        tensor = input_tensor.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            output = self.model(tensor)
            if class_idx < 0 or class_idx >= output.shape[1]:
                raise ValueError(f"Class index {class_idx} out of bounds for output shape {output.shape}")
            score = output[0, class_idx]
            score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM hook failed: activations or gradients were not captured. "
                "Ensure the target layer is part of the computation graph."
            )

        # Global average pooling of gradients -> channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of forward activation maps
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]  # [H, W]
        cam_np = cam.cpu().numpy().astype(np.float32)

        # Min-max normalization
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    def close(self):
        """Remove hooks to prevent memory leaks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def create_gradcam_overlay(
    original_pil: Image.Image,
    cam: np.ndarray,
    target_size: tuple = (224, 224),
    alpha: float = 0.5,
) -> Image.Image:
    """
    Generate an RGB image overlay of the Grad-CAM heatmap blended with
    the original fingerprint image.

    Args:
        original_pil: original PIL Image (any mode)
        cam: normalized 2D numpy array [H, W] in [0.0, 1.0]
        target_size: (width, height) to resize both image and heatmap
        alpha: blending weight for heatmap (0.0 = only image, 1.0 = only heatmap)

    Returns:
        PIL Image in RGB mode with the blended Grad-CAM heatmap
    """
    # Ensure original image is RGB and resized
    img_rgb = original_pil.convert("RGB").resize(target_size)
    img_np = np.array(img_rgb, dtype=np.uint8)

    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))

    # Apply JET colormap (blue = low attention, red = high attention)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blend
    overlay = np.clip(
        (1.0 - alpha) * img_np.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32),
        0.0,
        255.0,
    ).astype(np.uint8)

    return Image.fromarray(overlay)
