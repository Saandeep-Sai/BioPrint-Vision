"""
BioPrint Vision — Interactive Gradio Web Application

Provides a clean, professional web interface for fingerprint-based
blood-group classification.

Features:
  - Fingerprint image upload (PNG, JPG, BMP)
  - 1-click test examples from verified dataset
  - Image quality assessment (blur, contrast, brightness, resolution)
  - 8-class probability distribution
  - Predicted blood group and model confidence
  - Grad-CAM spatial attention visualization
  - Explicit scientific and medical disclaimers

Launch:
    python app.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import BloodGroupPredictor

# ── Paths & Predictor Initialization ──────────────────────────────────────────
MODEL_PATH = Path("models/best_model.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")

predictor: Optional[BloodGroupPredictor] = None
init_error: Optional[str] = None

try:
    if MODEL_PATH.exists():
        predictor = BloodGroupPredictor(
            model_path=MODEL_PATH,
            class_names_path=CLASS_NAMES_PATH,
        )
    else:
        init_error = f"Checkpoint not found at '{MODEL_PATH.resolve()}'. Train or provide the model weights."
except Exception as e:
    init_error = f"Failed to load model: {e}"


# ── Prediction Handler ────────────────────────────────────────────────────────
def analyze_fingerprint(
    image: Optional[Image.Image],
) -> Tuple[str, Dict[str, float], Optional[Image.Image], str]:
    """
    Callback executed when the user clicks 'Analyze'.

    Returns:
        (prediction_summary_markdown, probabilities_dict, gradcam_image, quality_markdown)
    """
    if init_error is not None or predictor is None:
        error_msg = f"### ⚠️ System Unavailable\n\n{init_error}"
        return error_msg, {}, None, "Quality check unavailable."

    if image is None:
        warning_msg = (
            "### ⚠️ No Image Provided\n\n"
            "Please upload a fingerprint image (BMP, PNG, or JPEG) or select one of the examples below."
        )
        return warning_msg, {}, None, "Please upload an image to assess quality."

    start_time = time.perf_counter()

    # Execute inference pipeline
    result = predictor.predict(image, return_gradcam=True)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

    if result.get("error"):
        error_md = (
            f"### ⚠️ Validation Error\n\n"
            f"**Details:** {result['error']}\n\n"
            f"Please verify that the uploaded file is a legible, uncorrupted fingerprint image."
        )
        return error_md, {}, None, "Assessment failed due to invalid image."

    # Extract prediction details
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    quality = result.get("quality") or {}
    q_status = quality.get("status", "Unknown")
    q_score = quality.get("score", 0.0)
    q_issues = quality.get("issues", [])

    # Format Prediction Summary Markdown
    status_icon = "🟢" if q_status == "Good" else ("🟡" if q_status == "Fair" else "🔴")
    summary_md = (
        f"## 🩸 Predicted Blood Group: `{pred_class}`\n\n"
        f"**Model Confidence:** `{confidence:.1f}%`  *(Latency: {elapsed_ms} ms)*\n\n"
        f"**Image Quality:** {status_icon} `{q_status}` (Score: `{q_score:.2f}`)\n\n"
        f"> **Notice:** High model confidence reflects pattern activation in the trained network, "
        f"not clinical confirmation. Laboratory blood typing remains the gold standard."
    )

    # Format Quality Details Markdown
    if q_issues:
        issues_list = "\n".join([f"- ⚠️ {issue}" for issue in q_issues])
        quality_md = (
            f"### Quality Assessment: {q_status} (Score: {q_score:.2f})\n\n"
            f"**Detected Considerations:**\n{issues_list}\n\n"
            f"*Statistics:* Resolution `{quality.get('stats', {}).get('width')}x{quality.get('stats', {}).get('height')}` | "
            f"Sharpness `{quality.get('stats', {}).get('blur_laplacian_var', 0)}` | "
            f"Contrast `{quality.get('stats', {}).get('std_intensity', 0)}`"
        )
    else:
        quality_md = (
            f"### Quality Assessment: {q_status} (Score: {q_score:.2f})\n\n"
            f"All quality indicators (sharpness, contrast, exposure, resolution) are within acceptable thresholds."
        )

    # Format probabilities for Gradio Label component (values between 0.0 and 1.0)
    probabilities_label = {
        cls_name: prob / 100.0 for cls_name, prob in result["probabilities"].items()
    }

    gradcam_img = result.get("gradcam_overlay")

    return summary_md, probabilities_label, gradcam_img, quality_md


# ── UI Construction ───────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """Build and return the Gradio interface."""

    # Select available sample images for the Examples component
    sample_images = []
    sample_candidates = [
        "data/raw/A+/cluster_0_1001.BMP",
        "data/raw/A-/cluster_1_0.BMP",
        "data/raw/B+/cluster_2_10.BMP",
        "data/raw/B-/cluster_3_1017.BMP",
        "data/raw/AB+/cluster_4_100.BMP",
        "data/raw/AB-/cluster_5_1003.BMP",
        "data/raw/O+/cluster_6_1004.BMP",
        "data/raw/O-/cluster_7_1002.BMP",
    ]
    for p in sample_candidates:
        if Path(p).exists():
            sample_images.append([p])

    theme = gr.themes.Soft(
        primary_hue="rose",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="BioPrint Vision: Blood Group Prediction") as demo:
        # Title and Scientific Disclaimer Banner
        gr.Markdown(
            """
            # 🩸 BioPrint Vision (V1 Prototype)
            ### Deep Learning Classifier for Fingerprint-to-Blood-Group Prediction

            > ⚠️ **Scientific & Medical Disclaimer:** This software is an **experimental research prototype** 
            > developed for biometric pattern investigation. It is **NOT** a certified medical diagnostic device 
            > and does **NOT** replace serological laboratory blood typing. Do not make medical decisions based on this output.
            """
        )

        with gr.Row():
            # Left Column: Input and Controls
            with gr.Column(scale=5):
                image_input = gr.Image(
                    label="Fingerprint Input Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=320,
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔬 Analyze Fingerprint", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

                if sample_images:
                    gr.Examples(
                        examples=sample_images,
                        inputs=[image_input],
                        label="Sample Fingerprints (Verified Dataset)",
                        examples_per_page=4,
                    )

            # Right Column: Prediction Results & Probabilities
            with gr.Column(scale=6):
                result_display = gr.Markdown(
                    value="*Upload a fingerprint image and click **Analyze Fingerprint** to see predictions.*",
                    label="Prediction Status",
                )

                prob_display = gr.Label(
                    num_top_classes=8,
                    label="8-Class Probability Distribution",
                )

        # Explainability & Quality Row
        with gr.Row():
            with gr.Column(scale=5):
                gradcam_display = gr.Image(
                    label="Grad-CAM Saliency Map (Model Attention)",
                    type="pil",
                    height=300,
                )
                gr.Markdown(
                    """
                    <small><b>Grad-CAM Explainability:</b> Warmer regions (red/yellow) indicate spatial minutiae 
                    that contributed most strongly to the predicted class. Saliency maps illustrate neural activation 
                    patterns, not biological biomarkers.</small>
                    """
                )

            with gr.Column(scale=6):
                quality_display = gr.Markdown(
                    value="*Quality assessment details will appear after image analysis.*",
                    label="Image Quality Analysis",
                )

        # Technical Details & Held-out Benchmark Accordion
        with gr.Accordion("ℹ️ Model Architecture & Benchmark Metrics (Held-Out Test Set)", open=False):
            gr.Markdown(
                """
                | Metric | Benchmark Result | Notes |
                |---|---|---|
                | **Architecture** | ResNet-50 (Transfer Learning) | Backbone pre-trained on ImageNet-1K |
                | **Input Resolution** | 224 × 224 × 3 | ImageNet standard normalization |
                | **Test Set Accuracy** | **87.78%** | Evaluated on 900 held-out images |
                | **Top-2 Accuracy** | **98.00%** | Ground truth in top 2 predictions |
                | **Macro Precision** | **88.56%** | Balanced across all 8 blood groups |
                | **Macro Recall** | **87.71%** | Balanced across all 8 blood groups |
                | **Macro F1-Score** | **87.92%** | Harmonic mean of precision & recall |
                | **Target Classes** | A+, A-, AB+, AB-, B+, B-, O+, O- | Strict 8-class classification |
                | **Dataset** | 6,000 Verified BMP Images | 70% Train (4,199) / 15% Val (901) / 15% Test (900) |
                | **Known Limitation** | Subject IDs unavailable | Split performed by image; generalization across novel subjects is experimental |
                """
            )

        # Event Handlers
        analyze_btn.click(
            fn=analyze_fingerprint,
            inputs=[image_input],
            outputs=[result_display, prob_display, gradcam_display, quality_display],
        )

        def clear_outputs():
            return (
                None,
                "*Upload a fingerprint image and click **Analyze Fingerprint** to see predictions.*",
                {},
                None,
                "*Quality assessment details will appear after image analysis.*",
            )

        clear_btn.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[image_input, result_display, prob_display, gradcam_display, quality_display],
        )

    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(
        primary_hue="rose",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    app = build_ui()
    server_port = int(os.getenv("PORT", "7860"))
    app.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        theme=theme,
        share=False,
        show_error=True,
    )
