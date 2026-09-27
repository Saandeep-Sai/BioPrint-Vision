# 🩸 BioPrint Vision (V1 Prototype)

> **Experimental AI-Based Blood-Group Classification from Fingerprint Images**  
> *Research & Screening Prototype — Not for Clinical Diagnostic Use.*

---

## 📌 Project Overview

**BioPrint Vision** is an end-to-end deep learning prototype designed to investigate correlation patterns between fingerprint minutiae and human ABO/Rh blood groups. The system accepts a digital fingerprint image, runs automated quality assessment, applies standardized preprocessing, executes an ImageNet-pretrained **ResNet-50** deep classifier, and delivers an 8-class probability distribution accompanied by **Grad-CAM** saliency heatmaps via a clean, interactive **Gradio** web application.

```
Fingerprint Image (BMP / PNG / JPEG)
        │
        ▼
Validation & Quality Assessment (Sharpness, Contrast, Exposure, Resolution)
        │
        ▼
Preprocessing Pipeline (Resize 224×224, Tensor Conversion, ImageNet Normalization)
        │
        ▼
Deep Convolutional Classifier (ResNet-50 Backbone + Custom MLP Head)
        │
        ▼
Softmax Probability Distribution (8 Blood Groups: A+, A-, AB+, AB-, B+, B-, O+, O-)
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
Predicted Blood Group        Model Confidence (%)          Grad-CAM Saliency Map
        │                             │                             │
        └─────────────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
                        Gradio Web Interface (Localhost)
```

---

## ⚠️ Scientific & Medical Disclaimer

> **IMPORTANT:** BioPrint Vision is an **experimental research prototype**. 
> - It is **NOT** a certified medical diagnostic device.
> - It does **NOT** diagnose, confirm, or replace standard serological blood-group testing.
> - High classification accuracy on public datasets does not prove clinical validity or biological causation.
> - Grad-CAM heatmaps highlight spatial regions of neural network activation, **not** verified medical biomarkers.
> - **Never base transfusion, donation, or clinical decisions on this software.**

---

## 📊 Dataset Audit & Provenance

The model was trained and evaluated on a verified local dataset of **6,000 raw fingerprint images**:

* **Total Images:** 6,000 bitmap (`.BMP`) files
* **Target Classes (8):** `A+`, `A-`, `AB+`, `AB-`, `B+`, `B-`, `O+`, `O-`
* **Integrity:** 0 corrupted or unreadable images
* **Resolution Profile:** 5,956 images at 96×103 pixels, 44 images at 241×298 pixels
* **Duplicate Check:** 15 identical files detected across 11 groups (verified via MD5 hashing)

### Class Distribution

| Blood Group | Sample Count | Percentage |
|:---:|:---:|:---:|
| **A-** | 1,009 | 16.82% |
| **O+** | 852 | 14.20% |
| **AB-** | 761 | 12.68% |
| **B-** | 741 | 12.35% |
| **O-** | 712 | 11.87% |
| **AB+** | 708 | 11.80% |
| **B+** | 652 | 10.87% |
| **A+** | 565 | 9.42% |
| **Total** | **6,000** | **100.0%** |

### ⚠️ Data Leakage & Subject ID Limitation
The dataset filenames follow a cluster-indexing format (`cluster_X_YYYY.BMP`). **Subject / individual identifiers are NOT available** in this dataset. As a result, the train/validation/test split was performed in a **stratified random manner by image**, not by subject. It is therefore possible that multiple fingerprint impressions from the same individual exist across both the training and test sets. **True subject-independent generalization remains unverified and is documented as a primary limitation.**

---

## 🔬 Benchmark Evaluation Results (Held-Out Test Set)

The model was trained for 25 epochs using AdamW (`lr=1e-4`), CrossEntropyLoss with class weights, and early stopping. Evaluation was conducted on a **strictly held-out test set of 900 samples (15% split)**:

### Overall Metrics

| Metric | Result | Description |
|---|:---:|---|
| **Test Accuracy** | **87.78%** | 790 of 900 test samples correctly classified |
| **Top-2 Accuracy** | **98.00%** | Ground truth label is within the top 2 predicted classes |
| **Macro Precision** | **88.56%** | Unweighted average precision across all 8 classes |
| **Macro Recall** | **87.71%** | Unweighted average recall across all 8 classes |
| **Macro F1-Score** | **87.92%** | Harmonic mean of macro precision and recall |
| **Weighted F1-Score**| **87.79%** | F1-score weighted by class support |
| **Mean Confidence** | **91.83%** | Average softmax confidence across all test predictions |
| **Median Confidence**| **99.32%** | Demonstrates sharp confidence distribution |
| **Misclassified Count** | **110** | 110 of 900 test samples |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|:---:|:---:|:---:|:---:|:---:|
| **A+**  | 97.26% | 83.53% | **89.87%** | 85 |
| **A-**  | 85.35% | 88.74% | **87.01%** | 151 |
| **AB+** | 86.61% | 91.51% | **88.99%** | 106 |
| **AB-** | 94.79% | 79.82% | **86.67%** | 114 |
| **B+**  | 85.98% | 93.88% | **89.76%** | 98 |
| **B-**  | 92.31% | 86.49% | **89.30%** | 111 |
| **O+**  | 85.19% | 89.84% | **87.45%** | 128 |
| **O-**  | 81.03% | 87.85% | **84.30%** | 107 |

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.11.9)
* Git

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/BioPrint-Vision.git
cd BioPrint-Vision

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: For systems without a dedicated NVIDIA GPU, PyTorch installs with CPU support by default. If a CUDA GPU is present, it will automatically be utilized for inference.*

---

## 🚀 Running the Prototype

### 1. Launch the Gradio Web Application (Recommended)

```bash
python app.py
```
Open your browser at **`http://127.0.0.1:7860`**.

* Upload any fingerprint image (PNG, JPG, or BMP).
* Alternatively, click any of the **Sample Fingerprints** in the interface to test instantly.
* Click **🔬 Analyze Fingerprint** to inspect:
  * Predicted Blood Group
  * Model Confidence percentage
  * Image Quality rating (Score, Status, and detected issues)
  * Full 8-Class Probability Distribution
  * Grad-CAM Saliency Heatmap

### 2. Run CLI Inference on a Single Image

```bash
python inference.py --image data/raw/A+/cluster_0_1001.BMP
```

### 3. Run the Automated Verification Test Suite

```bash
python test_pipeline.py
```

### 4. Run Dataset Audit (Optional)

```bash
python -m training.audit_dataset --data-dir data/raw --output dataset_audit.json
```

---

## 🧠 Model Architecture & Methodology

```
Input Tensor: [Batch, 3, 224, 224]
  │
  ├── ResNet-50 Feature Backbone (ImageNet Pretrained)
  │     ├── conv1, bn1, layer1, layer2 (Frozen during fine-tuning)
  │     └── layer3, layer4             (Fine-tuned)
  │
  ├── AdaptiveAvgPool2d (Output: 2048-dim embedding)
  │
  └── Multi-Stage Classification Head (nn.Sequential)
        ├── Linear(2048 → 1024) → BatchNorm1d → ReLU → Dropout(0.4)
        ├── Linear(1024 → 512)  → ReLU → Dropout(0.2)
        ├── Linear(512 → 128)   → ReLU → Dropout(0.1)
        ├── Linear(128 → 64)    → ReLU
        └── Linear(64 → 8)      → Logits [A+, A-, AB+, AB-, B+, B-, O+, O-]
```

* **Exact Contract Order:** `['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']`
* **Target Layer for Grad-CAM:** `model.layer4[-1]` (last bottleneck residual block)

---

## 📁 Repository Structure

```
BioPrint-Vision/
├── app.py                         # Interactive Gradio application
├── inference.py                   # Standalone inference module & CLI
├── test_pipeline.py               # Comprehensive verification test suite
├── requirements.txt               # Minimal pinned dependencies
├── dataset_audit.json             # Verified dataset audit artifact
├── README.md                      # Complete project documentation
├── .gitignore                     # Git ignore rules
│
├── data/
│   └── raw/                       # 6,000 raw BMP fingerprint images
│       ├── A+/ ... O-/            # 8 class directories
│
├── models/
│   ├── best_model.pth             # Trained ResNet-50 checkpoint (Epoch 25)
│   ├── class_names.json           # Ordered 8-class mapping contract
│   ├── split_indices.json         # Exact train/val/test indices (reproducibility)
│   └── training_log.json          # Complete 25-epoch training loss & accuracy history
│
├── preprocessing/
│   ├── __init__.py
│   ├── image_preprocess.py        # Shared deterministic transform & CLAHE
│   └── quality.py                 # Image quality assessment (blur, contrast, etc.)
│
├── training/
│   ├── __init__.py
│   ├── audit_dataset.py           # Dataset integrity & duplicate audit script
│   ├── train.py                   # ResNet-50 fine-tuning pipeline
│   └── evaluate.py                # Test set evaluation & confusion matrix generator
│
├── explainability/
│   ├── __init__.py
│   └── gradcam.py                 # Grad-CAM saliency generator & overlay blending
│
└── evaluation/
    ├── metrics.json               # Test metrics (Accuracy: 87.78%, Top-2: 98.00%)
    ├── confusion_matrix.png       # 8x8 Confusion matrix heatmap
    ├── classification_report.txt  # Detailed per-class precision/recall/F1 table
    └── misclassified_samples.json # Record of 110 misclassified test samples
```

---

## ⚖️ License & Ethical Use

This project is released under the **MIT License**. It is strictly intended for scientific investigation, biometric research, and computer vision education. Commercial medical claims or clinical deployment without rigorous independent prospective validation and regulatory approval are prohibited.
