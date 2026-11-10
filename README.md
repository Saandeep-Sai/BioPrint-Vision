# 🩸 BioPrint Vision (V2 Healthcare & Biometric Platform)

> **AI-Powered Fingerprint-to-Blood-Group Classification & Clinical Decision Support System**  
> *Research & Biometric Screening Platform : Not for Clinical Diagnostic Use.*

---

## 📌 Platform Overview

**BioPrint Vision** is an end-to-end computer vision and healthcare support platform that predicts human ABO/Rh blood groups from fingerprint impressions. The platform features two coexisting interfaces powered by the same verified **ResNet-50** deep convolutional checkpoint:

1. **Gradio Standalone Prototype (`app.py` on `http://127.0.0.1:7860`)**: Single-image research interface with instant quality scoring, 8-class softmax distributions, and Grad-CAM interpretability heatmaps.
2. **Flask Multi-User Web Platform (`run_flask.py` on `http://127.0.0.1:5000`)**: Complete healthcare web application featuring automated multi-fingerprint sheet detection, probability vector averaging, intra-sheet agreement scoring, ABO/Rh transfusion compatibility rules, donor pre-screening questionnaires, synthetic hospital/blood bank inventory search, hospital management portal, admin suite, contextual AI assistant, and ReportLab PDF report generation.

---

## 🏗️ Architecture & Processing Pipeline

```
Photograph (Single Fingerprint or Multi-Fingerprint Card)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Classical CV Region Detection & Segmentation                │
│ (Grayscale → CLAHE → Otsu Threshold → Morphology → NMS)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Single Fingerprint          Multiple Fingerprints (2+)
    (Direct Isolation)          (Bounding Box Crops)
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Fingerprint Quality Assessment                              │
│ (Laplacian Variance, Contrast, Brightness, Dynamic Range)   │
│ Threshold: Score >= 0.35 required for aggregation           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ResNet-50 Inference per Crop (TorchScript / PyTorch)        │
│ Layer4 Grad-CAM Activation Heatmaps                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Mathematical Aggregation (Raw Softmax Probability Averaging)│
│                                                             │
│   P_final = (1 / N) * SUM_{i=1}^N P_i                       │
│   Final Class = argmax(P_final)                             │
│   Agreement % = (max_count / N) * 100                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   Flask Web App     PDF Report Generator   Gradio Prototype
  (Port 5000)          (ReportLab)            (Port 7860)
```

---

## ⚠️ Scientific & Medical Disclaimer

> **IMPORTANT:** BioPrint Vision is an **experimental research prototype**. 
> - It is **NOT** a certified medical diagnostic device.
> - It does **NOT** diagnose, confirm, or replace standard serological agglutination blood tests.
> - Classification performance reflects statistical correlation on biometric datasets, **not** verified medical causation.
> - Grad-CAM heatmaps highlight spatial regions of neural network activation, **not** certified biological biomarkers.
> - **Hospital inventory data is synthetic demonstration data (`is_demo=True`)** and is not connected to live medical inventory.
> - **Never base transfusion, surgery, or clinical decisions on this software.**

---

## 🔬 Deep Learning Model & Benchmark Performance

* **Architecture:** ResNet-50 Backbone with custom dense classification head (2048 → 512 → Dropout(0.4) → 8 classes)
* **Checkpoint:** `models/best_model.pth` (Epoch 25, Early Stopping)
* **Exact Class Contract:** `['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']`
* **Dataset Size:** 6,000 raw fingerprint impressions

### Test Set Benchmark (900 Held-Out Samples)

| Metric | Score | Description |
|---|:---:|---|
| **Test Accuracy** | **87.78%** | 790 of 900 test samples correctly classified |
| **Top-2 Accuracy** | **98.00%** | True blood group is within the top 2 model predictions |
| **Macro Precision** | **88.56%** | Balanced precision across all 8 classes |
| **Macro Recall** | **87.71%** | Balanced recall across all 8 classes |
| **Macro F1-Score** | **87.92%** | Harmonic mean of macro precision and recall |
| **Weighted F1-Score**| **87.79%** | Support-weighted F1 across classes |
| **Validation Accuracy**| **91.34%** | Best validation accuracy at Epoch 25 |

---

## 🚀 Quickstart & How to Run

### 1. Prerequisites & Environment Setup

Ensure the project virtual environment is active:

```powershell
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

Or execute commands directly via the virtual environment interpreter:
`d:\BioPrint-Vision\.venv\Scripts\python.exe`

### 2. Run the Gradio Research Prototype (Port 7860)

```powershell
python app.py
```
Open your browser to: **`http://127.0.0.1:7860`**

### 3. Run the Full Flask Healthcare Platform (Port 5000)

```powershell
python run_flask.py
```
Open your browser to: **`http://127.0.0.1:5000`**

---

## 🔑 Pre-Configured Demo Accounts

The database includes pre-seeded demo accounts for technical evaluation:

| Role | Email Address | Password | Permissions |
|---|---|---|---|
| **Admin** | `admin@bioprint.local` | `AdminPassword@123` | Full administrative control, system rules, user management, facility registration, audit logs |
| **Hospital** | `hospital@citygeneral.local` | `HospitalPassword@123` | Hospital portal, 8-group stock updates, facility contact profile |
| **Hospital (Alt)** | `bloodbank@metro.local` | `BloodBankPassword@123` | Blood bank facility manager |
| **User (Patient)**| `user@example.com` | `UserPassword@123` | Fingerprint analysis, history, PDF reports, AI assistant, donor pre-screening |

---

## 🔌 Physical Biometric Scanner Plug-and-Play SDK

BioPrint Vision includes a hardware device driver abstraction layer (`services/scanner/`) enabling direct plug-and-play biometric capture from physical optical/capacitive scanners and local device bridges:

### Supported Hardware Drivers

1. **SecuGen Hamster Series (`secugen_webapi`):**
   * **Models:** Hamster Pro 20, Hamster Plus, Hamster IV, Hamster Air, Unity 20.
   * **Interface:** Local SecuGen FDx WebAPI service on `http://127.0.0.1:8084` or `https://127.0.0.1:8443`.
   * **Specs:** 500 DPI uncompressed BMP output, Auto-On touch detection, Fake Finger Rejection.

2. **Mantra MFS100 (`mantra_mfs100`):**
   * **Models:** MFS100 V54, MFS100 V61, MFS500.
   * **Interface:** Local Mantra RD Service on `http://127.0.0.1:11100` / `11101`.
   * **Specs:** STQC-certified 500 DPI optical platen, automated minutiae capture.

3. **DigitalPersona U.are.U Series (`digitalpersona_uareu`):**
   * **Models:** U.are.U 4500, 5160, 5300.
   * **Interface:** DigitalPersona Web Client / One Touch service on `http://127.0.0.1:52181`.
   * **Specs:** 512 DPI optical sensor with blue platen illumination.

4. **UART / Serial Optical Modules (`serial_optical_r307`):**
   * **Models:** Grow / Synochip R307, R502, R503, AS608.
   * **Interface:** Standard USB-to-UART bridge (CP2102, CH340, FTDI) on COM ports.
   * **Specs:** 500 DPI DSP module using standard Synochip 0xEF01 packet protocol.

5. **BioPrint Virtual Optical Scanner (`simulator_sensor`):**
   * **Models:** BioPrint V-Scan 500 (Virtual Sensor).
   * **Interface:** Built-in hardware simulation engine.
   * **Specs:** High-fidelity 500 DPI optical platen simulation with sensor Gaussian noise and touch latency (~450ms). Guarantees immediate plug-and-play operation for development, automated testing, and CI environments without requiring physical USB hardware attached.

### Scanner REST APIs

* **Device Enumeration:** `GET /api/scanner/devices` returns detected physical and virtual scanners with real-time connectivity status.
* **Live Sensor Capture:** `POST /api/scanner/capture` triggers optical acquisition and returns Base64-encoded 500 DPI BMP impression with quality metrics.
* **1-Click Scan & Predict:** `POST /api/scanner/analyze` triggers sensor capture, executes the ResNet-50 classification pipeline, stores database records, and returns the analysis view URL.

### Web Interface Integration

Navigate to **New Analysis (`/analysis/new`)** to use the interactive **Live Biometric Scanner** tab:
* Auto-detects connected USB devices.
* Live optical sensor platen with laser sweep feedback.
* **Preview Sensor Image Only:** Inspect impression clarity, 500 DPI resolution, and quality before analysis.
* **Scan & Predict Blood Group:** 1-click end-to-end acquisition, classification, and redirection to clinical reports.

---

## 🧪 Verification & Test Suites

The repository contains four complete, passing test suites:

### 1. Biometric Scanner Plug-and-Play SDK Tests
Validates device discovery, driver fallback, 500 DPI BMP generation, quality assessment, and REST analysis routes:
```powershell
d:\BioPrint-Vision\.venv\Scripts\python.exe tests\test_scanner_sdk.py
```
*(8 of 8 tests passing)*

### 2. End-to-End Flask Platform Integration Tests
Validates all routes, authentication, role restrictions, analysis workflow, PDF generation, AI assistant, and admin operations:
```powershell
d:\BioPrint-Vision\.venv\Scripts\python.exe tests\test_full_platform.py
```
*(11 of 11 tests passing)*

### 2. Multi-Fingerprint Computer Vision & Aggregation Tests
Validates automated contour detection, bounding box isolation, quality filtering, and probability averaging:
```powershell
d:\BioPrint-Vision\.venv\Scripts\python.exe tests\test_multi_fingerprint.py
```
*(All tests passing)*

### 3. Baseline Model & Gradio Pipeline Tests
Validates the frozen ResNet-50 checkpoint, Grad-CAM generation, and Gradio interface:
```powershell
d:\BioPrint-Vision\.venv\Scripts\python.exe test_pipeline.py
```
*(All tests passing)*

---

## 🗄️ Database Architecture

* **Database Engine:** MySQL 8.x Relational Database (`mysql+pymysql://root:password@localhost/bioprint_vision`).
* **DDL Schema:** The complete production enterprise database schema is defined in `database/schema.sql`.
* **Entities:**
  * `users`: Authentication, bcrypt password hashing, roles (`USER`, `HOSPITAL`, `ADMIN`).
  * `facilities`: Medical directory with GPS coordinates, contact info, operating hours, and `is_demo=True` tag.
  * `blood_inventory`: 8 stock records per facility with units available and automated status (`Available`, `Low Stock`, `Critical`, `Unavailable`).
  * `analyses`: Parent biometric run record (final prediction, confidence, agreement %, mode, image paths).
  * `fingerprint_results`: Child crop record (crop image, Grad-CAM, quality score, 8-class probabilities, exclusion reason).
  * `reports`: Generated PDF report metadata and static relative paths.

---

## 📁 Repository Structure

```
BioPrint-Vision/
├── app.py                         # FROZEN: Standalone Gradio interface (:7860)
├── run_flask.py                   # Full Flask platform entrypoint (:5000)
├── config.py                      # Application configuration & thresholds
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # Comprehensive documentation
│
├── database/                      # Database Layer
│   ├── connection.py              # SQLAlchemy engine initialization
│   ├── models.py                  # User, Facility, BloodInventory, Analysis, etc.
│   ├── schema.sql                 # MySQL 8.x production DDL schema
│   └── seed_demo_data.py          # Demo user & facility seeder (is_demo=True)
│
├── ml/                            # Machine Learning & CV Engines
│   ├── multi_fingerprint.py       # Automated multi-print CV splitter & aggregator
│   └── ...
│
├── models/                        # ML Checkpoints & Contracts
│   ├── best_model.pth             # ResNet-50 checkpoint (Epoch 25, 87.78% Test Acc)
│   ├── class_names.json           # Ordered 8-class array contract
│   └── split_indices.json         # Train/Val/Test split indices
│
├── routes/                        # Flask Blueprints
│   ├── auth.py                    # Login, registration, role guards
│   ├── main.py                    # Home, about, user dashboard
│   ├── analysis.py                # Biometric analysis upload, view, REST APIs
│   ├── scanner.py                 # Biometric scanner discovery, live capture & 1-click analysis
│   ├── availability.py            # Blood stock directory & search APIs
│   ├── donor.py                   # Compatibility guide & donor pre-screening
│   ├── reports.py                 # PDF list & secure download
│   ├── assistant.py               # AI assistant conversational interface
│   ├── hospital.py                # Hospital manager inventory portal
│   └── admin.py                   # Admin dashboard, users, facilities, rules
│
├── services/                      # Healthcare Decision Support Services
│   ├── scanner/                   # Physical Biometric Scanner Plug-and-Play SDK
│   │   ├── base.py                # Abstract driver interface & capture models
│   │   ├── manager.py             # Hardware discovery, auto-detection & routing
│   │   └── drivers/               # SecuGen, Mantra, DigitalPersona, Serial & Simulator
│   ├── compatibility.py           # Deterministic ABO/Rh RBC & plasma matrix
│   ├── eligibility.py             # Deterministic donor pre-screening rules
│   ├── inventory.py               # Haversine distance, facility search & stock classifier
│   ├── rarity.py                  # Rare blood group engine (AB-, B-, O-, A-)
│   ├── reporting.py               # ReportLab PDF generator
│   └── assistant.py               # Context-aware non-hallucinating AI chat service
│
├── templates/                     # Jinja2 HTML5 Templates (Bootstrap 5)
│   ├── base.html                  # Responsive shell with navbar, footer & alerts
│   ├── index.html                 # Platform landing page
│   ├── about.html                 # Architecture & scientific documentation
│   ├── dashboard.html             # User control dashboard
│   ├── analysis/                  # new.html (Scanner & Upload tabs), view.html, history.html
│   ├── healthcare/                # compatibility.html, donor_screening.html
│   ├── availability/              # search.html, facility_detail.html
│   ├── assistant/                 # chat.html
│   ├── reports/                   # list.html
│   ├── hospital/                  # dashboard.html, inventory.html, profile.html
│   ├── admin/                     # dashboard.html, users.html, facilities.html, etc.
│   └── errors/                    # 404.html, 500.html
│
└── tests/                         # Test Suites
    ├── test_scanner_sdk.py        # Biometric scanner discovery, capture & analysis tests
    ├── test_multi_fingerprint.py  # Unit tests for CV detection & probability averaging
    └── test_full_platform.py      # End-to-end platform integration test suite
```

---

## ⚖️ License & Ethical Conduct

This project is licensed under the **MIT License**. It is strictly intended for scientific investigation, biometric research, and computer vision education. Commercial medical claims or clinical deployment without prospective clinical trials and regulatory certification are prohibited.
