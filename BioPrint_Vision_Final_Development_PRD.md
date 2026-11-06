# BioPrint Vision — Final Development PRD

**Version:** Final Development Specification
**Status:** Approved for Full-System Development

## Product Purpose

BioPrint Vision is an experimental AI-assisted fingerprint and blood-group analysis platform. The accepted prototype already contains a trained ResNet-50 fingerprint classifier. The full product must preserve that ML core and extend it into a complete multi-user healthcare-support prototype with multi-fingerprint analysis, compatibility guidance, donor pre-screening, configurable rare-group alerts, synthetic hospital/blood-stock availability, smart reports, authentication, history, administration, and an AI assistant.

The supplied project PPT identifies the proposed features of fingerprint quality assessment, multi-fingerprint verification, blood donor/receiver compatibility, rare blood-group alerts, donor eligibility, smart report generation, and AI chat assistance. It also specifies user/admin authentication, MySQL, and PDF report generation in its technology requirements.

## Critical Implementation Principles

- The existing trained model is the source of truth for ML inference. Do not retrain merely to implement the full application.
- Preserve the exact trained class order: `A+`, `A-`, `AB+`, `AB-`, `B+`, `B-`, `O+`, `O-`.
- V2 multi-fingerprint aggregation must average raw probability vectors and then take `argmax`; do not replace this with majority voting.
- Synthetic hospital and inventory data must be explicitly labeled as demo/simulated data.
- Do not claim live hospital availability unless a real live data source is actually integrated.
- Compatibility and donor-screening rules must be deterministic/configurable rather than invented by an LLM.
- Do not present model confidence, agreement, or AI output as clinical certainty.
- Fingerprint images are biometric data; minimize storage and protect access.

## Current ML Evaluation Baseline

The completed model evaluation reported:

- Validation accuracy: **91.34%**
- Test accuracy: **87.78%**
- Top-2 accuracy: **98.00%**
- Macro precision: **88.56%**
- Macro recall: **87.71%**
- Macro F1: **87.92%**
- Weighted F1: **87.79%**
- Test set size: **900**
- Best epoch: **25**
- Mean confidence: **91.83%**
- Median confidence: **99.32%**
- Test errors: **110**

The model is therefore capable of being highly confident when wrong. The final UI must use the wording **Model Confidence**, not diagnostic certainty.

---

# BioPrint Vision V2
## Multi-Fingerprint Verification & Intelligent Blood-Group Analysis

## 1. Version Objective

BioPrint Vision V2 extends the V1 fingerprint blood-group classification prototype with **multi-fingerprint analysis from a single photograph**.

Instead of sending one complete image directly to the classifier, V2 accepts a photograph containing multiple fingerprints, automatically identifies and splits the individual fingerprint regions, runs the trained model independently on each fingerprint crop, aggregates the resulting probability distributions, and produces one final blood-group prediction.

### Core V2 workflow

```text
Photograph containing multiple fingerprints
                    ↓
             Image validation
                    ↓
        Fingerprint region detection
                    ↓
       Automatic fingerprint splitting
                    ↓
        Individual fingerprint crops
                    ↓
      Same V1 preprocessing pipeline
                    ↓
      Trained model on each fingerprint
                    ↓
        Per-fingerprint probabilities
                    ↓
          Probability aggregation
                    ↓
            Final blood group
                    ↓
        Confidence + verification
                    ↓
              Gradio result
```

---

# 2. V1 Baseline

V1 already provides:

- Fingerprint image upload
- Image validation
- Preprocessing
- Fingerprint quality assessment
- ResNet-50 inference
- Eight blood-group probabilities
- Blood-group prediction
- Model confidence
- Grad-CAM
- Gradio interface

The V1 trained model is retained and reused.

### Existing model

```text
Model: ResNet-50
Input: 224 × 224
Classes:
A+
A-
AB+
AB-
B+
B-
O+
O-
```

Do not retrain the model merely for V2.

---

# 3. Primary V2 Feature: Multi-Fingerprint Analysis

## 3.1 User Input

The user uploads **one photograph containing multiple fingerprints**.

Example:

```text
┌──────────────────────────────────┐
│                                  │
│   Fingerprint 1                  │
│                                  │
│        Fingerprint 2             │
│                                  │
│   Fingerprint 3                  │
│                                  │
│             Fingerprint 4        │
│                                  │
└──────────────────────────────────┘
```

The number of fingerprints may vary.

The system must not require the user to manually crop each fingerprint.

---

# 4. Fingerprint Detection and Splitting

The application must automatically locate individual fingerprint regions in the uploaded photograph.

The detection pipeline should:

1. Validate the input image.
2. Normalize the image for detection.
3. Detect candidate fingerprint regions.
4. Determine bounding boxes/regions of interest.
5. Filter invalid or excessively small regions.
6. Crop each fingerprint separately.
7. Preserve sufficient fingerprint detail.
8. Assign each detected crop a sequential ID.

The implementation should be robust to:

- different spacing between fingerprints
- moderate rotation
- moderate perspective variation
- different finger sizes
- uneven lighting
- background regions
- partial non-fingerprint content

---

# 5. Detection Strategy

Start with practical computer-vision techniques before introducing another detection model.

Candidate approach:

```text
Photograph
   ↓
Grayscale
   ↓
Denoising / contrast enhancement
   ↓
Fingerprint texture enhancement
   ↓
Foreground/ROI segmentation
   ↓
Connected components / contours
   ↓
Candidate regions
   ↓
Geometric + texture filtering
   ↓
Fingerprint crops
```

Where classical segmentation is insufficient, the agent may implement a lightweight detection/segmentation model or another practical method.

Choose the simplest approach that works reliably on representative prototype photographs.

Do not add an object-detection model merely because it is more sophisticated.

---

# 6. Per-Fingerprint Processing

Every detected fingerprint must pass through the **same inference preprocessing used by V1**.

For each crop:

```text
Fingerprint crop
      ↓
Quality check
      ↓
V1 preprocessing
      ↓
224 × 224
      ↓
ResNet-50
      ↓
8 probabilities
```

The system must preserve each fingerprint's individual result.

Example:

```text
Fingerprint 1
Prediction: B+
B+: 0.82

Fingerprint 2
Prediction: B+
B+: 0.91

Fingerprint 3
Prediction: O+
B+: 0.55
O+: 0.31

Fingerprint 4
Prediction: B+
B+: 0.88
```

---

# 7. Aggregation Algorithm

This is the defining V2 requirement.

The final result must be based on the **average of the per-fingerprint probability distributions**.

For N usable fingerprints:

```text
P_final(class)
=
( P1(class) + P2(class) + ... + PN(class) ) / N
```

The final prediction is:

```text
Final class = argmax(average probabilities)
```

This must be implemented as **probability averaging**, not majority vote.

Do not average already-rounded display percentages. Use the raw model probabilities.

---

# 8. Aggregated Confidence

The displayed final confidence must be derived from the final averaged probability distribution.

Example:

```text
Final Prediction: B+
Aggregated Model Confidence: 79.0%
```

Do not describe this as diagnostic certainty.

---

# 9. Fingerprint-Level Results

The UI should show the contribution of every detected fingerprint.

Example:

```text
Fingerprint 1 → B+ → 82.4%
Fingerprint 2 → B+ → 91.2%
Fingerprint 3 → O+ → 55.0%
Fingerprint 4 → B+ → 88.1%
```

This allows the user to see whether individual fingerprints agree.

---

# 10. Consistency Analysis

V2 should calculate a simple consistency indicator.

Possible measures:

- number of fingerprints supporting the final prediction
- percentage of fingerprints whose top prediction matches the final prediction
- dispersion of per-fingerprint probabilities
- agreement score

Example:

```text
4 fingerprints analyzed

3/4 top predictions = B+

Consistency: 75%
```

This is an analysis indicator, not a clinically validated reliability score.

---

# 11. Quality-Aware Handling

Every fingerprint crop must receive an individual quality assessment.

Example:

```text
Fingerprint 1 — Good
Fingerprint 2 — Good
Fingerprint 3 — Poor
Fingerprint 4 — Good
```

A clearly unusable fingerprint may be excluded from aggregation according to a deterministic quality rule.

The UI must show when a fingerprint was excluded.

Example:

```text
4 fingerprints detected
3 usable for final aggregation
1 excluded due to poor quality
```

Do not silently remove fingerprints.

---

# 12. Minimum Usable Fingerprints

Recommended prototype behavior:

- 1 usable fingerprint → allow analysis as single-fingerprint mode
- 2+ usable fingerprints → multi-fingerprint analysis
- 0 usable fingerprints → stop and request a clearer image

The threshold should be configurable.

---

# 13. Detection Visualization

Display the original photograph with detected fingerprint regions overlaid.

Example:

```text
Original photograph

┌───────────────────────────────┐
│ ┌───────┐                     │
│ │ F1    │                     │
│ └───────┘     ┌───────┐       │
│               │ F2    │       │
│               └───────┘       │
│ ┌───────┐                     │
│ │ F3    │       ┌───────┐     │
│ └───────┘       │ F4    │     │
│                 └───────┘     │
└───────────────────────────────┘
```

Each bounding box should have a corresponding fingerprint ID.

---

# 14. Per-Fingerprint Grad-CAM

Where feasible, generate Grad-CAM separately for each accepted fingerprint crop.

The UI should allow inspection of:

```text
Fingerprint 1
Original + Grad-CAM

Fingerprint 2
Original + Grad-CAM

Fingerprint 3
Original + Grad-CAM
```

Do not generate Grad-CAM at the whole-photo level because the trained classifier expects an individual fingerprint.

---

# 15. Final V2 Result

## Summary

```text
BIOPRINT VISION

Fingerprints detected: 4
Fingerprints analyzed: 4
Agreement: 75%

Final Prediction
B+

Aggregated Model Confidence
79.0%
```

## Detailed analysis

```text
Fingerprint 1
B+ — 82.4%

Fingerprint 2
B+ — 91.2%

Fingerprint 3
O+ — 55.0%

Fingerprint 4
B+ — 88.1%
```

Also show:

- averaged probability distribution
- detection visualization
- optional Grad-CAM
- quality warnings
- disclaimer

---

# 16. Gradio UI Structure

Recommended V2 interface:

```text
┌───────────────────────────────────────────────┐
│              BIOPRINT VISION                 │
│       Multi-Fingerprint Blood Analysis       │
├───────────────────────────────────────────────┤
│                                               │
│ Upload Photograph                             │
│                                               │
│       [ Fingerprint Photograph ]              │
│                                               │
│               [ ANALYZE ]                    │
│                                               │
├───────────────────────────────────────────────┤
│ DETECTION                                     │
│                                               │
│ 4 fingerprints detected                       │
│ [Annotated image]                             │
│                                               │
├───────────────────────────────────────────────┤
│ FINGERPRINT ANALYSIS                          │
│                                               │
│ Finger 1 → B+ → 82.4%                         │
│ Finger 2 → B+ → 91.2%                         │
│ Finger 3 → O+ → 55.0%                         │
│ Finger 4 → B+ → 88.1%                         │
│                                               │
├───────────────────────────────────────────────┤
│ FINAL RESULT                                  │
│                                               │
│                 B+                            │
│       Aggregated Confidence: 79.0%            │
│                                               │
│       Agreement: 75%                          │
│                                               │
│       [Probability Distribution]              │
│                                               │
├───────────────────────────────────────────────┤
│ EXPLAINABILITY                                │
│                                               │
│       [Grad-CAM results]                      │
│                                               │
├───────────────────────────────────────────────┤
│ Experimental research/prototype disclaimer    │
└───────────────────────────────────────────────┘
```

---

# 17. Backend API

Provide:

```python
result = analyze_multiple_fingerprints(image)
```

Recommended structure:

```python
{
    "detected_count": 4,
    "analyzed_count": 4,
    "excluded_count": 0,

    "fingerprints": [
        {
            "id": 1,
            "crop": ...,
            "quality": {...},
            "prediction": "B+",
            "confidence": 0.824,
            "probabilities": {...},
            "gradcam": ...
        }
    ],

    "final_prediction": "B+",
    "final_confidence": 0.790,
    "final_probabilities": {...},

    "agreement": {
        "supporting_count": 3,
        "total_count": 4,
        "percentage": 75.0
    }
}
```

Use typed structures/dataclasses where beneficial.

---

# 18. Performance Requirements

The system should:

- load the model once
- perform batch inference where practical
- avoid unnecessary model reloads
- reuse preprocessing components
- use GPU automatically when available
- remain functional on CPU

If multiple crops are available, evaluate whether batch inference improves performance.

---

# 19. Error Handling

Handle:

- no image
- image with no detected fingerprints
- only one detectable region
- very poor image quality
- overlapping fingerprint regions
- unreadable crop
- model inference failure
- Grad-CAM failure

Useful messages include:

> No clear fingerprint regions were detected. Please upload a clearer photograph.

> Only one usable fingerprint was detected. The system will perform single-fingerprint analysis.

> Two fingerprints were detected, but one was excluded because of poor image quality.

---

# 20. Testing

Create automated/basic tests for:

### Detection

- detection pipeline runs
- detected bounding boxes are valid
- empty/no-fingerprint images fail gracefully

### Aggregation

Given:

```text
P1
P2
P3
```

verify:

```text
P_final = mean(P1, P2, P3)
```

Verify:

```text
Final prediction = argmax(P_final)
```

Verify probabilities sum to approximately 1.

### Consistency

Verify agreement calculations.

### Quality filtering

Verify unusable crops are handled according to the documented rule.

### End-to-end

Test:

```text
photo
 ↓
detection
 ↓
crops
 ↓
individual predictions
 ↓
probability averaging
 ↓
final prediction
 ↓
Gradio output
```

---

# 21. Backward Compatibility

V1 single-fingerprint input must continue to work.

The application should support:

```text
Single fingerprint image
```

and:

```text
Photograph containing multiple fingerprints
```

Where possible, automatically determine the mode based on the number of valid detected regions.

---

# 22. Model Integrity

Do not retrain or modify the V1 trained model as part of V2 unless explicitly required.

V2 changes the input acquisition and aggregation layer, not the underlying classifier.

This makes V2 directly comparable with V1.

---

# 23. Scientific and Medical Disclaimer

The application must continue to state:

> **Experimental AI prediction for research/prototype purposes. This result does not replace laboratory blood-group testing.**

Multi-fingerprint averaging must not be described as clinical confirmation.

"Agreement", "consistency", and "confidence" are model-analysis indicators and must not be represented as clinically validated reliability measures.

---

# 24. Future Scope — V3+

The following remain outside V2:

- donor/receiver compatibility
- donor eligibility
- rare blood-group alerts
- AI chat assistant
- smart PDF reports
- user authentication
- admin panel
- database
- physical fingerprint scanner integration
- smartphone biometric sensor integration
- clinical validation

---

# 25. V2 Definition of Done

[ ] V1 functionality remains intact.

[ ] One photograph can be uploaded.

[ ] Multiple fingerprint regions can be detected.

[ ] Fingerprints are automatically cropped.

[ ] Each usable crop is independently classified.

[ ] Each crop receives its own probability distribution.

[ ] Final probabilities are calculated by arithmetic mean across usable fingerprints.

[ ] Final prediction is argmax of averaged probabilities.

[ ] Final confidence is taken from the averaged probability distribution.

[ ] Fingerprint-level predictions are displayed.

[ ] Agreement/consistency is displayed.

[ ] Poor-quality crops are handled transparently.

[ ] Annotated detection image is shown.

[ ] Per-fingerprint Grad-CAM is available where technically feasible.

[ ] Gradio UI works end-to-end.

[ ] Single-fingerprint V1 mode continues to work.

[ ] Automated/basic tests pass.

[ ] README documents V2 behavior and aggregation math.

[ ] No unsupported medical claims are introduced.


# 3. Final Full-System Scope

The final product consists of six major layers:

1. **AI/Computer Vision** — preprocessing, quality assessment, fingerprint detection/splitting, ResNet-50 classification, multi-fingerprint probability aggregation, Grad-CAM.
2. **Analysis** — single and multi-fingerprint analysis, individual results, agreement, final result.
3. **Healthcare Support** — compatibility, donor pre-screening, rare-group alerts, emergency information.
4. **Blood Availability** — hospital/blood-bank directory, facility creation, synthetic inventory, inventory updates, availability search.
5. **Platform** — authentication, roles, MySQL persistence, analysis history, reports, administration.
6. **AI Assistant** — controlled explanation and navigation using actual application context.

# 4. End-to-End User Workflow

```text
Register / Login
      ↓
User Dashboard
      ↓
New Analysis
      ↓
Single Fingerprint OR Multiple Fingerprints Photo
      ↓
Image Validation
      ↓
Quality Assessment
      ↓
Fingerprint Detection / Splitting
      ↓
Individual Fingerprint Crops
      ↓
ResNet-50 Inference Per Crop
      ↓
Raw 8-Class Probability Vectors
      ↓
Average Probability Vectors
      ↓
Final Blood Group + Model Confidence
      ↓
Agreement + Quality + Grad-CAM
      ↓
Compatibility + Donor Screening + Rare Alert
      ↓
Blood Availability Search
      ↓
AI Assistant / Smart Report
      ↓
Save Analysis + Report
      ↓
History
```

# 5. User Roles

## End User

- Register/login/logout
- Perform single-fingerprint analysis
- Perform multi-fingerprint analysis
- View quality and model results
- View compatibility
- Run donor pre-screening
- View rare-group alerts
- Search facility availability
- View reports/history
- Use AI assistant
- Manage profile

## Hospital / Blood Bank User

- Login
- Create/manage facility profile
- Add/update blood inventory
- Update stock status and timestamps
- View availability records

## Super Admin

- Manage users
- Manage facilities
- Approve/deactivate facilities
- Manage inventory records
- View analyses and reports
- Configure rules/settings
- Monitor application/model status

# 6. Authentication

Implement secure registration and login.

Required behavior:

- Password hashing
- Session/authentication handling
- Role-based authorization
- Logout
- Protected routes
- Server-side validation

Do not store plaintext passwords.

# 7. User Dashboard

Navigation should include:

```text
Dashboard
├── New Analysis
├── Analysis History
├── Blood Availability
├── Compatibility
├── Donor Screening
├── Reports
├── AI Assistant
└── Profile
```

# 8. Analysis Module

Support two modes:

### Single Fingerprint

One image → quality → preprocessing → model → prediction.

### Multiple Fingerprints

One photograph → detect fingerprints → crop → quality → model per crop → average raw probabilities → final prediction.

# 9. Multi-Fingerprint Detection

The system must automatically detect and crop individual fingerprints from one photograph.

Preferred first implementation:

```text
Photo
 ↓
Grayscale
 ↓
Contrast/texture enhancement
 ↓
Foreground segmentation
 ↓
Contours / connected components
 ↓
Geometric and texture filtering
 ↓
Fingerprint crops
```

Use a separate detection model only if the practical CV pipeline is insufficient.

Display an annotated source image showing F1, F2, F3, etc.

# 10. Multi-Fingerprint Aggregation

For N usable fingerprints:

```text
P_final(c) = (P1(c) + P2(c) + ... + PN(c)) / N
```

Then:

```text
Final Prediction = argmax(P_final)
```

Retain the full raw probability vectors for every fingerprint.

Example:

```text
F1 → B+ 0.82
F2 → B+ 0.91
F3 → O+ 0.55
F4 → B+ 0.88

Final B+ probability = mean of B+ probabilities across all usable fingerprints.
```

The final displayed confidence is the maximum value of the averaged probability vector.

# 11. Agreement

Show:

- detected count
- analyzed count
- excluded count
- number supporting final prediction
- agreement percentage

Agreement is a model-analysis indicator only.

# 12. Fingerprint Quality

Assess:

- blur
- contrast
- brightness
- usable area
- noise
- ridge visibility

For multiple fingerprints, assess each crop independently.

If a crop is excluded, show the reason. Never silently exclude it.

# 13. Model Inference

Load `models/best_model.pth` once per application worker where practical.

Requirements:

- ResNet-50
- exact class mapping
- 224×224 model input
- same preprocessing contract used during training
- CPU fallback
- CUDA when available
- inference/no-gradient mode

# 14. Explainability

Provide Grad-CAM for individual fingerprint crops when technically supported by the trained model.

Do not imply Grad-CAM proves a biological blood-group mechanism.

# 15. Compatibility Module

After prediction, show deterministic ABO/Rh compatibility information.

The compatibility engine must be implemented as maintainable rules/configuration, not generated by an LLM.

The UI must clearly label it as compatibility guidance and not a substitute for professional medical confirmation.

# 16. Donor Pre-Screening

Provide a structured questionnaire and configurable rule engine.

The PPT requires donor eligibility but does not define clinical thresholds. Therefore the implementation must not invent clinical policy.

Possible result labels:

- Potentially Eligible
- Medical Review Required
- Not Eligible Based on Screening

Display a disclaimer that final donor eligibility is determined by qualified blood-bank/medical staff.

# 17. Rare Blood-Group Alert

Implement a configurable rare-group rule.

Because the PPT does not define the exact rarity threshold or population, do not hard-code an unsupported definition.

Example:

```text
Rare Blood Group Alert
AB-

Please confirm through laboratory testing and contact an appropriate blood facility if necessary.
```

# 18. Hospital Directory

Provide facility records with:

- name
- type
- address
- city
- state
- pincode
- latitude
- longitude
- phone
- email
- operating hours
- status
- demo/simulated flag

Facility types:

- Hospital
- Blood Bank
- Blood Centre

# 19. Add Hospital Page

Authorized hospital/admin users can create a facility.

Fields:

```text
Facility Name
Facility Type
Address
City
State
Pincode
Latitude
Longitude
Phone
Email
Operating Hours
Status
```

# 20. Blood Inventory Management

Each facility can maintain stock for:

```text
A+ A- AB+ AB- B+ B- O+ O-
```

Fields:

```text
Blood Group
Units Available
Stock Status
Last Updated
```

Possible stock states:

- Available
- Low Stock
- Critical
- Unavailable

Use configurable thresholds where practical.

# 21. Synthetic Data

Seed the prototype with internally consistent demo hospitals and inventory.

Each seeded record must have a demo flag.

Every relevant UI page must state:

> **Demo availability data — not connected to live hospital inventory.**

Do not represent synthetic stock as real hospital inventory.

# 22. Blood Availability Search

From the final predicted blood group, provide a one-click availability search.

The user can also manually search a different blood group where appropriate.

Search/filter options:

- blood group
- city/location
- stock status
- facility type
- distance where coordinates exist

Display:

- facility name
- location
- matching blood group
- units
- stock status
- last updated
- demo indicator
- contact information
- map/directions link where implemented

# 23. Emergency Support Information

The system can provide an informational emergency-support section:

```text
AI Prediction
 ↓
Compatibility guidance
 ↓
Availability directory
 ↓
Facility contact/location
 ↓
Laboratory confirmation reminder
```

Do not automate emergency medical decisions.

# 24. AI Assistant

The assistant should use structured application context such as:

- current predicted blood group
- per-fingerprint predictions
- confidence/probabilities
- image quality
- agreement
- compatibility results
- availability results
- report data

Example questions:

- What blood group was predicted?
- Why did the fingerprints disagree?
- What can B+ receive?
- Which demo facilities have B+ stock?
- Why was a fingerprint excluded?
- Explain my report.

The assistant must not invent results, override deterministic rules, or present synthetic availability as real.

# 25. Smart PDF Reports

Use ReportLab or equivalent deterministic PDF generation.

Include:

- user information
- date/time
- analysis ID
- detected/analyzed/excluded fingerprints
- quality results
- individual predictions
- final prediction
- aggregated confidence
- probability distribution
- agreement
- compatibility
- donor pre-screening result
- rare-group alert
- availability results when selected
- Grad-CAM where appropriate
- full disclaimer

Clearly identify demo inventory data.

# 26. Analysis History

Store and display:

- analysis ID
- date/time
- final blood group
- confidence
- number of fingerprints
- agreement
- report

Do not expose another user's analysis.

# 27. Database

Use MySQL for the complete multi-user application.

## users

```text
id
name
email
password_hash
role
status
created_at
updated_at
```

## facilities

```text
id
name
type
address
city
state
pincode
latitude
longitude
phone
email
operating_hours
is_demo
status
created_at
updated_at
```

## blood_inventory

```text
id
facility_id
blood_group
units_available
stock_status
last_updated
is_demo
```

## analyses

```text
id
user_id
final_blood_group
final_confidence
fingerprints_detected
fingerprints_analyzed
fingerprints_excluded
agreement_percentage
created_at
```

## fingerprint_results

```text
id
analysis_id
finger_number
prediction
confidence
quality_score
quality_status
included_in_aggregation
probabilities_json
```

## reports

```text
id
analysis_id
file_path
created_at
```

Configuration/rules may be stored in database or versioned configuration files for compatibility, rarity, donor screening, and inventory thresholds.

# 28. Privacy

Fingerprint images are sensitive biometric information.

Default design should:

- avoid permanent raw-image storage unless required
- minimize personally identifiable data
- protect reports
- enforce role-based access
- avoid public raw-image URLs
- log access/actions without unnecessarily storing biometric content

# 29. Full Application Architecture

```text
                         Browser
                            │
                            ▼
                     Flask Application
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
 Authentication         Analysis API        Management API
                            │                    │
                  ┌─────────┼─────────┐         │
                  ▼         ▼         ▼         │
              Detection  ML Model  Grad-CAM     │
                  │         │                    │
                  └────┬────┘                    │
                       ▼                         │
                 Analysis Engine                │
                       │                         │
          ┌────────────┼────────────┐            │
          ▼            ▼            ▼            ▼
     Compatibility Eligibility  Availability  Admin/Rules
          │            │            │            │
          └────────────┼────────────┴────────────┘
                       ▼
                  Report Engine
                       │
                       ▼
                     MySQL
```

## Technology Direction

For the complete multi-page system:

- Frontend: HTML5, CSS3, JavaScript, Bootstrap
- Backend: Python + Flask
- ML inference: PyTorch + torchvision
- Image processing: OpenCV + Pillow/NumPy
- Database: MySQL
- PDF reports: ReportLab

The accepted Gradio prototype may remain as a standalone demo/debug interface, but the full product should expose the complete feature set through the main web application.

# 30. Main Pages

## Public

```text
/
/about
/login
/register
```

## User

```text
/dashboard
/analysis/new
/analysis/<id>
/history
/reports
/availability
/facilities/<id>
/compatibility
/donor-screening
/assistant
/profile
```

## Hospital

```text
/hospital/dashboard
/hospital/profile
/hospital/inventory
/hospital/availability
```

## Admin

```text
/admin/dashboard
/admin/users
/admin/facilities
/admin/inventory
/admin/analyses
/admin/reports
/admin/rules
/admin/system
```

# 31. Recommended API Surface

```text
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me

POST /api/analyze/single
POST /api/analyze/multiple
GET  /api/analysis/<id>
GET  /api/analysis/history

GET  /api/availability
GET  /api/facilities/<id>
POST /api/facilities
PUT  /api/facilities/<id>
GET  /api/facilities/<id>/inventory
POST /api/facilities/<id>/inventory
PUT  /api/inventory/<id>

POST /api/donor-screening
GET  /api/donor-screening/<id>

POST /api/reports/generate
GET  /api/reports/<id>

POST /api/assistant/chat
```

Add admin endpoints for users, facilities, analyses, inventory, and configuration as required.

# 32. Security

Implement:

- password hashing
- role-based access control
- server-side validation
- secure upload handling
- parameterized database access/ORM
- CSRF protection where applicable
- protected report access
- secure session/cookie configuration
- environment variables for secrets

# 33. Testing

## ML/Inference

- checkpoint loads
- class order is correct
- preprocessing shape is correct
- probabilities sum to approximately 1
- predicted class equals highest probability

## Multi-Fingerprint

- photograph processing runs
- crop bounding boxes are valid
- no-detection case is handled
- raw-probability averaging is mathematically correct
- argmax final prediction is correct
- agreement is correct
- exclusion behavior is correct

## Compatibility

Test every blood group against the configured rule table.

## Inventory

- create facility
- create inventory
- update stock
- filter by blood group
- demo flag works

## Authentication

- registration
- login
- logout
- role protection

## Reports

- report generation
- data correctness
- file access control

## End-to-End

```text
Login
 ↓
New Analysis
 ↓
Upload photo
 ↓
Detect fingerprints
 ↓
Split crops
 ↓
Model inference
 ↓
Probability averaging
 ↓
Final result
 ↓
Compatibility
 ↓
Availability
 ↓
Generate report
 ↓
History
```

# 34. Error Handling

Handle gracefully:

- invalid image
- corrupt image
- no fingerprints detected
- only one fingerprint detected
- poor quality
- unreadable crop
- model loading error
- inference error
- database failure
- report generation failure
- unavailable facility

Never expose raw stack traces to end users.

# 35. Performance

- load model once per worker
- use inference mode
- batch multiple crops where useful
- cache/read configuration efficiently
- use indexes for frequent database queries
- support CPU and GPU
- avoid permanent image processing duplicates

# 36. Demo Data Requirements

Provide a script such as:

```text
seed_demo_data.py
```

It should create:

- demo users
- demo hospitals/blood banks
- demo blood inventory

All demo facilities and inventory must be flagged as simulated.

# 37. Documentation

README must include:

- project overview
- architecture
- installation
- environment variables
- model setup
- database setup
- demo-data seeding
- application startup
- user roles
- analysis workflow
- multi-fingerprint aggregation formula
- compatibility behavior
- availability behavior
- report generation
- evaluation metrics
- limitations
- privacy considerations
- disclaimer

# 38. Medical/Scientific Disclaimer

Display everywhere a final AI result is shown:

> **Experimental AI prediction for research/prototype purposes. This result does not replace laboratory blood-group testing.**

For simulated inventory:

> **Demo availability data — not connected to live hospital inventory.**

For donor screening:

> **Pre-screening only. Final donor eligibility must be determined by qualified medical/blood-bank personnel.**

# 39. Non-Goals

Do not implement or claim:

- clinical diagnosis
- laboratory confirmation
- guaranteed blood-group accuracy
- real-time hospital stock without a real source
- direct access to a smartphone's secure built-in fingerprint unlock sensor
- automated emergency medical decisions
- guaranteed donor eligibility
- guaranteed blood availability

# 40. Definition of Done

The final product is complete only when:

### ML

[ ] Existing trained checkpoint loads.

[ ] Exact class mapping is preserved.

[ ] Single-fingerprint inference works.

[ ] Multi-fingerprint detection works.

[ ] Individual crops are analyzed.

[ ] Raw probabilities are averaged.

[ ] Final class is argmax of averaged probabilities.

[ ] Confidence and agreement are displayed correctly.

[ ] Image quality works.

[ ] Grad-CAM works where supported.

### Platform

[ ] Registration works.

[ ] Login/logout works.

[ ] Role-based access works.

[ ] User dashboard works.

[ ] Hospital dashboard works.

[ ] Admin dashboard works.

[ ] MySQL persistence works.

[ ] Analysis history works.

[ ] PDF reports work.

### Healthcare Support

[ ] Compatibility works.

[ ] Donor pre-screening works.

[ ] Rare-group alerts work.

[ ] Emergency information works.

### Blood Availability

[ ] Hospital can be added.

[ ] Inventory can be added/updated.

[ ] All eight groups are supported.

[ ] Availability search works.

[ ] Synthetic records are visibly marked.

[ ] Predicted blood group can drive the availability search.

### Assistant

[ ] AI assistant uses current system context.

[ ] AI assistant does not fabricate results.

### Quality

[ ] Invalid inputs are handled.

[ ] End-to-end testing passes.

[ ] README is complete.

[ ] No unsupported medical claims are present.

# 41. Final Product Narrative

The completed system should tell one coherent story:

> A user signs in and uploads a fingerprint or a photograph containing multiple fingerprints. BioPrint Vision validates the image, evaluates quality, automatically isolates usable fingerprint regions, and runs the trained ResNet-50 model on each fingerprint. The system averages the raw probability vectors across usable fingerprints and produces one final experimental blood-group prediction with model confidence, agreement, and explainability. It then provides deterministic blood compatibility guidance, a donor pre-screening workflow, configurable rare-group alerts, and a hospital/blood-bank directory containing clearly labelled demo blood-stock data. Users can explore the result through an AI assistant, generate a smart PDF report, and revisit prior analyses. Hospital users manage their simulated inventory, while administrators manage the platform and configuration.\n