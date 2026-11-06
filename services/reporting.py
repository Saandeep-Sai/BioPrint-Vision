"""
BioPrint Vision — Smart PDF Report Generation Service
Builds comprehensive medical-style PDF summary reports using ReportLab.
Adheres strictly to PRD Section 25 requirements.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from config import Config
from services.compatibility import get_compatibility_info
from services.rarity import evaluate_blood_rarity


def generate_analysis_pdf(analysis_data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """
    Generate a complete, professional PDF report for a completed analysis.

    Args:
        analysis_data: Dict containing analysis metadata, predictions, fingerprints, user, etc.
        output_path: Optional path to save the generated PDF file.

    Returns:
        Path to the generated PDF report.
    """
    Config.REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    analysis_id = analysis_data.get("id", "PROTOTYPE-1")

    if output_path is None:
        filename = f"BioPrint_Report_{analysis_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = Config.REPORT_FOLDER / filename
    else:
        output_path = Path(output_path)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#be123c"),  # Rose-700
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4b5563"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#991b1b"),
        backColor=colors.HexColor("#fee2e2"),
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8,
    )

    story = []

    # 1. Header with Brand & Document Metadata
    header_table_data = [
        [
            Paragraph("<b>🩸 BioPrint Vision</b><br/><font size=8 color='#64748b'>Deep Learning Fingerprint-to-Blood-Group Analysis</font>", title_style),
            Paragraph(
                f"<b>Analysis ID:</b> #{analysis_id}<br/>"
                f"<b>Date:</b> {analysis_data.get('created_at', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))}<br/>"
                f"<b>User:</b> {analysis_data.get('user_name', 'Authorized User')}<br/>"
                f"<b>Mode:</b> {analysis_data.get('mode', 'single').capitalize()}",
                subtitle_style,
            ),
        ]
    ]
    header_table = Table(header_table_data, colWidths=[4.0 * inch, 3.2 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))

    # 2. Mandatory Scientific & Medical Disclaimer
    disclaimer_text = (
        "<b>⚠️ CRITICAL MEDICAL & SCIENTIFIC DISCLAIMER:</b> This document presents experimental AI predictions "
        "derived from biometric dermatoglyphic analysis. It is <b>NOT</b> a certified diagnostic medical laboratory report. "
        "High model confidence does not guarantee clinical certainty. This screening prototype does <b>NOT</b> replace "
        "standard laboratory serological ABO/Rh testing. Never base transfusion, surgery, or clinical decisions on this report."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 6))

    # 3. Core Prediction Card
    final_bg = analysis_data.get("final_blood_group", "Unknown")
    confidence = analysis_data.get("final_confidence", 0.0)
    agreement = analysis_data.get("agreement_percentage", 100.0)

    # Rarity assessment
    rarity_info = evaluate_blood_rarity(final_bg)

    pred_card_data = [
        [
            Paragraph(f"<font size=10 color='#64748b'>PREDICTED BLOOD GROUP</font><br/><font size=28 color='#be123c'><b>{final_bg}</b></font>", body_style),
            Paragraph(
                f"<b>Model Confidence:</b> {confidence:.1f}%<br/>"
                f"<b>Multi-Finger Agreement:</b> {agreement:.1f}%<br/>"
                f"<b>Rarity Tier:</b> {rarity_info['tier']} ({rarity_info['prevalence']})<br/>"
                f"<b>Fingerprints Usable:</b> {analysis_data.get('fingerprints_analyzed', 1)} of {analysis_data.get('fingerprints_detected', 1)} detected",
                body_style,
            ),
        ]
    ]
    pred_table = Table(pred_card_data, colWidths=[3.2 * inch, 4.0 * inch])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 10))

    # 4. Fingerprint-Level Results Table
    fingerprints = analysis_data.get("fingerprints", [])
    if fingerprints:
        story.append(Paragraph("<b>Fingerprint Region Analysis (Individual Crops)</b>", section_heading))
        fp_table_data = [
            ["ID", "Crop Quality", "Crop Prediction", "Confidence", "Aggregation Status", "Notes / Exclusion"]
        ]
        for fp in fingerprints:
            f_num = f"F{fp.get('finger_number', fp.get('id', 1))}"
            q_status = fp.get("quality_status", "Good")
            q_score = fp.get("quality_score", 1.0)
            pred = fp.get("prediction", "N/A")
            conf = f"{fp.get('confidence', 0.0):.1f}%" if fp.get("confidence") else "N/A"
            status = "Included" if fp.get("included_in_aggregation", fp.get("included", True)) else "Excluded"
            reason = fp.get("exclusion_reason") or "Normal ridge flow"

            fp_table_data.append([
                f_num,
                f"{q_status} ({q_score:.2f})",
                pred,
                conf,
                status,
                reason[:35] + ("..." if len(reason) > 35 else ""),
            ])

        fp_table = Table(fp_table_data, colWidths=[0.6 * inch, 1.4 * inch, 1.2 * inch, 1.0 * inch, 1.2 * inch, 1.8 * inch])
        fp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(fp_table)
        story.append(Spacer(1, 10))

    # 5. 8-Class Probability Distribution
    probabilities = analysis_data.get("probabilities", {})
    if not probabilities and fingerprints:
        # Check first fingerprint probabilities or compute
        probabilities = analysis_data.get("final_probabilities", {})

    if probabilities:
        story.append(Paragraph("<b>8-Class Aggregated Probability Distribution</b>", section_heading))
        prob_headers = list(probabilities.keys())
        prob_values = [f"{float(probabilities[k]):.1f}%" for k in prob_headers]
        prob_table_data = [prob_headers, prob_values]
        prob_table = Table(prob_table_data, colWidths=[0.9 * inch] * len(prob_headers))
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 10))

    # 6. Compatibility & Donor Guidelines
    compat = get_compatibility_info(final_bg)
    if "error" not in compat:
        story.append(Paragraph("<b>Blood Transfusion Compatibility Guidelines</b>", section_heading))
        can_recv = ", ".join(compat.get("can_receive_from", []))
        can_don = ", ".join(compat.get("can_donate_to", []))
        compat_table_data = [
            [Paragraph("<b>Can Receive Red Cells From:</b>", body_style), Paragraph(can_recv, body_style)],
            [Paragraph("<b>Can Donate Red Cells To:</b>", body_style), Paragraph(can_don, body_style)],
            [Paragraph("<b>Universal Status:</b>", body_style), Paragraph(
                "Universal Red Blood Cell Donor" if compat.get("is_universal_donor") else (
                    "Universal Red Blood Cell Recipient" if compat.get("is_universal_recipient") else "Standard ABO/Rh Type"
                ), body_style)],
            [Paragraph("<b>Antigen / Antibody Profile:</b>", body_style), Paragraph(
                f"{compat.get('antigens')} | Antibodies: {compat.get('antibodies')}", body_style)],
        ]
        compat_table = Table(compat_table_data, colWidths=[2.5 * inch, 4.7 * inch])
        compat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(compat_table)
        story.append(Spacer(1, 10))

    # 7. Rare Blood Group Alert Banner (If Applicable)
    if rarity_info["is_rare"]:
        rare_alert_p = Paragraph(
            f"<b>⚠️ {rarity_info['alert_message']}</b>",
            ParagraphStyle(
                "RareAlert",
                parent=styles["Normal"],
                fontSize=8.5,
                leading=12,
                textColor=colors.HexColor("#92400e"),
                backColor=colors.HexColor("#fef3c7"),
                borderPadding=6,
            )
        )
        story.append(rare_alert_p)
        story.append(Spacer(1, 8))

    # 8. Demo Facility Availability Snapshot
    demo_facilities = analysis_data.get("facilities_snapshot", [])
    if demo_facilities:
        story.append(Paragraph("<b>Nearby Blood Facilities (Demo Stock Directory)</b>", section_heading))
        fac_table_data = [["Facility Name", "City", "Type", "Units Available", "Stock Status", "Contact Phone"]]
        for f in demo_facilities[:4]:
            fac_table_data.append([
                f.get("facility_name", "")[:28],
                f.get("city", ""),
                f.get("facility_type", "Hospital"),
                str(f.get("units_available", 0)),
                f.get("stock_status", "Available"),
                f.get("phone", ""),
            ])
        fac_table = Table(fac_table_data, colWidths=[2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.2 * inch])
        fac_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(fac_table)
        story.append(Paragraph("<font size=7 color='#64748b'><i>*Demo availability data — not connected to live hospital inventory. For demonstration purposes only.</i></font>", body_style))
        story.append(Spacer(1, 10))

    # 9. Embedded Grad-CAM Visualization (if present on disk)
    gradcam_path = analysis_data.get("gradcam_path")
    if gradcam_path and Path(gradcam_path).exists():
        try:
            story.append(Paragraph("<b>Grad-CAM Explainability (Neural Saliency Heatmap)</b>", section_heading))
            story.append(RLImage(str(gradcam_path), width=2.0 * inch, height=2.0 * inch))
            story.append(Paragraph(
                "<font size=7 color='#64748b'><i>Heatmap illustrates convolutional feature activations of ResNet-50 layer4; "
                "it highlights network attention areas rather than verified biological minutiae.</i></font>",
                body_style,
            ))
            story.append(Spacer(1, 8))
        except Exception:
            pass

    # 10. Footer / End of Document
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    footer_text = (
        "BioPrint Vision Prototype v2.0 • ResNet-50 Deep Learning Engine • "
        "Report Generated for Authorized Screening Research • All Rights Reserved."
    )
    story.append(Paragraph(f"<font size=7 color='#94a3b8'>{footer_text}</font>", body_style))

    doc.build(story)
    return output_path
