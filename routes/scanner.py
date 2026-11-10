"""
BioPrint Vision : Hardware Fingerprint Scanner Blueprint & REST APIs

Provides endpoints for plug-and-play biometric sensor discovery, hardware live capture,
and seamless end-to-end analysis from optical, capacitive, serial, or simulated scanners.
"""

from datetime import datetime
from pathlib import Path
from flask import (
    Blueprint,
    request,
    jsonify,
    g,
    url_for,
    current_app,
)
from PIL import Image

from routes.auth import login_required
from database.connection import db
from database.models import Analysis, FingerprintResult
from services.scanner import get_scanner_manager
from ml.multi_fingerprint import MultiFingerprintAnalyzer
from config import Config

scanner_bp = Blueprint("scanner", __name__, url_prefix="/api/scanner")

# Global analyzer cache
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = MultiFingerprintAnalyzer()
    return _analyzer


@scanner_bp.route("/devices", methods=["GET"])
def list_devices():
    """
    Enumerate all registered physical and simulated fingerprint scanners.
    Returns real-time connectivity status and device specifications.
    """
    try:
        mgr = get_scanner_manager()
        devices = mgr.list_devices(probe_hardware=True)
        active_id = mgr.get_active_device_id()

        return jsonify({
            "status": "success",
            "active_device_id": active_id,
            "device_count": len(devices),
            "devices": [d.to_dict() for d in devices],
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Failed to enumerate scanner devices: {str(e)}",
        }), 500


@scanner_bp.route("/capture", methods=["POST"])
def capture_live():
    """
    Trigger a live capture cycle on the selected biometric sensor.
    Returns the raw uncompressed 500 DPI bitmap (Base64) with quality metrics.
    """
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    timeout_sec = float(data.get("timeout", 10.0))

    try:
        mgr = get_scanner_manager()
        res = mgr.capture(device_id=device_id, timeout_sec=timeout_sec)

        if not res.success:
            return jsonify({
                "status": "error",
                "device_id": res.device_id,
                "device_name": res.device_name,
                "error": res.error_message or "Scanner capture failed",
            }), 400

        return jsonify({
            "status": "success",
            **res.to_dict(include_base64=True),
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Scanner capture exception: {str(e)}",
        }), 500


@scanner_bp.route("/analyze", methods=["POST"])
@login_required
def scan_and_analyze():
    """
    1-Click Plug-and-Play Biometric Flow:
      1. Trigger hardware capture on the selected scanner.
      2. Save 500 DPI impression into secure uploads folder.
      3. Execute multi-fingerprint / single-print classification pipeline.
      4. Store analysis record and individual fingerprint results.
      5. Return prediction summary and view redirect URL.
    """
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    force_mode = data.get("mode", "single")
    if force_mode == "auto":
        force_mode = None

    try:
        mgr = get_scanner_manager()
        capture_res, saved_path = mgr.capture_and_save(
            device_id=device_id,
            output_folder=Config.FINGERPRINT_FOLDER,
        )

        if not capture_res.success or not saved_path:
            return jsonify({
                "status": "error",
                "error": capture_res.error_message or "Failed to capture fingerprint from scanner",
            }), 400

        # Run ResNet-50 classification pipeline
        analyzer = get_analyzer()
        res = analyzer.analyze(saved_path, return_gradcam=True, force_mode=force_mode)

        # Handle annotated overview if multiple impressions detected
        annotated_rel_path = None
        if res.get("annotated_image"):
            annotated_filename = f"annotated_scan_{saved_path.stem}.jpg"
            annotated_path = Config.FINGERPRINT_FOLDER / annotated_filename
            res["annotated_image"].save(str(annotated_path), "JPEG")
            annotated_rel_path = f"uploads/fingerprints/{annotated_filename}"

        clean_rel_path = f"uploads/fingerprints/{saved_path.name}"

        # Create database record
        analysis = Analysis(
            user_id=g.user.id,
            final_blood_group=res["final_prediction"],
            final_confidence=res["final_confidence"],
            fingerprints_detected=res["detected_count"],
            fingerprints_analyzed=res["analyzed_count"],
            fingerprints_excluded=res["excluded_count"],
            agreement_percentage=res["agreement"]["percentage"],
            mode=res["mode"],
            image_path=clean_rel_path,
            annotated_image_path=annotated_rel_path,
            created_at=datetime.utcnow(),
        )
        db.session.add(analysis)
        db.session.flush()

        # Save individual crop and Grad-CAM results
        for fp in res["fingerprints"]:
            crop_rel_path = None
            gradcam_rel_path = None

            if fp.get("crop"):
                crop_fname = f"crop_scan_{analysis.id}_f{fp['id']}.jpg"
                crop_path = Config.FINGERPRINT_FOLDER / crop_fname
                fp["crop"].save(str(crop_path), "JPEG")
                crop_rel_path = f"uploads/fingerprints/{crop_fname}"

            if fp.get("gradcam"):
                gcam_fname = f"gradcam_scan_{analysis.id}_f{fp['id']}.jpg"
                gcam_path = Config.FINGERPRINT_FOLDER / gcam_fname
                fp["gradcam"].save(str(gcam_path), "JPEG")
                gradcam_rel_path = f"uploads/fingerprints/{gcam_fname}"

            fp_rec = FingerprintResult(
                analysis_id=analysis.id,
                finger_number=fp["id"],
                prediction=fp["prediction"],
                confidence=fp["confidence"],
                quality_score=fp["quality"]["score"],
                quality_status=fp["quality"]["status"],
                included_in_aggregation=fp["included"],
                exclusion_reason=fp.get("exclusion_reason"),
                crop_image_path=crop_rel_path,
                gradcam_image_path=gradcam_rel_path,
            )
            fp_rec.set_probabilities(fp["probabilities"])
            db.session.add(fp_rec)

        db.session.commit()

        return jsonify({
            "status": "success",
            "analysis_id": analysis.id,
            "redirect_url": url_for("analysis.view_analysis", analysis_id=analysis.id),
            "final_blood_group": res["final_prediction"],
            "final_confidence": res["final_confidence"],
            "quality_status": capture_res.quality_status,
            "quality_score": capture_res.quality_score,
            "device_used": capture_res.device_name,
            "capture_time_ms": capture_res.capture_time_ms,
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "status": "error",
            "error": f"Scanner analysis failed: {str(e)}",
        }), 500
