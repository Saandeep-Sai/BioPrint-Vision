"""
BioPrint Vision — Analysis Blueprint
Handles single and multi-fingerprint analysis execution, result rendering, and history.
"""

import uuid
from pathlib import Path
from datetime import datetime
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    g,
    current_app,
)
from werkzeug.utils import secure_filename
from PIL import Image

from routes.auth import login_required
from database.connection import db
from database.models import Analysis, FingerprintResult, Report
from ml.multi_fingerprint import MultiFingerprintAnalyzer
from services.compatibility import get_compatibility_info
from services.rarity import evaluate_blood_rarity
from services.inventory import search_blood_availability
from services.reporting import generate_analysis_pdf
from config import Config

analysis_bp = Blueprint("analysis", __name__)

# Initialize analyzer instance (loads model once)
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = MultiFingerprintAnalyzer()
    return _analyzer


@analysis_bp.route("/analysis/new", methods=["GET", "POST"])
@login_required
def new_analysis():
    if request.method == "POST":
        file = request.files.get("fingerprint_image")
        force_mode = request.form.get("mode")  # 'auto', 'single', 'multiple'
        if force_mode == "auto":
            force_mode = None

        if not file or file.filename == "":
            flash("Please upload a valid fingerprint image file.", "danger")
            return render_template("analysis/new.html")

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in Config.ALLOWED_EXTENSIONS:
            flash(f"Unsupported format '.{ext}'. Supported: {', '.join(Config.ALLOWED_EXTENSIONS)}", "danger")
            return render_template("analysis/new.html")

        # Save uploaded image safely
        unique_id = str(uuid.uuid4())[:8]
        clean_name = f"upload_{unique_id}_{secure_filename(file.filename)}"
        upload_path = Config.FINGERPRINT_FOLDER / clean_name
        file.save(str(upload_path))

        try:
            analyzer = get_analyzer()
            res = analyzer.analyze(upload_path, return_gradcam=True, force_mode=force_mode)

            # Save annotated image if multi-fingerprint mode
            annotated_rel_path = None
            if res.get("annotated_image"):
                annotated_name = f"annotated_{unique_id}.jpg"
                annotated_path = Config.FINGERPRINT_FOLDER / annotated_name
                res["annotated_image"].save(str(annotated_path), "JPEG")
                annotated_rel_path = f"uploads/fingerprints/{annotated_name}"

            # Create Database Record for Analysis
            analysis = Analysis(
                user_id=g.user.id,
                final_blood_group=res["final_prediction"],
                final_confidence=res["final_confidence"],
                fingerprints_detected=res["detected_count"],
                fingerprints_analyzed=res["analyzed_count"],
                fingerprints_excluded=res["excluded_count"],
                agreement_percentage=res["agreement"]["percentage"],
                mode=res["mode"],
                image_path=f"uploads/fingerprints/{clean_name}",
                annotated_image_path=annotated_rel_path,
                created_at=datetime.utcnow(),
            )
            db.session.add(analysis)
            db.session.flush()

            # Save individual crop results and per-crop Grad-CAM images
            for fp in res["fingerprints"]:
                crop_rel_path = None
                gradcam_rel_path = None

                if fp.get("crop"):
                    c_name = f"crop_{analysis.id}_f{fp['id']}.jpg"
                    c_path = Config.FINGERPRINT_FOLDER / c_name
                    fp["crop"].save(str(c_path), "JPEG")
                    crop_rel_path = f"uploads/fingerprints/{c_name}"

                if fp.get("gradcam"):
                    g_name = f"gradcam_{analysis.id}_f{fp['id']}.jpg"
                    g_path = Config.FINGERPRINT_FOLDER / g_name
                    fp["gradcam"].save(str(g_path), "JPEG")
                    gradcam_rel_path = f"uploads/fingerprints/{g_name}"

                fp_record = FingerprintResult(
                    analysis_id=analysis.id,
                    finger_number=fp["id"],
                    prediction=fp["prediction"],
                    confidence=fp["confidence"],
                    quality_score=fp["quality"]["score"],
                    quality_status=fp["quality"]["status"],
                    included_in_aggregation=fp["included"],
                    exclusion_reason=fp["exclusion_reason"],
                    crop_image_path=crop_rel_path,
                    gradcam_image_path=gradcam_rel_path,
                )
                fp_record.set_probabilities(fp["probabilities"])
                db.session.add(fp_record)

            db.session.commit()

            # Pre-generate PDF report
            analysis_dict = analysis.to_dict(include_fingerprints=True)
            analysis_dict["final_probabilities"] = res["final_probabilities"]
            # Attach first crop gradcam for report if available
            first_gradcam = next((fp.gradcam_image_path for fp in analysis.fingerprints if fp.gradcam_image_path), None)
            if first_gradcam:
                analysis_dict["gradcam_path"] = str(Config.BASE_DIR / "static" / first_gradcam)

            # Attach matching demo facilities snapshot
            matching_facs = search_blood_availability(blood_group=res["final_prediction"])
            analysis_dict["facilities_snapshot"] = matching_facs.get("results", [])[:3]

            pdf_path = generate_analysis_pdf(analysis_dict)
            rel_pdf_path = f"uploads/reports/{pdf_path.name}"

            report_record = Report(
                analysis_id=analysis.id,
                file_path=rel_pdf_path,
                created_at=datetime.utcnow(),
            )
            db.session.add(report_record)
            db.session.commit()

            flash("Fingerprint analysis completed successfully!", "success")
            return redirect(url_for("analysis.view_analysis", analysis_id=analysis.id))

        except Exception as e:
            db.session.rollback()
            flash(f"Analysis failed: {str(e)}", "danger")
            return render_template("analysis/new.html")

    # Sample images from verified dataset for 1-click testing
    sample_images = [
        {"name": "Sample A+", "path": "data/raw/A+/cluster_0_1001.BMP", "label": "Single A+"},
        {"name": "Sample B+", "path": "data/raw/B+/cluster_2_10.BMP", "label": "Single B+"},
        {"name": "Sample O+", "path": "data/raw/O+/cluster_6_1004.BMP", "label": "Single O+"},
    ]
    return render_template("analysis/new.html", samples=sample_images)


@analysis_bp.route("/analysis/<int:analysis_id>")
@login_required
def view_analysis(analysis_id: int):
    analysis = Analysis.query.get_or_404(analysis_id)

    # Protect privacy: users only see their own analyses, unless admin
    if analysis.user_id != g.user.id and g.user.role != "ADMIN":
        flash("You do not have permission to view this analysis.", "danger")
        return redirect(url_for("main.dashboard"))

    # Compute healthcare information
    compat = get_compatibility_info(analysis.final_blood_group)
    rarity = evaluate_blood_rarity(analysis.final_blood_group)

    # Search demo facilities with matching stock
    stock_info = search_blood_availability(blood_group=analysis.final_blood_group)
    nearby_facilities = stock_info.get("results", [])[:5]

    # Retrieve report if generated
    report = Report.query.filter_by(analysis_id=analysis.id).first()

    return render_template(
        "analysis/view.html",
        analysis=analysis,
        compatibility=compat,
        rarity=rarity,
        facilities=nearby_facilities,
        report=report,
    )


@analysis_bp.route("/history")
@login_required
def history():
    user = g.user
    query = Analysis.query.filter_by(user_id=user.id)

    # Optional filter by blood group
    bg_filter = request.args.get("blood_group")
    if bg_filter and bg_filter != "All":
        query = query.filter_by(final_blood_group=bg_filter)

    analyses = query.order_by(Analysis.created_at.desc()).all()
    return render_template("analysis/history.html", analyses=analyses, bg_filter=bg_filter)


# ── REST API Endpoints ─────────────────────────────────────────────────────────

@analysis_bp.route("/api/analyze/single", methods=["POST"])
def api_analyze_single():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        analyzer = get_analyzer()
        pil_img = Image.open(file.stream)
        res = analyzer.analyze(pil_img, return_gradcam=False, force_mode="single")
        return jsonify({
            "predicted_class": res["final_prediction"],
            "confidence": res["final_confidence"],
            "probabilities": res["final_probabilities"],
            "quality": res["fingerprints"][0]["quality"] if res["fingerprints"] else {},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@analysis_bp.route("/api/analyze/multiple", methods=["POST"])
def api_analyze_multiple():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        analyzer = get_analyzer()
        pil_img = Image.open(file.stream)
        res = analyzer.analyze(pil_img, return_gradcam=False, force_mode="multiple")
        return jsonify({
            "detected_count": res["detected_count"],
            "analyzed_count": res["analyzed_count"],
            "excluded_count": res["excluded_count"],
            "final_prediction": res["final_prediction"],
            "final_confidence": res["final_confidence"],
            "final_probabilities": res["final_probabilities"],
            "agreement": res["agreement"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
