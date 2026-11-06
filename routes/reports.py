"""
BioPrint Vision — Reports Blueprint
Handles listing and secure downloading of generated PDF analysis reports.
"""

from pathlib import Path
from flask import Blueprint, render_template, send_file, abort, g, flash, redirect, url_for
from routes.auth import login_required
from database.models import Report, Analysis
from config import Config

reports_bp = Blueprint("reports", __name__)


@reports_bp.route("/reports")
@login_required
def list_reports():
    """List all reports generated for the logged-in user."""
    reports = (
        Report.query.join(Analysis)
        .filter(Analysis.user_id == g.user.id)
        .order_by(Report.created_at.desc())
        .all()
    )
    return render_template("reports/list.html", reports=reports)


@reports_bp.route("/reports/<int:report_id>/download")
@login_required
def download_report(report_id: int):
    """Download PDF report file securely."""
    report = Report.query.get_or_404(report_id)

    # Privacy check: only owner or admin can download
    if report.analysis.user_id != g.user.id and g.user.role != "ADMIN":
        flash("You are not authorized to access this report.", "danger")
        return redirect(url_for("reports.list_reports"))

    file_path = Config.BASE_DIR / "static" / report.file_path
    if not file_path.exists():
        abort(404, description="Report PDF file not found on disk.")

    return send_file(
        str(file_path),
        as_attachment=True,
        download_name=file_path.name,
        mimetype="application/pdf",
    )
