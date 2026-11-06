"""
BioPrint Vision — Main Pages Blueprint
Provides landing page, about page, and user dashboard.
"""

from flask import Blueprint, render_template, redirect, url_for, g
from routes.auth import login_required
from database.models import Analysis, Facility
from services.compatibility import COMPATIBILITY_RULES

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Public landing page showcasing BioPrint Vision platform features."""
    return render_template("index.html")


@main_bp.route("/about")
def about():
    """About page with technical architecture, dataset audit, and model benchmarks."""
    return render_template("about.html")


@main_bp.route("/dashboard")
@login_required
def dashboard():
    """User dashboard summarizing recent analyses and quick healthcare tools."""
    user = g.user
    recent_analyses = (
        Analysis.query.filter_by(user_id=user.id)
        .order_by(Analysis.created_at.desc())
        .limit(5)
        .all()
    )
    total_analyses = Analysis.query.filter_by(user_id=user.id).count()
    facility_count = Facility.query.filter_by(status="ACTIVE").count()

    latest_analysis = recent_analyses[0] if recent_analyses else None

    return render_template(
        "dashboard.html",
        user=user,
        recent_analyses=recent_analyses,
        total_analyses=total_analyses,
        facility_count=facility_count,
        latest_analysis=latest_analysis,
        compatibility_rules=COMPATIBILITY_RULES,
    )
