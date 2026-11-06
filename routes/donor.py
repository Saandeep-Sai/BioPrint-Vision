"""
BioPrint Vision — Donor & Compatibility Blueprint
Handles interactive blood compatibility exploration and donor pre-screening questionnaires.
"""

from flask import Blueprint, render_template, request, flash
from services.compatibility import get_compatibility_info, COMPATIBILITY_RULES
from services.eligibility import evaluate_donor_eligibility

donor_bp = Blueprint("donor", __name__)


@donor_bp.route("/compatibility")
def compatibility_explorer():
    """Interactive compatibility guide for all 8 blood groups."""
    selected_bg = request.args.get("blood_group", "B+").strip().upper()
    info = get_compatibility_info(selected_bg)

    return render_template(
        "healthcare/compatibility.html",
        selected_bg=selected_bg,
        info=info,
        all_rules=COMPATIBILITY_RULES,
    )


@donor_bp.route("/donor-screening", methods=["GET", "POST"])
def donor_screening():
    """Donor pre-screening questionnaire with instant evaluation."""
    evaluation = None
    answers = {}

    if request.method == "POST":
        answers = {
            "age": request.form.get("age"),
            "weight": request.form.get("weight"),
            "hemoglobin": request.form.get("hemoglobin"),
            "days_since_last_donation": request.form.get("days_since_last_donation"),
            "recent_tattoo_or_piercing": "recent_tattoo" in request.form,
            "recent_major_surgery": "recent_surgery" in request.form,
            "active_infection_or_antibiotics": "active_infection" in request.form,
            "pregnant_or_lactating": "pregnant" in request.form,
            "chronic_conditions": "chronic" in request.form,
        }

        evaluation = evaluate_donor_eligibility(answers)

    return render_template(
        "healthcare/donor_screening.html",
        evaluation=evaluation,
        answers=answers,
    )
