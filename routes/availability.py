"""
BioPrint Vision — Availability Blueprint
Handles hospital/blood-bank directory browsing, stock filtering, and facility detail views.
"""

from flask import Blueprint, render_template, request, jsonify
from database.models import Facility, BloodInventory
from services.inventory import search_blood_availability
from config import Config

availability_bp = Blueprint("availability", __name__)


@availability_bp.route("/availability")
def browse_availability():
    """Browse and filter blood facilities and stock."""
    blood_group = request.args.get("blood_group", "")
    city = request.args.get("city", "")
    facility_type = request.args.get("type", "")
    stock_status = request.args.get("status", "")

    # Retrieve all distinct cities for dropdown filter
    all_cities = [c[0] for c in Facility.query.with_entities(Facility.city).distinct().all()]

    search_data = search_blood_availability(
        blood_group=blood_group if blood_group else None,
        city=city if city else None,
        facility_type=facility_type if facility_type else None,
        stock_status=stock_status if stock_status else None,
    )

    return render_template(
        "availability/search.html",
        results=search_data["results"],
        total_count=search_data["count"],
        demo_notice=search_data["demo_notice"],
        cities=all_cities,
        selected_bg=blood_group,
        selected_city=city,
        selected_type=facility_type,
        selected_status=stock_status,
    )


@availability_bp.route("/facilities/<int:facility_id>")
def facility_detail(facility_id: int):
    """View full profile of a facility including all 8 blood inventory levels."""
    facility = Facility.query.get_or_404(facility_id)
    inventory = BloodInventory.query.filter_by(facility_id=facility.id).all()

    return render_template(
        "availability/facility_detail.html",
        facility=facility,
        inventory=inventory,
        demo_notice="Demo availability data — not connected to live hospital inventory.",
    )


@availability_bp.route("/api/availability")
def api_availability():
    """JSON API for blood stock search."""
    bg = request.args.get("blood_group")
    city = request.args.get("city")
    ftype = request.args.get("type")
    status = request.args.get("status")

    data = search_blood_availability(blood_group=bg, city=city, facility_type=ftype, stock_status=status)
    return jsonify(data)
