"""
BioPrint Vision — Hospital Management Blueprint
Provides facility managers with tools to maintain blood stock levels and update facility profiles.
"""

from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, g
from routes.auth import role_required
from database.connection import db
from database.models import Facility, BloodInventory
from services.inventory import update_inventory_units
from config import Config

hospital_bp = Blueprint("hospital", __name__)


@hospital_bp.route("/hospital/dashboard")
@role_required("HOSPITAL", "ADMIN")
def dashboard():
    """Hospital manager dashboard showing inventory status and critical stock alerts."""
    user = g.user
    # Find facility associated with this user, or first facility
    facility = Facility.query.filter_by(user_id=user.id).first()
    if not facility:
        facility = Facility.query.first()

    if not facility:
        flash("No facility registered for this account.", "warning")
        return redirect(url_for("main.dashboard"))

    inventory = BloodInventory.query.filter_by(facility_id=facility.id).all()
    critical_items = [i for i in inventory if i.stock_status in ["Critical", "Unavailable"]]
    total_units = sum(i.units_available for i in inventory)

    return render_template(
        "hospital/dashboard.html",
        facility=facility,
        inventory=inventory,
        critical_items=critical_items,
        total_units=total_units,
    )


@hospital_bp.route("/hospital/inventory", methods=["GET", "POST"])
@role_required("HOSPITAL", "ADMIN")
def manage_inventory():
    """Update stock units for all 8 blood groups."""
    user = g.user
    facility = Facility.query.filter_by(user_id=user.id).first() or Facility.query.first()

    if not facility:
        flash("No facility found.", "warning")
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        for bg in ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]:
            field_name = f"units_{bg}"
            if field_name in request.form:
                try:
                    units = int(request.form.get(field_name, 0))
                    update_inventory_units(facility.id, bg, units)
                except ValueError:
                    pass

        flash("Blood inventory updated successfully!", "success")
        return redirect(url_for("hospital.manage_inventory"))

    inventory = BloodInventory.query.filter_by(facility_id=facility.id).all()
    return render_template("hospital/inventory.html", facility=facility, inventory=inventory)


@hospital_bp.route("/hospital/profile", methods=["GET", "POST"])
@role_required("HOSPITAL", "ADMIN")
def facility_profile():
    """Update facility contact and operational details."""
    user = g.user
    facility = Facility.query.filter_by(user_id=user.id).first() or Facility.query.first()

    if not facility:
        flash("No facility found.", "warning")
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        facility.name = request.form.get("name", facility.name).strip()
        facility.phone = request.form.get("phone", facility.phone).strip()
        facility.email = request.form.get("email", facility.email).strip()
        facility.address = request.form.get("address", facility.address).strip()
        facility.operating_hours = request.form.get("operating_hours", facility.operating_hours).strip()

        db.session.commit()
        flash("Facility profile updated successfully.", "success")
        return redirect(url_for("hospital.facility_profile"))

    return render_template("hospital/profile.html", facility=facility)
