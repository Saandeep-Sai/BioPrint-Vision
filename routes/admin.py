"""
BioPrint Vision — Admin Blueprint
Provides platform management: user management, facility registry, audit logs, and configuration.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, g
from routes.auth import role_required
from database.connection import db
from database.models import User, Facility, Analysis, BloodInventory, Report
from services.inventory import determine_stock_status
from config import Config

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/dashboard")
@role_required("ADMIN")
def dashboard():
    """Admin overview dashboard."""
    stats = {
        "users_count": User.query.count(),
        "facilities_count": Facility.query.count(),
        "analyses_count": Analysis.query.count(),
        "reports_count": Report.query.count(),
    }
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(5).all()

    return render_template(
        "admin/dashboard.html",
        stats=stats,
        recent_users=recent_users,
        recent_analyses=recent_analyses,
    )


@admin_bp.route("/admin/users", methods=["GET", "POST"])
@role_required("ADMIN")
def manage_users():
    """List and manage user accounts and role permissions."""
    if request.method == "POST":
        user_id = request.form.get("user_id", type=int)
        new_role = request.form.get("role")
        new_status = request.form.get("status")

        user = User.query.get(user_id)
        if user:
            if new_role in ["USER", "HOSPITAL", "ADMIN"]:
                user.role = new_role
            if new_status in ["ACTIVE", "INACTIVE", "SUSPENDED"]:
                user.status = new_status
            db.session.commit()
            flash(f"Updated account for {user.email}.", "success")
        return redirect(url_for("admin.manage_users"))

    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/users.html", users=users)


@admin_bp.route("/admin/facilities")
@role_required("ADMIN")
def manage_facilities():
    """List and manage registered medical facilities."""
    facilities = Facility.query.order_by(Facility.created_at.desc()).all()
    return render_template("admin/facilities.html", facilities=facilities)


@admin_bp.route("/admin/facilities/add", methods=["GET", "POST"])
@role_required("ADMIN")
def add_facility():
    """Register a new medical facility or blood bank with default 8 inventory groups."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        fac_type = request.form.get("type", "Hospital").strip()
        address = request.form.get("address", "").strip()
        city = request.form.get("city", "").strip()
        state = request.form.get("state", "").strip()
        pincode = request.form.get("pincode", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        hours = request.form.get("operating_hours", "24/7").strip()

        if not name or not address or not city or not phone:
            flash("Facility name, address, city, and phone number are required.", "danger")
            return render_template("admin/add_facility.html")

        new_fac = Facility(
            name=name,
            type=fac_type,
            address=address,
            city=city,
            state=state,
            pincode=pincode,
            phone=phone,
            email=email,
            operating_hours=hours,
            is_demo=True,
            status="ACTIVE",
            user_id=g.user.id,
        )
        db.session.add(new_fac)
        db.session.flush()

        # Initialize all 8 blood inventory slots with default 10 units
        for bg in ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]:
            inv = BloodInventory(
                facility_id=new_fac.id,
                blood_group=bg,
                units_available=10,
                stock_status="Low Stock",
                is_demo=True,
            )
            db.session.add(inv)

        db.session.commit()
        flash(f"Facility '{name}' created with 8 stock inventory records.", "success")
        return redirect(url_for("admin.manage_facilities"))

    return render_template("admin/add_facility.html")


@admin_bp.route("/admin/analyses")
@role_required("ADMIN")
def list_all_analyses():
    """System-wide analysis audit history."""
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(50).all()
    return render_template("admin/analyses.html", analyses=analyses)


@admin_bp.route("/admin/rules", methods=["GET", "POST"])
@role_required("ADMIN")
def manage_rules():
    """Configure business rules (rare blood groups, stock thresholds)."""
    if request.method == "POST":
        rare_groups = request.form.getlist("rare_groups")
        if rare_groups:
            Config.RARE_BLOOD_GROUPS = rare_groups

        avail_thresh = request.form.get("avail_threshold", type=int)
        low_thresh = request.form.get("low_threshold", type=int)
        if avail_thresh is not None:
            Config.STOCK_THRESHOLD_AVAILABLE = avail_thresh
        if low_thresh is not None:
            Config.STOCK_THRESHOLD_LOW = low_thresh

        flash("System rules updated successfully.", "success")
        return redirect(url_for("admin.manage_rules"))

    return render_template(
        "admin/rules.html",
        rare_groups=Config.RARE_BLOOD_GROUPS,
        avail_thresh=Config.STOCK_THRESHOLD_AVAILABLE,
        low_thresh=Config.STOCK_THRESHOLD_LOW,
        all_groups=["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"],
    )
