"""
BioPrint Vision — Authentication Blueprint
Handles user registration, login, logout, profile updates, and role-based access control.
"""

from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, g
from database.connection import db
from database.models import User

auth_bp = Blueprint("auth", __name__)


def get_current_user():
    """Retrieve logged-in user from session, or None."""
    user_id = session.get("user_id")
    if user_id:
        return db.session.get(User, user_id)
    return None


def login_required(f):
    """Decorator to protect routes requiring authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def role_required(*allowed_roles):
    """Decorator to enforce role-based access control."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                flash("Please sign in to access this page.", "warning")
                return redirect(url_for("auth.login", next=request.path))

            user = get_current_user()
            if not user or user.role not in allowed_roles:
                flash("Access denied: You do not possess the required authorization for this area.", "danger")
                return redirect(url_for("main.dashboard"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@auth_bp.before_app_request
def load_logged_in_user():
    """Attach current user to Flask global `g` for templates and routes."""
    g.user = get_current_user()


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        role = request.form.get("role", "USER").strip().upper()

        if role not in ["USER", "HOSPITAL"]:
            role = "USER"

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("auth/register.html", name=name, email=email, role=role)

        if len(password) < 6:
            flash("Password must be at least 6 characters in length.", "danger")
            return render_template("auth/register.html", name=name, email=email, role=role)

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("auth/register.html", name=name, email=email, role=role)

        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("An account with this email address already exists. Please log in.", "warning")
            return redirect(url_for("auth.login"))

        new_user = User(name=name, email=email, role=role, status="ACTIVE")
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! You may now sign in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET" and g.user:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        next_url = request.args.get("next") or request.form.get("next")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            if user.status != "ACTIVE":
                flash("Your account has been deactivated. Please contact administrator.", "danger")
                return render_template("auth/login.html", email=email)

            session.clear()
            session["user_id"] = user.id
            session["user_role"] = user.role
            session["user_name"] = user.name

            flash(f"Welcome back, {user.name}!", "success")

            if next_url and next_url.startswith("/"):
                return redirect(next_url)

            if user.role == "ADMIN":
                return redirect(url_for("admin.dashboard"))
            elif user.role == "HOSPITAL":
                return redirect(url_for("hospital.dashboard"))
            return redirect(url_for("main.dashboard"))
        else:
            flash("Invalid email or password. Please try again.", "danger")
            return render_template("auth/login.html", email=email)

    return render_template("auth/login.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("main.index"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = g.user
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        new_password = request.form.get("new_password", "")

        if name:
            user.name = name

        if new_password:
            if len(new_password) < 6:
                flash("New password must be at least 6 characters long.", "danger")
                return render_template("auth/profile.html", user=user)
            user.set_password(new_password)
            flash("Password updated successfully.", "success")

        db.session.commit()
        session["user_name"] = user.name
        flash("Profile updated successfully.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("auth/profile.html", user=user)
