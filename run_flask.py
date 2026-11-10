"""
BioPrint Vision : Full Flask Application Server
Launches the complete multi-user healthcare platform:
  - Authentication & Role-Based Access (User, Hospital, Admin)
  - Single & Multi-Fingerprint Automated Analysis
  - Probability Vector Averaging & Agreement Scoring
  - ABO/Rh Compatibility Engine & Donor Pre-Screening
  - Blood Availability Search & Facility Directory
  - Hospital Inventory Management Portal
  - Contextual AI Assistant
  - Smart PDF Report Generation
  - Admin Management Dashboard
  - Physical & Simulated Fingerprint Scanner SDK Integration

Launch via:
    python run_flask.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from flask import Flask, render_template
from config import Config, init_app_dirs
from database.connection import db, init_db
from database.models import User
from database.seed_demo_data import seed_database
from routes import (
    auth_bp,
    main_bp,
    analysis_bp,
    availability_bp,
    donor_bp,
    reports_bp,
    assistant_bp,
    hospital_bp,
    admin_bp,
    scanner_bp,
)


def create_app(config_class=Config):
    """Application factory for BioPrint Vision."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure upload directories exist
    init_app_dirs(app)

    # Initialize Database
    init_db(app)

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(availability_bp)
    app.register_blueprint(donor_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(assistant_bp)
    app.register_blueprint(hospital_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(scanner_bp)

    # Custom Jinja Filters
    @app.template_filter("badge_color")
    def badge_color(status):
        status_map = {
            "Available": "success",
            "Low Stock": "warning",
            "Critical": "danger",
            "Unavailable": "secondary",
            "Good": "success",
            "Fair": "warning",
            "Poor": "danger",
            "Potentially Eligible": "success",
            "Medical Review Required": "warning",
            "Not Eligible Based on Screening": "danger",
            "ACTIVE": "success",
            "INACTIVE": "secondary",
        }
        return status_map.get(str(status), "primary")

    # Error Handlers (never expose raw stack traces)
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template("errors/500.html"), 500

    return app


if __name__ == "__main__":
    app = create_app()

    # Automatically check if demo data is present; if not, seed it
    with app.app_context():
        if User.query.count() == 0:
            print("[INFO] No users found in database. Seeding initial demo data...")
            seed_database()

    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"

    print("\n" + "=" * 65)
    print("  [+] BIOPRINT VISION -- FLASK HEALTHCARE PLATFORM")
    print("=" * 65)
    print(f"  Server URL: http://127.0.0.1:{port}")
    print("  Demo Logins:")
    print("    [-] Admin:    admin@bioprint.local      / AdminPassword@123")
    print("    [-] Hospital: hospital@citygeneral.local / HospitalPassword@123")
    print("    [-] User:     user@example.com          / UserPassword@123")
    print("=" * 65 + "\n")

    app.run(host="127.0.0.1", port=port, debug=debug_mode)
