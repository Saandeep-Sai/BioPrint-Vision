"""
BioPrint Vision — End-to-End Platform Integration Test Suite
Validates the complete web platform:
  1. Unauthenticated & Authenticated Route Access
  2. Authentication & Session Management (User, Hospital, Admin)
  3. Single & Multi-Fingerprint Analysis Pipeline with PDF Generation
  4. Compatibility Matrix & Donor Pre-Screening Engine
  5. Blood Availability Directory & JSON Search APIs
  6. Hospital Management Portal & Inventory Updates
  7. Admin Dashboard, User Role Management, Facility Registration, & Rules
  8. Contextual AI Assistant Query Engine
  9. REST Analysis APIs
"""

import os
import sys
import io
import unittest
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from run_flask import create_app
from database.connection import db
from database.models import User, Facility, Analysis, BloodInventory, Report
from database.seed_demo_data import seed_database
from config import Config


class TestBioPrintPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configure app for testing
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.app.test_client()

        with cls.app.app_context():
            # Seed demo data if database is empty
            if User.query.count() == 0:
                seed_database()

    def login(self, email, password):
        """Helper to log in via form POST."""
        return self.client.post("/login", data={
            "email": email,
            "password": password
        }, follow_redirects=True)

    def logout(self):
        """Helper to log out."""
        return self.client.get("/logout", follow_redirects=True)

    def tearDown(self):
        """Ensure clean session after every test."""
        self.logout()

    # ── 1. PUBLIC ROUTES ────────────────────────────────────────────────────────

    def test_01_public_pages(self):
        """Verify home, about, and compatibility matrix are publicly accessible."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"BioPrint Vision", resp.data)

        resp = self.client.get("/about")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"ResNet-50", resp.data)

        resp = self.client.get("/compatibility?blood_group=A+")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Red Blood Cell", resp.data)

    def test_02_auth_protection(self):
        """Verify protected routes redirect unauthenticated users to login."""
        protected_routes = [
            "/dashboard",
            "/analysis/new",
            "/history",
            "/reports",
            "/hospital/dashboard",
            "/admin/dashboard"
        ]
        for route in protected_routes:
            resp = self.client.get(route, follow_redirects=False)
            self.assertEqual(resp.status_code, 302, f"Route {route} should redirect to /login")
            self.assertIn("/login", resp.headers.get("Location", ""))

    # ── 2. AUTHENTICATION & ROLES ───────────────────────────────────────────────

    def test_03_login_user(self):
        """Verify patient/standard user login and dashboard access."""
        resp = self.login("user@example.com", "UserPassword@123")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Demo Patient / Volunteer", resp.data)

        # Standard user attempting admin dashboard should be blocked / redirected
        admin_resp = self.client.get("/admin/dashboard", follow_redirects=True)
        self.assertIn(b"Access denied", admin_resp.data)

    def test_04_login_hospital(self):
        """Verify hospital role login and inventory management access."""
        resp = self.login("hospital@citygeneral.local", "HospitalPassword@123")
        self.assertEqual(resp.status_code, 200)

        # Hospital portal access
        dash = self.client.get("/hospital/dashboard")
        self.assertEqual(dash.status_code, 200)
        self.assertIn(b"Hospital Portal", dash.data)

        self.logout()

    def test_05_login_admin(self):
        """Verify admin login and access to admin suite."""
        resp = self.login("admin@bioprint.local", "AdminPassword@123")
        self.assertEqual(resp.status_code, 200)

        dash = self.client.get("/admin/dashboard")
        self.assertEqual(dash.status_code, 200)
        self.assertIn(b"System Administration", dash.data)

        self.logout()

    # ── 3. DONOR PRE-SCREENING ──────────────────────────────────────────────────

    def test_06_donor_prescreening(self):
        """Test donor eligibility questionnaire rules."""
        # Test eligible candidate
        resp = self.client.post("/donor-screening", data={
            "age": "26",
            "weight": "72",
            "hemoglobin": "14.5",
            "days_since_last_donation": "100"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Potentially Eligible", resp.data)

        # Test ineligible candidate (underweight + recent tattoo)
        resp2 = self.client.post("/donor-screening", data={
            "age": "20",
            "weight": "42",
            "recent_tattoo": "on"
        })
        self.assertEqual(resp2.status_code, 200)
        self.assertIn(b"Not Eligible Based on Screening", resp2.data)

    # ── 4. BLOOD AVAILABILITY & DIRECTORY ───────────────────────────────────────

    def test_07_availability_search(self):
        """Test directory browsing and JSON API."""
        resp = self.client.get("/availability?blood_group=O+")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Search Results", resp.data)
        self.assertIn(b"Demo", resp.data)

        # Test JSON API
        api_resp = self.client.get("/api/availability?blood_group=B+")
        self.assertEqual(api_resp.status_code, 200)
        json_data = api_resp.get_json()
        self.assertIn("results", json_data)
        self.assertIn("demo_notice", json_data)

    # ── 5. FULL ANALYSIS WORKFLOW WITH SAMPLE IMAGE ────────────────────────────

    def test_08_end_to_end_analysis_and_pdf(self):
        """Upload a sample fingerprint, trigger ML prediction, and verify PDF creation."""
        self.login("user@example.com", "UserPassword@123")

        # Locate sample fingerprint from dataset
        sample_path = BASE_DIR / "data" / "raw" / "B+" / "cluster_2_10.BMP"
        if not sample_path.exists():
            # Fallback to any .BMP in data/raw
            bmps = list((BASE_DIR / "data" / "raw").rglob("*.BMP"))
            self.assertTrue(len(bmps) > 0, "No sample BMP images found in dataset")
            sample_path = bmps[0]

        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        # Submit analysis form
        post_data = {
            "fingerprint_image": (io.BytesIO(img_bytes), "test_fingerprint.bmp"),
            "mode": "auto"
        }
        resp = self.client.post(
            "/analysis/new",
            data=post_data,
            content_type="multipart/form-data",
            follow_redirects=True
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Biometric Blood Group Analysis", resp.data)
        self.assertIn(b"Predicted Blood Group", resp.data)

        # Verify DB record created
        with self.app.app_context():
            latest_analysis = Analysis.query.order_by(Analysis.id.desc()).first()
            self.assertIsNotNone(latest_analysis)
            self.assertIn(latest_analysis.final_blood_group, ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"])
            self.assertGreater(latest_analysis.final_confidence, 0.0)

            # Check Report created
            report = Report.query.filter_by(analysis_id=latest_analysis.id).first()
            self.assertIsNotNone(report)
            pdf_file = Config.BASE_DIR / "static" / report.file_path
            self.assertTrue(pdf_file.exists(), f"PDF report {pdf_file} should exist on disk")

            analysis_id = latest_analysis.id
            report_id = report.id

        # Test View Analysis page
        view_resp = self.client.get(f"/analysis/{analysis_id}")
        self.assertEqual(view_resp.status_code, 200)
        self.assertIn(b"RESEARCH PROTOTYPE DISCLAIMER", view_resp.data)

        # Test PDF Report Download
        dl_resp = self.client.get(f"/reports/{report_id}/download")
        self.assertEqual(dl_resp.status_code, 200)
        self.assertEqual(dl_resp.content_type, "application/pdf")
        self.assertTrue(len(dl_resp.data) > 1000)

        # Test History Page
        hist_resp = self.client.get("/history")
        self.assertEqual(hist_resp.status_code, 200)
        self.assertIn(b"Fingerprint Analysis History", hist_resp.data)

        # Test AI Assistant with this analysis ID attached
        ai_resp = self.client.post("/api/assistant/chat", json={
            "message": "What is my predicted blood group and who can I donate to?",
            "analysis_id": analysis_id
        })
        self.assertEqual(ai_resp.status_code, 200)
        ai_data = ai_resp.get_json()
        self.assertIn("reply", ai_data)
        self.assertTrue(len(ai_data["reply"]) > 20)

        self.logout()

    # ── 6. HOSPITAL PORTAL ACTIONS ─────────────────────────────────────────────

    def test_09_hospital_inventory_update(self):
        """Test hospital stock update through form POST."""
        self.login("hospital@citygeneral.local", "HospitalPassword@123")

        update_resp = self.client.post("/hospital/inventory", data={
            "units_A+": "18",
            "units_A-": "6",
            "units_AB+": "12",
            "units_AB-": "2",
            "units_B+": "25",
            "units_B-": "4",
            "units_O+": "30",
            "units_O-": "1",
        }, follow_redirects=True)
        self.assertEqual(update_resp.status_code, 200)
        self.assertIn(b"Blood inventory updated successfully", update_resp.data)

        # Verify DB changes
        with self.app.app_context():
            facility = Facility.query.filter_by(name="City General Hospital").first()
            if facility:
                inv_o_minus = BloodInventory.query.filter_by(facility_id=facility.id, blood_group="O-").first()
                self.assertEqual(inv_o_minus.units_available, 1)
                self.assertEqual(inv_o_minus.stock_status, "Low Stock")

        self.logout()

    # ── 7. ADMIN SUITE ACTIONS ─────────────────────────────────────────────────

    def test_10_admin_operations(self):
        """Test admin user role modifications and rules updates."""
        self.login("admin@bioprint.local", "AdminPassword@123")

        # 1. View Users
        users_resp = self.client.get("/admin/users")
        self.assertEqual(users_resp.status_code, 200)
        self.assertIn(b"Platform User Accounts", users_resp.data)

        # 2. Add New Medical Facility
        add_fac_resp = self.client.post("/admin/facilities/add", data={
            "name": "St. Jude Metro Blood Bank",
            "type": "Blood Bank",
            "address": "45 Park Street, Central District",
            "city": "Kolkata",
            "state": "West Bengal",
            "pincode": "700016",
            "phone": "+91 33 2222 3333",
            "email": "contact@stjude.local",
            "operating_hours": "24/7"
        }, follow_redirects=True)
        self.assertEqual(add_fac_resp.status_code, 200)
        self.assertIn(b"created with 8 stock inventory records", add_fac_resp.data)

        # 3. View All Analyses
        all_ana_resp = self.client.get("/admin/analyses")
        self.assertEqual(all_ana_resp.status_code, 200)
        self.assertIn(b"System-Wide Analysis Audit Log", all_ana_resp.data)

        # 4. Update Rules
        rules_resp = self.client.post("/admin/rules", data={
            "rare_groups": ["AB-", "B-", "O-"],
            "avail_threshold": 10,
            "low_threshold": 2
        }, follow_redirects=True)
        self.assertEqual(rules_resp.status_code, 200)
        self.assertIn(b"System rules updated successfully", rules_resp.data)

        self.logout()

    # ── 8. REST ANALYSIS APIS ──────────────────────────────────────────────────

    def test_11_rest_analysis_apis(self):
        """Test headless POST /api/analyze/single and /api/analyze/multiple."""
        sample_path = BASE_DIR / "data" / "raw" / "A+" / "cluster_0_1001.BMP"
        if not sample_path.exists():
            bmps = list((BASE_DIR / "data" / "raw").rglob("*.BMP"))
            sample_path = bmps[0]

        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        # Single API
        resp = self.client.post("/api/analyze/single", data={
            "image": (io.BytesIO(img_bytes), "sample.bmp")
        }, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertIn("quality", data)

        # Multiple API
        resp_m = self.client.post("/api/analyze/multiple", data={
            "image": (io.BytesIO(img_bytes), "sample.bmp")
        }, content_type="multipart/form-data")
        self.assertEqual(resp_m.status_code, 200)
        data_m = resp_m.get_json()
        self.assertIn("final_prediction", data_m)
        self.assertIn("agreement", data_m)


if __name__ == "__main__":
    unittest.main()
