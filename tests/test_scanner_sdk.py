"""
BioPrint Vision : Physical Scanner Plug-and-Play SDK Test Suite

Validates:
  1. Scanner Device Manager & Driver Registry
  2. Device Enumeration & Connectivity Status
  3. Simulator Sensor Capture, 500 DPI BMP Generation & Quality Assessment
  4. Device Capture & Disk Persistence
  5. Error Handling & Non-Existent Device Failover
  6. REST API: GET /api/scanner/devices
  7. REST API: POST /api/scanner/capture
  8. End-to-End Biometric Scan & Analysis Flow: POST /api/scanner/analyze
  9. Authentication Guardrails
"""

import os
import sys
import json
import unittest
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from run_flask import create_app
from database.connection import db
from database.models import User, Analysis, FingerprintResult
from database.seed_demo_data import seed_database
from services.scanner import (
    ScannerDeviceManager,
    get_scanner_manager,
    ScannerDeviceInfo,
    ScannerCaptureResult,
)
from config import Config

VALID_CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


class TestScannerSDK(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.app.test_client()

        with cls.app.app_context():
            if User.query.count() == 0:
                seed_database()

    def login_user(self):
        return self.client.post("/login", data={
            "email": "user@example.com",
            "password": "UserPassword@123",
        }, follow_redirects=True)

    def logout_user(self):
        return self.client.get("/logout", follow_redirects=True)

    def tearDown(self):
        self.logout_user()

    # ── 1. DRIVER & MANAGER UNIT TESTS ────────────────────────────────────────

    def test_01_device_manager_enumeration(self):
        """Verify device manager registers all supported drivers."""
        mgr = get_scanner_manager()
        devices = mgr.list_devices(probe_hardware=False)
        self.assertGreaterEqual(len(devices), 5)

        device_ids = [d.device_id for d in devices]
        self.assertIn("simulator_sensor", device_ids)
        self.assertIn("secugen_webapi", device_ids)
        self.assertIn("mantra_mfs100", device_ids)
        self.assertIn("digitalpersona_uareu", device_ids)
        self.assertIn("serial_optical_r307", device_ids)

        # Simulator should always report connected and ready
        sim_dev = next(d for d in devices if d.device_id == "simulator_sensor")
        self.assertTrue(sim_dev.is_connected)
        self.assertEqual(sim_dev.dpi, 500)

    def test_02_simulator_capture(self):
        """Verify simulator captures realistic 500 DPI BMP impression with quality score."""
        mgr = get_scanner_manager()
        res = mgr.capture("simulator_sensor", simulate_latency=False)

        self.assertTrue(res.success)
        self.assertIsNone(res.error_message)
        self.assertEqual(res.dpi, 500)
        self.assertEqual(res.raw_format, "BMP")
        self.assertGreaterEqual(res.width, 200)
        self.assertGreaterEqual(res.height, 200)
        self.assertGreaterEqual(res.quality_score, 0.40)
        self.assertIn(res.quality_status, ["Good", "Fair"])
        self.assertIsNotNone(res.pil_image)
        self.assertIsNotNone(res.image_bytes)

        # Verify BMP magic bytes 'BM'
        self.assertEqual(res.image_bytes[:2], b"BM")

    def test_03_capture_and_save_disk(self):
        """Verify capture_and_save persists valid BMP file to filesystem."""
        mgr = get_scanner_manager()
        res, saved_path = mgr.capture_and_save(
            device_id="simulator_sensor",
            simulate_latency=False,
        )

        self.assertTrue(res.success)
        self.assertIsNotNone(saved_path)
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.suffix.lower(), ".bmp")

        # Verify PIL can read the saved file
        with Image.open(saved_path) as img:
            self.assertEqual(img.format, "BMP")
            self.assertGreaterEqual(img.width, 200)

        # Clean up temp test capture
        if saved_path.exists():
            saved_path.unlink()

    def test_04_unknown_device_handling(self):
        """Verify requesting invalid device ID returns graceful error."""
        mgr = get_scanner_manager()
        res = mgr.capture("non_existent_device_xyz", simulate_latency=False)
        self.assertFalse(res.success)
        self.assertIn("No driver registered", res.error_message)

    # ── 2. REST API TESTS ─────────────────────────────────────────────────────

    def test_05_api_list_devices(self):
        """Verify GET /api/scanner/devices returns valid JSON with active device."""
        resp = self.client.get("/api/scanner/devices")
        self.assertEqual(resp.status_code, 200)

        data = json.loads(resp.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("active_device_id", data)
        self.assertIsInstance(data["devices"], list)
        self.assertGreaterEqual(len(data["devices"]), 5)

    def test_06_api_capture_live(self):
        """Verify POST /api/scanner/capture triggers sensor capture and returns Base64 BMP."""
        payload = {
            "device_id": "simulator_sensor",
            "timeout": 5.0,
        }
        resp = self.client.post(
            "/api/scanner/capture",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        data = json.loads(resp.data)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["device_id"], "simulator_sensor")
        self.assertIn("image_base64", data)
        self.assertTrue(data["image_base64"].startswith("data:image/bmp;base64,"))
        self.assertEqual(data["dpi"], 500)

    def test_07_api_analyze_unauthenticated_rejected(self):
        """Verify POST /api/scanner/analyze redirects unauthenticated requests."""
        resp = self.client.post(
            "/api/scanner/analyze",
            data=json.dumps({"device_id": "simulator_sensor"}),
            content_type="application/json",
            follow_redirects=False,
        )
        # Should redirect to login (302)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers.get("Location", ""))

    def test_08_api_scan_and_analyze_end_to_end(self):
        """Verify 1-click end-to-end scan -> inference -> DB record -> redirect flow."""
        self.login_user()

        payload = {
            "device_id": "simulator_sensor",
            "mode": "single",
        }
        resp = self.client.post(
            "/api/scanner/analyze",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        data = json.loads(resp.data)
        self.assertEqual(data["status"], "success")
        self.assertIn(data["final_blood_group"], VALID_CLASSES)
        self.assertGreater(data["final_confidence"], 0.0)
        self.assertIn("redirect_url", data)
        self.assertIn("analysis_id", data)

        # Verify analysis record was created in SQLite DB
        with self.app.app_context():
            rec = db.session.get(Analysis, data["analysis_id"])
            self.assertIsNotNone(rec)
            self.assertEqual(rec.final_blood_group, data["final_blood_group"])
            self.assertGreater(len(rec.fingerprints), 0)


if __name__ == "__main__":
    unittest.main()
