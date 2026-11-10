"""
BioPrint Vision — Client Presentation Screenshot Capture
Captures high-resolution, full-fidelity screenshots of all primary outputs
and interfaces across the BioPrint Vision platform for client reporting.
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(r"d:\BioPrint-Vision\client_screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME_PATH = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
BROWSER_EXE = EDGE_PATH if os.path.exists(EDGE_PATH) else CHROME_PATH

BASE_URL = "http://127.0.0.1:5000"

PAGES_TO_CAPTURE = [
    {
        "id": "01_Landing_Hero_Platform",
        "url": f"{BASE_URL}/",
        "title": "BioPrint Vision Landing Page & System Overview",
        "scroll": False,
    },
    {
        "id": "02_Biometric_Scanner_Capture",
        "url": f"{BASE_URL}/analysis/new",
        "title": "Biometric Scanner Input & Hardware Driver Interface",
        "scroll": False,
    },
    {
        "id": "03_AI_Analysis_Output_A_Positive",
        "url": f"{BASE_URL}/analysis/20",
        "title": "Clinical Output: A+ Prediction, Grad-CAM Heatmap & Probabilities",
        "scroll": False,
    },
    {
        "id": "04_AI_Analysis_Output_B_Positive",
        "url": f"{BASE_URL}/analysis/15",
        "title": "Clinical Output: B+ Prediction & Compatibility Profile",
        "scroll": False,
    },
    {
        "id": "05_Patient_Analysis_History",
        "url": f"{BASE_URL}/history",
        "title": "Patient Biometric Analysis History & Clinical Audit Trail",
        "scroll": False,
    },
    {
        "id": "06_Blood_Inventory_Availability",
        "url": f"{BASE_URL}/availability",
        "title": "Real-Time Blood Bank Availability & Facility Stock Locator",
        "scroll": False,
    },
    {
        "id": "07_Clinical_Donor_Screening",
        "url": f"{BASE_URL}/donor/eligibility",
        "title": "Clinical Donor Screening & Transfusion Qualification",
        "scroll": False,
    },
    {
        "id": "08_Hospital_Emergency_Portal",
        "url": f"{BASE_URL}/hospital/dashboard",
        "title": "Hospital Emergency Portal & Critical Inventory Monitor",
        "scroll": False,
    },
    {
        "id": "09_Admin_Platform_Dashboard",
        "url": f"{BASE_URL}/admin/dashboard",
        "title": "Administrator Operations Dashboard & System Health",
        "scroll": False,
    },
    {
        "id": "10_AI_Clinical_Assistant",
        "url": f"{BASE_URL}/assistant",
        "title": "BioPrint AI Clinical Assistant & Transfusion Guidance",
        "scroll": False,
    },
]


def capture_all():
    print(f"[+] Starting client screenshot capture session...")
    print(f"[+] Browser Executable: {BROWSER_EXE}")
    print(f"[+] Target Output Directory: {OUTPUT_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=BROWSER_EXE,
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
            ],
        )

        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=1.25,
        )
        page = context.new_page()

        # Step 1: Login as Administrator to unlock all clinical, hospital, and admin interfaces
        print("[*] Authenticating as Administrator (admin@bioprint.local)...")
        page.goto(f"{BASE_URL}/login", timeout=20000)
        page.wait_for_load_state("networkidle")
        
        page.fill('input[name="email"]', "admin@bioprint.local")
        page.fill('input[name="password"]', "AdminPassword@123")
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
        time.sleep(1.5)
        print("[+] Logged in successfully!")

        captured_files = []

        # Step 2: Capture each designated view
        for item in PAGES_TO_CAPTURE:
            url = item["url"]
            file_name = f"{item['id']}.png"
            out_file = OUTPUT_DIR / file_name

            print(f"[*] Capturing: {item['title']} -> {url}")
            try:
                page.goto(url, timeout=25000)
                page.wait_for_load_state("networkidle")
                time.sleep(1.5)  # Allow charts and animations to render smoothly

                page.screenshot(
                    path=str(out_file),
                    full_page=False,
                )
                file_size_kb = out_file.stat().st_size / 1024
                print(f"    [OK] Saved {file_name} ({file_size_kb:.1f} KB)")
                captured_files.append((file_name, item["title"], file_size_kb))
            except Exception as e:
                print(f"    [FAIL] Could not capture {url}: {e}")

        browser.close()
        print(f"\n[+] Finished capturing {len(captured_files)} screenshots successfully!")
        return captured_files


if __name__ == "__main__":
    capture_all()
