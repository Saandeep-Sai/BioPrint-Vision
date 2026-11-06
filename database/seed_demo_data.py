"""
BioPrint Vision — Seed Demo Data Script
Populates the database with realistic demonstration data:
  - Demo Users (Admin, Hospital, Regular User)
  - Demo Facilities (Hospitals, Blood Banks, Blood Centres)
  - Demo Blood Inventory (All 8 blood groups with realistic stock)

All facilities and inventory records are explicitly tagged with `is_demo=True`.
Run via CLI:
    python -m database.seed_demo_data
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from flask import Flask
from config import Config
from database.connection import db, init_db
from database.models import User, Facility, BloodInventory

BLOOD_GROUPS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

DEMO_USERS = [
    {
        "name": "System Administrator",
        "email": "admin@bioprint.local",
        "password": "AdminPassword@123",
        "role": "ADMIN",
    },
    {
        "name": "City General Hospital Admin",
        "email": "hospital@citygeneral.local",
        "password": "HospitalPassword@123",
        "role": "HOSPITAL",
    },
    {
        "name": "Metro Blood Bank Staff",
        "email": "bloodbank@metro.local",
        "password": "BloodBankPassword@123",
        "role": "HOSPITAL",
    },
    {
        "name": "Demo Patient / Volunteer",
        "email": "user@example.com",
        "password": "UserPassword@123",
        "role": "USER",
    },
]

DEMO_FACILITIES = [
    {
        "name": "City General Hospital & Trauma Centre",
        "type": "Hospital",
        "address": "45 Healthcare Boulevard, Bandra West",
        "city": "Mumbai",
        "state": "Maharashtra",
        "pincode": "400050",
        "latitude": 19.0596,
        "longitude": 72.8295,
        "phone": "+91 22 2640 1000",
        "email": "bloodbank@citygeneral.local",
        "operating_hours": "24/7",
        "inventory": {
            "A+": 24, "A-": 6, "B+": 32, "B-": 4, "AB+": 12, "AB-": 2, "O+": 45, "O-": 8
        },
    },
    {
        "name": "Red Cross Regional Blood Bank",
        "type": "Blood Bank",
        "address": "12 Red Cross Road, Connaught Place",
        "city": "Delhi",
        "state": "Delhi",
        "pincode": "110001",
        "latitude": 28.6289,
        "longitude": 77.2065,
        "phone": "+91 11 2371 6441",
        "email": "donations@redcrossdelhi.local",
        "operating_hours": "08:00 AM - 08:00 PM",
        "inventory": {
            "A+": 40, "A-": 15, "B+": 55, "B-": 12, "AB+": 20, "AB-": 5, "O+": 68, "O-": 18
        },
    },
    {
        "name": "Apollo Life & Research Hospital",
        "type": "Hospital",
        "address": "154/11 Bannerghatta Main Road",
        "city": "Bangalore",
        "state": "Karnataka",
        "pincode": "560076",
        "latitude": 12.8954,
        "longitude": 77.5988,
        "phone": "+91 80 2630 4050",
        "email": "emergency@apolloblr.local",
        "operating_hours": "24/7",
        "inventory": {
            "A+": 18, "A-": 4, "B+": 22, "B-": 3, "AB+": 8, "AB-": 1, "O+": 30, "O-": 4
        },
    },
    {
        "name": "Narayana Blood Centre",
        "type": "Blood Centre",
        "address": "258/A Bommasandra Industrial Area",
        "city": "Bangalore",
        "state": "Karnataka",
        "pincode": "560099",
        "latitude": 12.8123,
        "longitude": 77.6890,
        "phone": "+91 80 7122 2222",
        "email": "info@narayanablood.local",
        "operating_hours": "24/7",
        "inventory": {
            "A+": 35, "A-": 10, "B+": 48, "B-": 8, "AB+": 15, "AB-": 3, "O+": 52, "O-": 11
        },
    },
    {
        "name": "Metropolis Multi-Speciality Blood Bank",
        "type": "Blood Bank",
        "address": "88 Jubilee Hills Road No. 36",
        "city": "Hyderabad",
        "state": "Telangana",
        "pincode": "500033",
        "latitude": 17.4319,
        "longitude": 78.4073,
        "phone": "+91 40 2355 7799",
        "email": "support@metropolisblood.local",
        "operating_hours": "07:00 AM - 10:00 PM",
        "inventory": {
            "A+": 16, "A-": 2, "B+": 28, "B-": 5, "AB+": 7, "AB-": 0, "O+": 38, "O-": 6
        },
    },
    {
        "name": "Ruby Hall Clinic & Transfusion Centre",
        "type": "Hospital",
        "address": "40 Sassoon Road, Sangamvadi",
        "city": "Pune",
        "state": "Maharashtra",
        "pincode": "411001",
        "latitude": 18.5308,
        "longitude": 73.8777,
        "phone": "+91 20 6645 5100",
        "email": "transfusion@rubyhall.local",
        "operating_hours": "24/7",
        "inventory": {
            "A+": 20, "A-": 5, "B+": 25, "B-": 4, "AB+": 9, "AB-": 2, "O+": 34, "O-": 5
        },
    },
    {
        "name": "Fortis Malar Blood Centre",
        "type": "Blood Centre",
        "address": "52 1st Main Road, Gandhi Nagar, Adyar",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "pincode": "600020",
        "latitude": 13.0067,
        "longitude": 80.2575,
        "phone": "+91 44 4289 2222",
        "email": "blood@fortismalar.local",
        "operating_hours": "24/7",
        "inventory": {
            "A+": 22, "A-": 7, "B+": 30, "B-": 6, "AB+": 11, "AB-": 3, "O+": 41, "O-": 9
        },
    },
    {
        "name": "Heritage Health City Blood Bank",
        "type": "Blood Bank",
        "address": "17 Park Street, Taltala",
        "city": "Kolkata",
        "state": "West Bengal",
        "pincode": "700016",
        "latitude": 22.5511,
        "longitude": 88.3524,
        "phone": "+91 33 2229 5555",
        "email": "heritage@kolkatablood.local",
        "operating_hours": "08:00 AM - 09:00 PM",
        "inventory": {
            "A+": 19, "A-": 3, "B+": 36, "B-": 7, "AB+": 10, "AB-": 1, "O+": 44, "O-": 4
        },
    },
]


def determine_stock_status(units: int) -> str:
    """Classify units into Available, Low Stock, Critical, or Unavailable."""
    if units >= Config.STOCK_THRESHOLD_AVAILABLE:
        return "Available"
    elif units >= Config.STOCK_THRESHOLD_LOW:
        return "Low Stock"
    elif units > 0:
        return "Critical"
    else:
        return "Unavailable"


def seed_database():
    """Populate database with demo users, facilities, and inventory."""
    app = Flask(__name__)
    app.config.from_object(Config)
    init_db(app)

    with app.app_context():
        print("\n" + "=" * 60)
        print("  BIOPRINT VISION -- SEEDING DEMO DATABASE")
        print("=" * 60)

        # 1. Seed Demo Users
        print("\n[1/3] Seeding demo users...")
        for u_data in DEMO_USERS:
            existing = User.query.filter_by(email=u_data["email"]).first()
            if not existing:
                user = User(
                    name=u_data["name"],
                    email=u_data["email"],
                    role=u_data["role"],
                    status="ACTIVE",
                )
                user.set_password(u_data["password"])
                db.session.add(user)
                print(f"  + Added User: {u_data['email']} ({u_data['role']})")
            else:
                print(f"  * User already exists: {u_data['email']}")

        db.session.commit()

        # Find a hospital user to associate
        hospital_user = User.query.filter_by(role="HOSPITAL").first()
        h_uid = hospital_user.id if hospital_user else None

        # 2. Seed Demo Facilities and Inventory
        print("\n[2/3] Seeding demo facilities and blood inventory...")
        for fac_data in DEMO_FACILITIES:
            existing_fac = Facility.query.filter_by(name=fac_data["name"]).first()
            if not existing_fac:
                facility = Facility(
                    name=fac_data["name"],
                    type=fac_data["type"],
                    address=fac_data["address"],
                    city=fac_data["city"],
                    state=fac_data["state"],
                    pincode=fac_data["pincode"],
                    latitude=fac_data["latitude"],
                    longitude=fac_data["longitude"],
                    phone=fac_data["phone"],
                    email=fac_data["email"],
                    operating_hours=fac_data["operating_hours"],
                    is_demo=True,
                    status="ACTIVE",
                    user_id=h_uid,
                )
                db.session.add(facility)
                db.session.flush()  # to obtain facility.id

                # Add inventory for all 8 blood groups
                for bg in BLOOD_GROUPS:
                    units = fac_data["inventory"].get(bg, 0)
                    status = determine_stock_status(units)
                    inv = BloodInventory(
                        facility_id=facility.id,
                        blood_group=bg,
                        units_available=units,
                        stock_status=status,
                        last_updated=datetime.utcnow(),
                        is_demo=True,
                    )
                    db.session.add(inv)

                print(f"  + Added Facility: {fac_data['name']} ({fac_data['city']}) with 8 stock records")
            else:
                print(f"  * Facility already exists: {fac_data['name']}")

        db.session.commit()
        print("\n[3/3] Demo data seed complete!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    seed_database()
