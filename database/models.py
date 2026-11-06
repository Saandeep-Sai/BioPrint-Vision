"""
BioPrint Vision — Database Models Module
Defines SQLAlchemy ORM models corresponding to the PRD specifications:
  - users
  - facilities
  - blood_inventory
  - analyses
  - fingerprint_results
  - reports
"""

import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from database.connection import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="USER", index=True)  # USER, HOSPITAL, ADMIN
    status = db.Column(db.String(20), nullable=False, default="ACTIVE")          # ACTIVE, INACTIVE, SUSPENDED
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    analyses = db.relationship("Analysis", backref="user", lazy=True, cascade="all, delete-orphan")
    facilities = db.relationship("Facility", backref="owner", lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Facility(db.Model):
    __tablename__ = "facilities"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(180), nullable=False, index=True)
    type = db.Column(db.String(50), nullable=False, default="Hospital", index=True)  # Hospital, Blood Bank, Blood Centre
    address = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(100), nullable=False, index=True)
    state = db.Column(db.String(100), nullable=False)
    pincode = db.Column(db.String(20), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    phone = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    operating_hours = db.Column(db.String(120), nullable=False, default="24/7")
    is_demo = db.Column(db.Boolean, nullable=False, default=True, index=True)
    status = db.Column(db.String(20), nullable=False, default="ACTIVE", index=True)  # ACTIVE, PENDING, INACTIVE
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inventory = db.relationship("BloodInventory", backref="facility", lazy=True, cascade="all, delete-orphan")

    def to_dict(self, include_inventory: bool = False):
        data = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "pincode": self.pincode,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "phone": self.phone,
            "email": self.email,
            "operating_hours": self.operating_hours,
            "is_demo": self.is_demo,
            "status": self.status,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_inventory:
            data["inventory"] = [item.to_dict() for item in self.inventory]
        return data


class BloodInventory(db.Model):
    __tablename__ = "blood_inventory"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    facility_id = db.Column(db.Integer, db.ForeignKey("facilities.id", ondelete="CASCADE"), nullable=False, index=True)
    blood_group = db.Column(db.String(10), nullable=False, index=True)  # A+, A-, AB+, AB-, B+, B-, O+, O-
    units_available = db.Column(db.Integer, nullable=False, default=0)
    stock_status = db.Column(db.String(20), nullable=False, default="Available", index=True)  # Available, Low Stock, Critical, Unavailable
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_demo = db.Column(db.Boolean, nullable=False, default=True)

    def to_dict(self):
        return {
            "id": self.id,
            "facility_id": self.facility_id,
            "facility_name": self.facility.name if self.facility else None,
            "facility_city": self.facility.city if self.facility else None,
            "facility_phone": self.facility.phone if self.facility else None,
            "blood_group": self.blood_group,
            "units_available": self.units_available,
            "stock_status": self.stock_status,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_demo": self.is_demo,
        }


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    final_blood_group = db.Column(db.String(10), nullable=False, index=True)
    final_confidence = db.Column(db.Float, nullable=False)
    fingerprints_detected = db.Column(db.Integer, nullable=False, default=1)
    fingerprints_analyzed = db.Column(db.Integer, nullable=False, default=1)
    fingerprints_excluded = db.Column(db.Integer, nullable=False, default=0)
    agreement_percentage = db.Column(db.Float, nullable=False, default=100.0)
    mode = db.Column(db.String(20), nullable=False, default="single")  # single, multiple
    image_path = db.Column(db.String(255), nullable=True)
    annotated_image_path = db.Column(db.String(255), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    fingerprints = db.relationship("FingerprintResult", backref="analysis", lazy=True, cascade="all, delete-orphan")
    reports = db.relationship("Report", backref="analysis", lazy=True, cascade="all, delete-orphan")

    def to_dict(self, include_fingerprints: bool = True):
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else "Anonymous",
            "final_blood_group": self.final_blood_group,
            "final_confidence": round(self.final_confidence, 2),
            "fingerprints_detected": self.fingerprints_detected,
            "fingerprints_analyzed": self.fingerprints_analyzed,
            "fingerprints_excluded": self.fingerprints_excluded,
            "agreement_percentage": round(self.agreement_percentage, 1),
            "mode": self.mode,
            "image_path": self.image_path,
            "annotated_image_path": self.annotated_image_path,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }
        if include_fingerprints:
            data["fingerprints"] = [fp.to_dict() for fp in self.fingerprints]
        return data


class FingerprintResult(db.Model):
    __tablename__ = "fingerprint_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    finger_number = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(10), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    quality_score = db.Column(db.Float, nullable=False, default=1.0)
    quality_status = db.Column(db.String(20), nullable=False, default="Good")  # Good, Fair, Poor
    included_in_aggregation = db.Column(db.Boolean, nullable=False, default=True)
    exclusion_reason = db.Column(db.String(255), nullable=True)
    probabilities_json = db.Column(db.Text, nullable=True)
    crop_image_path = db.Column(db.String(255), nullable=True)
    gradcam_image_path = db.Column(db.String(255), nullable=True)

    def get_probabilities(self):
        if self.probabilities_json:
            try:
                return json.loads(self.probabilities_json)
            except Exception:
                return {}
        return {}

    def set_probabilities(self, probs: dict):
        self.probabilities_json = json.dumps(probs)

    def to_dict(self):
        return {
            "id": self.id,
            "finger_number": self.finger_number,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 2) if self.confidence else None,
            "quality_score": round(self.quality_score, 2),
            "quality_status": self.quality_status,
            "included_in_aggregation": self.included_in_aggregation,
            "exclusion_reason": self.exclusion_reason,
            "probabilities": self.get_probabilities(),
            "crop_image_path": self.crop_image_path,
            "gradcam_image_path": self.gradcam_image_path,
        }


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "file_path": self.file_path,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }
