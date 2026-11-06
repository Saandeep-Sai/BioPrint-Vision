"""
BioPrint Vision — Configuration Module
Handles environment variables, database connections (MySQL with SQLite fallback),
upload directory paths, and medical/inventory business rules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration."""

    SECRET_KEY = os.getenv("SECRET_KEY", "bioprint-vision-secure-secret-key-2026-v2")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    # Base directories
    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    REPORT_FOLDER = UPLOAD_FOLDER / "reports"
    FINGERPRINT_FOLDER = UPLOAD_FOLDER / "fingerprints"

    # ML Model configuration
    MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
    CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
    MODEL_IMG_SIZE = 224

    # Database Configuration (Local storage engine; documentation specifies MySQL 8.x schema)
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "bioprint_vision")

    # Primary Database URI: Local storage database file
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'bioprint_vision.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Inventory stock status thresholds (Units)
    STOCK_THRESHOLD_AVAILABLE = int(os.getenv("STOCK_THRESHOLD_AVAILABLE", "15"))
    STOCK_THRESHOLD_LOW = int(os.getenv("STOCK_THRESHOLD_LOW", "5"))

    # Rare blood group configuration (Default rare in broad populations)
    RARE_BLOOD_GROUPS = ["AB-", "B-", "O-", "A-"]

    # Maximum file upload size: 16 MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}


def init_app_dirs(app=None):
    """Ensure required upload and artifact directories exist."""
    Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    Config.REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    Config.FINGERPRINT_FOLDER.mkdir(parents=True, exist_ok=True)
