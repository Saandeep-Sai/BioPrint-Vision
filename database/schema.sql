-- =============================================================================
-- BioPrint Vision — MySQL 8.x Database Schema
-- Defines tables, indexes, constraints, and relationships for the full platform
-- =============================================================================

CREATE DATABASE IF NOT EXISTS bioprint_vision CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE bioprint_vision;

-- -----------------------------------------------------------------------------
-- 1. USERS TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'USER',
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_email (email),
    INDEX idx_user_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 2. FACILITIES (Hospitals, Blood Banks, Blood Centres)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS facilities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(180) NOT NULL,
    type VARCHAR(50) NOT NULL DEFAULT 'Hospital',
    address VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    pincode VARCHAR(20) NOT NULL,
    latitude DOUBLE NULL,
    longitude DOUBLE NULL,
    phone VARCHAR(50) NOT NULL,
    email VARCHAR(120) NOT NULL,
    operating_hours VARCHAR(120) NOT NULL DEFAULT '24/7',
    is_demo BOOLEAN NOT NULL DEFAULT TRUE,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    user_id INT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_facility_city (city),
    INDEX idx_facility_type (type),
    INDEX idx_facility_is_demo (is_demo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 3. BLOOD INVENTORY
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS blood_inventory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    facility_id INT NOT NULL,
    blood_group VARCHAR(10) NOT NULL,
    units_available INT NOT NULL DEFAULT 0,
    stock_status VARCHAR(20) NOT NULL DEFAULT 'Available',
    last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_demo BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (facility_id) REFERENCES facilities(id) ON DELETE CASCADE,
    INDEX idx_inventory_facility (facility_id),
    INDEX idx_inventory_blood_group (blood_group),
    INDEX idx_inventory_status (stock_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 4. ANALYSES TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analyses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    final_blood_group VARCHAR(10) NOT NULL,
    final_confidence FLOAT NOT NULL,
    fingerprints_detected INT NOT NULL DEFAULT 1,
    fingerprints_analyzed INT NOT NULL DEFAULT 1,
    fingerprints_excluded INT NOT NULL DEFAULT 0,
    agreement_percentage FLOAT NOT NULL DEFAULT 100.0,
    mode VARCHAR(20) NOT NULL DEFAULT 'single',
    image_path VARCHAR(255) NULL,
    annotated_image_path VARCHAR(255) NULL,
    notes TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_analysis_user (user_id),
    INDEX idx_analysis_blood_group (final_blood_group),
    INDEX idx_analysis_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 5. FINGERPRINT RESULTS (Per-crop analysis)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fingerprint_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id INT NOT NULL,
    finger_number INT NOT NULL,
    prediction VARCHAR(10) NULL,
    confidence FLOAT NULL,
    quality_score FLOAT NOT NULL DEFAULT 1.0,
    quality_status VARCHAR(20) NOT NULL DEFAULT 'Good',
    included_in_aggregation BOOLEAN NOT NULL DEFAULT TRUE,
    exclusion_reason VARCHAR(255) NULL,
    probabilities_json TEXT NULL,
    crop_image_path VARCHAR(255) NULL,
    gradcam_image_path VARCHAR(255) NULL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    INDEX idx_finger_analysis (analysis_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 6. REPORTS TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id INT NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    INDEX idx_report_analysis (analysis_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
