"""
BioPrint Vision — Blood Inventory & Facility Availability Service
Handles facility queries, stock filtering, distance calculations, and status updates.
"""

import math
from typing import List, Dict, Any, Optional
from datetime import datetime
from database.connection import db
from database.models import Facility, BloodInventory
from config import Config


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two coordinates in kilometers."""
    R = 6371.0  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 1)


def search_blood_availability(
    blood_group: Optional[str] = None,
    city: Optional[str] = None,
    facility_type: Optional[str] = None,
    stock_status: Optional[str] = None,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Search and filter facilities with matching blood availability.

    Returns:
        Dict with 'results', 'total_count', 'filters_applied', and mandatory demo data disclaimer.
    """
    query = db.session.query(BloodInventory).join(Facility)

    # Filter by active facility
    query = query.filter(Facility.status == "ACTIVE")

    if blood_group:
        query = query.filter(BloodInventory.blood_group == blood_group.strip().upper())

    if city and city.strip():
        query = query.filter(Facility.city.ilike(f"%{city.strip()}%"))

    if facility_type and facility_type.strip() and facility_type != "All":
        query = query.filter(Facility.type == facility_type.strip())

    if stock_status and stock_status.strip() and stock_status != "All":
        query = query.filter(BloodInventory.stock_status == stock_status.strip())

    # Sort primarily by units available descending
    inventory_records = query.order_by(BloodInventory.units_available.desc()).all()

    results = []
    for item in inventory_records:
        fac = item.facility
        dist = None
        if user_lat is not None and user_lon is not None and fac.latitude and fac.longitude:
            dist = haversine_distance(user_lat, user_lon, fac.latitude, fac.longitude)

        results.append({
            "inventory_id": item.id,
            "facility_id": fac.id,
            "facility_name": fac.name,
            "facility_type": fac.type,
            "city": fac.city,
            "state": fac.state,
            "address": fac.address,
            "pincode": fac.pincode,
            "phone": fac.phone,
            "email": fac.email,
            "operating_hours": fac.operating_hours,
            "blood_group": item.blood_group,
            "units_available": item.units_available,
            "stock_status": item.stock_status,
            "last_updated": item.last_updated.strftime("%Y-%m-%d %H:%M") if item.last_updated else None,
            "is_demo": item.is_demo,
            "distance_km": dist,
        })

    # If coordinates provided, sort by distance
    if user_lat is not None and user_lon is not None:
        results = sorted(results, key=lambda r: (r["distance_km"] if r["distance_km"] is not None else 999999))

    return {
        "results": results,
        "count": len(results),
        "demo_notice": "Demo availability data — not connected to live hospital inventory.",
        "filters": {
            "blood_group": blood_group,
            "city": city,
            "facility_type": facility_type,
            "stock_status": stock_status,
        },
    }


def determine_stock_status(units: int) -> str:
    """Classify stock status based on configured unit thresholds."""
    if units >= Config.STOCK_THRESHOLD_AVAILABLE:
        return "Available"
    elif units >= Config.STOCK_THRESHOLD_LOW:
        return "Low Stock"
    elif units > 0:
        return "Critical"
    else:
        return "Unavailable"


def update_inventory_units(
    facility_id: int,
    blood_group: str,
    units: int,
) -> BloodInventory:
    """Update stock count and automatically classify stock_status."""
    bg = blood_group.strip().upper()
    inv = BloodInventory.query.filter_by(facility_id=facility_id, blood_group=bg).first()

    if not inv:
        inv = BloodInventory(facility_id=facility_id, blood_group=bg)
        db.session.add(inv)

    inv.units_available = max(0, units)
    inv.stock_status = determine_stock_status(inv.units_available)
    inv.last_updated = datetime.utcnow()
    db.session.commit()
    return inv

