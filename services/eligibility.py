"""
BioPrint Vision — Donor Pre-Screening Module
Implements deterministic, configurable rule evaluation for preliminary blood donation eligibility.
"""

from typing import Dict, Any, List


def evaluate_donor_eligibility(answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate user-supplied health and lifestyle responses against standard
    blood donation pre-screening guidelines (e.g. WHO and National Blood Transfusion Council).

    Expected keys in answers:
      - age: int (Years)
      - weight: float (kg)
      - hemoglobin: Optional[float] (g/dL)
      - days_since_last_donation: Optional[int] (Days, or None if first time)
      - recent_tattoo_or_piercing: bool (Within last 6 months)
      - recent_major_surgery: bool (Within last 6 months)
      - active_infection_or_antibiotics: bool (Currently sick or taking antibiotics)
      - pregnant_or_lactating: bool (Currently pregnant or breastfeeding)
      - chronic_conditions: bool (History of Hepatitis B/C, HIV, or heart disease)

    Returns:
      Dict containing:
        - status: 'Potentially Eligible', 'Medical Review Required', or 'Not Eligible Based on Screening'
        - flags: list of reasons/considerations
        - recommendations: list of actionable advice
        - disclaimer: mandatory screening disclaimer
    """
    flags = []
    recommendations = []
    hard_disqualifier = False
    review_needed = False

    # 1. Age Evaluation (Standard 18 - 65)
    age = answers.get("age")
    if age is not None:
        try:
            age = int(age)
            if age < 18:
                flags.append(f"Under minimum eligible age of 18 (Current: {age}).")
                hard_disqualifier = True
            elif age > 65:
                flags.append(f"Over standard donation age threshold of 65 (Current: {age}).")
                review_needed = True
                recommendations.append("Donors over 65 require annual medical clearance from a certified physician.")
        except (ValueError, TypeError):
            flags.append("Invalid age provided.")
            review_needed = True

    # 2. Weight Evaluation (Minimum 50 kg)
    weight = answers.get("weight")
    if weight is not None:
        try:
            weight = float(weight)
            if weight < 45.0:
                flags.append(f"Weight is below safe donation limit (Current: {weight} kg, minimum 50 kg).")
                hard_disqualifier = True
            elif weight < 50.0:
                flags.append(f"Borderline weight (Current: {weight} kg). Minimum 50 kg required for standard 350ml/450ml draw.")
                hard_disqualifier = True
        except (ValueError, TypeError):
            flags.append("Invalid weight provided.")
            review_needed = True

    # 3. Chronic Infectious or Cardiovascular Conditions
    if answers.get("chronic_conditions"):
        flags.append("Reported chronic medical conditions (Hepatitis B/C, HIV, severe cardiovascular disease).")
        hard_disqualifier = True
        recommendations.append("Permanent deferral is standard for chronic blood-borne conditions to safeguard recipients.")

    # 4. Pregnancy / Lactation
    if answers.get("pregnant_or_lactating"):
        flags.append("Current pregnancy or breastfeeding within the last 6 months.")
        hard_disqualifier = True
        recommendations.append("Temporary deferral applies until at least 6 months post-delivery or post-lactation.")

    # 5. Active Infection / Antibiotics
    if answers.get("active_infection_or_antibiotics"):
        flags.append("Active bacterial/viral infection or currently on antibiotics course.")
        review_needed = True
        recommendations.append("Must be symptom-free and complete antibiotic course for at least 72 hours before donation.")

    # 6. Recent Tattoos or Piercings (6-Month Deferral)
    if answers.get("recent_tattoo_or_piercing"):
        flags.append("Tattoo, body piercing, or acupuncture within the last 6 months.")
        review_needed = True
        recommendations.append("Standard 6-month temporary deferral applies to rule out window-period infections.")

    # 7. Recent Major Surgery
    if answers.get("recent_major_surgery"):
        flags.append("Major surgical procedure within the last 6 months.")
        review_needed = True
        recommendations.append("Consult with surgical team; standard deferral is 6 months post full recovery.")

    # 8. Donation Interval (Minimum 90 days for whole blood for males, 120 days for females)
    days_since = answers.get("days_since_last_donation")
    if days_since is not None and days_since != "":
        try:
            days = int(days_since)
            if days < 90:
                flags.append(f"Recent blood donation was only {days} days ago (Minimum 90 days required).")
                hard_disqualifier = True
                recommendations.append(f"Please wait {90 - days} more days to allow iron stores and red cell recovery.")
        except (ValueError, TypeError):
            pass

    # 9. Hemoglobin (Optional check, minimum 12.5 g/dL)
    hb = answers.get("hemoglobin")
    if hb is not None and hb != "":
        try:
            hb_val = float(hb)
            if hb_val < 12.5:
                flags.append(f"Hemoglobin level is low ({hb_val} g/dL, minimum required is 12.5 g/dL).")
                review_needed = True
                recommendations.append("Consume iron-rich foods (spinach, legumes, dates) and retest at blood bank.")
        except (ValueError, TypeError):
            pass

    # Determine Overall Status
    if hard_disqualifier:
        status = "Not Eligible Based on Screening"
    elif review_needed:
        status = "Medical Review Required"
    else:
        status = "Potentially Eligible"
        recommendations.append("Ensure adequate hydration, eat a light healthy meal, and bring a valid government ID.")

    return {
        "status": status,
        "flags": flags,
        "recommendations": recommendations,
        "disclaimer": (
            "Pre-screening only. Final donor eligibility must be determined by qualified "
            "medical/blood-bank personnel on the day of donation."
        ),
    }
