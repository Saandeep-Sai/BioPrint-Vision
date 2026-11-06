"""
BioPrint Vision — Rare Blood Group Alert Service
Configurable rule engine to identify rare blood phenotypes and issue appropriate advisories.
"""

from typing import Dict, Any
from config import Config


# General epidemiological distribution approximations (India & global composite)
RARITY_METRICS: Dict[str, Dict[str, Any]] = {
    "AB-": {
        "tier": "Ultra Rare",
        "prevalence": "0.6% - 1.0%",
        "description": "Least common ABO/Rh phenotype. Severely constrained supply in regional emergency banks.",
        "alert_level": "CRITICAL_ALERT",
    },
    "B-": {
        "tier": "Very Rare",
        "prevalence": "1.5% - 2.0%",
        "description": "Very low representation. High emergency vulnerability.",
        "alert_level": "HIGH_ALERT",
    },
    "O-": {
        "tier": "Rare (Universal Red Cell Donor)",
        "prevalence": "2.0% - 3.0%",
        "description": "Universal emergency red cell donor. Continually subject to critical hospital shortages.",
        "alert_level": "HIGH_ALERT",
    },
    "A-": {
        "tier": "Rare",
        "prevalence": "2.5% - 3.5%",
        "description": "Uncommon phenotype requiring proactive donor registry awareness.",
        "alert_level": "MODERATE_ALERT",
    },
    "AB+": {
        "tier": "Uncommon",
        "prevalence": "7.0% - 9.0%",
        "description": "Universal red cell recipient. Universal plasma donor.",
        "alert_level": "INFORMATIONAL",
    },
    "A+": {
        "tier": "Common",
        "prevalence": "21.0% - 25.0%",
        "description": "Widely available and in constant daily surgical demand.",
        "alert_level": "STANDARD",
    },
    "O+": {
        "tier": "Very Common",
        "prevalence": "31.0% - 36.0%",
        "description": "Highly demanded across all emergency and planned procedures.",
        "alert_level": "STANDARD",
    },
    "B+": {
        "tier": "Very Common",
        "prevalence": "32.0% - 38.0%",
        "description": "Predominant blood group in South Asian demographics.",
        "alert_level": "STANDARD",
    },
}


def evaluate_blood_rarity(blood_group: str) -> Dict[str, Any]:
    """
    Check if a blood group qualifies for a rare blood group alert.

    Args:
        blood_group: Blood group label (e.g. 'AB-', 'B+')

    Returns:
        Dict containing rarity tier, prevalence, alert flag, and clinical guidance.
    """
    bg = blood_group.strip().upper()
    info = RARITY_METRICS.get(bg, {
        "tier": "Unknown",
        "prevalence": "N/A",
        "description": "Uncataloged blood group.",
        "alert_level": "INFORMATIONAL",
    })

    # Check against configurable rare group list in Config
    is_rare = bg in Config.RARE_BLOOD_GROUPS

    if is_rare:
        alert_message = (
            f"Rare Blood Group Alert: {bg} ({info['tier']}). "
            f"Please verify this result through certified laboratory serological testing. "
            f"Consider registering with a recognized regional donor registry to assist in local emergencies."
        )
    else:
        alert_message = f"{bg} is a widely occurring blood group ({info['tier']})."

    return {
        "blood_group": bg,
        "is_rare": is_rare,
        "tier": info["tier"],
        "prevalence": info["prevalence"],
        "description": info["description"],
        "alert_level": info["alert_level"],
        "alert_message": alert_message,
        "disclaimer": (
            "Application-level rarity classification based on regional population statistics. "
            "Does not replace certified clinical blood typing."
        ),
    }
