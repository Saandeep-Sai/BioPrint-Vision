"""
BioPrint Vision — Blood Compatibility Service
Provides deterministic ABO and Rh compatibility rules for red blood cells and plasma.
"""

from typing import Dict, List, Any


COMPATIBILITY_RULES: Dict[str, Dict[str, Any]] = {
    "O-": {
        "can_receive_from": ["O-"],
        "can_donate_to": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
        "is_universal_donor": True,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
        "plasma_can_donate_to": ["O-", "O+"],
        "antigens": "None (No A, B, or Rh antigens)",
        "antibodies": "Anti-A, Anti-B, and Anti-Rh (if sensitized)",
        "clinical_note": "Universal red blood cell donor. In extreme emergencies when blood type is unknown, O- is the safest choice.",
    },
    "O+": {
        "can_receive_from": ["O+", "O-"],
        "can_donate_to": ["O+", "A+", "B+", "AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"],
        "plasma_can_donate_to": ["O+", "O-"],
        "antigens": "Rh antigen (No A or B antigens)",
        "antibodies": "Anti-A, Anti-B",
        "clinical_note": "Most common blood group in many populations. Critical for emergency transfusions for Rh-positive patients.",
    },
    "A-": {
        "can_receive_from": ["A-", "O-"],
        "can_donate_to": ["A-", "A+", "AB-", "AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["A-", "A+", "AB-", "AB+"],
        "plasma_can_donate_to": ["A-", "A+", "O-", "O+"],
        "antigens": "A antigen",
        "antibodies": "Anti-B",
        "clinical_note": "Can donate red blood cells to A and AB recipients regardless of Rh status.",
    },
    "A+": {
        "can_receive_from": ["A+", "A-", "O+", "O-"],
        "can_donate_to": ["A+", "AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["A+", "A-", "AB+", "AB-"],
        "plasma_can_donate_to": ["A+", "A-", "O+", "O-"],
        "antigens": "A and Rh antigens",
        "antibodies": "Anti-B",
        "clinical_note": "Can receive red blood cells from A and O donors.",
    },
    "B-": {
        "can_receive_from": ["B-", "O-"],
        "can_donate_to": ["B-", "B+", "AB-", "AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["B-", "B+", "AB-", "AB+"],
        "plasma_can_donate_to": ["B-", "B+", "O-", "O+"],
        "antigens": "B antigen",
        "antibodies": "Anti-A",
        "clinical_note": "Relatively rare blood group; high demand in regional blood banks.",
    },
    "B+": {
        "can_receive_from": ["B+", "B-", "O+", "O-"],
        "can_donate_to": ["B+", "AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["B+", "B-", "AB+", "AB-"],
        "plasma_can_donate_to": ["B+", "B-", "O+", "O-"],
        "antigens": "B and Rh antigens",
        "antibodies": "Anti-A",
        "clinical_note": "One of the most widely represented blood types in South Asian populations.",
    },
    "AB-": {
        "can_receive_from": ["AB-", "A-", "B-", "O-"],
        "can_donate_to": ["AB-", "AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": False,
        "plasma_can_receive_from": ["AB-", "AB+"],
        "plasma_can_donate_to": ["AB-", "AB+", "A-", "A+", "B-", "B+", "O-", "O+"],
        "antigens": "A and B antigens",
        "antibodies": "None against A or B",
        "clinical_note": "The rarest common blood group. Universal plasma donor for Rh-negative patients.",
    },
    "AB+": {
        "can_receive_from": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
        "can_donate_to": ["AB+"],
        "is_universal_donor": False,
        "is_universal_recipient": True,
        "plasma_can_receive_from": ["AB+"],
        "plasma_can_donate_to": ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"],
        "antigens": "A, B, and Rh antigens",
        "antibodies": "None",
        "clinical_note": "Universal red blood cell recipient. Universal plasma donor across all blood types.",
    },
}


def get_compatibility_info(blood_group: str) -> Dict[str, Any]:
    """
    Retrieve comprehensive deterministic compatibility guidance for a given blood group.

    Args:
        blood_group: Standard blood group string (e.g. 'B+', 'O-')

    Returns:
        Dict containing can_receive_from, can_donate_to, plasma rules, and clinical guidance.
    """
    bg = blood_group.strip().upper()
    if bg in ["A", "B", "AB", "O"] or (bg + "+" in COMPATIBILITY_RULES):
        bg = bg + "+"

    if bg not in COMPATIBILITY_RULES:
        return {
            "blood_group": bg,
            "error": f"Unknown blood group '{bg}'. Must be one of {list(COMPATIBILITY_RULES.keys())}",
            "can_receive_from": [],
            "can_donate_to": [],
            "red_cells": {"can_donate_to": [], "can_receive_from": []},
            "plasma": {"can_donate_to": [], "can_receive_from": []},
            "universal_donor": False,
            "universal_recipient": False,
        }

    info = dict(COMPATIBILITY_RULES[bg])
    info["blood_group"] = bg
    info["red_cells"] = {
        "can_donate_to": info["can_donate_to"],
        "can_receive_from": info["can_receive_from"],
    }
    info["plasma"] = {
        "can_donate_to": info["plasma_can_donate_to"],
        "can_receive_from": info["plasma_can_receive_from"],
    }
    info["universal_donor"] = info.get("is_universal_donor", False)
    info["universal_recipient"] = info.get("is_universal_recipient", False)
    info["disclaimer"] = (
        "Guidance is for informational/screening reference only. "
        "Clinical transfusions mandate formal laboratory crossmatching and antibody screening."
    )
    return info
