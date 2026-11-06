"""
BioPrint Vision — Contextual AI Assistant Service
Answers user questions using real application state, compatibility rules,
donor pre-screening logic, and demo inventory data without hallucinating.
"""

from typing import Dict, Any, Optional
from services.compatibility import get_compatibility_info
from services.rarity import evaluate_blood_rarity
from services.inventory import search_blood_availability


class AIAssistant:
    """
    Context-aware assistant for BioPrint Vision.
    Processes user queries against the active analysis context and system knowledge.
    """

    def __init__(self):
        self.disclaimer = (
            "⚠️ Note: I am an automated AI assistant providing screening context only. "
            "I cannot provide medical advice, diagnosis, or clinical clearance. "
            "Always consult qualified healthcare professionals and accredited laboratories."
        )

    def answer_query(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Formulate an accurate, evidence-bounded response.

        Args:
            message: User query string
            context: Current analysis context (if available)

        Returns:
            Dict containing 'response', 'suggested_actions', and 'disclaimer'.
        """
        msg_lower = message.strip().lower()
        context = context or {}

        current_bg = context.get("final_blood_group")
        confidence = context.get("final_confidence")
        agreement = context.get("agreement_percentage")
        fingerprints = context.get("fingerprints", [])

        # 1. Query: What blood group was predicted?
        if any(kw in msg_lower for kw in ["predicted", "my blood group", "result", "what is my blood type", "prediction"]):
            if current_bg:
                conf_str = f" with a model confidence of {confidence:.1f}%" if confidence else ""
                agree_str = f" (Multi-fingerprint agreement: {agreement:.1f}%)" if agreement is not None else ""
                reply = (
                    f"Based on your latest biometric analysis, the predicted blood group is **{current_bg}**{conf_str}{agree_str}.\n\n"
                    f"Please remember that this is an experimental AI prediction from dermatoglyphic ridge patterns. "
                    f"It serves as a prototype screening indicator and **does not replace laboratory serological blood testing**."
                )
                suggestions = ["Tell me about compatibility for this blood group", "Where can I find blood near me?", "Explain my quality score"]
            else:
                reply = (
                    "No active analysis was found in your current session. "
                    "Please upload a fingerprint image or a multi-fingerprint photograph on the **New Analysis** page to receive a prediction."
                )
                suggestions = ["How does fingerprint analysis work?", "What blood groups are supported?"]

        # 2. Query: Why did the fingerprints disagree?
        elif any(kw in msg_lower for kw in ["disagree", "agreement", "why different", "different prediction"]):
            if fingerprints and len(fingerprints) > 1:
                details = []
                for fp in fingerprints:
                    f_id = fp.get("id", fp.get("finger_number", 1))
                    pred = fp.get("prediction", "N/A")
                    c = fp.get("confidence", 0.0)
                    details.append(f"• **Fingerprint {f_id}**: Predicted `{pred}` ({c:.1f}%)")
                details_text = "\n".join(details)
                reply = (
                    f"During multi-fingerprint analysis, each detected impression is analyzed independently by our deep learning model:\n\n"
                    f"{details_text}\n\n"
                    f"**Why variations happen:** Fingerprint ridge minutiae vary slightly across different fingers of the same individual. "
                    f"Minor differences in ink pressure, contact angle, or micro-blurring can lead to different softmax activations. "
                    f"To maximize robustness, BioPrint Vision **averages the raw probability vectors across all usable fingers** "
                    f"to produce the final consolidated prediction of **{current_bg or 'the final group'}**."
                )
                suggestions = ["How was the final prediction calculated?", "Were any fingers excluded?"]
            else:
                reply = (
                    "In single-fingerprint mode or when all fingers agree, individual predictions align. "
                    "When multiple fingerprints are provided, the system averages raw probabilities to prevent single-impression bias."
                )
                suggestions = ["Explain the aggregation formula", "Check blood compatibility"]

        # 3. Query: Why was a fingerprint excluded?
        elif any(kw in msg_lower for kw in ["exclude", "excluded", "why rejected", "poor quality"]):
            excluded_fps = [
                fp for fp in fingerprints
                if not fp.get("included_in_aggregation", fp.get("included", True))
            ]
            if excluded_fps:
                reasons = []
                for fp in excluded_fps:
                    f_id = fp.get("id", fp.get("finger_number", 1))
                    reason = fp.get("exclusion_reason", "Low contrast or excessive blur")
                    reasons.append(f"• **Fingerprint {f_id}**: Excluded due to *{reason}*")
                reasons_text = "\n".join(reasons)
                reply = (
                    f"To ensure statistical reliability, our automated quality screening excluded the following fingerprint(s):\n\n"
                    f"{reasons_text}\n\n"
                    f"Substandard crops with excessive blur, washed-out contrast, or heavy noise are omitted from the final probability average to avoid degrading the prediction."
                )
                suggestions = ["What makes a good fingerprint image?", "Can I re-analyze my photo?"]
            else:
                reply = (
                    "Good news! None of your detected fingerprints were excluded. "
                    "All detected fingerprint impressions met our sharpness, contrast, and resolution quality standards."
                )
                suggestions = ["Show probability distribution", "Find matching blood banks"]

        # 4. Query: Blood Compatibility (Receive / Donate)
        elif any(kw in msg_lower for kw in ["compatible", "receive", "donate", "can i give", "can i take", "who can donate"]):
            bg_target = current_bg
            # Check if user mentioned a specific blood group in message
            for bg in ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]:
                if bg.lower() in msg_lower:
                    bg_target = bg
                    break

            if bg_target:
                compat = get_compatibility_info(bg_target)
                can_recv = ", ".join(compat["can_receive_from"])
                can_don = ", ".join(compat["can_donate_to"])
                universal = ""
                if compat["is_universal_donor"]:
                    universal = "\n\n⭐ **Universal Donor:** O- red blood cells can be safely transfused to individuals of ANY blood type in emergencies!"
                elif compat["is_universal_recipient"]:
                    universal = "\n\n⭐ **Universal Recipient:** AB+ individuals can receive red blood cells of ANY blood type!"

                reply = (
                    f"### 🩸 Compatibility Profile for **{bg_target}**\n\n"
                    f"• **Can receive red blood cells from:** `{can_recv}`\n"
                    f"• **Can donate red blood cells to:** `{can_don}`\n"
                    f"• **Antigen profile:** {compat['antigens']}\n"
                    f"• **Antibodies present:** {compat['antibodies']}"
                    f"{universal}\n\n"
                    f"*Clinical Note:* In actual hospital practice, strict serological crossmatching is always performed before any transfusion."
                )
                suggestions = ["Search blood availability for this type", "Am I eligible to donate?"]
            else:
                reply = (
                    "Please specify which blood group you'd like compatibility details for (e.g., 'What can O- donate to?' or 'Who can receive B+?'), "
                    "or run an analysis to view your specific compatibility."
                )
                suggestions = ["What is the universal donor?", "Who can AB+ receive from?"]

        # 5. Query: Blood Availability / Hospital Stock
        elif any(kw in msg_lower for kw in ["availability", "hospital", "blood bank", "stock", "units", "where to get"]):
            bg_search = current_bg or "B+"
            for bg in ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]:
                if bg.lower() in msg_lower:
                    bg_search = bg
                    break

            try:
                stock_res = search_blood_availability(blood_group=bg_search)
                facilities = stock_res.get("results", [])[:3]
                if facilities:
                    lines = []
                    for f in facilities:
                        lines.append(
                            f"• **{f['facility_name']}** ({f['city']}): `{f['units_available']} units` ({f['stock_status']}) — Tel: `{f['phone']}`"
                        )
                    fac_text = "\n".join(lines)
                    reply = (
                        f"Here are nearby demonstration facilities with stock for **{bg_search}**:\n\n"
                        f"{fac_text}\n\n"
                        f"*(Demo availability data — not connected to live hospital inventory. For demonstration purposes only.)*"
                    )
                    suggestions = ["Open full Blood Availability Directory", "Can I filter by city?"]
                else:
                    reply = f"No demonstration facilities currently show available stock for {bg_search}."
                    suggestions = ["Check compatibility", "View all facilities"]
            except Exception as e:
                reply = f"You can explore blood availability directly on our **Blood Availability** search page for {bg_search}."
                suggestions = ["Open Blood Availability", "View compatibility"]

        # 6. Query: Donor Eligibility / Pre-screening
        elif any(kw in msg_lower for kw in ["eligible", "eligibility", "can i donate", "donation rules", "requirements"]):
            reply = (
                "### 🩸 Blood Donation Pre-Screening Guidelines\n\n"
                "To donate blood safely, basic criteria typically include:\n"
                "1. **Age:** 18 to 65 years\n"
                "2. **Weight:** Minimum 50 kg (110 lbs)\n"
                "3. **Interval:** At least 90 days (3 months) since your last whole blood donation\n"
                "4. **Health:** Free from active infections, fever, or antibiotic treatments for at least 72 hours\n"
                "5. **Tattoos / Piercings:** At least 6 months post-procedure to rule out window infections\n\n"
                "You can take our interactive questionnaire under **Donor Pre-Screening** for an instant evaluation!"
            )
            suggestions = ["Take Donor Pre-Screening Questionnaire", "Check my blood compatibility"]

        # 7. Query: Rarity of Blood Group
        elif any(kw in msg_lower for kw in ["rare", "rarity", "how common"]):
            bg_target = current_bg or "AB-"
            rarity = evaluate_blood_rarity(bg_target)
            reply = (
                f"### Rarity Assessment for **{bg_target}**\n\n"
                f"• **Rarity Tier:** {rarity['tier']}\n"
                f"• **Estimated Population Prevalence:** {rarity['prevalence']}\n"
                f"• **Profile:** {rarity['description']}\n\n"
                f"{rarity['alert_message']}"
            )
            suggestions = ["Check blood compatibility", "Find blood facilities"]

        # 8. Default fallback
        else:
            reply = (
                "Hello! I am your **BioPrint Vision AI Assistant**. I can help you with:\n\n"
                "• **Analysis Insights**: Explaining your predicted blood group, model confidence, or multi-fingerprint agreement\n"
                "• **Compatibility**: Guidance on red cell and plasma donor/recipient matching\n"
                "• **Blood Availability**: Finding demonstration facilities with matching blood units\n"
                "• **Donor Eligibility**: Explaining pre-screening criteria and deferral guidelines\n"
                "• **Report Assistance**: Explaining sections of your generated PDF report\n\n"
                "What would you like to know?"
            )
            suggestions = [
                "What blood group was predicted?",
                "Who can I receive blood from?",
                "Check blood availability near me",
                "Am I eligible to donate blood?",
            ]

        return {
            "response": reply,
            "reply": reply,
            "suggested_actions": suggestions,
            "suggested_followups": suggestions,
            "disclaimer": self.disclaimer,
        }
