"""
BioPrint Vision — Assistant Blueprint
Handles interactive AI chat interface and contextual response generation.
"""

from flask import Blueprint, render_template, request, jsonify, g
from routes.auth import login_required
from database.models import Analysis
from services.assistant import AIAssistant

assistant_bp = Blueprint("assistant", __name__)
_ai_service = AIAssistant()


@assistant_bp.route("/assistant")
@login_required
def chat_view():
    """Contextual AI Assistant page."""
    analysis_id = request.args.get("analysis_id", type=int)
    active_analysis = None
    if analysis_id:
        active_analysis = Analysis.query.filter_by(id=analysis_id, user_id=g.user.id).first()

    return render_template("assistant/chat.html", active_analysis=active_analysis)


@assistant_bp.route("/api/assistant/chat", methods=["POST"])
@login_required
def chat_api():
    """API endpoint for assistant conversation."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    analysis_id = data.get("analysis_id")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    context = {}
    if analysis_id:
        analysis = Analysis.query.filter_by(id=analysis_id, user_id=g.user.id).first()
        if analysis:
            context = analysis.to_dict(include_fingerprints=True)
            # Add probabilities dict if available
            context["probabilities"] = context["fingerprints"][0]["probabilities"] if context["fingerprints"] else {}

    reply = _ai_service.answer_query(message, context=context)
    return jsonify(reply)
