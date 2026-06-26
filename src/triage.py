"""
triage.py — Business triage logic for AutoClaim AI.

Translates a model prediction + confidence into an actionable insurance
triage recommendation.  Thresholds are read from config.py so they can
be tuned without touching this file.

Triage levels:
    PRIORITY ASSESSMENT  — severe damage + confidence ≥ 70%
    FAST-TRACK           — minor damage (dent/scratch) + confidence ≥ 80%
    HUMAN REVIEW         — everything else (low confidence, unclear image)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Decision logic ───────────────────────────────────────────────────────────

def get_triage_decision(predicted_class: str, confidence: float) -> dict:
    """
    Apply business rules to produce a triage recommendation.

    Rules (ordered by priority):
    1. Severe class AND confidence >= SEVERE threshold → "Priority Assessment"
       Safety-critical damage (tire flat, glass shatter, structural crack,
       broken lamp) is escalated immediately for human inspection.

    2. Minor class AND confidence >= HIGH threshold → "Fast-Track"
       High-confidence dent/scratch predictions are routed automatically,
       saving ~15 min per claim in manual first review.

    3. Otherwise → "Human Review Required"
       Covers: severe damage below the priority threshold, minor damage below
       the fast-track threshold, and unclear or low-quality images.

    Parameters
    ----------
    predicted_class : str   top predicted damage category
    confidence      : float probability [0, 1] of the top prediction

    Returns
    -------
    dict  {decision, level, reason, icon}
    """
    cls = predicted_class.lower()

    # Rule 1 — severe damage with sufficient confidence → priority
    if cls in config.SEVERE_CLASSES and confidence >= config.TRIAGE_SEVERE_CONFIDENCE:
        return {
            "decision": "Priority Assessment",
            "level":    "priority",
            "reason":   (
                f"'{predicted_class}' is classified as severe / safety-critical damage "
                f"with {confidence:.1%} confidence. "
                "This claim is flagged for priority handling and technical inspection."
            ),
            "icon": "🔴",
        }

    # Rule 2 — minor damage with high confidence → fast-track
    if cls not in config.SEVERE_CLASSES and confidence >= config.TRIAGE_HIGH_CONFIDENCE:
        return {
            "decision": "Fast-Track Claim",
            "level":    "fast_track",
            "reason":   (
                f"'{predicted_class}' is low-severity damage detected with "
                f"{confidence:.1%} confidence. "
                "This claim can proceed through the standard automated pipeline."
            ),
            "icon": "🟢",
        }

    # Rule 3 — everything else → human review
    return {
        "decision": "Human Review Required",
        "level":    "review",
        "reason":   (
            f"Prediction is '{predicted_class}' at {confidence:.1%} confidence. "
            "Confidence is insufficient to automate this decision. "
            "A claims handler should review the image."
        ),
        "icon": "🟡",
    }


# ─── Batch triage helper ──────────────────────────────────────────────────────

def triage_batch(predictions: list) -> list:
    """Apply get_triage_decision to a list of predict_image() outputs."""
    return [
        {**pred, "triage": get_triage_decision(pred["predicted_class"], pred["confidence"])}
        for pred in predictions
    ]


# ─── Explanation text for UI ──────────────────────────────────────────────────

TRIAGE_EXPLANATIONS = {
    "priority": (
        "This damage type is considered severe or safety-critical. "
        "The claim has been routed to a senior assessor for priority review. "
        "Expected response time: within 4 business hours."
    ),
    "fast_track": (
        "This appears to be minor cosmetic damage with high model confidence. "
        "The claim is eligible for automated fast-track processing. "
        "A final human spot-check will be performed on a random sample."
    ),
    "review": (
        "The model could not make a high-confidence determination. "
        "A trained claims handler will review this image manually. "
        "Expected response time: 1–2 business days."
    ),
}


def get_triage_explanation(level: str) -> str:
    return TRIAGE_EXPLANATIONS.get(level, "No explanation available.")
