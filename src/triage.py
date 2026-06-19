"""
triage.py — Business triage logic for AutoClaim AI.

Translates a model prediction + confidence into an actionable insurance
triage recommendation.  Thresholds are read from config.py so they can
be tuned without touching this file.

Triage levels:
    PRIORITY ASSESSMENT  — severe damage + sufficient confidence
    FAST-TRACK           — minor damage + high confidence
    HUMAN REVIEW         — low confidence or ambiguous case
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
    1. Confidence below LOW threshold → "Human Review Required"
       Rationale: a model that is uncertain should not make automated decisions.
       The risk of a wrong automatic routing exceeds the cost of one human review.

    2. Severe class AND confidence >= SEVERE threshold → "Priority Assessment"
       Rationale: safety-critical damage (tire flat, glass shatter, structural
       crack, broken lamp) carries liability risk.  Even at moderate confidence
       these cases should be escalated quickly.

    3. Minor class AND confidence >= HIGH threshold → "Fast-Track"
       Rationale: a high-confidence prediction on low-severity damage (scratch,
       dent) can be automatically routed to standard processing, saving ~15 min
       per claim in manual first review.

    4. Otherwise → "Human Review Required"
       Rationale: covers edge cases not matched above (moderate confidence on
       severe class below the severe threshold, or moderate confidence on minor
       class below the fast-track threshold).

    Parameters
    ----------
    predicted_class : str   top predicted damage category
    confidence      : float probability [0, 1] of the top prediction

    Returns
    -------
    dict  {decision, level, reason, icon}
    """
    cls = predicted_class.lower()

    # Rule 1 — low confidence always goes to human review
    if confidence < config.TRIAGE_LOW_CONFIDENCE:
        return {
            "decision": "Human Review Required",
            "level":    "review",
            "reason":   (
                f"Model confidence is {confidence:.1%}, which is below the "
                f"minimum threshold of {config.TRIAGE_LOW_CONFIDENCE:.0%}. "
                "A claims handler should verify this image manually."
            ),
            "icon": "⚠️",
        }

    # Rule 2 — severe damage with sufficient confidence → priority
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

    # Rule 3 — minor damage with high confidence → fast-track
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

    # Rule 4 — everything else
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
