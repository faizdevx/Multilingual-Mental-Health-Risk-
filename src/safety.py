"""Safety wording and crisis-resource pointers.

Design rules
  * Never diagnose. Never say "you have <condition>". Only describe *language signals* in the text.
  * The only risk dataset (Dreaddit) labels *stress* in Reddit posts. It has NO suicide / self-harm / high-risk category,
    so the model can never emit a "high-risk" label. HIGH_RISK_LABELS is therefore empty; the high-risk wording below is
    wired in for a future dataset that genuinely supports such a category and is never triggered today.
  * No phone numbers are shipped (they differ by country and go stale). We link only to two international directories
    whose pages were fetched and confirmed to be what they claim on VERIFIED_ON; the user chooses their own country.
"""
from __future__ import annotations

from src.config import DISCLAIMER, PROBABILITY_NOTE

HIGH_RISK_LABELS: frozenset[str] = frozenset()  # empty: no dataset used here defines a high-risk category
ELEVATED_LABELS = frozenset({"stress"})
ELEVATED_NOTICE = ("This result indicates language associated with elevated distress. "
                   "This tool is not a medical diagnostic system.")
HIGH_RISK_NOTICE = ("High-risk language signal detected. This classifier is not a clinical assessment. If this reflects an "
                    "immediate safety concern, seek help from a qualified professional or local emergency/crisis service.")
VERIFIED_ON = "2026-10-07"
CRISIS_RESOURCES = [
    {"name": "Find a Helpline (ThroughLine)", "url": "https://findahelpline.com",
     "description": "Directory of free, confidential helplines; choose your own country/region."},
    {"name": "IASP - Crisis Centres & Helplines", "url": "https://www.iasp.info/crisis-centres-helplines/",
     "description": "International Association for Suicide Prevention directory of crisis centres by country."},
]
NOT_VALIDATED_NOTE = ("The risk (stress-language) head was trained and evaluated on English text only. For other languages "
                      "or scripts its output is an unvalidated zero-shot transfer and should not be relied on.")


def build_safety(risk_label: str | None, risk_probability: float | None, risk_validated: bool) -> dict:
    high = risk_label in HIGH_RISK_LABELS
    elevated = risk_label in ELEVATED_LABELS
    notice = HIGH_RISK_NOTICE if high else ELEVATED_NOTICE if elevated else None
    return {
        "disclaimer": DISCLAIMER,
        "probability_note": PROBABILITY_NOTE,
        "level": "high_risk_language" if high else "elevated_distress_language" if elevated else "none_detected",
        "notice": notice,
        "show_resources": bool(high or elevated),
        "resources": CRISIS_RESOURCES if (high or elevated) else [],
        "resources_verified_on": VERIFIED_ON,
        "risk_validated_for_language": risk_validated,
        "validation_note": None if risk_validated else NOT_VALIDATED_NOTE,
    }


def risk_signal_text(label: str) -> str:
    """Human wording that matches the actual model label (never a diagnosis)."""
    return {"stress": "Elevated distress-related (stress) language signal",
            "non_stress": "No elevated stress-language signal detected"}.get(label, label)
