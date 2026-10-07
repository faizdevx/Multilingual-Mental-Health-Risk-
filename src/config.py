"""Central paths, label schemas and constants.

Label schemas are copied from the dataset cards / files (see DATASET.md), not invented.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CKPT_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_VERSION = "0.1.0"
TRANSFORMER_NAME = "google/muril-base-cased"  # Apache-2.0; see DATASET.md / README for verification

# ---- task label schemas -------------------------------------------------
TASKS = ("sentiment", "emotion", "risk")

# cardiffnlp/tweet_sentiment_multilingual: 0 negative, 1 neutral, 2 positive
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
# BRIGHTER (SemEval-2025 Task 11, track A): six binary (multi-label) columns
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# Dreaddit: label 1 = text annotated as expressing stress, 0 = not
RISK_LABELS = ["non_stress", "stress"]

TASK_LABELS = {"sentiment": SENTIMENT_LABELS, "emotion": EMOTION_LABELS, "risk": RISK_LABELS}
TASK_KIND = {"sentiment": "multiclass", "emotion": "multilabel", "risk": "multiclass"}
POSITIVE_RISK_LABEL = "stress"

# Languages for which each task has labelled evaluation data (language tag = BCP-47 language + ISO-15924 script)
TASK_LANGUAGES = {
    "sentiment": ["en-Latn", "hi-Latn"],  # NB: the "hindi" tweet config is Romanized (Latin-script) Hindi
    "emotion": ["en-Latn", "hi-Deva"],
    "risk": ["en-Latn"],
}
EVALUATED_LANGUAGES = ["en-Latn", "hi-Deva", "hi-Latn"]
LANGUAGE_NAMES = {"en-Latn": "English", "hi-Deva": "Hindi (Devanagari)", "hi-Latn": "Hinglish / Romanized Hindi"}

# Max sequence length per task (Dreaddit posts are long; tweets/sentences are short)
MAX_LEN = {"sentiment": 64, "emotion": 64, "risk": 128}

DISCLAIMER = (
    "This tool is an NLP research and screening system. It is not a medical diagnostic tool "
    "and should not be used to make clinical decisions."
)
PROBABILITY_NOTE = "Model probability is a statistical model output and is not a clinical probability."
