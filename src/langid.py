"""Language + script identification.

Strategy (documented limits in README):
  1. Script from Unicode blocks (deterministic).
  2. Non-Latin scripts: language from the script (Devanagari is disambiguated hi/mr with `lingua`; other Indic scripts
     map to their principal language - note Bengali script is also used for Assamese, Arabic script for Urdu/Persian etc.).
  3. Latin script: a small char-n-gram classifier decides English vs Romanized Hindi ("hi-Latn", Hinglish). It is trained on
     public data (see scripts/train_langid.py) and evaluated honestly in reports/metrics/langid.json. If it says "not
     Romanized Hindi", `lingua` checks for clearly non-English Latin-script languages (fr/de/es/...).
Only en-Latn, hi-Deva and hi-Latn have labelled evaluation data in this project; anything else is flagged unsupported.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.config import CKPT_DIR, EVALUATED_LANGUAGES

SCRIPT_RANGES = {
    "Deva": (0x0900, 0x097F), "Beng": (0x0980, 0x09FF), "Guru": (0x0A00, 0x0A7F), "Gujr": (0x0A80, 0x0AFF),
    "Orya": (0x0B00, 0x0B7F), "Taml": (0x0B80, 0x0BFF), "Telu": (0x0C00, 0x0C7F), "Knda": (0x0C80, 0x0CFF),
    "Mlym": (0x0D00, 0x0D7F), "Arab": (0x0600, 0x06FF), "Cyrl": (0x0400, 0x04FF), "Hani": (0x4E00, 0x9FFF),
    "Hira": (0x3040, 0x309F), "Kana": (0x30A0, 0x30FF), "Hang": (0xAC00, 0xD7AF),
}
SCRIPT_LANG = {"Beng": "bn", "Guru": "pa", "Gujr": "gu", "Orya": "or", "Taml": "ta", "Telu": "te", "Knda": "kn", "Mlym": "ml",
               "Cyrl": "ru", "Hani": "zh", "Hira": "ja", "Kana": "ja", "Hang": "ko"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
              "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "ur": "Urdu", "ar": "Arabic", "fa": "Persian",
              "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish",
              "pt": "Portuguese", "it": "Italian", "id": "Indonesian", "tr": "Turkish", "nl": "Dutch", "und": "Undetermined"}

_LATIN_CLF = CKPT_DIR / "langid" / "latn_en_hi.joblib"
_STRIP = re.compile(r"<url>|<email>|<phone>|@user|https?://\S+|#\w+|[^a-z\s]")


def latin_features_text(text: str) -> str:
    """Style-robust view for the Latin classifier: lowercase letters and spaces only."""
    return re.sub(r"\s+", " ", _STRIP.sub(" ", text.lower())).strip()


def detect_script(text: str) -> tuple[str, float]:
    """Dominant script among alphabetic characters and its share. Returns ('Zyyy', 0) if no letters."""
    counts: dict[str, int] = {}
    n = 0
    for ch in text:
        if not ch.isalpha():
            continue
        n += 1
        cp = ord(ch)
        script = "Latn" if (ch.isascii() or 0x00C0 <= cp <= 0x024F) else next((s for s, (a, b) in SCRIPT_RANGES.items() if a <= cp <= b), "Zzzz")
        counts[script] = counts.get(script, 0) + 1
    if n == 0:
        return "Zyyy", 0.0
    top = max(counts, key=counts.get)
    return top, counts[top] / n


@lru_cache(maxsize=1)
def _lingua_all():
    from lingua import Language as L, LanguageDetectorBuilder

    langs = [L.ENGLISH, L.FRENCH, L.GERMAN, L.SPANISH, L.PORTUGUESE, L.ITALIAN, L.INDONESIAN, L.TURKISH, L.DUTCH, L.HINDI]
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build(), L


@lru_cache(maxsize=1)
def _lingua_deva():
    from lingua import Language as L, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(L.HINDI, L.MARATHI).build(), L


@lru_cache(maxsize=1)
def _latin_clf():
    if not _LATIN_CLF.exists():
        return None
    import joblib

    return joblib.load(_LATIN_CLF)


def _result(lang, script, conf, method, note=""):
    tag = f"{lang}-{script}" if script not in ("Zyyy", "Zzzz") else lang
    return {"language": lang, "language_name": LANG_NAMES.get(lang, lang), "script": script, "language_tag": tag,
            "confidence": round(float(conf), 4), "method": method, "evaluated": tag in EVALUATED_LANGUAGES, "note": note}


def detect_language(text: str) -> dict:
    script, share = detect_script(text)
    if script in ("Zyyy",):
        return _result("und", "Zyyy", 0.0, "none", "no alphabetic characters")
    if script == "Deva":
        det, L = _lingua_deva()
        vals = {c.language.iso_code_639_1.name.lower(): c.value for c in det.compute_language_confidence_values(text)}
        lang = max(vals, key=vals.get)
        return _result(lang, "Deva", vals[lang] * share, "script+lingua")
    if script == "Arab":
        return _result("ur", "Arab", share * 0.5, "script", "Arabic script is shared by Arabic/Urdu/Persian; language not resolved (default 'ur' is a guess)")
    if script in SCRIPT_LANG:
        note = "Bengali script is also used for Assamese" if script == "Beng" else ""
        return _result(SCRIPT_LANG[script], script, share, "script", note)
    if script != "Latn":
        return _result("und", script, share, "script")
    # Latin script: English vs Romanized Hindi, then other Latin-script languages
    clf = _latin_clf()
    p_hi = None
    if clf is not None:
        feats = latin_features_text(text)
        p_hi = float(clf.predict_proba([feats])[0][list(clf.classes_).index("hi-Latn")]) if feats else 0.0
        if p_hi >= 0.5:
            return _result("hi", "Latn", p_hi, "char-ngram-classifier", "Romanized Hindi / Hinglish; classifier trained on one public YouTube-comment corpus")
    det, L = _lingua_all()
    vals = {c.language: c.value for c in det.compute_language_confidence_values(text)}
    top = max(vals, key=vals.get)
    if top != L.ENGLISH and top != L.HINDI and vals[top] >= 0.8:
        return _result(top.iso_code_639_1.name.lower(), "Latn", vals[top], "lingua")
    conf = (1 - p_hi) if p_hi is not None else vals.get(L.ENGLISH, 0.5)
    return _result("en", "Latn", conf, "char-ngram-classifier+lingua" if clf else "lingua")
