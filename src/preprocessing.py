"""Text preprocessing and light PII scrubbing. Pure functions, no I/O."""
from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
_MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9_]{2,}")
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​‌‍﻿]")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def clean_text(text: str) -> str:
    """Normalise unicode, mask URLs/emails/phone numbers/@mentions, collapse whitespace.

    Masking is also a privacy measure: identifiers never reach the model or the logs.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CTRL.sub(" ", text)
    text = _URL.sub("<url>", text)
    text = _EMAIL.sub("<email>", text)
    text = _PHONE.sub("<phone>", text)
    text = _MENTION.sub("@user", text)
    return _WS.sub(" ", text).strip()


def simple_tokenize(text: str) -> list[str]:
    """Lower-cased word/punctuation tokens; works for Latin and Indic scripts via \\w."""
    return _TOKEN.findall(text.lower())


def length_bucket(n_tokens: int) -> str:
    if n_tokens < 8:
        return "short(<8)"
    if n_tokens < 40:
        return "medium(8-39)"
    return "long(40+)"
