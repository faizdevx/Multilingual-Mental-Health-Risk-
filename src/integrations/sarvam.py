"""OPTIONAL Sarvam AI speech-to-text adapter. The core project works without it.

Pipeline: audio -> Sarvam STT (external API) -> local classifier (this project) -> results.

STATUS: written against Sarvam's public STT docs (POST https://api.sarvam.ai/speech-to-text, header `api-subscription-key`,
multipart field `file`, JSON field `transcript`; checked 2026-10-07). It is covered by unit tests with a MOCKED HTTP
client only. It has NOT been run against the live Sarvam service (no API key was available), so do not assume it works
end-to-end until you have tested it with your own key.

PRIVACY: calling `transcribe` uploads audio to a third party. It only runs if you call it explicitly AND set SARVAM_API_KEY.
Nothing is logged. TTS is intentionally not implemented.
"""
from __future__ import annotations

import os

STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.environ.get("SARVAM_API_KEY"))


def transcribe(audio_bytes: bytes, filename: str = "audio.wav", language_code: str | None = None, model: str | None = None, client=None) -> str:
    """Return the transcript string. `client` (httpx-like, with .post) is injectable for tests."""
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SarvamNotConfigured("SARVAM_API_KEY is not set; Sarvam integration is disabled")
    import httpx

    data = {k: v for k, v in (("language_code", language_code), ("model", model)) if v}
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        r = client.post(STT_URL, headers={"api-subscription-key": key}, files={"file": (filename, audio_bytes)}, data=data)
        r.raise_for_status()
        return r.json()["transcript"]
    finally:
        if own:
            client.close()


def analyze_audio(audio_bytes: bytes, predictor, model: str = "transformer", **kw) -> dict:
    """Transcribe then classify locally. The transcript is returned to the caller but never logged."""
    text = transcribe(audio_bytes, **kw)
    return {"transcript": text, "analysis": predictor.analyze(text, model)}
