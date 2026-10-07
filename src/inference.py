"""Inference service: language ID + multi-task model + calibration + safety wording. All outputs come from model inference."""
from __future__ import annotations

import time

import numpy as np

from src.config import EMOTION_LABELS, MODEL_VERSION, RISK_LABELS, SENTIMENT_LABELS, TASK_LANGUAGES
from src.evaluation.adapters import DISPLAY, Adapter, available_models, load_adapter
from src.explain import integrated_gradients
from src.langid import LANG_NAMES, detect_language
from src.preprocessing import clean_text
from src.safety import build_safety, risk_signal_text
from src.training.calibration import sigmoid, softmax

MAX_CHARS = 5000


class ModelRegistry:
    """Loads whichever checkpoints exist; missing ones are reported, not faked."""

    def __init__(self, names: list[str] | None = None, device: str | None = None, adapters: dict[str, Adapter] | None = None):
        self.adapters: dict[str, Adapter] = dict(adapters or {})
        self.errors: dict[str, str] = {}
        if adapters is None:
            for n in names or available_models():
                try:
                    self.adapters[n] = load_adapter(n, device)
                except Exception as e:  # corrupt/incompatible checkpoint
                    self.errors[n] = type(e).__name__

    def get(self, name: str) -> Adapter:
        if name not in self.adapters:
            raise KeyError(name)
        return self.adapters[name]

    def names(self) -> list[str]:
        return list(self.adapters)

    @property
    def device(self) -> str:
        for a in self.adapters.values():
            if getattr(a, "device", None):
                return a.device
        return "cpu"


def _lang_from_hint(hint: str | None, text: str) -> dict:
    if not hint or hint.lower() == "auto":
        return detect_language(text)
    lang, _, script = hint.partition("-")
    script = script or ("Deva" if lang == "hi" else "Latn")
    tag = f"{lang}-{script}"
    return {"language": lang, "language_name": LANG_NAMES.get(lang, lang), "script": script, "language_tag": tag, "confidence": 1.0,
            "method": "user_supplied", "evaluated": tag in {t for ts in TASK_LANGUAGES.values() for t in ts}, "note": ""}


class Predictor:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def _task_outputs(self, adapter: Adapter, texts: list[str]) -> dict:
        cal = adapter.calibration()
        T = cal.get("temperature", {})
        thr = np.array(cal.get("emotion_thresholds") or [0.5] * len(EMOTION_LABELS))
        out = {}
        out["sentiment"] = softmax(adapter.predict_logits("sentiment", texts), T.get("sentiment", 1.0))
        out["emotion"] = sigmoid(adapter.predict_logits("emotion", texts), T.get("emotion", 1.0))
        out["risk"] = softmax(adapter.predict_logits("risk", texts), T.get("risk", 1.0))
        out["_thr"] = thr
        out["_calibrated"] = bool(T)
        return out

    def analyze_many(self, texts: list[str], model: str = "transformer", language_hints: list[str | None] | None = None) -> list[dict]:
        adapter = self.registry.get(model)
        t0 = time.perf_counter()
        cleaned = [clean_text(t)[:MAX_CHARS] for t in texts]
        outs = self._task_outputs(adapter, cleaned)
        per_item_ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
        results = []
        for i, text in enumerate(cleaned):
            lang = _lang_from_hint(language_hints[i] if language_hints else None, text)
            tag = lang["language_tag"]
            sp, ep, rp = outs["sentiment"][i], outs["emotion"][i], outs["risk"][i]
            above = [(EMOTION_LABELS[j], float(ep[j])) for j in range(len(EMOTION_LABELS)) if ep[j] >= outs["_thr"][j]]
            above.sort(key=lambda x: -x[1])
            risk_idx = int(rp.argmax())
            risk_label = RISK_LABELS[risk_idx]
            p_stress = float(rp[RISK_LABELS.index("stress")])
            validated = {t: tag in TASK_LANGUAGES[t] for t in TASK_LANGUAGES}
            results.append({
                "language": lang["language"], "script": lang["script"], "language_tag": tag, "language_name": lang["language_name"],
                "language_confidence": lang["confidence"], "language_method": lang["method"], "language_note": lang["note"],
                "sentiment": SENTIMENT_LABELS[int(sp.argmax())],
                "sentiment_probabilities": {l: round(float(p), 4) for l, p in zip(SENTIMENT_LABELS, sp)},
                "emotion": above[0][0] if above else "none",
                "emotions_detected": [{"label": l, "probability": round(p, 4)} for l, p in above],
                "emotion_probabilities": {l: round(float(p), 4) for l, p in zip(EMOTION_LABELS, ep)},
                "risk": risk_label,
                "risk_signal": risk_signal_text(risk_label),
                "risk_probability": round(p_stress, 4),
                "risk_predictions": [{"label": l, "probability": round(float(p), 4)} for l, p in zip(RISK_LABELS, rp)],
                "validated_for_language": validated,
                "probabilities_calibrated": outs["_calibrated"],
                "model": model, "model_display_name": DISPLAY.get(adapter.kind, model), "model_version": MODEL_VERSION,
                "inference_latency_ms": round(per_item_ms, 1),
                "device": getattr(adapter, "device", "cpu"),
                "safety": build_safety(risk_label, p_stress, validated["risk"]),
            })
        return results

    def analyze(self, text: str, model: str = "transformer", language_hint: str | None = None, explain: bool = False) -> dict:
        r = self.analyze_many([text], model, [language_hint])[0]
        if explain:
            adapter = self.registry.get(model)
            if hasattr(adapter, "model"):  # neural models only (baseline has no embedding layer)
                cleaned = clean_text(text)[:MAX_CHARS]
                r["explanation"] = {
                    "risk": integrated_gradients(adapter.model, adapter.tok, cleaned, "risk", RISK_LABELS.index("stress"), device=adapter.device),
                    "sentiment": integrated_gradients(adapter.model, adapter.tok, cleaned, "sentiment", device=adapter.device),
                }
            else:
                r["explanation"] = None
        return r
