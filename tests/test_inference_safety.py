import pytest

from src.safety import CRISIS_RESOURCES, HIGH_RISK_LABELS, build_safety, risk_signal_text


def test_output_schema_and_ranges(predictor):
    r = predictor.analyze("I went to the market today and it was fine.", "bilstm")
    for k in ("language", "script", "sentiment", "emotion", "risk", "risk_signal", "risk_probability", "model", "inference_latency_ms", "safety"):
        assert k in r
    assert 0.0 <= r["risk_probability"] <= 1.0
    assert abs(sum(p["probability"] for p in r["risk_predictions"]) - 1) < 1e-3
    assert abs(sum(r["sentiment_probabilities"].values()) - 1) < 1e-3
    assert r["probabilities_calibrated"] is True  # calibration.json fixture was loaded and applied


def test_probability_comes_from_model_not_label_mapping(predictor):
    probs = {predictor.analyze(f"sample text number {i} about the weather and work", "transformer")["risk_probability"] for i in range(6)}
    assert len(probs) > 1  # varies continuously with input, i.e. not a fixed per-label constant


def test_batch_matches_single(predictor):
    texts = ["the weather is nice", "yaar aaj mood off hai", "यह एक परीक्षण वाक्य है"]
    many = predictor.analyze_many(texts, "bilstm")
    single = [predictor.analyze(t, "bilstm") for t in texts]
    assert [m["risk_probability"] for m in many] == pytest.approx([s["risk_probability"] for s in single], abs=1e-4)
    assert [m["language_tag"] for m in many][2] == "hi-Deva"


def test_unvalidated_language_is_flagged(predictor):
    r = predictor.analyze("यह एक परीक्षण वाक्य है और मैं ठीक हूँ", "bilstm")
    assert r["validated_for_language"]["risk"] is False and r["safety"]["validation_note"]
    assert predictor.analyze("the weather is lovely today and I feel fine", "bilstm")["validated_for_language"]["risk"] is True


def test_language_hint_overrides_detection(predictor):
    assert predictor.analyze("the weather is nice", "bilstm", language_hint="hi-Latn")["language_tag"] == "hi-Latn"


def test_explanation_attribution(predictor):
    for m in ("bilstm", "transformer"):
        r = predictor.analyze("i am so worried about work today", m, explain=True)
        ex = r["explanation"]["risk"]
        assert ex["method"] == "integrated_gradients" and ex["tokens"] and "not clinical reasoning" in ex["caveat"].lower()
        assert all(-1.0001 <= t["attribution"] <= 1.0001 for t in ex["tokens"])


def test_safety_wording_never_diagnoses():
    s = build_safety("stress", 0.9, True)
    assert s["level"] == "elevated_distress_language" and "not a medical diagnostic system" in s["notice"]
    assert s["show_resources"] and all(r["url"].startswith("https://") for r in s["resources"])
    blob = str(s).lower()
    assert "you have" not in blob and "diagnos" in blob
    quiet = build_safety("non_stress", 0.1, True)
    assert quiet["notice"] is None and quiet["resources"] == []
    assert "not a medical diagnostic tool" in quiet["disclaimer"]
    assert "stress" in risk_signal_text("stress").lower()


def test_no_fabricated_phone_numbers_and_no_high_risk_label():
    import json, re
    assert not re.search(r"\d{3}[\s-]?\d{3}[\s-]?\d{3,4}", json.dumps(CRISIS_RESOURCES))
    assert HIGH_RISK_LABELS == frozenset()  # no dataset here defines a high-risk class


def test_high_risk_notice_wired_for_future_datasets(monkeypatch):
    monkeypatch.setattr("src.safety.HIGH_RISK_LABELS", frozenset({"x"}))
    s = build_safety("x", 0.9, True)
    assert s["level"] == "high_risk_language" and "not a clinical assessment" in s["notice"]
