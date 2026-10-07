import pytest

from src.langid import _LATIN_CLF, detect_language, detect_script


def test_script_detection():
    assert detect_script("hello world")[0] == "Latn"
    assert detect_script("यह एक वाक्य है")[0] == "Deva"
    assert detect_script("இது ஒரு வாக்கியம்")[0] == "Taml"
    assert detect_script("12345 !!!")[0] == "Zyyy"


def test_devanagari_hindi():
    r = detect_language("यह एक परीक्षण वाक्य है और मैं अच्छा हूँ")
    assert (r["language"], r["script"], r["language_tag"]) == ("hi", "Deva", "hi-Deva") and r["evaluated"]


def test_other_indic_script_flagged_unevaluated():
    r = detect_language("இது ஒரு சோதனை வாக்கியம்")
    assert r["language"] == "ta" and r["script"] == "Taml" and r["evaluated"] is False


def test_english_and_empty():
    assert detect_language("The weather is lovely and I am going for a long walk today")["language_tag"] == "en-Latn"
    assert detect_language("!!! 123")["language"] == "und"


@pytest.mark.skipif(not _LATIN_CLF.exists(), reason="Romanized-Hindi classifier not trained (scripts/train_langid.py)")
def test_romanized_hindi_detected():
    assert detect_language("Yaar aaj mera mood bilkul off hai, kuch karne ka mann nahi hai")["language_tag"] == "hi-Latn"
