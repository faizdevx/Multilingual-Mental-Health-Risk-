import pytest

from src.integrations import sarvam


class FakeResp:
    def raise_for_status(self): pass
    def json(self): return {"transcript": "hello there", "request_id": "r"}


class FakeClient:
    def __init__(self): self.calls = []
    def post(self, url, **kw): self.calls.append((url, kw)); return FakeResp()


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert sarvam.is_configured() is False
    with pytest.raises(sarvam.SarvamNotConfigured):
        sarvam.transcribe(b"x")


def test_transcribe_with_mocked_http(monkeypatch):
    # MOCKED ONLY: this does not exercise the live Sarvam service.
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    c = FakeClient()
    assert sarvam.transcribe(b"audio", language_code="hi-IN", client=c) == "hello there"
    url, kw = c.calls[0]
    assert url == sarvam.STT_URL and kw["headers"] == {"api-subscription-key": "test-key"} and kw["data"] == {"language_code": "hi-IN"}


def test_analyze_audio_pipeline(monkeypatch, predictor):
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    out = sarvam.analyze_audio(b"a", predictor, "bilstm", client=FakeClient())
    assert out["transcript"] == "hello there" and "risk_probability" in out["analysis"]


def test_no_hardcoded_key_in_source():
    import inspect
    src = inspect.getsource(sarvam)
    assert "SARVAM_API_KEY" in src and "sk_" not in src
